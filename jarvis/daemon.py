"""Jarvis-0 daemon: wires collectors, brain, executor, reminders and the UI together."""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

from .collectors.focus import FocusCollector
from .collectors.folders import FolderWatcher
from .config import Config
from .dsl.grammar import DSLError, canonical, parse
from .actions.base import ActionResult
from .executor import Executor
from .nlu.brain import Brain
from .persona import Persona, smalltalk
from .persona import mood
from . import briefing
from .safety import Tier
from .store import Store

log = logging.getLogger("jarvis")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


HELP = """Commands:
  /fix <program>  save the correct program for your last command (trains the next model)
  /improve        fine-tune on your corrections now (switches only if better)
  /models         model versions · /rollback [vNNN] go back to an earlier one
  /good           confirm the last result was right (also saved as training data)
  /wrong          flag the last command as wrong, to label later
  /status         model backend, accuracy, autostart and encryption status
  /settings       open the settings window
  /update         pull the latest version from GitHub and restart
  /hud            show or hide the HUD overlay
  /help           this message"""


class Jarvis:
    def __init__(self, config: Config, confirm, data_root: Path | None = None):
        self.config = config
        self.store = Store(config.db_path, encrypt=config.encrypt_logs)
        self.brain = Brain(PROJECT_ROOT / config.model_checkpoint, PROJECT_ROOT / config.tokenizer_path,
                           min_confidence=config.min_confidence, prefer=config.model_backend)
        self.hooks: dict = {}   # UI callbacks: settings, restart
        self.persona = Persona(config.persona_name, config.persona_address, config.persona_tone,
                               model=self._load_reply_model() if config.persona_model else None)
        self.executor = Executor(config, self.store, confirm, describe=self.persona.confirm_question)
        self.data_root = data_root or PROJECT_ROOT / "data"
        self.last_text: str | None = None
        self.monitor = None
        self.sampler = None
        self.last_program: str | None = None
        from .executor import REGISTRY
        REGISTRY["briefing"] = self._briefing_action
        self.briefings = briefing.Scheduler(config.data_dir / "briefing_state.json", config.briefing_morning,
                                            config.briefing_evening, enabled=config.briefing_auto)

    # ------------------------------------------------------------ briefings (phase 24)
    @property
    def job_logs(self) -> dict:
        ck = PROJECT_ROOT / "checkpoints"
        return {"Training command model": ck / "train_log.txt", "Training reply model": ck / "reply_log.txt",
                "Self-improvement": ck / "improve_log.txt"}

    # ------------------------------------------------------------ self-improving loop (phase 14)
    def start_improver(self, deliver) -> threading.Thread:
        """At night (improve_at, within 3 h), if you're away, the PC is plugged in, no training is running and
        there's something new to learn: fine-tune in the background. deliver(text) gets the outcome."""
        from datetime import datetime
        from .actions.training import start_improving
        state_path = self.config.data_dir / "improve_auto.json"

        def due(now: datetime) -> bool:
            if not self.config.improve_auto:
                return False
            try:
                h, m = (int(x) for x in self.config.improve_at.split(":"))
            except ValueError:
                h, m = 2, 0
            if (now.hour * 60 + now.minute - (h * 60 + m)) % 1440 >= 180:
                return False                                     # only within 3 h after improve_at
            try:
                if json.loads(state_path.read_text()).get("date") == now.strftime("%Y-%m-%d"):
                    return False
            except (OSError, ValueError):
                pass
            t = time.time()
            if self.store.events_since(t - 1200, "focus") or self.store.events_since(t - 1200, "command"):
                return False                                     # you're using the PC
            snap = self.sampler.snapshot if self.sampler else None
            if snap is not None and snap.battery is not None and not snap.charging:
                return False                                     # on battery
            ck = PROJECT_ROOT / "checkpoints"
            for log_name in ("train_log.txt", "reply_log.txt"):
                p = ck / log_name
                if p.exists() and t - p.stat().st_mtime < 600:
                    return False                                 # a training run is going
            import sys as _sys
            _sys.path.insert(0, str(PROJECT_ROOT))
            from training.loop import what_is_new
            return bool(what_is_new())

        def loop():
            while True:
                try:
                    now = datetime.now()
                    if due(now):
                        state_path.write_text(json.dumps({"date": now.strftime("%Y-%m-%d")}))
                        self.store.log_event("improve", {"auto": True})
                        proc = start_improving(force=False)
                        if proc is not None:
                            proc.wait()
                            text = self.improve_summary()
                            if text:
                                deliver(text)
                except Exception as e:
                    log.warning("self-improvement check failed: %s", e)
                time.sleep(300)

        t = threading.Thread(target=loop, daemon=True, name="improver")
        t.start()
        return t

    def improve_summary(self, max_age_h: float = 18) -> str:
        """One line about the last improvement run, for alerts and briefings."""
        rp = PROJECT_ROOT / "checkpoints" / "improve_report.json"
        try:
            r = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if time.time() - rp.stat().st_mtime > max_age_h * 3600:
            return ""
        return briefing.improve_line(r)

    def briefing_text(self, kind: str, now: float | None = None) -> str:
        try:
            downloads = self.config.resolve_folder("Downloads")
        except Exception:
            downloads = None
        facts = briefing.gather(kind, self.store, self.sampler.snapshot if self.sampler else None, self.job_logs,
                                PROJECT_ROOT / "checkpoints" / "eval_report.txt", downloads, now)
        return briefing.render(facts, self.config.persona_name, self.config.persona_address, self.config.persona_tone)

    def _briefing_action(self, args, ctx, dry_run: bool):
        from datetime import datetime
        kind = str(args.get("kind", "")) or ("morning" if datetime.now().hour < 15 else "evening")
        text = self.briefing_text(kind)
        if not dry_run:
            self.briefings.mark(kind, datetime.now())
            self.store.log_event("briefing", {"kind": kind, "auto": False})
        return ActionResult(briefing.short(text), kind="briefing", data={"text": text, "kind": kind})

    def start_briefings(self, deliver) -> threading.Thread:
        """deliver(kind, text): the morning one the first time you're at the PC after briefing_morning,
        the evening one after briefing_evening, each once a day."""
        from datetime import datetime

        def loop():
            while True:
                try:
                    now = time.time()
                    active = bool(self.store.events_since(now - 300, "focus") or self.store.events_since(now - 300, "command"))
                    kind = self.briefings.due(datetime.now(), active)
                    if kind:
                        self.briefings.mark(kind, datetime.now())
                        self.store.log_event("briefing", {"kind": kind, "auto": True})
                        deliver(kind, self.briefing_text(kind))
                except Exception as e:                 # a briefing must never take Jarvis down
                    log.warning("briefing failed: %s", e)
                time.sleep(60)

        t = threading.Thread(target=loop, daemon=True, name="briefings")
        t.start()
        return t

    @staticmethod
    def _load_reply_model():
        ck = PROJECT_ROOT / "checkpoints"
        enc, dec, tok = ck / "reply_encoder.onnx", ck / "reply_decoder.onnx", ck / "reply_tokenizer.json"
        if not (enc.exists() and dec.exists() and tok.exists()):
            return None
        try:
            from .nlg.generator import ReplyModel
            m = ReplyModel(enc, dec, tok)
            log.info("reply model loaded (ONNX)")
            return m
        except Exception as e:
            log.warning("reply model unavailable (%s); using templates", e)
            return None

    def _append(self, name: str, row: dict) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        with open(self.data_root / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _meta(self, text: str) -> str:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd == "/help":
            return HELP
        if cmd == "/status":
            return self.status()
        if cmd == "/hud":
            if "hud_toggle" in self.hooks:
                self.hooks["hud_toggle"]()
                return "Toggled the HUD."
            return "The HUD needs the command bar (not available in --cli mode)."
        if cmd == "/settings":
            if "settings" in self.hooks:
                self.hooks["settings"]()
                return "Opening settings."
            return f"Settings live in {self.config.data_dir / 'config.json'} (the window needs the command bar)."
        if cmd in ("/improve", "/rollback", "/models"):
            import sys as _sys
            _sys.path.insert(0, str(PROJECT_ROOT))
            if cmd == "/improve":
                from .actions.training import start_improving
                return ("Improving in the background (about 30 minutes). I'll switch only if the new model is better."
                        if start_improving(force=True) else "An improvement run is already going.")
            from training import loop as L
            if cmd == "/rollback":
                return L.rollback(arg.strip() or None)
            live = L.current_version()
            rows = [f"{'*' if v.name == live else ' '} {v.name}  {json.loads((v / 'report.json').read_text()).get('reason', '') if (v / 'report.json').exists() else 'starting point'}"
                    for v in L.versions()]
            return "\n".join(rows) if rows else "No model versions yet: the first improvement run creates them."
        if cmd == "/update":
            from .updater import update
            changed, msg = update()
            if changed and "restart" in self.hooks:
                self.hooks["restart"]()
                return msg + "\nRestarting..."
            return msg + ("\nRestart Jarvis to use the new version." if changed else "")
        if self.last_text is None:
            return "No previous command to label yet."
        if cmd == "/fix":
            try:
                program = canonical(arg.strip())
            except DSLError as e:
                return f"That program isn't valid: {e}"
            self._append("user_corrections.jsonl", {"text": self.last_text, "program": program})
            return f'Saved. Next training run will learn: "{self.last_text}" -> {program}'
        if cmd == "/good":
            if not self.last_program:
                return "The last command wasn't understood, so there's nothing to confirm. Use /fix instead."
            self._append("user_corrections.jsonl", {"text": self.last_text, "program": self.last_program})
            return "Thanks, saved as a confirmed example."
        if cmd == "/wrong":
            self._append("needs_label.jsonl", {"text": self.last_text, "predicted": self.last_program})
            return "Flagged. Use /fix <program> to give the right answer, or label it later in data/needs_label.jsonl."
        return f"Unknown command {cmd}. Type /help."

    def status(self) -> str:
        from . import autostart
        lines = [f"Model: {self.brain.backend}"]
        meta = PROJECT_ROOT / self.config.model_checkpoint
        onnx_meta = meta.parent / "intent_onnx.json"
        if onnx_meta.exists():
            m = json.loads(onnx_meta.read_text(encoding="utf-8")).get("metrics") or {}
            if m:
                lines.append(f"  validation exact {m.get('exact', 0):.2%}, unseen phrasing {m.get('val_ood_exact', 0):.2%}")
        k = getattr(self.brain.model, "knows", None) if self.brain.model is not None else None
        if k:
            lines.append(f"  trained on {len(k.get('actions', []))} commands + {', '.join(k.get('features', []))}")
        elif self.brain.model is not None:
            lines.append("  trained before Jarvis recorded what models learn: rules cover the newest commands"
                         " until you retrain")
        today = self.store.events_since(time.time() - 86400, "command")
        by_model = sum(1 for _, _, p in today if str(p.get("source", "")).startswith("model"))
        by_rules = sum(1 for _, _, p in today if str(p.get("source", "")).startswith("rules"))
        if by_model or by_rules:
            lines.append(f"  last 24 h: {by_model} commands understood by the model, {by_rules} by backup rules")
        rm = self.persona.model
        if rm is not None:
            st = rm.stats
            lines.append(f"Reply model: on ({st['generated']} generated, {st['rejected']} rejected by the fact check)")
        else:
            lines.append("Reply model: off (templates)")
        if self.monitor is not None:
            learned = sum(1 for st in self.monitor.baseline.stats.values() if st.n >= self.monitor.baseline.min_samples)
            lines.append(f"Monitoring: on ({learned} of {len(self.monitor.baseline.stats)} hourly baselines learned)")
        else:
            lines.append("Monitoring: off")
        lines.append(f"Confidence gate: {self.config.min_confidence:.2f}")
        import sys as _sys
        enc = "yes (Windows DPAPI)" if self.config.encrypt_logs and _sys.platform == "win32" else \
            ("no (DPAPI is Windows-only)" if self.config.encrypt_logs else "no (turned off in settings)")
        lines.append(f"Activity log encrypted: {enc}")
        lines.append(f"Autostart: {'on' if autostart.is_enabled() else 'off'}")
        return "\n".join(lines)

    def start_monitor(self, deliver) -> tuple:
        """Start the metrics sampler and the alert monitor. deliver(alert, text) shows the alert."""
        from .hud.metrics import MetricsSampler
        from .monitor.engine import Monitor, Thresholds
        ck = PROJECT_ROOT / "checkpoints"
        sampler = MetricsSampler(self.job_logs)
        sampler.start()
        self.sampler = sampler
        monitor = None
        if self.config.monitor_enabled:
            c = self.config
            th = Thresholds(gpu_temp=c.alert_gpu_temp, disk_free_gb=c.alert_disk_free_gb, battery=c.alert_battery,
                            cooldown_min=c.alert_cooldown_min)

            def on_alert(a):
                text = self.persona.alert(a)
                self.store.log_event("alert", {"key": a.key, "level": a.level})
                deliver(a, text)

            monitor = Monitor(sampler, logs, on_alert, th, baseline_path=c.data_dir / "baseline.json", store=self.store)
            monitor.start()
        self.monitor = monitor
        return sampler, logs

    def _hud(self, state: str | None = None, last: str | None = None, hold: float = 0.0):
        if state and "state" in self.hooks:
            self.hooks["state"](state, hold)
        if last and "last" in self.hooks:
            self.hooks["last"](last)

    def handle(self, text: str) -> str:
        if text.strip().startswith("/"):
            return self._meta(text.strip())
        self._hud("thinking")
        try:
            reply = self._handle(text)
        except Exception:
            self._hud("error", hold=3)
            raise
        return reply

    def _mood(self, text: str, prog=None) -> str | None:
        """How you feel -> a kind reply, and music that fits (now, or after you say yes).
        Fixed rules first (they alone handle crisis words), then the intent model's mood(...) reading."""
        offer, self._mood_offer = getattr(self, "_mood_offer", None), None
        if offer is not None and mood.YES.match(text.strip()):
            return self._play_mood(offer, empathy_first=False)
        m = mood.detect(text)
        if m is not None and m.kind != "crisis" and self.brain.knows("mood"):
            m = None                                   # the trained model decides; only crisis words stay fixed
        m = m or mood.from_program(prog, text)
        if m is None:
            return None
        self.store.log_event("mood", {"kind": m.kind})       # only the kind, never the text
        if m.kind == "crisis":
            reply = mood.crisis_reply(m, self.persona)
            self._hud("idle", reply, hold=4)
            return reply
        if m.wants_music:
            return self._play_mood(m)
        self._mood_offer = m
        reply = f"{mood.empathy(m, self.persona)} {mood.offer_line(m, self.persona)}"
        self._hud("idle", reply, hold=3)
        return self._with_details(reply, "mood", None)

    def _play_mood(self, m, empathy_first: bool = True) -> str:
        prog = parse(m.program)
        self.last_program = str(prog)
        report = self.executor.run(prog)
        if report.failures:
            reply = self.persona.report(report)
            self._hud("error", reply, hold=2)
        else:
            line = mood.playing_line(m, self.persona)
            reply = f"{mood.empathy(m, self.persona)} {line}" if empathy_first else line
            self._hud("success", reply, hold=2)
        return self._with_details(reply, "mood", prog, report)

    def _handle(self, text: str) -> str:
        prog, source = self.brain.resolve(text)
        # Feelings win only when the sentence isn't a real command ("i'm tired of these pdfs, delete them" stays a delete);
        # crisis words always win.
        is_command = prog is not None and any(c.action not in ("unknown", "unsupported", "play_media", "search_web", "mood")
                                              for c in prog.calls)
        if not is_command or mood.detect(text) is not None and mood.detect(text).kind == "crisis":
            moody = self._mood(text, prog)
            if moody is not None:
                return moody
            if prog is not None and any(c.action == "mood" for c in prog.calls):
                prog = None                                     # a mood call never reaches the executor
        else:
            self._mood_offer = None
        self.last_text, self.last_program = text.strip(), (str(prog) if prog else None)
        self.store.log_event("command", {"text_len": len(text), "source": source,
                                         "program": str(prog) if prog else None})
        if prog is None or str(prog) == "unknown()":
            reply = smalltalk(text, self.persona)
            if reply is not None:
                source = "smalltalk"
            else:
                reply = self.persona.result(ActionResult("", kind="unknown")) if prog else self.persona.unparsed()
            self._hud("success" if source == "smalltalk" else "error", reply, hold=2)
            return self._with_details(reply, source, prog)
        report = self.executor.run(prog)
        reply = self.persona.report(report)
        state = "error" if report.failures else ("idle" if report.declined and not report.results else "success")
        self._hud(state, reply, hold=2)
        return self._with_details(reply, source, prog, report)

    def _with_details(self, reply: str, source: str, prog, report=None) -> str:
        if not self.config.show_details:
            return reply
        detail = f"[{source}] {prog}" if prog is not None else f"[{source}]"
        detail += f"  · reply: {self.persona.last_source}"
        return f"{reply}\n\n{detail}"

    def start_background(self, on_reminder) -> list[threading.Thread]:
        threads: list[threading.Thread] = [
            FocusCollector(self.store, self.config.focus_poll_seconds, keep_titles=self.config.keep_window_titles),
            FolderWatcher(self.store, [self.config.resolve_folder(f) for f in self.config.watch_folders]),
            threading.Thread(target=self._reminder_loop, args=(on_reminder,), daemon=True, name="reminders"),
        ]
        for t in threads:
            t.start()
        return threads

    def _reminder_loop(self, on_reminder):
        while True:
            for _id, text in self.store.due_reminders(time.time()):
                on_reminder(self.persona.reminder_due(text))
            time.sleep(10)


def _cli_confirm(tier: Tier, summary: str, items: list[str]) -> bool:
    print(summary)
    for i in items[:15]:
        print("  -", i)
    if len(items) > 15:
        print(f"  ...and {len(items) - 15} more")
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(prog="jarvis")
    ap.add_argument("--cli", action="store_true", help="terminal mode instead of the command bar")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--autostart", choices=["on", "off"], help="start Jarvis when you log in to Windows")
    args = ap.parse_args()
    if args.autostart:
        from . import autostart
        print(autostart.enable() if args.autostart == "on" else autostart.disable())
        return
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = Config.load(args.config)

    if args.cli:
        jarvis = Jarvis(config, _cli_confirm)
        jarvis.start_background(lambda text: print(f"\n[REMINDER] {text}\n> ", end=""))
        jarvis.start_monitor(lambda a, text: print(f"\n[{a.level.upper()}] {text}\n> ", end=""))
        jarvis.start_briefings(lambda kind, text: print(f"\n[{kind.upper()} BRIEFING] {text}\n> ", end=""))
        jarvis.start_improver(lambda text: print(f"\n[MODEL] {text}\n> ", end=""))
        print("Jarvis-0 (cli). Ctrl+C to quit.")
        try:
            while True:
                print(jarvis.handle(input("> ")))
        except (KeyboardInterrupt, EOFError):
            return

    from .ui.command_bar import CommandBar
    from .ui.hotkey import HotkeyThread

    from .ui.settings import SettingsWindow
    from .ui.tray import start_tray
    from .updater import restart, update

    bar_ref: dict = {}
    jarvis = Jarvis(config, lambda *a: bar_ref["bar"].confirm(*a))
    bar = CommandBar(jarvis.handle)
    bar_ref["bar"] = bar
    jarvis.hooks["settings"] = lambda: bar.post("call", lambda: SettingsWindow(bar.root, config))
    jarvis.hooks["restart"] = lambda: bar.post("call", lambda: (bar.root.destroy(), restart()))
    from .actions.control import remember_foreground

    def on_hotkey():
        remember_foreground()          # the window you were in, for "snap this left" etc.
        bar.post("toggle")

    hk = HotkeyThread(config.hotkey, on_hotkey)
    hk.start()

    def tray_update():
        changed, msg = update()
        bar.post("message", msg + ("\nRestarting..." if changed else ""))
        if changed:
            jarvis.hooks["restart"]()

    hud = None
    ui: dict = {"icon": None}

    def deliver_alert(a, text):
        """Monitoring alert -> HUD flash + last line, Windows notification (tray), command bar if no tray."""
        if hud is not None:
            state = "alert" if a.level in ("critical", "warning") else "success"
            bar.post("call", lambda: (hud.set_state(state, 6), hud.set_last(text)))
        icon = ui["icon"]
        if icon is not None:
            try:
                icon.notify(text, f"{config.persona_name}")
            except Exception:
                pass
        elif a.level == "critical" or hud is None:
            bar.post("message", text)

    sampler, _logs = jarvis.start_monitor(deliver_alert)
    if config.hud_enabled:
        from .hud.view import HUD
        try:
            hud = HUD(bar.root, sampler, name=config.persona_name,
                      reminders=lambda: jarvis.store.upcoming_reminders(time.time()),
                      corner=config.hud_corner, opacity=config.hud_opacity,
                      click_through=config.hud_click_through)

            # commands run on a worker thread: every HUD change goes through the UI thread's queue
            jarvis.hooks["state"] = lambda state, hold=0.0: bar.post("call", lambda: hud.set_state(state, hold))
            jarvis.hooks["last"] = lambda text: bar.post("call", lambda: hud.set_last(text))
            jarvis.hooks["hud_toggle"] = lambda: bar.post("call", hud.toggle)
            HotkeyThread(config.hud_hotkey, lambda: bar.post("call", hud.toggle), hotkey_id=2).start()
        except Exception as e:     # never let the HUD stop Jarvis from starting
            log.warning("HUD unavailable: %s", e)
            hud = None

    def on_reminder(text):
        bar.post("reminder", text)
        if hud is not None:
            bar.post("call", lambda: (hud.set_state("alert", 4), hud.set_last(text)))

    jarvis.start_background(on_reminder)

    def deliver_briefing(kind, text):
        """Full text in the command bar, the gist on the HUD and as a Windows notification."""
        bar.post("message", text)
        gist = briefing.short(text)
        if hud is not None:
            bar.post("call", lambda: (hud.set_state("success", 4), hud.set_last(gist)))
        if ui["icon"] is not None:
            try:
                ui["icon"].notify(gist, f"{config.persona_name}: {kind} briefing")
            except Exception:
                pass

    jarvis.start_briefings(deliver_briefing)
    from .ui.command_bar import Toast
    jarvis.start_improver(lambda text: bar.post("call", lambda: Toast(bar.root, "Model update", text, seconds=30)))

    icon = ui["icon"] = start_tray({
        "open": lambda: bar.post("open"),
        "hud": (lambda: bar.post("call", hud.toggle)) if hud else None,
        "settings": jarvis.hooks["settings"],
        "update": lambda: threading.Thread(target=tray_update, daemon=True).start(),
        "quit": lambda: bar.post("quit"),
    })
    log.info("Jarvis-0 running (%s model). Press %s to open the command bar.", jarvis.brain.backend, config.hotkey)
    if hk.error:
        log.warning(hk.error)
    bar.show()
    try:
        bar.mainloop()
    finally:
        if icon is not None:
            icon.stop()


if __name__ == "__main__":
    main()
