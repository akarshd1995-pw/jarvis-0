"""Jarvis-0 daemon: wires collectors, brain, executor, reminders and the UI together."""
from __future__ import annotations

import re
import argparse
import sys
import os
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
  /doctor         health check: tests every part of Jarvis and repairs what it can
  /perf           where the time and memory go: start-up, each step of a command, models in memory
  /words          your nicknames, typos, projects and folders I know; /words refresh looks again
  /setup          what I found on this PC: folders (OneDrive, moved), apps, browser, monitors, battery
  /review         go through the commands I wasn't sure about (also: "review my commands")
  /usage          how sure I've been this week; /usage forget wipes the saved commands
  /backup         back up your settings, history, corrections and models now
  /models         model versions · /rollback [vNNN] go back to an earlier one
  /good           confirm the last result was right (also saved as training data)
  /wrong          flag the last command as wrong, to label later
  /status         model backend, accuracy, autostart and encryption status
  /settings       open the settings window
  /update         get the latest version: it's tested first (tests + self-test) and only then switched to;
                  if it can't start, I go back by myself. /update force skips the tests.
  /quit           close Jarvis completely (same as tray > Quit Jarvis)
  /selftest       run every action in a throwaway folder to check everything works
  /hud            show or hide the HUD overlay
  /help           this message"""


class Jarvis:
    def __init__(self, config: Config, confirm, data_root: Path | None = None, background_models: bool = False):
        self.config = config
        self.store = Store(config.db_path, encrypt=config.encrypt_logs)
        from . import onnxutil
        from .lazy import Lazy
        from .perf import PERF
        onnxutil.USE_INT8 = config.use_int8
        self.brain = Brain(PROJECT_ROOT / config.model_checkpoint, PROJECT_ROOT / config.tokenizer_path,
                           min_confidence=config.min_confidence, prefer=config.model_backend,
                           budget_s=config.command_budget_ms / 1000 if config.command_budget_ms else None,
                           background=background_models)
        self.hooks: dict = {}   # UI callbacks: settings, restart
        self.persona = Persona(config.persona_name, config.persona_address, config.persona_tone,
                               model=Lazy(self._load_reply_model, "reply model") if config.persona_model else None)
        self.executor = Executor(config, self.store, confirm, describe=self.persona.confirm_question)
        # your corrections etc.: in the project folder, or (installed app, read-only folder) in your data folder
        self.data_root = data_root or (config.data_dir / "data" if getattr(sys, "frozen", False)
                                       else PROJECT_ROOT / "data")
        self.last_text: str | None = None
        self.monitor = None
        self.sampler = None
        self.last_results: list[dict] = []
        self.last_program: str | None = None
        from .executor import REGISTRY
        REGISTRY["briefing"] = self._briefing_action
        REGISTRY["maintenance"] = self._maintenance_action
        self.watchdog = None
        try:                                           # phase 40: finish or roll back interrupted file steps
            self.recovered = self.executor.recover()
        except Exception as e:
            log.warning("recovery check failed: %s", e)
            self.recovered = {}
        from .usage import UsageLog                    # phase 39: learn from how you actually talk
        self.usage = UsageLog(config.db_path, encrypt=config.encrypt_logs, enabled=config.learn_from_usage,
                              corrections=self.data_root / "user_corrections.jsonl")
        try:
            self.usage.prune()
        except Exception as e:
            log.warning("usage prune failed: %s", e)
        REGISTRY["training"] = self._training_action
        from .actions.control import LAST_EXTERNAL
        from .context import ScreenContext           # phase 38: what "this", "these" and "here" mean right now
        self.screen = ScreenContext(hwnd=lambda: LAST_EXTERNAL.get("hwnd"))
        self.executor.screen = self.screen
        REGISTRY["open_result"] = self._open_result
        from . import focus as F                      # phase 32: focus & gaming mode
        self.focus = F.Engine(log=lambda kind, data: self.store.log_event(kind, data),
                              sites=list(config.focus_block_sites), apps=list(config.focus_block_apps))
        F.CURRENT = self.focus
        from .vocab import Vocab                       # phase 43: your words
        self.vocab = Vocab.load(config.data_dir)
        from . import vocab as _V
        _V.CURRENT = self.vocab
        config.learned_places = self.vocab.places()
        REGISTRY["vocab"] = self._vocab_action
        from . import wellbeing as W                   # phase 34: breaks, screen time, wind-down
        self.wellbeing = W.Wellbeing(config.data_dir / "screentime.json", limits=config.screen_limits,
                                     eyes=config.wellbeing_eyes, stretch=config.wellbeing_stretch,
                                     winddown=config.wellbeing_winddown, enforce=config.screen_limits_enforce)
        W.CURRENT = (self.wellbeing, config)
        REGISTRY["screen_time"] = self._screen_time_action
        REGISTRY["breaks"] = self._breaks_action
        REGISTRY["focus"] = self._focus_action
        REGISTRY["gaming"] = self._gaming_action
        from .dialog import Conversation, DialogModel  # phase 6: follow-ups and asking back
        dialog = Lazy(lambda: DialogModel.load(PROJECT_ROOT / "checkpoints"), "dialog model") \
            if config.dialog_model and (PROJECT_ROOT / "checkpoints" / "dialog_encoder.onnx").exists() else None
        self.conv = Conversation(dialog, clarify=config.clarify)
        PERF.mark("jarvis ready")
        self.briefings = briefing.Scheduler(config.data_dir / "briefing_state.json", config.briefing_morning,
                                            config.briefing_evening, enabled=config.briefing_auto)

    # ------------------------------------------------------------ phase 44: this PC
    def setup_check(self, force: bool = False) -> str:
        """The first-run check (also after Jarvis's data moves to another PC). Returns the summary ('' if not due)."""
        from . import apps, machine
        if not force and not machine.needs_setup(self.config.data_dir):
            return ""
        idx = apps.get_index(self.config.data_dir) if sys.platform == "win32" else None
        if idx is not None and force:
            idx.refresh()
        report = machine.scan(self.config, idx)
        machine.save_report(self.config.data_dir, report)
        log.info("setup check: %s", report.get("notes"))
        return machine.summary(report)

    def recovery_message(self) -> str:
        r = self.recovered or {}
        fin, back, miss = r.get("finished", []), r.get("rolled_back", []), r.get("missing", [])
        if not (fin or back or miss):
            return ""
        parts = []
        if fin:
            parts.append(f"finished {len(fin)} file move{'s' * (len(fin) != 1)}")
        if back:
            parts.append(f"rolled back {len(back)} that hadn't completed (the originals are where they were)")
        if miss:
            parts.append(f"couldn't find {len(miss)}: " + ", ".join(miss[:3]))
        return "I was interrupted in the middle of a file operation last time, so I " + "; ".join(parts) + "."

    def take_results(self) -> list[dict]:
        out, self.last_results = self.last_results, []
        return out

    # ------------------------------------------------------------ memory index (phase 7)
    def memory_roots(self) -> list:
        roots = []
        for name in self.config.memory_folders:
            try:
                p = self.config.resolve_folder(name)
            except Exception:
                continue
            if p.is_dir() and p not in roots:
                roots.append(p)
        return roots

    def start_indexer(self, every_min: float = 20) -> threading.Thread:
        """Keep the memory index fresh: files every 20 minutes (only new/changed ones are read), browser history too."""
        from .memory import browser, index as MI

        def loop():
            idx = MI.get(self.config.data_dir)
            while True:
                if self.config.memory_enabled:
                    try:
                        st = idx.crawl(self.memory_roots())
                        if self.config.memory_browser:
                            idx.add_history(browser.history())
                        from . import machine
                        if machine.heavy_work_ok(self.config) and not getattr(self, "busy", False):
                            self._ocr_pass(idx)                  # phase 44/41: not on battery, not while gaming
                            self._embed_pass(idx)
                        self._release_heavy_models()             # phase 41: give the memory back until next time
                        log.info("memory index: %s (%s)", idx.count(), st)
                    except Exception as e:
                        log.warning("memory indexing failed: %s", e)
                time.sleep(every_min * 60)

        t = threading.Thread(target=loop, daemon=True, name="indexer")
        t.start()
        return t

    def start_yield(self, notify=None, every_s: float = 15) -> threading.Thread:
        """Phase 41: while a full-screen game runs (or the laptop is on battery), pause training runs and skip
        OCR / meaning passes so the game gets the GPU and CPU. Only pauses it made itself are lifted again;
        a pause you asked for stays."""
        from . import machine
        from .actions.training import PAUSE_FLAG, jobs
        AUTO = "auto: "

        def mine() -> bool:
            try:
                return PAUSE_FLAG.read_text(encoding="utf-8").startswith(AUTO)
            except OSError:
                return False

        def loop():
            while True:
                try:
                    why = None
                    if self.config.yield_when_gaming:
                        why = {"game": "a game is running", "fullscreen": "something is running full screen"}.get(
                            machine.fullscreen_busy() or "")
                    if why is None and not machine.heavy_work_ok(self.config):
                        why = "the laptop is on battery"
                    self.busy = why is not None
                    training = any(j["state"] == "running" for j in jobs())
                    if why and training and not PAUSE_FLAG.exists():
                        PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                        PAUSE_FLAG.write_text(AUTO + why + "\n", encoding="utf-8")
                        log.info("paused training: %s", why)
                        if notify:
                            notify(f"I paused training because {why}. It carries on by itself afterwards.")
                    elif not why and mine():
                        PAUSE_FLAG.unlink(missing_ok=True)
                        log.info("resumed training")
                except Exception as e:
                    log.warning("yield check failed: %s", e)
                time.sleep(every_s)

        t = threading.Thread(target=loop, daemon=True, name="yield")
        t.start()
        return t

    # ------------------------------------------------------------ phase 34: wellbeing
    def _screen_time_action(self, args, ctx, dry_run: bool):
        from .briefing import focus_minutes
        from .wellbeing import day_key
        act = str(args["action"])
        w, cfg = self.wellbeing, self.config
        if act == "report":
            now = time.time()
            start = datetime_day_start(now)
            per_app = focus_minutes(self.store.events_since(start), now)
            total = sum(per_app.values()) / 60
            top = sorted(per_app.items(), key=lambda x: -x[1])[:4]
            used = w.usage_today()
            limits = {a: {"limit": m, "used": round(used.get(a, 0))} for a, m in cfg.screen_limits.items()}
            app = str(args.get("app", "")).strip().lower()
            return ActionResult("screen time", kind="screen_time", data={
                "action": "report", "total_min": round(total), "top": [(a, round(s / 60)) for a, s in top],
                "limits": limits, "app": app, "app_used": round(used.get(app, 0)) if app else None,
                "app_minutes": round(per_app.get(app, 0) / 60) if app else None})
        app = str(args.get("app", "")).strip().lower()
        if not app:
            raise ValueError("which app? e.g. \"limit youtube to an hour a day\"")
        before = cfg.screen_limits.get(app)
        if act == "limit":
            minutes = int(args.get("minutes") or 60)
            if not dry_run:
                cfg.screen_limits[app] = minutes
                w.limits = cfg.screen_limits
                cfg.save()
                ctx.store.record(ctx.batch_id, "limit", {"op": "limit_set", "app": app, "minutes": before})
            return ActionResult(f"limit {app} {minutes}", kind="screen_time",
                                data={"action": "limit", "app": app, "minutes": minutes, "before": before,
                                      "used": round(w.usage_today().get(app, 0)), "enforce": cfg.screen_limits_enforce})
        if not dry_run and before is not None:
            cfg.screen_limits.pop(app, None)
            w.limits = cfg.screen_limits
            cfg.save()
            ctx.store.record(ctx.batch_id, "limit", {"op": "limit_set", "app": app, "minutes": before})
        return ActionResult(f"unlimit {app}", kind="screen_time", data={"action": "unlimit", "app": app,
                                                                        "had": before is not None})

    def _breaks_action(self, args, ctx, dry_run: bool):
        act = str(args["action"])
        kind = str(args.get("kind", "all"))
        w, cfg = self.wellbeing, self.config
        if act == "snooze":
            minutes = int(args.get("minutes") or 60)
            if not dry_run:
                w.snooze(minutes)
            return ActionResult("breaks snoozed", kind="breaks", data={"action": "snooze", "minutes": minutes})
        if act in ("on", "off") and not dry_run:
            on = act == "on"
            if kind in ("eyes", "all"):
                w.eyes = cfg.wellbeing_eyes = on
            if kind in ("stretch", "all"):
                w.stretch = cfg.wellbeing_stretch = on
            cfg.save()
        return ActionResult("breaks", kind="breaks", data={"action": act, "kind": kind, "eyes": w.eyes,
                                                           "stretch": w.stretch,
                                                           "active_min": round(w.active_s / 60)})

    def start_wellbeing(self, say, every_s: float = 5.0) -> threading.Thread:
        """Every few seconds: how long you've been at the screen, what's in front, and your usual hours."""
        from . import machine, winlayout
        from .wellbeing import usual_end
        cache = {"usual": None, "at": 0.0, "saved": time.time()}

        def loop():
            while True:
                try:
                    now = time.time()
                    if now - cache["at"] > 3600:
                        ts = [t for t, _, _ in self.store.events_since(now - 15 * 86400)]
                        cache["usual"], cache["at"] = usual_end(ts, now), now
                    w, c = self.wellbeing, self.config                   # Settings changes apply right away
                    w.limits, w.enforce = c.screen_limits, c.screen_limits_enforce
                    w.eyes, w.stretch, w.winddown = c.wellbeing_eyes, c.wellbeing_stretch, c.wellbeing_winddown
                    fg = winlayout.foreground()
                    quiet = (self.focus.session is not None and self.focus.session.phase == "work") or \
                        self.focus.gaming is not None
                    events = self.wellbeing.tick(machine.idle_seconds(), fg.title if fg else "", fg.exe if fg else "",
                                                 quiet=quiet, usual_end=cache["usual"])
                    for ev, d in events:
                        if ev == "minimize" and fg is not None:
                            winlayout.minimize(fg.hwnd)
                            continue
                        say(ev, self.persona.say(*self._wellbeing_words(ev, d)))
                    if now - cache["saved"] > 60:
                        self.wellbeing.save()
                        cache["saved"] = now
                except Exception as e:
                    log.warning("wellbeing: %s", e)
                time.sleep(every_s)

        t = threading.Thread(target=loop, daemon=True, name="wellbeing")
        t.start()
        return t

    @staticmethod
    def _wellbeing_words(ev: str, d: dict) -> tuple[str, dict]:
        if ev == "eyes":
            return "well_eyes", {}
        if ev == "stretch":
            return "well_stretch", {"minutes": d["minutes"]}
        if ev == "limit_warn":
            return "well_limit_warn", {"app": d["app"].title(), "left": d["left"]}
        if ev == "limit_hit":
            return "well_limit_hit", {"app": d["app"].title(), "minutes": d["minutes"]}
        return "well_winddown", {"now": d["now"], "usual": d["usual"]}

    # ------------------------------------------------------------ phase 43: your words
    def _vocab_action(self, args, ctx, dry_run: bool):
        act = str(args["action"])
        v = self.vocab
        if act == "list":
            d = v.data
            return ActionResult("your words", kind="vocab", data={
                "action": "list", "aliases": dict(d.get("aliases") or {}), "typos": dict(d.get("typos") or {}),
                "apps": len(d.get("apps") or []), "projects": sorted((d.get("projects") or {}))[:12],
                "folders": len(d.get("folders") or {})})
        word = str(args.get("word", ""))
        if act == "teach":
            means = str(args.get("means", ""))
            if dry_run:
                return ActionResult(f"{word} = {means}", kind="vocab", data={"action": "teach", "word": word,
                                                                            "means": means, "dry": True})
            w, m, before = v.teach(word, means)
            ctx.store.record(ctx.batch_id, "vocab", {"op": "vocab_set", "word": w, "means": before})
            return ActionResult(f"{w} = {m}", kind="vocab", data={"action": "teach", "word": w, "means": m,
                                                                 "replaced": before})
        before = None if dry_run else v.forget(word)
        if before is not None:
            ctx.store.record(ctx.batch_id, "vocab", {"op": "vocab_set", "word": word.lower().strip(), "means": before})
        return ActionResult(f"forgot {word}", kind="vocab", data={"action": "forget", "word": word,
                                                                 "known": before is not None})

    def refresh_vocab(self, typos: bool = True) -> dict:
        """Look again at your apps, projects and folders (and typos in your past commands)."""
        known, texts = None, None
        if typos:
            try:
                from .vocab import known_words
                known = known_words(8000)
                texts = [r["text"] for r in self.usage.real_texts(days=120)]
            except Exception as e:
                log.info("typo learning skipped: %s", e)
        stats = self.vocab.refresh(self.config, known, texts)
        self.config.learned_places = self.vocab.places()
        return stats

    # ------------------------------------------------------------ phase 32: focus & gaming
    def _focus_action(self, args, ctx, dry_run: bool):
        from . import focus as F
        act = str(args["action"])
        e = self.focus
        if act == "status":
            return ActionResult("focus status", kind="focus", data={"action": "status", "status": e.status(),
                                                                     "gaming": e.gaming is not None})
        if act == "stop":
            summary = None if dry_run else e.stop("stopped")
            return ActionResult("focus stopped", kind="focus", data={"action": "stop", "summary": summary})
        minutes = int(args.get("minutes") or self.config.focus_minutes)
        mode = str(args.get("mode", "single"))
        block = str(args.get("block")) == "yes" if args.get("block") is not None else self.config.focus_block
        total = str(args.get("quiet", "normal")) == "total"
        data = {"action": "start", "minutes": minutes, "mode": mode, "block": block, "total": total,
                "blocking": (e.sites + [a[:-4] for a in e.apps]) if block else []}
        if dry_run:
            return ActionResult(f"Focus for {minutes} min" + (f", pausing {', '.join(data['blocking'][:6])}"
                                                              if block else ""), kind="focus", data={**data, "dry": True})
        e.sites, e.apps = list(self.config.focus_block_sites), list(self.config.focus_block_apps)  # latest Settings
        s = e.start(minutes, mode, block, total)
        ctx.store.record(ctx.batch_id, "focus", {"op": "focus_stop"})        # "undo" ends it (and the blocking)
        data["until"] = s.phase_end
        self._focus_hud()
        return ActionResult(f"focus {minutes} min", kind="focus", data=data)

    def _gaming_action(self, args, ctx, dry_run: bool):
        on = str(args["action"]) == "on"
        if dry_run:
            return ActionResult("gaming", kind="gaming", data={"action": "on" if on else "off", "dry": True})
        if on:
            self.focus.gaming_on(auto=False)
            if "hud_hide" in self.hooks:
                self.hooks["hud_hide"](True)
            return ActionResult("gaming on", kind="gaming", data={"action": "on"})
        summary = self.focus.gaming_off()
        if "hud_hide" in self.hooks:
            self.hooks["hud_hide"](False)
        return ActionResult("gaming off", kind="gaming", data={"action": "off", "summary": summary})

    def _focus_hud(self):
        if "focus_hud" in self.hooks:
            self.hooks["focus_hud"](self.focus.countdown())

    def notify_or_hold(self, kind: str, text: str) -> bool:
        """True: show it now. False: focus / gaming is on, it was kept for the summary afterwards."""
        return not self.focus.hold(kind, text)

    def start_focus_ticker(self, say, every_s: float = 1.0) -> threading.Thread:
        """Countdown, breaks, the end of a session, gentle blocking and automatic gaming mode."""
        from . import focus as F
        from . import machine, winlayout
        state = {"last_block": 0.0, "game_seen": 0.0}

        def loop():
            while True:
                try:
                    for ev, d in self.focus.tick():
                        say(self.persona.say(*self._focus_event(ev, d)))
                    if self.focus.session is not None or self.focus.gaming is not None:
                        self._focus_hud()
                    s = self.focus.session
                    if s is not None and s.block and s.phase == "work":
                        fg = winlayout.foreground()
                        name = self.focus.blocked(fg.title, fg.exe) if fg else None
                        if name:
                            winlayout.minimize(fg.hwnd)
                            if time.time() - state["last_block"] > 20:
                                state["last_block"] = time.time()
                                until = time.strftime("%H:%M", time.localtime(s.phase_end))
                                say(self.persona.say("focus_blocked", name=name, until=until))
                    if self.config.gaming_auto:                  # a full-screen game: gaming mode by itself
                        busy = machine.fullscreen_busy()
                        g = self.focus.gaming
                        if busy == "game" and g is None:
                            self.focus.gaming_on(auto=True)
                            if "hud_hide" in self.hooks:
                                self.hooks["hud_hide"](True)
                        elif busy != "game" and g is not None and g.auto:
                            summary = self.focus.gaming_off()
                            if "hud_hide" in self.hooks:
                                self.hooks["hud_hide"](False)
                            if summary and (summary["held"] or summary["minutes"] >= 10):
                                say(self.persona.say(*self._gaming_end(summary)))
                except Exception as e:
                    log.warning("focus ticker: %s", e)
                time.sleep(every_s)

        t = threading.Thread(target=loop, daemon=True, name="focus")
        t.start()
        return t

    def _focus_event(self, ev: str, d: dict) -> tuple[str, dict]:
        from .focus import held_summary
        held = held_summary(d.get("held") or [])
        extra = f" Meanwhile {held}." if held else ""
        if ev == "break":
            return ("focus_break_long" if d["long"] else "focus_break"), {"minutes": d["minutes"], "extra": extra}
        if ev == "work":
            return "focus_back", {"minutes": d["minutes"], "cycle": d["cycle"]}
        return "focus_done", {"minutes": d["minutes"], "extra": extra}

    def _gaming_end(self, summary: dict) -> tuple[str, dict]:
        from .focus import held_summary
        held = held_summary(summary.get("held") or [])
        return "gaming_done", {"minutes": summary["minutes"], "extra": f" While you played, {held}." if held else ""}

    def _release_heavy_models(self) -> None:
        """OCR and the meaning model are only needed during an indexing pass (and a search): free them after."""
        from .perf import PERF
        if getattr(self, "_ocr", None) is not None:
            self._ocr = None
            PERF.model_released("OCR text finder")
            PERF.model_released("OCR reader")
        from .memory import meaning
        if time.time() - getattr(meaning, "last_used", 0) > 600 and meaning._loaded.get("emb") is not None:
            meaning._loaded.clear()
            PERF.model_released("meaning model")
        import gc
        gc.collect()

    def _ocr_pass(self, idx, limit: int = 150) -> int:
        """Read the text in up to `limit` new screenshots/images with our OCR models (if trained)."""
        if not self.config.memory_ocr:
            return 0
        if getattr(self, "_ocr", None) is None:
            from .memory.ocr import OCR
            try:
                self._ocr = OCR.load(PROJECT_ROOT / "checkpoints")
            except Exception as e:
                log.warning("OCR unavailable: %s", e)
                self._ocr = None
            if self._ocr is None:
                return 0
        done = 0
        for path in idx.needs_ocr(limit):
            try:
                text = self._ocr.read(path)
            except Exception:
                text = ""                        # unreadable image: don't try again until it changes
            idx.set_text(path, text)
            done += 1
        return done

    def _embed_pass(self, idx, limit: int = 3000) -> int:
        """Meaning vectors for new/changed items (if the meaning model is trained)."""
        if not self.config.memory_meaning:
            return 0
        from .memory.meaning import chunks, get_embedder
        emb = get_embedder(PROJECT_ROOT / "checkpoints")
        if emb is None:
            return 0
        done = 0
        for doc_id, title, body in idx.needs_embed(limit):
            try:
                idx.add_vectors(doc_id, emb.embed(chunks(title, body)))
            except Exception as e:
                log.warning("embedding failed for %s: %s", title, e)
                continue
            done += 1
            if done % 200 == 0:
                idx.commit()
        idx.commit()
        return done

    # ------------------------------------------------------------ briefings (phase 24)
    @property
    def job_logs(self) -> dict:
        ck = PROJECT_ROOT / "checkpoints"
        return {"Training command model": ck / "train_log.txt", "Training reply model": ck / "reply_log.txt",
                "Self-improvement": ck / "improve_log.txt", "Training OCR": ck / "ocr_log.txt",
                "Training meaning model": ck / "meaning_log.txt"}

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
            from . import machine
            if not machine.heavy_work_ok(self.config):
                return False                                     # battery saver
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
        facts.learning = self.learning_lines(now)
        if kind == "evening" and getattr(self, "wellbeing", None) is not None:
            used = self.wellbeing.usage_today()
            facts.limits = [(a, int(used.get(a, 0)), int(m)) for a, m in (self.config.screen_limits or {}).items()]
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
        if cmd == "/doctor":
            return self.health_report()
        if cmd == "/backup":
            from . import backup
            p = backup.create(self.config)
            return f"Backed up: {p}"
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
            changed, msg = update(progress=self.hooks.get("message", lambda s: None), data_dir=self.config.data_dir,
                                  test=arg.strip().lower() != "force")
            if changed and "restart" in self.hooks:
                self.hooks["restart"]()
                return msg + "\nRestarting..."
            return msg + ("\nRestart Jarvis to use the new version." if changed else "")
        if cmd in ("/quit", "/exit"):                  # close Jarvis completely (the supervisor won't restart it)
            if "quit" in self.hooks:
                threading.Timer(0.8, self.hooks["quit"]).start()
                return "Shutting down. Start me again from the Start menu or run_jarvis.bat."
            return "Use the tray icon > Quit Jarvis."
        if cmd == "/selftest":
            from . import selftest
            return selftest.summary(selftest.run(log=lambda s: None))
        if cmd == "/words":
            if arg.strip() == "refresh":
                s = self.refresh_vocab()
                return (f"Looked again: {s['apps']} apps, {s['projects']} projects, {s['folders']} folders, "
                        f"{s['typos']} typos of yours.")
            from .actions.base import Ctx as _Ctx
            res = self._vocab_action({"action": "list"}, _Ctx(self.config, self.store, "words"), True)
            return self.persona.result(res)
        if cmd == "/perf":
            from .perf import PERF
            from . import onnxutil
            ck = PROJECT_ROOT / "checkpoints"
            int8 = [p.name.split(".")[0] for p in sorted(ck.glob("*.int8.onnx"))] if ck.exists() else []
            note = (f"\nint8 copies on disk: {', '.join(int8)}" if int8 else
                    "\nNo int8 copies yet: run quantize.bat once (smaller, faster models).")
            if not onnxutil.USE_INT8:
                note += " (off in settings)"
            return PERF.report() + note
        if cmd == "/setup":
            return self.setup_check(force=True)
        if cmd == "/review":
            return self._review_cmd(arg)
        if cmd == "/usage":
            return self._usage_cmd(arg)
        if self.last_text is None:
            return "No previous command to label yet."
        if cmd == "/fix":
            try:
                program = canonical(arg.strip())
            except DSLError as e:
                return f"That program isn't valid: {e}"
            self._append("user_corrections.jsonl", {"text": self.last_text, "program": program})
            self._safe(self.usage.mark_last, "fixed", program)
            return f'Saved. Next training run will learn: "{self.last_text}" -> {program}'
        if cmd == "/good":
            if not self.last_program:
                return "The last command wasn't understood, so there's nothing to confirm. Use /fix instead."
            self._append("user_corrections.jsonl", {"text": self.last_text, "program": self.last_program})
            self._safe(self.usage.mark_last, "right", self.last_program)
            return "Thanks, saved as a confirmed example."
        if cmd == "/wrong":
            self._append("needs_label.jsonl", {"text": self.last_text, "predicted": self.last_program})
            self._safe(self.usage.mark_last, "new")
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
        if rm:
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
        logs = self.job_logs
        sampler = MetricsSampler(logs)
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

    def read_as(self, text: str) -> str:
        """Phase 43: Indian scripts in Hinglish lettering, then your nicknames, before anything reads the command."""
        from .indic import to_latin
        self.vocab.reload_if_changed()
        text = to_latin(text)
        if re.match(r"^(?:forget|unlearn|stop using|when i say|learn that|remember that|jab main)\b|^\S+ means\b|.* isn'?t a nickname$",
                    text.strip(), re.I):
            return text                     # a command about a word: read it literally, not as its nickname
        return self.vocab.expand(text)

    def handle(self, text: str) -> str:
        if text.strip().startswith("/"):
            return self._meta(text.strip())
        text = self.read_as(text)
        from .perf import PERF
        self._hud("thinking")
        PERF.begin()
        self._source = ""
        try:
            reply = self._handle(text)
        except Exception:
            self._hud("error", hold=3)
            raise
        finally:
            rec = PERF.command_done(self._source)
            if rec and rec["total"] > 3:
                log.info("slow command (%.1fs): %s", rec["total"],
                         ", ".join(f"{k} {v:.2f}s" for k, v in rec["stages"].items()))
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

    def _converse(self, text: str) -> str | None:
        """A reply to what just happened ("no, to Desktop", "the second one", "yes"), or None for a fresh command."""
        c = self.conv.active()
        if c is None:
            return None
        from .perf import PERF
        with PERF.stage("follow-up"):
            program, source = self.conv.read(text)
        if program is None:
            return None
        self._source = source
        if program == "unknown()":
            if c.kind == "prev":
                return None                              # not a follow-up: a new command
            prog, how = self.brain.resolve(text)         # ignored the question and asked something else?
            self.conv.clear()
            conf = (getattr(self.brain, "last_detail", None) or {}).get("conf")
            sure = how in ("dsl", "rules") or how.startswith("rules (model not") or \
                (how.startswith("model") and conf is not None and conf >= 0.8)
            if sure and prog is not None and any(x.action not in ("unknown", "unsupported", "mood") for x in prog.calls):
                return None
            if c.usage_id is not None:
                self._safe(self.usage.skip, c.usage_id)
            reply = self.persona.say("clarify_cancel")
            self._hud("idle", reply, hold=2)
            return self._with_details(reply, source, None)
        try:
            prog = parse(program)
        except DSLError:
            return None
        if c.kind in ("ask", "confirm") and c.usage_id is not None:
            # your answer is the right reading of what you first said: a training example (phase 39)
            self._safe(self.usage.fix, c.usage_id, program)
        self.conv.clear()
        # an answer labels what you first said; a bare follow-up ("same for Desktop") means nothing on its own
        self.last_text, self.last_program = (c.text.strip() if c.kind != "prev" else None), program
        self.store.log_event("command", {"text_len": len(text), "source": source, "program": program})
        if any(x.action == "undo" for x in prog.calls):
            self._safe(self.usage.mark_undone)
        return self._execute(prog, source)

    @staticmethod
    def _file_row(path: str) -> dict:
        p = Path(path)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0
        return {"title": p.name, "path": str(p), "source": "files", "ext": p.suffix.lstrip(".").lower(),
                "mtime": mtime}

    def _open_result(self, args, ctx, dry_run: bool):
        """open_result(n=2): the second of the results I just showed you."""
        from .context import ContextError
        rows, n = self.screen.fresh_results(), int(args["n"])
        if not rows:                                # "open this" with a file selected in File Explorer
            try:
                files, _ = self.screen.this_files()
                rows = [self._file_row(f) for f in files]
            except ContextError:
                raise ContextError("I haven't shown you any results lately, and nothing is selected. "
                                   "Search or list files first") from None
        if not 1 <= n <= len(rows):
            raise ContextError(f"I only showed {len(rows)} result{'s' * (len(rows) != 1)}")
        row = rows[n - 1]
        if row.get("source") in ("note", "task"):      # "open the second one" after a list: read it out
            return ActionResult(row.get("text", row["title"]), kind="note_read", data={"text": row.get("text", "")})
        if not dry_run:
            if row.get("source") == "web":
                import webbrowser
                webbrowser.open(row["path"])
            else:
                if not os.path.exists(row["path"]):
                    raise FileNotFoundError(f"file not found: {row['path']}")
                os.startfile(row["path"])                       # noqa: Windows only
        return ActionResult(f"Opened {row['title']}", kind="open", data={"what": "url", "name": row["title"]})

    def _execute(self, prog, source: str) -> str:
        from .perf import PERF
        self._source = source
        with PERF.stage("act"):
            report = self.executor.run(prog)
        for r in report.results:                    # search / file list: rows the command bar can show and open
            rows = None
            if r.kind in ("memory", "notes", "tasks") and r.data.get("results"):
                rows = r.data["results"]
            elif r.kind == "list" and r.data.get("paths"):
                rows = [self._file_row(p) for p in r.data["paths"]]
            if rows:
                self.last_results = rows
                self.screen.set_results(rows)       # "open the second one", "move these to Archive"
        if not report.failures and not any(c.action == "undo" for c in prog.calls):
            self.conv.ran(str(prog))
        else:
            self.conv.clear()
        with PERF.stage("reply"):
            reply = self.persona.report(report)
        state = "error" if report.failures else ("idle" if report.declined and not report.results else "success")
        self._hud(state, reply, hold=2)
        return self._with_details(reply, source, prog, report)

    def _handle(self, text: str) -> str:
        from .perf import PERF
        followed = self._converse(text)
        if followed is not None:
            return followed
        with PERF.stage("understand"):
            prog, source = self.brain.resolve(text)
        self._source = source
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
            self._learn(text, prog, source)
            self._hud("success" if source == "smalltalk" else "error", reply, hold=2)
            return self._with_details(reply, source, prog)
        if any(c.action == "undo" for c in prog.calls):
            self._safe(self.usage.mark_undone)          # undoing right away = it probably misunderstood
        self._learn(text, prog, source)
        ask = self.conv.should_ask(str(prog), source, getattr(self.brain, "last_detail", None))
        if ask is not None:                         # torn between readings, or an unsure move: ask, don't guess
            from .nlu.explain import explain
            kind, options = ask
            self.conv.asking(kind, options, text, self.usage.last_id if self.config.learn_from_usage else None)
            if kind == "ask":
                opts = "  ".join(f"{i}. {explain(p)}" for i, p in enumerate(options, 1))
                reply = self.persona.say("clarify_ask", opts=opts)
            else:
                w = explain(options[0])
                reply = self.persona.say("clarify_confirm", what=w[:1].upper() + w[1:])
            self._hud("idle", reply, hold=3)
            return self._with_details(reply, source, prog)
        return self._execute(prog, source)

    @staticmethod
    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:                       # learning must never break a command
            log.warning("usage log: %s", e)
            return None

    def _learn(self, text: str, prog, source: str) -> None:
        self._safe(self.usage.record, text, str(prog) if prog is not None else None, source,
                   getattr(self.brain, "last_detail", None))

    # ------------------------------------------------------------ phase 39: weekly review
    def review_items(self, limit: int = 10):
        return self._safe(self.usage.to_review, limit) or []

    def review_parse(self, text: str) -> str | None:
        prog, _ = self.brain.resolve(text)
        return str(prog) if prog is not None and str(prog) != "unknown()" else None

    def _review_state(self) -> dict:
        try:
            return json.loads((self.config.data_dir / "review_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def review_done(self, summary: str = "") -> None:
        st = self._review_state()
        st["last"] = time.time()
        try:
            (self.config.data_dir / "review_state.json").write_text(json.dumps(st), encoding="utf-8")
        except OSError:
            pass

    def learning_lines(self, now: float | None = None) -> list[str]:
        """For the briefing: a weekly nudge to review, and a drift alert when I'm suddenly less sure."""
        from . import usage as U
        if not self.config.learn_from_usage:
            return []
        now = now or time.time()
        out = []
        d = self._safe(self.usage.drift, now)
        if d:
            out.append(U.drift_line(d))
        last = self._review_state().get("last", 0)
        if self.config.review_weekly and (d or now - last >= 6 * 86400):
            n = len(self.review_items())
            if n >= 3:
                out.append(U.review_line(n))
        return out

    def _training_action(self, args, ctx, dry_run: bool):
        from .actions import training as tr
        if str(args.get("action")) != "review":
            return tr.training(args, ctx, dry_run)
        items = self.review_items()
        if dry_run or not items:
            return ActionResult(f"{len(items)} to review", kind="training",
                                data={"action": "review", "n": len(items), "off": not self.config.learn_from_usage})
        opened = "review" in self.hooks
        if opened:
            self.hooks["review"]()
        return ActionResult(f"{len(items)} to review", kind="training",
                            data={"action": "review", "n": len(items), "opened": opened,
                                  "text": self._review_text(items) if not opened else ""})

    def _review_text(self, items) -> str:
        from .nlu.explain import explain
        rows = [f"{n}. \"{it.text}\" -> I did: {explain(it.program)}" for n, it in enumerate(items, 1)]
        return "\n".join(rows) + ("\n\n/review ok N  ·  /review fix N <clearer wording or program>  ·  /review skip N")

    def _review_cmd(self, arg: str) -> str:
        """Text-mode review (CLI): /review, /review ok 2, /review fix 2 play super machi song, /review skip 2."""
        items = self.review_items()
        parts = arg.split(maxsplit=2)
        if not parts:
            if not items:
                return "Nothing to review: I was sure about everything lately."
            return self._review_text(items)
        verb = parts[0].lower()
        try:
            it = items[int(parts[1]) - 1]
        except (IndexError, ValueError):
            return "Which one? e.g. /review ok 2"
        if verb in ("ok", "right", "good"):
            self.usage.confirm(it.id) if it.program else self.usage.fix(it.id, "unknown()")
            done = f'Confirmed: "{it.text}".'
        elif verb == "skip":
            self.usage.skip(it.id)
            done = f'Skipped "{it.text}".'
        elif verb == "fix" and len(parts) == 3:
            try:
                program = canonical(parts[2])
            except DSLError:
                program = self.review_parse(parts[2])
            if not program:
                return "I couldn't understand that either. Try other words, or give the program."
            self.usage.fix(it.id, program)
            from .nlu.explain import explain
            done = f'Saved: "{it.text}" means {explain(program)}.'
        else:
            return "Use /review ok N, /review fix N <what you meant>, or /review skip N."
        self.review_done()
        return done + " I'll learn it in the next improvement run."

    def _usage_cmd(self, arg: str) -> str:
        if arg.strip() == "forget":
            n = self.usage.forget()
            return f"Forgot {n} saved command{'s' * (n != 1)}. (Turn learning off in Settings > Privacy to stop saving.)"
        now = time.time()
        w, u = self.usage.stats(now - 7 * 86400, now), self.usage.stats(now - 35 * 86400, now - 7 * 86400)
        if not w["n"] and not u["n"]:
            return "No commands saved yet." if self.config.learn_from_usage else "Learning from your commands is off."
        def f(s):
            c = f", model {s['conf']:.0%} sure on average" if s["conf"] is not None else ""
            return f"{s['n']} commands, {s['unsure']:.0%} unsure{c}, {s['undone']} undone"
        return f"This week: {f(w)}.\nThe 4 weeks before: {f(u)}.\nTo review: {len(self.review_items())}."

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

    def restart_reminders(self, on_reminder) -> threading.Thread:
        t = threading.Thread(target=self._reminder_loop, args=(on_reminder,), daemon=True, name="reminders")
        t.start()
        return t

    def start_maintenance(self) -> threading.Thread:
        """Weekly backup of your Jarvis (when you're not using the PC)."""
        from . import backup

        def loop():
            while True:
                try:
                    idle = not self.store.events_since(time.time() - 1200, "command")
                    if idle and backup.due(self.config):
                        p = backup.create(self.config)
                        log.info("weekly backup saved: %s", p)
                except Exception as e:
                    log.warning("backup failed: %s", e)
                time.sleep(3600)

        t = threading.Thread(target=loop, daemon=True, name="backups")
        t.start()
        return t

    def health_report(self) -> str:
        from . import doctor
        wd = getattr(self, "watchdog", None)
        return doctor.summary(doctor.run(self.config, wd.status() if wd else None), self.config.persona_address)

    def _maintenance_action(self, args, ctx, dry_run: bool):
        action = str(args.get("action", "health_check"))
        if action == "backup":
            from . import backup
            if dry_run:
                return ActionResult("Would back up Jarvis", kind="report", data={"text": "I'd make a backup now."})
            p = backup.create(self.config)
            text = f"Backed up, {self.config.persona_address}: {p.name} in {p.parent} ({p.stat().st_size / 2**20:.0f} MB)."
            return ActionResult(text, kind="report", data={"text": text})
        text = self.health_report()
        return ActionResult(text.splitlines()[0], kind="report", data={"text": text})

    def _reminder_loop(self, on_reminder):
        while True:
            for _id, text in self.store.due_reminders(time.time()):
                on_reminder(self.persona.reminder_due(text))
            try:
                self.nudge_overdue(on_reminder)
            except Exception as e:
                log.warning("task nudge failed: %s", e)
            time.sleep(10)

    def nudge_overdue(self, say, now: float | None = None) -> int:
        """Phase 33: a task with a time just went past its deadline: say so once."""
        from . import dates as D
        from .actions.tasks import overdue_to_nudge
        now = time.time() if now is None else now
        due = overdue_to_nudge(self.store, now)
        for t in due:
            self.store.update_task(t["id"], nudged=now)
            short = " ".join(t["text"].split()[:3])
            say(self.persona.say("task_overdue", text=t["text"], due=D.nice(t["due"], True, now), short=short))
        return len(due)

    def hud_items(self, now: float | None = None) -> list[tuple]:
        """Reminders, then tasks due today or overdue, for the HUD's list."""
        from .actions.tasks import select
        now = time.time() if now is None else now
        items: list[tuple] = list(self.store.upcoming_reminders(now))
        for t in select(self.store, "today", now)[:3]:
            label = "overdue" if t["due"] < now else ("today" if not t["has_time"] else None)
            items.append((t["due"], "\u2610 " + t["text"], label))
        return items


def datetime_day_start(now: float) -> float:
    """5 am today (or yesterday, before 5 am): where a screen-time day begins."""
    from datetime import datetime, timedelta
    d = datetime.fromtimestamp(now)
    start = d.replace(hour=5, minute=0, second=0, microsecond=0)
    if d < start:
        start -= timedelta(days=1)
    return start.timestamp()


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
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)       # run by the supervisor
    ap.add_argument("--safe-mode", action="store_true", help="rules only, no HUD or background services")
    ap.add_argument("--no-supervisor", action="store_true", help="don't restart automatically after a crash")
    ap.add_argument("--selftest-exe", action="store_true", help=argparse.SUPPRESS)  # checks a built Jarvis.exe
    args = ap.parse_args()
    if args.selftest_exe:                              # build_installer.bat: does the packaged app work?
        import tempfile
        from . import selftest
        checks = selftest.run(log=lambda s: None)
        (Path(tempfile.gettempdir()) / "jarvis_selftest.txt").write_text(selftest.summary(checks), encoding="utf-8")
        sys.exit(0 if all(c.ok for c in checks) else 1)
    if args.autostart:
        from . import autostart
        print(autostart.enable() if args.autostart == "on" else autostart.disable())
        return
    config = Config.load(args.config)
    if not (args.child or args.cli or args.no_supervisor or os.environ.get("JARVIS_SUPERVISED")):
        from .supervisor import run as supervise                # keeps Jarvis running (phase 29)
        child = (["--config", str(args.config)] if args.config else []) + (["-v"] if args.verbose else [])
        sys.exit(supervise(child, config.data_dir))
    from .reliability import SingleInstance, Watchdog, setup_logging
    setup_logging(config.data_dir, args.verbose)
    if args.safe_mode:
        config.model_backend, config.persona_model, config.hud_enabled = "rules", False, False
        config.monitor_enabled = config.memory_enabled = config.briefing_auto = config.improve_auto = False
        log.warning("safe mode: rules only, no HUD or background services")
    watchdog = Watchdog()

    if args.cli:
        jarvis = Jarvis(config, _cli_confirm)
        jarvis.start_background(lambda text: print(f"\n[REMINDER] {text}\n> ", end=""))
        jarvis.start_monitor(lambda a, text: print(f"\n[{a.level.upper()}] {text}\n> ", end=""))
        jarvis.start_briefings(lambda kind, text: print(f"\n[{kind.upper()} BRIEFING] {text}\n> ", end=""))
        jarvis.start_improver(lambda text: print(f"\n[MODEL] {text}\n> ", end=""))
        jarvis.start_indexer()
        jarvis.start_maintenance()
        print("Jarvis-0 (cli). Ctrl+C to quit.")
        if jarvis.recovery_message():
            print(jarvis.recovery_message())
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

    single = SingleInstance()
    if not single.acquire():
        if single.ping_existing():
            log.info("Jarvis is already running: opened its command bar instead")
            return
        log.warning("port %s is taken by another program; starting anyway", single.port)
    bar_ref: dict = {}
    from .perf import PERF
    PERF.mark("settings loaded")
    jarvis = Jarvis(config, lambda *a: bar_ref["bar"].confirm(*a), background_models=True)
    jarvis.watchdog = watchdog
    bar = CommandBar(jarvis.handle)
    bar.saved_pos = tuple(config.bar_pos) if config.bar_pos else None        # where you left it
    bar.mini_pos = tuple(config.bar_mini_pos) if config.bar_mini_pos else None

    def remember_position(which, pos):
        if which == "bar":
            config.bar_pos = list(pos) if pos else None
        else:
            config.bar_mini_pos = list(pos) if pos else None
        try:
            config.save()
        except OSError as e:
            log.warning("couldn't save the bar position: %s", e)
    bar.on_position = remember_position
    bar_ref["bar"] = bar
    single.on_show = lambda: bar.post("open")
    jarvis.hooks["settings"] = lambda: bar.post("call", lambda: SettingsWindow(bar.root, config))
    jarvis.hooks["restart"] = lambda: bar.post("call", lambda: (bar.root.destroy(), restart()))
    jarvis.hooks["message"] = lambda text: bar.post("message", text)
    jarvis.hooks["quit"] = lambda: bar.post("quit")

    def open_review():
        from .ui.review import ReviewSession, ReviewWindow
        items = jarvis.review_items()
        if items:
            ReviewWindow(bar.root, ReviewSession(jarvis.usage, items, jarvis.review_parse), on_done=jarvis.review_done)
    jarvis.hooks["review"] = lambda: bar.post("call", open_review)
    from .actions.control import remember_foreground

    def on_hotkey():
        remember_foreground()          # the window you were in, for "snap this left" etc.
        bar.post("toggle")

    hk = HotkeyThread(config.hotkey, on_hotkey)
    hk.start()

    def tray_update():
        bar.post("message", "Checking for updates...")
        changed, msg = update(progress=lambda s: bar.post("message", s), data_dir=config.data_dir)
        bar.post("message", msg + ("\nRestarting..." if changed else ""))
        if changed:
            jarvis.hooks["restart"]()

    hud = None
    ui: dict = {"icon": None}

    def deliver_alert(a, text):
        """Monitoring alert -> HUD flash + last line, Windows notification (tray), command bar if no tray."""
        if a.level != "critical" and not jarvis.notify_or_hold("alert", text):
            return                                       # focus / gaming: kept for the summary afterwards
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

    sampler, _logs = jarvis.start_monitor(deliver_alert) if config.monitor_enabled else (None, {})
    if sampler is None:
        from .hud.metrics import MetricsSampler
        sampler = MetricsSampler({})
    if config.hud_enabled:
        from .hud.view import HUD
        try:
            hud = HUD(bar.root, sampler, name=config.persona_name,
                      reminders=jarvis.hud_items,
                      corner=config.hud_corner, opacity=config.hud_opacity,
                      click_through=config.hud_click_through)

            # commands run on a worker thread: every HUD change goes through the UI thread's queue
            jarvis.hooks["state"] = lambda state, hold=0.0: bar.post("call", lambda: hud.set_state(state, hold))
            jarvis.hooks["last"] = lambda text: bar.post("call", lambda: hud.set_last(text))
            jarvis.hooks["hud_toggle"] = lambda: bar.post("call", hud.toggle)
            jarvis.hooks["focus_hud"] = lambda text: bar.post("call", lambda: hud.set_focus(text))
            jarvis.hooks["hud_hide"] = lambda hide: bar.post("call", lambda: hud.game_hide(hide))
            HotkeyThread(config.hud_hotkey, lambda: bar.post("call", hud.toggle), hotkey_id=2).start()
        except Exception as e:     # never let the HUD stop Jarvis from starting
            log.warning("HUD unavailable: %s", e)
            hud = None

    def on_reminder(text):
        if not jarvis.notify_or_hold("reminder", text):  # only with "no interruptions at all"
            return
        bar.post("reminder", text)
        if hud is not None:
            bar.post("call", lambda: (hud.set_state("alert", 4), hud.set_last(text)))

    bg = jarvis.start_background(on_reminder)
    watchdog.register("reminders", bg[-1], lambda: jarvis.restart_reminders(on_reminder))

    def deliver_briefing(kind, text):
        """Full text in the command bar, the gist on the HUD and as a Windows notification."""
        if not jarvis.notify_or_hold("briefing", text):
            return
        bar.post("message", text)
        gist = briefing.short(text)
        if hud is not None:
            bar.post("call", lambda: (hud.set_state("success", 4), hud.set_last(gist)))
        if ui["icon"] is not None:
            try:
                ui["icon"].notify(gist, f"{config.persona_name}: {kind} briefing")
            except Exception:
                pass

    from .ui.command_bar import Toast
    def toast(title, text, seconds=10):
        """Jarvis's own notices: held back during focus / gaming, then summarised."""
        if jarvis.notify_or_hold("toast", f"{title}: {text}"):
            bar.post("call", lambda: Toast(bar.root, title, text, seconds=seconds))
    improver_msg = lambda text: toast("Model update", text, 30)  # noqa: E731
    if not args.safe_mode:
        watchdog.register("briefings", jarvis.start_briefings(deliver_briefing),
                          lambda: jarvis.start_briefings(deliver_briefing))
        watchdog.register("indexer", jarvis.start_indexer(), jarvis.start_indexer)
        watchdog.register("self-improvement", jarvis.start_improver(improver_msg),
                          lambda: jarvis.start_improver(improver_msg))
        watchdog.register("backups", jarvis.start_maintenance(), jarvis.start_maintenance)
        yield_msg = lambda text: toast("Training paused", text, 8)  # noqa: E731
        focus_say = lambda text: bar.post("message", text)  # noqa: E731

        def well_say(ev, text):                        # phase 34: a small, short notice; never the full bar
            if not jarvis.notify_or_hold("toast", text):
                return
            title = {"eyes": "Eye break", "stretch": "Stretch", "winddown": "Late night"}.get(ev, "Screen time")
            bar.post("call", lambda: Toast(bar.root, title, text, seconds=12))
            if hud is not None:
                bar.post("call", lambda: hud.set_last(text))
        watchdog.register("wellbeing", jarvis.start_wellbeing(well_say), lambda: jarvis.start_wellbeing(well_say))
        watchdog.register("focus", jarvis.start_focus_ticker(focus_say), lambda: jarvis.start_focus_ticker(focus_say))
        watchdog.register("yield", jarvis.start_yield(yield_msg), lambda: jarvis.start_yield(yield_msg))
    watchdog.start()
    bar.results_provider = jarvis.take_results

    icon = ui["icon"] = start_tray({
        "open": lambda: bar.post("open"),
        "hud": (lambda: bar.post("call", hud.toggle)) if hud else None,
        "settings": jarvis.hooks["settings"],
        "review": jarvis.hooks["review"],
        "update": lambda: threading.Thread(target=tray_update, daemon=True).start(),
        "quit": lambda: bar.post("quit"),
    })
    log.info("Jarvis-0 running (%s model). Press %s to open the command bar.", jarvis.brain.backend, config.hotkey)
    if hk.error:
        log.warning(hk.error)
    bar.show()
    PERF.mark("command bar ready")
    from . import release                              # phase 42: this version started fine -> trust it
    threading.Timer(release.CONFIRM_AFTER_S, lambda: release.confirm(config.data_dir)).start()
    rolled = release.message_once(config.data_dir)
    if rolled:
        bar.post("message", rolled)
    log.info("start-up: %s", ", ".join(f"{k} {v:.2f}s" for k, v in PERF.marks.items()))

    def warm_up():                                     # phase 41: load the reply model once you can already type
        time.sleep(3)
        if jarvis.persona.model is not None and hasattr(jarvis.persona.model, "warm"):
            jarvis.persona.model.warm()
    threading.Thread(target=warm_up, daemon=True, name="warm-up").start()
    if jarvis.recovery_message():
        bar.post("message", jarvis.recovery_message())

    def first_run():                                   # phase 44: a new PC (or Jarvis moved to one): look around
        try:
            if jarvis.vocab.stale():                   # phase 43: your apps, projects, folders, typos (weekly)
                log.info("your words: %s", jarvis.refresh_vocab())
        except Exception as e:
            log.warning("vocab refresh failed: %s", e)
        try:
            text = jarvis.setup_check()
        except Exception as e:
            log.warning("setup check failed: %s", e)
            return
        if text:
            bar.post("call", lambda: Toast(bar.root, "Set up for this PC", text + "\n\n/setup shows this again.",
                                           seconds=25))
    if not args.safe_mode:
        threading.Thread(target=first_run, daemon=True, name="setup-check").start()
    if args.safe_mode:
        bar.post("message", "I restarted in safe mode after crashing several times: rules only, no HUD or "
                            "background services. Type /doctor to check what's wrong; the next start is normal "
                            "once I've run without trouble for 15 minutes.")
    try:
        bar.mainloop()
    finally:
        single.release()
        if icon is not None:
            icon.stop()


if __name__ == "__main__":
    main()
