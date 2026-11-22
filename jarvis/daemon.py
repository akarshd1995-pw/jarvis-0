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
from .dsl.grammar import DSLError, canonical
from .executor import Executor
from .nlu.brain import Brain
from .safety import Tier
from .store import Store

log = logging.getLogger("jarvis")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


HELP = """Commands:
  /fix <program>  save the correct program for your last command (trains the next model)
  /good           confirm the last result was right (also saved as training data)
  /wrong          flag the last command as wrong, to label later
  /help           this message"""


class Jarvis:
    def __init__(self, config: Config, confirm, data_root: Path | None = None):
        self.config = config
        self.store = Store(config.db_path)
        self.brain = Brain(PROJECT_ROOT / config.model_checkpoint, PROJECT_ROOT / config.tokenizer_path)
        self.executor = Executor(config, self.store, confirm)
        self.data_root = data_root or PROJECT_ROOT / "data"
        self.last_text: str | None = None
        self.last_program: str | None = None

    def _append(self, name: str, row: dict) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        with open(self.data_root / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _meta(self, text: str) -> str:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd == "/help":
            return HELP
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

    def handle(self, text: str) -> str:
        if text.strip().startswith("/"):
            return self._meta(text.strip())
        prog, source = self.brain.resolve(text)
        self.last_text, self.last_program = text.strip(), (str(prog) if prog else None)
        self.store.log_event("command", {"text_len": len(text), "source": source,
                                         "program": str(prog) if prog else None})
        if prog is None:
            return "Sorry, I couldn't understand that yet. (Tip: you can also type raw DSL.)"
        report = self.executor.run(prog)
        return f"[{source}] {prog}\n{report.text()}"

    def start_background(self, on_reminder) -> list[threading.Thread]:
        threads: list[threading.Thread] = [
            FocusCollector(self.store, self.config.focus_poll_seconds),
            FolderWatcher(self.store, [self.config.resolve_folder(f) for f in self.config.watch_folders]),
            threading.Thread(target=self._reminder_loop, args=(on_reminder,), daemon=True, name="reminders"),
        ]
        for t in threads:
            t.start()
        return threads

    def _reminder_loop(self, on_reminder):
        while True:
            for _id, text in self.store.due_reminders(time.time()):
                on_reminder(text)
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = Config.load(args.config)

    if args.cli:
        jarvis = Jarvis(config, _cli_confirm)
        jarvis.start_background(lambda text: print(f"\n[REMINDER] {text}\n> ", end=""))
        print("Jarvis-0 (cli). Ctrl+C to quit.")
        try:
            while True:
                print(jarvis.handle(input("> ")))
        except (KeyboardInterrupt, EOFError):
            return

    from .ui.command_bar import CommandBar
    from .ui.hotkey import HotkeyThread

    bar_ref: dict = {}
    jarvis = Jarvis(config, lambda *a: bar_ref["bar"].confirm(*a))
    bar = CommandBar(jarvis.handle)
    bar_ref["bar"] = bar
    jarvis.start_background(lambda text: bar.post("reminder", text))
    hk = HotkeyThread(config.hotkey, lambda: bar.post("toggle"))
    hk.start()
    log.info("Jarvis-0 running. Press %s to open the command bar.", config.hotkey)
    bar.show()
    bar.mainloop()


if __name__ == "__main__":
    main()
