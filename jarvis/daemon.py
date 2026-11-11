"""Jarvis-0 daemon: wires collectors, brain, executor, reminders and the UI together."""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

from .collectors.focus import FocusCollector
from .collectors.folders import FolderWatcher
from .config import Config
from .executor import Executor
from .nlu.brain import Brain
from .safety import Tier
from .store import Store

log = logging.getLogger("jarvis")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Jarvis:
    def __init__(self, config: Config, confirm):
        self.config = config
        self.store = Store(config.db_path)
        self.brain = Brain(PROJECT_ROOT / config.model_checkpoint, PROJECT_ROOT / config.tokenizer_path)
        self.executor = Executor(config, self.store, confirm)

    def handle(self, text: str) -> str:
        prog, source = self.brain.resolve(text)
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
