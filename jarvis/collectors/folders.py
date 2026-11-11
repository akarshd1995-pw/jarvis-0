"""Polling folder watcher (dependency-free). Logs 'file_added' events for watched folders."""
from __future__ import annotations

import threading
from pathlib import Path

from ..store import Store


class FolderWatcher(threading.Thread):
    def __init__(self, store: Store, folders: list[Path], poll: float = 10.0):
        super().__init__(daemon=True, name="folder-watcher")
        self.store, self.folders, self.poll = store, folders, poll
        self._stop = threading.Event()
        self._seen: dict[Path, set[str]] = {}

    def stop(self):
        self._stop.set()

    def _snapshot(self, folder: Path) -> set[str]:
        try:
            return {p.name for p in folder.iterdir() if p.is_file()}
        except OSError:
            return set()

    def run(self):
        for f in self.folders:
            self._seen[f] = self._snapshot(f)
        while not self._stop.wait(self.poll):
            for f in self.folders:
                now = self._snapshot(f)
                for name in sorted(now - self._seen.get(f, set())):
                    self.store.log_event("file_added", {"folder": f.name, "ext": Path(name).suffix.lower()})
                self._seen[f] = now
