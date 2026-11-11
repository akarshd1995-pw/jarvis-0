"""Configuration: folder aliases, app aliases, safety allowlist, paths."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
DEFAULT_DATA_DIR = Path(os.environ.get("JARVIS_HOME", HOME / ".jarvis0"))


def _default_folders() -> dict[str, str]:
    names = ["Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos"]
    folders = {n: str(HOME / n) for n in names}
    folders["Archive"] = str(HOME / "Documents" / "Archive")
    return folders


def _default_apps() -> dict[str, str]:
    # alias -> executable / URI understood by os.startfile on Windows
    return {
        "vscode": "code",
        "chrome": "chrome",
        "edge": "msedge",
        "notepad": "notepad",
        "explorer": "explorer",
        "terminal": "wt",
        "spotify": "spotify:",
        "calculator": "calc",
        "settings": "ms-settings:",
    }


@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    folders: dict[str, str] = field(default_factory=_default_folders)
    apps: dict[str, str] = field(default_factory=_default_apps)
    # Jarvis may only touch files under these roots (resolved paths).
    allowed_roots: list[str] = field(default_factory=lambda: [str(HOME)])
    hotkey: str = "ctrl+alt+j"
    focus_poll_seconds: float = 1.0
    watch_folders: list[str] = field(default_factory=lambda: ["Downloads"])
    model_checkpoint: str = "checkpoints/intent.pt"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"

    def resolve_folder(self, name_or_path: str) -> Path:
        """Map an alias like 'Downloads' (case-insensitive) or a literal path to a Path."""
        for alias, p in self.folders.items():
            if alias.lower() == str(name_or_path).lower():
                return Path(p).expanduser()
        return Path(str(name_or_path)).expanduser()

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg = cls()
        path = path or (cfg.data_dir / "config.json")
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if k == "data_dir":
                    v = Path(v)
                if k in ("folders", "apps"):
                    getattr(cfg, k).update(v)
                elif hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.trash_dir.mkdir(parents=True, exist_ok=True)
        return cfg
