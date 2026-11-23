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
    persona_name: str = "Jarvis"
    persona_address: str = "sir"        # how Jarvis addresses you: "sir", "ma'am", your name, or ""
    persona_tone: str = "butler"        # butler | friendly | minimal
    show_details: bool = True           # also show the technical program under each reply
    min_confidence: float = 0.55        # below this the model's answer is not executed
    confirm_actions: list[str] = field(default_factory=lambda: ["move_files", "undo"])  # delete always asks
    encrypt_logs: bool = True           # DPAPI-encrypt the activity log (Windows)
    keep_window_titles: bool = False    # store full window titles (off: only app name + hash)
    model_backend: str = "auto"         # auto | onnx | torch
    model_checkpoint: str = "checkpoints/intent.pt"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    def save(self, path: Path | None = None) -> Path:
        """Write user-editable settings to config.json (used by the settings window)."""
        path = path or (self.data_dir / "config.json")
        keys = ["persona_name", "persona_address", "persona_tone", "show_details", "hotkey", "folders", "apps", "allowed_roots", "watch_folders", "min_confidence", "confirm_actions",
                "encrypt_logs", "keep_window_titles", "model_backend", "focus_poll_seconds"]
        path.write_text(json.dumps({k: getattr(self, k) for k in keys}, indent=2), encoding="utf-8")
        return path

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
