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
    hud_enabled: bool = True
    hud_hotkey: str = "ctrl+alt+h"
    hud_corner: str = "top-right"         # top-right | top-left | bottom-right | bottom-left
    hud_opacity: float = 0.9
    hud_click_through: bool = True
    monitor_enabled: bool = True
    alert_gpu_temp: float = 85.0
    alert_disk_free_gb: float = 10.0
    alert_battery: float = 20.0
    alert_cooldown_min: float = 30.0
    focus_poll_seconds: float = 1.0
    watch_folders: list[str] = field(default_factory=lambda: ["Downloads"])
    briefing_auto: bool = True            # morning / evening briefing on its own, once a day each
    briefing_morning: str = "07:00"       # first time you're at the PC after this (until noon)
    briefing_evening: str = "21:00"       # after this, if you're at the PC
    memory_enabled: bool = True           # phase 7: index files + browser history for "find ..." (all local)
    memory_browser: bool = True           # include Chrome/Edge/Brave/Firefox history
    memory_folders: list[str] = field(default_factory=lambda: ["Downloads", "Documents", "Desktop", "Pictures",
                                                               "Videos"])
    improve_auto: bool = True             # self-improving loop: fine-tune at night when there's something new
    improve_at: str = "02:00"             # earliest start (runs within 3 h of this, only when idle and plugged in)
    persona_name: str = "Jarvis"
    persona_address: str = "sir"        # how Jarvis addresses you: "sir", "ma'am", your name, or ""
    persona_tone: str = "butler"        # butler | friendly | minimal
    show_details: bool = True           # also show the technical program under each reply
    persona_model: bool = True          # use the trained reply model (fact-checked) when it's available
    min_confidence: float = 0.55        # below this the model's answer is not executed
    confirm_actions: list[str] = field(default_factory=lambda: ["move_files", "undo", "window_close", "radio_off"])
    encrypt_logs: bool = True           # DPAPI-encrypt the activity log (Windows)
    keep_window_titles: bool = False    # store full window titles (off: only app name + hash)
    model_backend: str = "auto"         # auto | onnx | torch
    model_checkpoint: str = "checkpoints/intent.pt"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    def save(self, path: Path | None = None) -> Path:
        """Write user-editable settings to config.json (used by the settings window)."""
        path = path or (self.data_dir / "config.json")
        keys = ["monitor_enabled", "alert_gpu_temp", "alert_disk_free_gb", "alert_battery", "alert_cooldown_min",
                "hud_enabled", "hud_hotkey", "hud_corner", "hud_opacity", "hud_click_through", "persona_name", "persona_address", "persona_tone", "show_details", "persona_model", "hotkey", "folders", "apps", "allowed_roots", "watch_folders", "min_confidence", "confirm_actions",
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
