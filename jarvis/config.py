"""Configuration: folder aliases, app aliases, safety allowlist, paths."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
DEFAULT_DATA_DIR = Path(os.environ.get("JARVIS_HOME", HOME / ".jarvis0"))


# Windows "known folders": where Desktop, Documents, ... really are (OneDrive, another drive, moved by the user)
KNOWN_FOLDER_IDS = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641", "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B", "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43", "Videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "Screenshots": "B7BEDE81-DF94-4682-A7D8-57A52620B86F",
}


def known_folder(name: str) -> Path | None:
    """The real location Windows uses for a standard folder (None off Windows or if unknown)."""
    if os.name != "nt" or name not in KNOWN_FOLDER_IDS:
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        u = uuid.UUID(KNOWN_FOLDER_IDS[name])
        g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out)) != 0:
            return None
        path = Path(out.value)
        ctypes.windll.ole32.CoTaskMemFree(out)
        return path
    except Exception:
        return None


def merge_folders(defaults: dict[str, str], saved: dict[str, str]) -> dict[str, str]:
    """Saved aliases win, except a standard folder saved at a path that doesn't exist while Windows knows
    where it really is (e.g. Desktop moved into OneDrive after the config was written)."""
    out = dict(defaults)
    out.update(saved)
    for name, real in defaults.items():
        if name in KNOWN_FOLDER_IDS and not Path(out[name]).exists() and Path(real).exists():
            out[name] = real
    return out


def _default_folders() -> dict[str, str]:
    names = ["Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos", "Screenshots"]
    folders = {}
    for n in names:
        real = known_folder(n)
        folders[n] = str(real or (HOME / "Pictures" / "Screenshots" if n == "Screenshots" else HOME / n))
    folders["Archive"] = str(Path(folders["Documents"]) / "Archive")
    return folders


def with_standard_folders(roots: list[str], folders: dict[str, str]) -> list[str]:
    """Your standard folders are always allowed, even when Windows keeps them outside your user folder
    (Documents moved to D:\\, Desktop in a OneDrive on another drive)."""
    out = list(roots)
    have = [Path(r).expanduser() for r in roots]
    for name, path in folders.items():
        p = Path(path).expanduser()
        inside = False
        for r in have:
            try:
                p.relative_to(r)
                inside = True
                break
            except ValueError:
                continue
        real = known_folder(name)
        if not inside and real is not None and Path(real) == p:       # Windows' own folder, at its real place
            out.append(str(p))
            have.append(p)
    return out


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
    pause_heavy_on_battery: bool = True
    bar_pos: list | None = None           # where you dragged the command bar (None: centre of your monitor)
    bar_mini_pos: list | None = None      # where its minimized orb sits
    use_int8: bool = True                 # phase 41: the smaller, faster int8 model copies when they exist
    command_budget_ms: int = 1500         # past this, the model's answer is dropped and the rules answer
    yield_when_gaming: bool = True        # pause training / OCR while a full-screen game or video runs   # phase 44: OCR / meaning vectors / self-training wait for the charger
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
    memory_meaning: bool = True           # search by meaning too, with our meaning model (once trained)
    memory_ocr: bool = True               # read the text in screenshots/images with our OCR models (once trained)
    memory_folders: list[str] = field(default_factory=lambda: ["Downloads", "Documents", "Desktop", "Pictures",
                                                               "Screenshots", "Videos"])
    places: dict = field(default_factory=lambda: {"home": "", "work": ""})   # "how long to get home"
    maps_api_key: str = ""                # your own Google Maps key (Routes API) for live-traffic times
    maps_osm: bool = True                 # without a key: OpenStreetMap estimate (normal traffic)
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
    clarify: bool = True                # phase 6: ask "did you mean A or B?" instead of guessing
    dialog_model: bool = True           # use the trained dialog model for follow-ups (rules otherwise)
    learn_from_usage: bool = True       # phase 39: keep your commands (on this PC) to find what to learn
    review_weekly: bool = True          # mention the 2-minute review in the briefing
    keep_window_titles: bool = False    # store full window titles (off: only app name + hash)
    model_backend: str = "auto"         # auto | onnx | torch
    model_checkpoint: str = "checkpoints/intent.pt"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    def __post_init__(self):
        self.allowed_roots = with_standard_folders(self.allowed_roots, self.folders)

    # Not settings: where Jarvis keeps its data, and the model files it was built with
    NOT_SAVED = ("data_dir", "model_checkpoint", "tokenizer_path")

    def save(self, path: Path | None = None) -> Path:
        """Write every setting to config.json (the settings window calls this). All fields are saved, so a new
        setting can never be forgotten; keys this version doesn't know (from a newer one) are kept."""
        from dataclasses import fields
        path = path or (self.data_dir / "config.json")
        try:
            out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            out = {}
        for f in fields(self):
            if f.name in self.NOT_SAVED:
                continue
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)                       # never a half-written settings file
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
                if k == "folders":
                    cfg.folders = merge_folders(cfg.folders, v)
                elif k == "memory_folders" and v == ["Downloads", "Documents", "Desktop", "Pictures", "Videos"]:
                    pass                                   # the old default: keep the new one (adds Screenshots)
                elif k == "apps":
                    getattr(cfg, k).update(v)
                elif hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.allowed_roots = with_standard_folders(cfg.allowed_roots, cfg.folders)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.trash_dir.mkdir(parents=True, exist_ok=True)
        return cfg
