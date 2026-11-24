"""The health check (phase 29): "run a health check" / /doctor.

Tests every part of Jarvis and repairs what it safely can:
  settings file broken      -> saved aside, defaults restored
  command model won't load  -> the last good version restored (from checkpoints/versions)
  memory index damaged      -> moved aside, rebuilt in the background
  activity database damaged -> moved aside, a fresh one started
Nothing is deleted: anything replaced is kept next to the original with a .broken-<time> name.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .updater import PROJECT_ROOT

CK = PROJECT_ROOT / "checkpoints"


@dataclass
class Check:
    name: str
    status: str          # ok | fixed | warn | fail | info
    detail: str = ""


def _aside(path: Path) -> Path:
    dest = path.with_name(f"{path.name}.broken-{dt.datetime.now():%Y%m%d-%H%M%S}")
    shutil.move(str(path), str(dest))
    return dest


def _onnx_loads(*paths: Path) -> str | None:
    """None if every file loads, else the error."""
    try:
        import onnxruntime as ort
    except ImportError:
        return "onnxruntime isn't installed (pip install -r requirements.txt)"
    for p in paths:
        if not p.exists():
            return f"{p.name} is missing"
        try:
            ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        except Exception as e:
            return f"{p.name} won't load: {str(e)[:120]}"
    return None


def check_config(config_path: Path) -> Check:
    if not config_path.exists():
        return Check("Settings", "ok", "using defaults")
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
        return Check("Settings", "ok")
    except (ValueError, OSError) as e:
        kept = _aside(config_path)
        return Check("Settings", "fixed", f"the settings file was broken ({e}); saved as {kept.name}, defaults restored")


def check_folders(cfg: Config) -> list[Check]:
    out = []
    missing = [n for n in cfg.memory_folders if not cfg.resolve_folder(n).is_dir()]
    if missing:
        out.append(Check("Folders", "warn", "not found: " + ", ".join(missing) +
                         " (Settings > Privacy > Folders to index)"))
    else:
        out.append(Check("Folders", "ok", f"{len(cfg.memory_folders)} folders indexed"))
    return out


def check_command_model(ck: Path = CK) -> Check:
    if not (ck / "tokenizer.json").exists():
        return Check("Command model", "warn", "not trained yet: rules only (run train_night.bat)")
    err = _onnx_loads(ck / "intent_encoder.onnx", ck / "intent_decoder.onnx")
    if err is None:
        return Check("Command model", "ok")
    versions = sorted((ck / "versions").glob("v*")) if (ck / "versions").exists() else []
    for v in reversed(versions):                           # newest good version
        if _onnx_loads(v / "intent_encoder.onnx", v / "intent_decoder.onnx") is None:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from training.loop import make_live
            make_live(v)
            return Check("Command model", "fixed", f"{err}; restored {v.name}")
    return Check("Command model", "fail", f"{err}. Run improve.bat or train_night.bat")


def check_optional_models(ck: Path = CK) -> list[Check]:
    out = []
    for name, files, how in [
        ("Reply model", ["reply_encoder.onnx", "reply_decoder.onnx"], "train_replies.bat (templates work without it)"),
        ("OCR", ["ocr_finder.onnx", "ocr_reader.onnx"], "train_ocr.bat (screenshots found by name only)"),
        ("Meaning model", ["meaning.onnx"], "train_meaning.bat (search by words only)"),
    ]:
        if not any((ck / f).exists() for f in files):
            out.append(Check(name, "info", "not trained: " + how))
            continue
        err = _onnx_loads(*[ck / f for f in files])
        out.append(Check(name, "ok") if err is None else Check(name, "fail", f"{err}. Re-run {how.split()[0]}"))
    return out


def check_sqlite(path: Path, name: str, rebuildable: bool) -> Check:
    if not path.exists():
        return Check(name, "ok", "will be created")
    try:
        con = sqlite3.connect(str(path))
        try:
            res = con.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        res = str(e)
    if res == "ok":
        return Check(name, "ok")
    if rebuildable:
        from .memory import index as MI
        MI.close(path.parent)                              # the running index lets go of the file
    try:
        kept = _aside(path)
    except OSError:
        return Check(name, "fail", f"it's damaged ({res}) and in use: quit Jarvis, then run python -m jarvis.doctor")
    how = "rebuilding it in the background" if rebuildable else "started a fresh one (restart Jarvis)"
    return Check(name, "fixed", f"it was damaged ({res}); kept as {kept.name}, {how}")


def check_disk(path: Path) -> Check:
    free = shutil.disk_usage(path.anchor or path).free / 2**30
    if free < 2:
        return Check("Disk space", "warn", f"only {free:.1f} GB free: training and indexing may fail")
    return Check("Disk space", "ok", f"{free:.0f} GB free")


def check_packages() -> Check:
    missing = []
    for mod in ("numpy", "onnxruntime", "PIL", "pystray", "pypdf"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return Check("Packages", "warn", "missing " + ", ".join(missing) + ": pip install -r requirements.txt")
    return Check("Packages", "ok")


def check_crashes(data_dir: Path) -> Check:
    from .reliability import recent_crashes
    crashes = recent_crashes(data_dir)
    if not crashes:
        return Check("Crashes", "ok", "none in the last 7 days")
    return Check("Crashes", "warn", f"{len(crashes)} in the last 7 days; latest report: {crashes[-1].name}")


def check_this_pc(cfg: Config) -> list[Check]:
    """Phase 44: standard folders where Windows says they are; power."""
    from . import machine
    from .config import known_folder
    out = []
    moved = []
    for name in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"):
        real = known_folder(name)
        saved = cfg.folders.get(name)
        if real and saved and Path(saved).expanduser() != real:
            moved.append(f"{name} is set to {saved} but Windows keeps it at {real}")
    if moved:
        out.append(Check("Standard folders", "warn", "; ".join(moved) + " (Settings > Folders & apps)"))
    p = machine.power()
    if p.on_battery is not None:
        state = "on battery" + (" (battery saver)" if p.saver else "") if p.on_battery or p.saver else "plugged in"
        out.append(Check("Power", "info", f"laptop, {state}"
                         + ("; heavy background work is waiting for the charger" if (p.on_battery or p.saver)
                            and cfg.pause_heavy_on_battery else "")))
    return out


def run(cfg: Config, services: dict[str, bool] | None = None, ck: Path = CK) -> list[Check]:
    checks = [check_config(cfg.data_dir / "config.json")]
    checks += check_folders(cfg)
    checks += check_this_pc(cfg)
    checks.append(check_command_model(ck))
    checks += check_optional_models(ck)
    checks.append(check_sqlite(cfg.data_dir / "memory.sqlite", "Memory index", rebuildable=True))
    checks.append(check_sqlite(cfg.db_path, "Activity log", rebuildable=False))
    checks.append(check_disk(cfg.data_dir))
    checks.append(check_packages())
    checks.append(check_crashes(cfg.data_dir))
    if services is not None:
        dead = [n for n, alive in services.items() if not alive]
        checks.append(Check("Background services", "warn" if dead else "ok",
                            ("stopped: " + ", ".join(dead) + " (the watchdog restarts them)") if dead
                            else f"{len(services)} running"))
    from . import autostart
    try:
        on = autostart.is_enabled()
        checks.append(Check("Start with Windows", "ok" if on else "info", "on" if on else "off (Settings > General)"))
    except Exception:
        pass
    return checks


def summary(checks: list[Check], address: str = "sir") -> str:
    n = {s: sum(1 for c in checks if c.status == s) for s in ("ok", "fixed", "warn", "fail", "info")}
    addr = f", {address}" if address else ""
    if n["fail"]:
        head = f"Health check done{addr}: {n['fail']} problem{'s' * (n['fail'] > 1)} I couldn't fix."
    elif n["fixed"]:
        head = f"Health check done{addr}. I fixed {n['fixed']} thing{'s' * (n['fixed'] > 1)}; everything else is fine."
    elif n["warn"]:
        head = f"All systems working{addr}, with {n['warn']} thing{'s' * (n['warn'] > 1)} worth a look."
    else:
        head = f"All systems healthy{addr}."
    icon = {"ok": "✓", "fixed": "🔧", "warn": "!", "fail": "✗", "info": "·"}
    lines = [f"{icon[c.status]} {c.name}" + (f": {c.detail}" if c.detail else "") for c in checks]
    return head + "\n" + "\n".join(lines)


if __name__ == "__main__":
    import sys
    cfg = Config.load()
    print(summary(run(cfg), cfg.persona_address))
    sys.exit(0)
