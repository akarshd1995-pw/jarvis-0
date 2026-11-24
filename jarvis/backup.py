"""Backup and restore of everything that makes your Jarvis yours (phase 29).

    python -m jarvis.backup                 # make a backup now
    python -m jarvis.backup --list
    python -m jarvis.backup --restore <zip> # restore (the current state is backed up first)

A backup holds your settings, activity log and reminders, /fix corrections, the live models and their
version history notes, and the self-improvement state. The memory index is left out (it rebuilds itself).
Backups go to Documents\\Jarvis backups; the newest 4 are kept. Jarvis makes one by itself every week.
Note: the activity log is encrypted to your Windows account, so a backup restores on this account only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
import zipfile
from pathlib import Path

from .config import Config
from .updater import PROJECT_ROOT

KEEP = 3
DATA_FILES = ["config.json", "jarvis.db", "briefing_state.json", "baseline.json", "improve_auto.json"]
PROJECT_FILES = ["data/user_corrections.jsonl", "data/regression.jsonl", "checkpoints/improve_state.json",
                 "checkpoints/tokenizer.json", "checkpoints/intent_encoder.onnx", "checkpoints/intent_decoder.onnx",
                 "checkpoints/intent_onnx.json", "checkpoints/intent.pt", "checkpoints/reply_tokenizer.json",
                 "checkpoints/reply_encoder.onnx", "checkpoints/reply_decoder.onnx", "checkpoints/reply_onnx.json",
                 "checkpoints/ocr_finder.onnx", "checkpoints/ocr_reader.onnx", "checkpoints/meaning.onnx",
                 "checkpoints/meaning_tokenizer.json"]


def backup_dir(cfg: Config) -> Path:
    d = cfg.resolve_folder("Documents") / "Jarvis backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def backups(cfg: Config) -> list[Path]:
    return sorted(backup_dir(cfg).glob("jarvis-backup-*.zip"))


def create(cfg: Config, label: str = "", keep: int = KEEP) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = backup_dir(cfg) / f"jarvis-backup-{stamp}{('-' + label) if label else ''}.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in DATA_FILES:
            p = cfg.data_dir / name
            if p.exists():
                z.write(p, f"data_dir/{name}")
        for rel in PROJECT_FILES:
            p = PROJECT_ROOT / rel
            if p.exists():
                z.write(p, f"project/{rel}")
        for rep in (PROJECT_ROOT / "checkpoints" / "versions").glob("*/report.json"):
            z.write(rep, f"project/checkpoints/versions/{rep.parent.name}/report.json")
    for old in backups(cfg)[:-keep]:
        old.unlink(missing_ok=True)
    return dest


def restore(cfg: Config, zip_path: Path) -> str:
    before = create(cfg, label="before-restore", keep=KEEP + 1)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("data_dir/"):
                target = cfg.data_dir / name[len("data_dir/"):]
            elif name.startswith("project/"):
                target = PROJECT_ROOT / name[len("project/"):]
            else:
                continue
            if ".." in Path(name).parts:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
            n += 1
    return f"Restored {n} files from {zip_path.name}. Your state before the restore is in {before.name}. Restart Jarvis."


def due(cfg: Config, every_days: float = 7) -> bool:
    b = backups(cfg)
    return not b or time.time() - b[-1].stat().st_mtime > every_days * 86400


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", type=Path)
    args = ap.parse_args()
    cfg = Config.load()
    if args.list:
        for b in backups(cfg):
            print(f"{b.name}  {b.stat().st_size / 2**20:.0f} MB")
        return
    if args.restore:
        print(restore(cfg, args.restore))
        return
    p = create(cfg)
    print(f"Backup saved: {p} ({p.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
