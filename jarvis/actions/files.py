"""File actions. Every mutating op journals its inverse so it can be undone."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..dsl.grammar import Duration
from ..safety import MAX_FILES_PER_ACTION, SafetyError, check_path
from .base import ActionResult, Ctx


def _unique_target(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = dest_dir / f"{stem} ({i}){suffix}"
        i += 1
    return target


def select_files(ctx: Ctx, args: dict) -> tuple[Path, list[Path]]:
    src = check_path(ctx.config.resolve_folder(str(args["src"])), ctx.config.allowed_roots)
    if not src.is_dir():
        raise FileNotFoundError(f"folder not found: {src}")
    now = time.time()
    ext = str(args["ext"]).lower().lstrip(".") if "ext" in args else None
    older: Duration | None = args.get("older_than")
    newer: Duration | None = args.get("newer_than")
    needle = args.get("name_contains", "").lower()

    out = []
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if ext and p.suffix.lower().lstrip(".") != ext:
            continue
        if needle and needle not in p.name.lower():
            continue
        age = now - p.stat().st_mtime
        if older and age < older.seconds():
            continue
        if newer and age > newer.seconds():
            continue
        out.append(p)
    if len(out) > MAX_FILES_PER_ACTION:
        raise SafetyError(f"{len(out)} files matched; cap is {MAX_FILES_PER_ACTION}. Narrow the filter.")
    return src, out


def list_files(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    src, files = select_files(ctx, args)
    return ActionResult(f"{len(files)} file(s) in {src}", [f.name for f in files])


def _transfer(args: dict, ctx: Ctx, dry_run: bool, copy: bool) -> ActionResult:
    src, files = select_files(ctx, args)
    dest = check_path(ctx.config.resolve_folder(str(args["dest"])), ctx.config.allowed_roots)
    verb = "copy" if copy else "move"
    preview = [f"{f.name} -> {dest}" for f in files]
    if dry_run:
        return ActionResult(f"Will {verb} {len(files)} file(s) from {src} to {dest}", preview)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = _unique_target(dest, f.name)
        if copy:
            shutil.copy2(f, target)
            ctx.store.record(ctx.batch_id, "copy", {"op": "remove", "path": str(target)})
        else:
            shutil.move(str(f), str(target))
            ctx.store.record(ctx.batch_id, "move", {"op": "move", "src": str(target), "dest": str(f)})
    return ActionResult(f"{verb.capitalize()}d {len(files)} file(s) to {dest}", preview)


def move_files(args, ctx, dry_run):
    return _transfer(args, ctx, dry_run, copy=False)


def copy_files(args, ctx, dry_run):
    return _transfer(args, ctx, dry_run, copy=True)


def delete_files(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    """'Delete' moves files into Jarvis's own trash so it is always undoable."""
    src, files = select_files(ctx, args)
    preview = [f.name for f in files]
    if dry_run:
        return ActionResult(f"Will delete {len(files)} file(s) from {src} (recoverable via undo)", preview)
    trash = ctx.config.trash_dir / ctx.batch_id
    trash.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = _unique_target(trash, f.name)
        shutil.move(str(f), str(target))
        ctx.store.record(ctx.batch_id, "delete", {"op": "move", "src": str(target), "dest": str(f)})
    return ActionResult(f"Deleted {len(files)} file(s) (kept in Jarvis trash)", preview)


def apply_inverse(inv: dict) -> None:
    if inv["op"] == "move":
        dest = Path(inv["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = dest if not dest.exists() else _unique_target(dest.parent, dest.name)
        shutil.move(inv["src"], str(target))
    elif inv["op"] == "remove":
        Path(inv["path"]).unlink(missing_ok=True)
    elif inv["op"] == "cancel_reminder":
        pass  # handled by the reminder store; kept for completeness
    else:
        raise ValueError(f"unknown inverse op {inv['op']}")
