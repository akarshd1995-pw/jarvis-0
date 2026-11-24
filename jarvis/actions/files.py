"""File actions. Every mutating op journals its inverse so it can be undone."""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from ..dsl.grammar import Duration
from .. import filetypes
from ..safety import MAX_FILES_PER_ACTION, SafetyError, check_path
from . import safeops
from .base import ActionResult, Ctx


def _unique_target(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = dest_dir / f"{stem} ({i}){suffix}"
        i += 1
    return target


def _facts(args: dict, count: int, op: str = "list", dry: bool = False) -> dict:
    return {"op": op, "count": count, "dry": dry, "src": str(args.get("src", "")),
            "dest": str(args["dest"]) if "dest" in args else (str(args["new_folder"]) if "new_folder" in args else None),
            "new_folder": "new_folder" in args,
            "ext": str(args["ext"]) if "ext" in args else None,
            "older_than": str(args["older_than"]) if "older_than" in args else None,
            "newer_than": str(args["newer_than"]) if "newer_than" in args else None,
            "name_contains": args.get("name_contains")}


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
        if ext and not filetypes.matches(p.suffix, ext):          # "video" = mp4, mkv, avi, mov, ...
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
    return ActionResult(f"{len(files)} file(s) in {src}", [f.name for f in files], kind="list",
                        data=_facts(args, len(files)))


def _transfer(args: dict, ctx: Ctx, dry_run: bool, copy: bool) -> ActionResult:
    src, files = select_files(ctx, args)
    if "new_folder" in args:                                  # "to a new folder videos": made inside the source
        name = re.sub(r'[<>:"/\\|?*]', "", str(args["new_folder"])).strip(" .") or "New folder"
        dest = check_path(src / name, ctx.config.allowed_roots)
    else:
        dest = check_path(ctx.config.resolve_folder(str(args["dest"])), ctx.config.allowed_roots)
    verb = "copy" if copy else "move"
    preview = [f"{f.name} -> {dest}" for f in files]
    if dry_run:
        return ActionResult(f"Will {verb} {len(files)} file(s) from {src} to {dest}", preview, kind="files",
                            data=_facts(args, len(files), op=verb, dry=True))
    done, failed = safeops.transfer(ctx.store, ctx.batch_id, files, dest, "copy" if copy else "move",
                                    _unique_target)
    data = _facts(args, len(done), op=verb)
    data.update(failed=failed, requested=len(files))
    return ActionResult(f"{verb.capitalize()}d {len(done)} of {len(files)} file(s) to {dest}"
                        + (f"; {len(failed)} failed" if failed else ""), preview, kind="files", data=data)


def move_files(args, ctx, dry_run):
    return _transfer(args, ctx, dry_run, copy=False)


def copy_files(args, ctx, dry_run):
    return _transfer(args, ctx, dry_run, copy=True)


def delete_files(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    """'Delete' moves files into Jarvis's own trash so it is always undoable."""
    src, files = select_files(ctx, args)
    preview = [f.name for f in files]
    if dry_run:
        return ActionResult(f"Will delete {len(files)} file(s) from {src} (recoverable via undo)", preview,
                            kind="files", data=_facts(args, len(files), op="delete", dry=True))
    trash = ctx.config.trash_dir / ctx.batch_id
    done, failed = safeops.transfer(ctx.store, ctx.batch_id, files, trash, "trash", _unique_target)
    data = _facts(args, len(done), op="delete")
    data.update(failed=failed, requested=len(files))
    return ActionResult(f"Deleted {len(done)} of {len(files)} file(s) (kept in Jarvis trash)", preview,
                        kind="files", data=data)


def apply_inverse(inv: dict) -> None:
    if inv["op"] == "move":
        src, dest = Path(inv["src"]), Path(inv["dest"])
        if not src.exists() and dest.exists():
            return                                       # already back where it was (an interrupted undo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = dest if not dest.exists() else _unique_target(dest.parent, dest.name)
        safeops._retry(lambda: shutil.move(str(src), str(target)))
    elif inv["op"] == "remove":
        Path(inv["path"]).unlink(missing_ok=True)
    elif inv["op"] == "cancel_reminder":
        pass  # handled by the reminder store; kept for completeness
    else:
        raise ValueError(f"unknown inverse op {inv['op']}")
