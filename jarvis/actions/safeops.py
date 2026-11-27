"""Crash-safe file operations (phase 40).

Every step is written to the journal as *pending* before the file is touched and marked *done* after it has
been verified at its destination. If Jarvis is interrupted (crash, power cut), recover() looks at the disk on
the next start and finishes or rolls back each pending step, so no file is ever lost or left half-copied.

Problems with one file (open in another app, gone, no permission, disk full, path too long) don't stop the
rest: each failure is reported with a plain reason. Copies that can't fit are refused before starting.
"""
from __future__ import annotations

import errno
import shutil
import time
from pathlib import Path

RESERVE = 200 * 2**20          # keep this much free on the destination drive


class NotEnoughSpace(OSError):
    pass


def reason(e: BaseException) -> str:
    """A plain-language reason for a failed file operation."""
    win = getattr(e, "winerror", None)
    if win in (32, 33) or (isinstance(e, PermissionError) and win is None and "being used" in str(e)):
        return "it's open in another app"
    if isinstance(e, FileNotFoundError):
        return "it was moved or deleted"
    if isinstance(e, NotEnoughSpace) or win == 112 or getattr(e, "errno", None) == errno.ENOSPC:
        return "there isn't enough disk space"
    if win in (206, 3) or getattr(e, "errno", None) == errno.ENAMETOOLONG:
        return "the path is too long"
    if isinstance(e, PermissionError):
        return "Windows didn't allow it (permission denied, or it's open in another app)"
    return str(e)[:120] or e.__class__.__name__


def _retry(fn, attempts: int = 3, wait: float = 0.5):
    """Files that are briefly busy (antivirus scan, sync, preview pane) usually free up within a second."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError as e:
            if i == attempts - 1 or getattr(e, "winerror", 32) not in (32, 33, 5, None):
                raise
            time.sleep(wait * (i + 1))


def _same_volume(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_dev == b.stat().st_dev
    except OSError:
        return False


def space_check(files: list[Path], dest: Path, copy: bool) -> None:
    if not copy and _same_volume(files[0].parent, dest):
        return                                           # a move on the same drive needs no space
    need = sum(f.stat().st_size for f in files if f.exists())
    free = shutil.disk_usage(dest).free
    if need > free - RESERVE:
        raise NotEnoughSpace(f"needs {need / 2**30:.1f} GB, {dest.drive or dest.anchor} has "
                             f"{max(0, free - RESERVE) / 2**30:.1f} GB free")


def verified(src: Path, target: Path, size: int, kind: str) -> bool:
    try:
        ok = target.exists() and target.stat().st_size == size
    except OSError:
        return False
    if kind in ("move", "trash"):
        ok = ok and not src.exists()
    return ok


def transfer(store, batch_id: str, files: list[Path], dest: Path, kind: str, unique,
             attempts: int = 3) -> tuple[list, list]:
    """kind: move | copy | trash. Returns (done [(src, target)], failed [(name, reason)]).
    attempts=1 for bulk cleanups, where a file in use should just be skipped, not waited for."""
    done, failed = [], []
    if not files:
        return done, failed
    dest.mkdir(parents=True, exist_ok=True)
    try:
        space_check(files, dest, copy=(kind == "copy"))
    except NotEnoughSpace as e:
        return done, [(f.name, f"there isn't enough disk space ({e})") for f in files]
    for f in files:
        try:
            size = f.stat().st_size
        except OSError as e:
            failed.append((f.name, reason(e)))
            continue
        target = unique(dest, f.name)
        if kind == "copy":
            inverse = {"op": "remove", "path": str(target)}
        else:
            inverse = {"op": "move", "src": str(target), "dest": str(f)}
        jid = store.begin(batch_id, kind if kind != "trash" else "delete", inverse,
                          {"kind": kind, "src": str(f), "target": str(target), "size": size})
        try:
            if kind == "copy":
                _retry(lambda: shutil.copy2(f, target), attempts)
            else:
                _retry(lambda: shutil.move(str(f), str(target)), attempts)
            if not verified(f, target, size, kind):
                _cleanup_partial(f, target, size, kind)
                raise OSError("it didn't arrive intact, so I left the original where it was")
        except Exception as e:                           # this file only: carry on with the rest
            if target.exists() and f.exists() and kind != "copy":
                _cleanup_partial(f, target, size, kind)
            store.abort(jid)
            failed.append((f.name, reason(e)))
            continue
        store.finish(jid)
        done.append((f, target))
    return done, failed


def _cleanup_partial(src: Path, target: Path, size: int, kind: str) -> None:
    """Never lose the original: a copy that is incomplete goes, the original stays."""
    try:
        if src.exists() and target.exists() and target.stat().st_size != size:
            target.unlink()
        elif kind == "copy" and target.exists() and target.stat().st_size != size:
            target.unlink()
    except OSError:
        pass


def recover(store) -> dict:
    """Finish or roll back steps that were interrupted. Returns counts and the names involved."""
    out = {"finished": [], "rolled_back": [], "missing": []}
    for jid, op, detail in store.pending():
        kind = detail.get("kind", "move")
        src, target, size = Path(detail["src"]), Path(detail["target"]), int(detail.get("size", -1))
        s_ok, t_ok = src.exists(), target.exists()
        t_full = t_ok and target.stat().st_size == size
        try:
            if kind == "copy":
                if t_full:
                    store.finish(jid)
                    out["finished"].append(src.name)
                else:
                    if t_ok:
                        target.unlink()
                    store.abort(jid)
                    out["rolled_back"].append(src.name)
                continue
            # move / trash
            if t_full and not s_ok:                          # it happened
                store.finish(jid)
                out["finished"].append(src.name)
            elif t_full and s_ok:                            # copied across drives, original not removed yet
                src.unlink()
                store.finish(jid)
                out["finished"].append(src.name)
            elif s_ok:                                       # never happened, or a partial copy: keep original
                if t_ok:
                    target.unlink()
                store.abort(jid)
                out["rolled_back"].append(src.name)
            else:
                store.abort(jid)
                out["missing"].append(src.name)
        except OSError:
            out["missing"].append(src.name)
    return out
