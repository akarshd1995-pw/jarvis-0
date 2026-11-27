"""The `care` action (phase 37): disk advisor, cleanups, startup apps, GPU temperatures. See jarvis/care.py."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from .. import care as C
from . import safeops
from .base import ActionResult, Ctx
from .files import _unique_target

REG = None            # tests swap in a fake registry
FOLDERS = None        # tests swap in fake startup folders
WHAT = ("temp", "installers", "duplicates")


def _temp_dir(ctx: Ctx) -> Path:
    return Path(ctx.config.care_temp_dir or tempfile.gettempdir())


def _downloads(ctx: Ctx) -> Path | None:
    try:
        p = ctx.config.resolve_folder("Downloads")
        return p if p.is_dir() else None
    except Exception:
        return None


def _skip_temp(p: Path, temp: Path) -> bool:
    """Never touch Jarvis's own temp work or a running app's unpacked files."""
    try:
        top = p.relative_to(temp).parts[0].lower()
    except (ValueError, IndexError):
        return True
    return top.startswith(("jarvis", "_mei", "pyinstaller"))


def candidates(ctx: Ctx, what: str, now: float) -> dict[str, list[tuple[Path, int]]]:
    out = {}
    kinds = WHAT if what in ("all", "", None) else (what,)
    if "temp" in kinds:
        temp = _temp_dir(ctx)
        out["temp"] = [(p, s) for p, s in C.temp_candidates(temp, now) if not _skip_temp(p, temp)]
    if "installers" in kinds:
        out["installers"] = C.installer_candidates(_downloads(ctx), now)
    if "duplicates" in kinds:
        out["duplicates"] = [(d, size) for _, extra, size in C.duplicate_groups(_downloads(ctx)) for d in extra]
    return out


def _reg():
    if REG is not None:
        return REG
    return C.WinRegistry() if sys.platform == "win32" else None


def care(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    act = str(args.get("action") or "disk")
    now = time.time()
    if act == "disk":
        return _disk(ctx, now)
    if act == "cleanup":
        return _cleanup(ctx, str(args.get("what") or "all"), now, dry_run)
    if act == "empty_trash":
        return _empty_trash(ctx, now, dry_run)
    if act == "temps":
        hist = C.temp_history(ctx.store.events_since(now - 7 * C.DAY, "metrics"), now)
        return ActionResult(f"gpu temps {hist['days']}", kind="care", data={"op": "temps", **hist})
    if act in ("startup_list", "startup_off", "startup_on"):
        return _startup(ctx, act, str(args.get("name") or ""), now, dry_run)
    raise ValueError(f"unknown care action {act}")


def _disk(ctx: Ctx, now: float) -> ActionResult:
    home = Path.home()
    scan = C.scan_sizes(home, max_depth=3, budget_s=8.0)
    big = [(str(p).replace(str(home), "~", 1), s) for p, s in C.biggest(scan, home, 5)]
    found = candidates(ctx, "all", now)
    sizes = {k: sum(s for _, s in v) for k, v in found.items()}
    counts = {k: len(v) for k, v in found.items()}
    old_trash = [(b, s) for b, s, age in C.trash_batches(ctx.config.trash_dir, now) if age >= C.TRASH_KEEP_D]
    rb = C.recycle_bin()
    data = {"op": "disk", "drives": [(d, t, f) for d, t, f in C.drives()], "biggest": big, "complete": scan.complete,
            "sizes": sizes, "counts": counts, "trash_old": sum(s for _, s in old_trash),
            "recycle": rb[0] if rb else 0}
    items = [f"{p}: {C.human(s)}" for p, s in big]
    return ActionResult("disk report", items, kind="care", data=data)


def _cleanup(ctx: Ctx, what: str, now: float, dry_run: bool) -> ActionResult:
    found = candidates(ctx, what, now)
    files = [(p, s, k) for k, v in found.items() for p, s in v][:C.MAX_FILES]
    total = sum(s for _, s, _ in files)
    names = [p.name for p, _, _ in sorted(files, key=lambda x: -x[1])[:15]]
    base = {"op": "cleanup", "what": what, "kinds": {k: len(v) for k, v in found.items()}}
    if dry_run:
        return ActionResult(f"Will clear {len(files)} file(s), {C.human(total)} (undo brings them back)", names,
                            kind="care", data={**base, "count": len(files), "bytes": total, "dry": True})
    if not files:
        return ActionResult("nothing to clean", kind="care", data={**base, "count": 0, "bytes": 0})
    trash = ctx.config.trash_dir / ctx.batch_id
    done, failed = safeops.transfer(ctx.store, ctx.batch_id, [p for p, _, _ in files], trash, "trash",
                                    _unique_target, attempts=1)
    if "temp" in found:
        C.remove_empty_dirs(_temp_dir(ctx))
    size_of = {p: s for p, s, _ in files}
    freed = sum(size_of.get(src, 0) for src, _ in done)
    in_use = sum(1 for _, r in failed if "open in another app" in r or "permission" in r.lower())
    return ActionResult(f"Cleared {len(done)} file(s), {C.human(freed)}", names, kind="care",
                        data={**base, "count": len(done), "bytes": freed, "skipped": len(failed), "in_use": in_use,
                              "requested": len(files)})


def _empty_trash(ctx: Ctx, now: float, dry_run: bool) -> ActionResult:
    old = [(b, s) for b, s, age in C.trash_batches(ctx.config.trash_dir, now) if age >= C.TRASH_KEEP_D]
    total = sum(s for _, s in old)
    if dry_run:
        return ActionResult(f"Will permanently delete {len(old)} old trash batch(es), {C.human(total)}",
                            [b.name for b, _ in old][:15], kind="care",
                            data={"op": "empty_trash", "count": len(old), "bytes": total, "dry": True})
    freed = 0
    for b, s in old:
        C.rmtree(b)
        if not b.exists():
            freed += s
            ctx.store.forget_batch(b.name)
    return ActionResult(f"Emptied {C.human(freed)} from Jarvis trash", kind="care",
                        data={"op": "empty_trash", "count": len(old), "bytes": freed})


def _startup(ctx: Ctx, act: str, name: str, now: float, dry_run: bool) -> ActionResult:
    reg = _reg()
    if reg is None:
        return ActionResult("startup apps are a Windows feature", kind="care", data={"op": "startup_na"})
    items = C.startup_items(reg, FOLDERS)
    if act == "startup_list":
        return ActionResult("startup apps", [f"{'on ' if i.enabled else 'off'}  {i.label}" for i in items], kind="care",
                            data={"op": "startup_list", "on": [i.label for i in items if i.enabled],
                                  "off": [i.label for i in items if not i.enabled]})
    want = act == "startup_on"
    hits = C.find_startup(items, name)
    if not hits:
        return ActionResult(f"no startup app called {name}", kind="care",
                            data={"op": "startup_missing", "name": name, "on": [i.label for i in items if i.enabled]})
    todo = [i for i in hits if i.enabled != want]
    if dry_run:
        return ActionResult(f"turn {'on' if want else 'off'} {hits[0].label} at startup", kind="care",
                            data={"op": "startup_set", "name": hits[0].label, "on": want, "dry": True})
    if not todo:
        return ActionResult("already", kind="care", data={"op": "startup_already", "name": hits[0].label, "on": want})
    changed, admin = [], []
    for it in todo:
        try:
            before = C.set_startup(reg, it, want, now)
        except PermissionError:
            admin.append(it.label)
            continue
        hive, key = it.approved
        ctx.store.record(ctx.batch_id, "startup", {"op": "startup_set", "hive": hive, "key": key, "name": it.name,
                                                   "raw": before.hex() if before is not None else None})
        changed.append(it.label)
    return ActionResult(f"startup {'on' if want else 'off'}: {changed}", kind="care",
                        data={"op": "startup_set", "name": (changed or admin)[0], "on": want, "changed": changed,
                              "admin": admin})


def undo_op(inverse: dict) -> None:
    reg = _reg()
    if reg is None:
        raise OSError("startup apps can only be changed on Windows")
    raw = inverse.get("raw")
    reg.set_binary(inverse["hive"], inverse["key"], inverse["name"], bytes.fromhex(raw) if raw else None)
