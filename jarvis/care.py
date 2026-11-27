"""System care (phase 37): disk space advisor, cleanup, startup apps, GPU temperature history.

Everything here only *finds* things; the `care` action decides what to do with them:
- cleanups move files into Jarvis's own trash (crash-safe journal, so "undo" brings them back) and always ask
  first, because to you they are deletes;
- emptying the Jarvis trash is the one permanent step, and it asks too;
- startup apps are switched off the same way Task Manager does it (the StartupApproved flag), never by deleting
  the entry, so switching back on — or "undo" — restores it exactly.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DAY = 86400
TEMP_MIN_AGE_D = 2               # temp files untouched for this long are safe to clear
INSTALLER_MIN_AGE_D = 14         # setup files you ran weeks ago
TRASH_KEEP_D = 30                # Jarvis trash batches older than this can be emptied
MAX_FILES = 5000                 # per cleanup run: keeps the journal (and undo) quick
INSTALLER_EXT = {".exe", ".msi", ".msix", ".msixbundle", ".appx", ".appxbundle"}
SKIP_DIRS = {"$recycle.bin", "system volume information", "windowsapps"}


def human(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("bytes", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ------------------------------------------------------------------ sizes
@dataclass
class Scan:
    sizes: dict          # {Path: bytes} for folders down to max_depth
    complete: bool       # False if the time budget ran out (sizes are then lower bounds)


def scan_sizes(root: Path, max_depth: int = 2, budget_s: float = 8.0, clock=time.monotonic) -> Scan:
    """Total size of every folder under root down to max_depth. Never follows links or junctions."""
    sizes: dict[Path, int] = {}
    deadline = clock() + budget_s
    complete = True

    def walk(p: Path, depth: int) -> int:
        nonlocal complete
        total = 0
        try:
            it = os.scandir(p)
        except OSError:
            return 0
        with it:
            for e in it:
                if clock() > deadline:
                    complete = False
                    break
                try:
                    if e.is_symlink() or _reparse(e):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        if e.name.lower() in SKIP_DIRS:
                            continue
                        total += walk(Path(e.path), depth + 1)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
        if depth <= max_depth:
            sizes[p] = total
        return total

    walk(Path(root), 0)
    return Scan(sizes, complete)


def _reparse(e) -> bool:
    """Junctions / mount points (Windows): never follow them, or "Application Data" loops back into AppData."""
    if hasattr(e, "is_junction") and e.is_junction():
        return True
    try:
        return bool(getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400)
    except OSError:
        return True


def rmtree(path: Path) -> None:
    """Delete a folder for good, clearing read-only flags that Windows would otherwise refuse on."""
    import stat

    def unlock(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=unlock) if sys.version_info < (3, 12) else shutil.rmtree(path, onexc=unlock)


def biggest(scan: Scan, root: Path, n: int = 5) -> list[tuple[Path, int]]:
    """The folders actually taking the space: a parent is replaced by its child when one child is most of it."""
    root = Path(root)
    kids: dict[Path, list[Path]] = {}
    for p in scan.sizes:
        if p != root:
            kids.setdefault(p.parent, []).append(p)
    out: list[tuple[Path, int]] = []
    for top in sorted(kids.get(root, []), key=lambda p: -scan.sizes[p]):
        p = top
        while kids.get(p):
            child = max(kids[p], key=lambda c: scan.sizes[c])
            if scan.sizes[child] < 0.7 * scan.sizes[p]:
                break
            p = child
        if scan.sizes[top] > 0:
            out.append((p, scan.sizes[p]))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ cleanup candidates
def _files(root: Path, limit: int = 50000):
    n = 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for f in filenames:
            n += 1
            if n > limit:
                return
            yield Path(dirpath) / f


def _stat(p: Path):
    try:
        return p.stat()
    except OSError:
        return None


def temp_candidates(temp_dir: Path | None, now: float, min_age_d: float = TEMP_MIN_AGE_D) -> list[tuple[Path, int]]:
    """Files in your temp folder nobody has touched for a couple of days. Files in use are skipped later."""
    if not temp_dir or not Path(temp_dir).is_dir():
        return []
    out = []
    for f in _files(Path(temp_dir)):
        st = _stat(f)
        if st and now - max(st.st_mtime, st.st_atime if sys.platform != "win32" else 0) >= min_age_d * DAY:
            out.append((f, st.st_size))
            if len(out) >= MAX_FILES:
                break
    return out


def installer_candidates(downloads: Path | None, now: float, min_age_d: float = INSTALLER_MIN_AGE_D):
    """Setup files in Downloads (top level only) that you downloaded weeks ago."""
    if not downloads or not Path(downloads).is_dir():
        return []
    out = []
    for f in Path(downloads).iterdir():
        name = f.name.lower()
        if f.suffix.lower() in INSTALLER_EXT or re.search(r"(setup|installer|install)[^/]*\.(zip|exe)$", name):
            st = _stat(f)
            if st and f.is_file() and now - st.st_mtime >= min_age_d * DAY:
                out.append((f, st.st_size))
    return sorted(out, key=lambda x: -x[1])


_COPY_SUFFIX = re.compile(r"(?: \(\d+\)| - copy(?: \(\d+\))?|_copy\d*| copy\d*)$", re.I)


def _digest(p: Path, head: int | None = None) -> str | None:
    h = hashlib.sha1()
    try:
        with open(p, "rb") as fh:
            if head:
                h.update(fh.read(head))
            else:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def duplicate_groups(folder: Path | None, min_size: int = 1024) -> list[tuple[Path, list[Path], int]]:
    """Identical files (same bytes) in a folder. Keeps the copy without '(1)'/'copy' in its name, else the oldest.
    Returns [(keep, [extra copies], size)], biggest waste first."""
    if not folder or not Path(folder).is_dir():
        return []
    by_size: dict[int, list[Path]] = {}
    for f in Path(folder).iterdir():
        st = _stat(f)
        if st and f.is_file() and st.st_size >= min_size:
            by_size.setdefault(st.st_size, []).append(f)
    groups = []
    for size, fs in by_size.items():
        if len(fs) < 2:
            continue
        by_head: dict[str, list[Path]] = {}
        for f in fs:
            d = _digest(f, 65536)
            if d:
                by_head.setdefault(d, []).append(f)
        for same in by_head.values():
            if len(same) < 2:
                continue
            by_full: dict[str, list[Path]] = {}
            for f in same:
                d = _digest(f) if size > 65536 else "head"
                if d:
                    by_full.setdefault(d, []).append(f)
            for dupes in by_full.values():
                if len(dupes) < 2:
                    continue
                dupes.sort(key=lambda p: (bool(_COPY_SUFFIX.search(p.stem)), _stat(p).st_mtime if _stat(p) else 0,
                                          len(p.name)))
                groups.append((dupes[0], dupes[1:], size))
    return sorted(groups, key=lambda g: -g[2] * len(g[1]))


def trash_batches(trash_dir: Path | None, now: float) -> list[tuple[Path, int, float]]:
    """[(batch folder, bytes, age in days)] in Jarvis's trash, oldest first."""
    if not trash_dir or not Path(trash_dir).is_dir():
        return []
    out = []
    for b in Path(trash_dir).iterdir():
        if not b.is_dir():
            continue
        size, newest = 0, 0.0
        for f in _files(b):
            st = _stat(f)
            if st:
                size += st.st_size
                newest = max(newest, st.st_mtime)
        st = _stat(b)
        newest = newest or (st.st_mtime if st else now)
        out.append((b, size, (now - newest) / DAY))
    return sorted(out, key=lambda x: -x[2])


def remove_empty_dirs(root: Path) -> None:
    for dirpath, dirnames, filenames in sorted(os.walk(root), key=lambda x: -len(x[0])):
        if dirpath != str(root) and not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def drives() -> list[tuple[str, int, int]]:
    """[(drive, total bytes, free bytes)] for fixed drives (Windows) or the home drive."""
    roots = []
    if sys.platform == "win32":
        try:
            roots = [d for d in os.listdrives() if _fixed(d)]           # Python 3.12+
        except AttributeError:
            roots = [f"{c}:\\" for c in "CDEFGH" if os.path.exists(f"{c}:\\") and _fixed(f"{c}:\\")]
    else:
        roots = [str(Path.home().anchor or "/")]
    out = []
    for r in roots:
        try:
            u = shutil.disk_usage(r)
            out.append((r.rstrip("\\/") or r, u.total, u.free))
        except OSError:
            continue
    return out


def _fixed(root: str) -> bool:
    try:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(root) == 3        # DRIVE_FIXED
    except Exception:
        return True


def recycle_bin() -> tuple[int, int] | None:
    """(bytes, items) in the Windows Recycle Bin, or None off Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Info(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("i64Size", ctypes.c_longlong), ("i64NumItems", ctypes.c_longlong)]
        info = Info()
        info.cbSize = ctypes.sizeof(Info)
        if ctypes.windll.shell32.SHQueryRecycleBinW(None, ctypes.byref(info)) == 0:
            return info.i64Size, info.i64NumItems
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ startup apps
RUN = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN32 = r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Run"
APPROVED = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved"
ENABLED = bytes([2] + [0] * 11)


@dataclass
class StartupItem:
    name: str            # value name or shortcut file name
    command: str
    source: str          # user_run | machine_run | machine_run32 | user_folder | machine_folder
    enabled: bool

    @property
    def approved(self) -> tuple[str, str]:
        """(hive, key) where Task Manager keeps this item's on/off flag."""
        sub = {"user_run": "Run", "machine_run": "Run", "machine_run32": "Run32",
               "user_folder": "StartupFolder", "machine_folder": "StartupFolder"}[self.source]
        return ("HKCU" if self.source.startswith("user") else "HKLM"), APPROVED + "\\" + sub

    @property
    def label(self) -> str:
        n = re.sub(r"\.lnk$", "", self.name, flags=re.I)
        exe = re.search(r"([^\\/\"]+?)\.exe", self.command, re.I)
        return n if n else (exe.group(1) if exe else self.name)


def disabled_value(now: float) -> bytes:
    filetime = int((now + 11644473600) * 10_000_000)                 # what Task Manager writes when you disable
    return bytes([3, 0, 0, 0]) + struct.pack("<Q", filetime)


def is_enabled(raw) -> bool:
    return not raw or not (bytes(raw)[0] & 1)


class WinRegistry:
    """The small slice of winreg that startup control needs (swapped for a fake in tests)."""

    def _hive(self, hive: str):
        import winreg
        return {"HKCU": winreg.HKEY_CURRENT_USER, "HKLM": winreg.HKEY_LOCAL_MACHINE}[hive]

    def values(self, hive: str, key: str) -> dict:
        import winreg
        out = {}
        try:
            with winreg.OpenKey(self._hive(hive), key) as k:
                i = 0
                while True:
                    try:
                        name, val, _ = winreg.EnumValue(k, i)
                    except OSError:
                        break
                    out[name] = val
                    i += 1
        except OSError:
            pass
        return out

    def set_binary(self, hive: str, key: str, name: str, data: bytes | None) -> None:
        import winreg
        with winreg.CreateKeyEx(self._hive(hive), key, 0, winreg.KEY_SET_VALUE) as k:
            if data is None:
                try:
                    winreg.DeleteValue(k, name)
                except FileNotFoundError:
                    pass
            else:
                winreg.SetValueEx(k, name, 0, winreg.REG_BINARY, data)


def startup_folders() -> dict[str, Path]:
    appdata, progdata = os.environ.get("APPDATA"), os.environ.get("PROGRAMDATA")
    tail = Path("Microsoft") / "Windows" / "Start Menu" / "Programs" / "Startup"
    out = {}
    if appdata:
        out["user_folder"] = Path(appdata) / tail
    if progdata:
        out["machine_folder"] = Path(progdata) / tail
    return out


def startup_items(reg=None, folders: dict[str, Path] | None = None) -> list[StartupItem]:
    if reg is None:
        if sys.platform != "win32":
            return []
        reg = WinRegistry()
    folders = startup_folders() if folders is None else folders
    items = []
    for source, hive, key in (("user_run", "HKCU", RUN), ("machine_run", "HKLM", RUN), ("machine_run32", "HKLM", RUN32)):
        for name, cmd in reg.values(hive, key).items():
            if name and isinstance(cmd, str):
                items.append(StartupItem(name, cmd, source, True))
    for source, folder in folders.items():
        try:
            for f in Path(folder).iterdir():
                if f.is_file() and f.name.lower() != "desktop.ini":
                    items.append(StartupItem(f.name, str(f), source, True))
        except OSError:
            continue
    cache: dict = {}
    for it in items:
        hive, key = it.approved
        if (hive, key) not in cache:
            cache[(hive, key)] = reg.values(hive, key)
        it.enabled = is_enabled(cache[(hive, key)].get(it.name))
    return sorted(items, key=lambda i: (not i.enabled, i.label.lower()))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_startup(items: list[StartupItem], name: str) -> list[StartupItem]:
    """Best matches for a spoken name: exact label, then starts-with, then contained (label or exe)."""
    want = _norm(name)
    if not want:
        return []
    for test in (lambda i: _norm(i.label) == want,
                 lambda i: _norm(i.label).startswith(want) or want.startswith(_norm(i.label)) and len(_norm(i.label)) > 2,
                 lambda i: want in _norm(i.label) or want in _norm(i.command)):
        hits = [i for i in items if test(i)]
        if hits:
            return hits
    return []


def set_startup(reg, item: StartupItem, enabled: bool, now: float | None = None) -> bytes | None:
    """Switch an item on/off the way Task Manager does. Returns the previous raw flag (for undo).
    Machine-wide items need administrator rights: PermissionError is raised for the caller to explain."""
    hive, key = item.approved
    before = reg.values(hive, key).get(item.name)
    reg.set_binary(hive, key, item.name, ENABLED if enabled else disabled_value(time.time() if now is None else now))
    return bytes(before) if before is not None else None


# ------------------------------------------------------------------ temperatures
def temp_history(events, now: float, days: int = 7) -> dict:
    """GPU temperature per day from the monitor's once-a-minute samples (event kind 'metrics')."""
    per_day: dict[str, list[tuple[float, float]]] = {}
    for ts, kind, p in events:
        t = (p or {}).get("gpu_temp") if kind == "metrics" else None
        if t is None or ts < now - days * DAY:
            continue
        per_day.setdefault(datetime.fromtimestamp(ts).strftime("%Y-%m-%d"), []).append((ts, float(t)))
    rows = []
    hottest = None
    for day in sorted(per_day):
        temps = [t for _, t in per_day[day]]
        top_ts, top = max(per_day[day], key=lambda x: x[1])
        rows.append({"day": day, "avg": round(statistics.fmean(temps)), "max": round(top),
                     "hot_min": sum(1 for t in temps if t >= 80)})       # one sample a minute
        if hottest is None or top > hottest[1]:
            hottest = (top_ts, top)
    latest = None
    for ts, kind, p in events:
        if kind == "metrics" and (p or {}).get("gpu_temp") is not None and (latest is None or ts > latest[0]):
            latest = (ts, float(p["gpu_temp"]))
    return {"days": rows, "hottest": (datetime.fromtimestamp(hottest[0]).strftime("%a %d %b, %I:%M %p").replace(" 0", " "),
                                      round(hottest[1])) if hottest else None,
            "now": round(latest[1]) if latest and now - latest[0] < 600 else None,
            "samples": sum(len(v) for v in per_day.values())}


def since_label(ts: float, now: float) -> str:
    d = timedelta(seconds=max(0, now - ts))
    return f"{d.days} days" if d.days else f"{d.seconds // 3600} hours"
