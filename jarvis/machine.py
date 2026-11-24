"""This PC (phase 44): what Jarvis needs to know to behave the same on any Windows setup.

  power()          on battery? battery saver? (heavy background work waits for the charger)
  work_area(x, y)  the usable part of the monitor at a point (taskbar excluded), for placing windows
  default_browser  the browser Windows opens links with ("open chrome" falls back to it if Chrome isn't installed)
  user_port()      one Jarvis per Windows user, even with several people signed in on the same PC
  scan(config)     the first-run check: folders (OneDrive / moved / other languages), apps, browser, monitors,
                   power; problems in plain words. /setup shows it; it runs by itself on a new PC.

Standard library only (ctypes, winreg).
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"
BASE_PORT = 47913


# ------------------------------------------------------------------ power
@dataclass
class Power:
    on_battery: bool | None          # None: no battery (desktop) or unknown
    saver: bool = False
    percent: int | None = None


def power() -> Power:
    if not IS_WINDOWS:
        return Power(None)
    try:
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return Power(None)
        no_battery = s.BatteryFlag == 128 or s.BatteryFlag == 255
        return Power(None if no_battery else s.ACLineStatus == 0, bool(s.SystemStatusFlag & 1),
                     None if s.BatteryLifePercent == 255 else int(s.BatteryLifePercent))
    except Exception:
        return Power(None)


def heavy_work_ok(config, p: Power | None = None) -> bool:
    """OCR, meaning vectors, self-training: only on the charger (or on a desktop), never in battery saver."""
    if not getattr(config, "pause_heavy_on_battery", True):
        return True
    p = p or power()
    return not (p.on_battery or p.saver)


# ------------------------------------------------------------------ monitors
def cursor_pos() -> tuple[int, int] | None:
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    except Exception:
        return None


def work_area(x: int | None = None, y: int | None = None) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of the usable area of the monitor at (x, y) (cursor if None; primary if
    neither), without the taskbar, wherever and however big it is. None off Windows."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MI(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]
        if x is None or y is None:
            pos = cursor_pos()
            x, y = pos if pos else (0, 0)
        u = ctypes.windll.user32
        u.MonitorFromPoint.restype = wintypes.HANDLE
        u.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        h = u.MonitorFromPoint(wintypes.POINT(int(x), int(y)), 2)          # MONITOR_DEFAULTTONEAREST
        mi = MI()
        mi.cbSize = ctypes.sizeof(MI)
        if not u.GetMonitorInfoW(h, ctypes.byref(mi)):
            return None
        r = mi.rcWork
        return r.left, r.top, r.right, r.bottom
    except Exception:
        return None


def primary_work_area() -> tuple[int, int, int, int] | None:
    return work_area(0, 0)


def monitors() -> list[dict]:
    """[{left, top, right, bottom, primary, scale}] for every monitor."""
    if not IS_WINDOWS:
        return []
    out: list[dict] = []
    try:
        import ctypes
        from ctypes import wintypes

        class MI(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]
        proc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HANDLE, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                  wintypes.LPARAM)

        def cb(h, hdc, rect, lp):
            mi = MI()
            mi.cbSize = ctypes.sizeof(MI)
            ctypes.windll.user32.GetMonitorInfoW(h, ctypes.byref(mi))
            scale = None
            try:
                dx, dy = ctypes.c_uint(), ctypes.c_uint()
                if ctypes.windll.shcore.GetDpiForMonitor(h, 0, ctypes.byref(dx), ctypes.byref(dy)) == 0:
                    scale = round(dx.value / 96 * 100)
            except Exception:
                pass
            r = mi.rcMonitor
            out.append({"left": r.left, "top": r.top, "right": r.right, "bottom": r.bottom,
                        "primary": bool(mi.dwFlags & 1), "scale": scale})
            return 1
        ctypes.windll.user32.EnumDisplayMonitors(None, None, proc(cb), 0)
    except Exception as e:
        log.info("monitor list unavailable: %s", e)
    return out


def place(width: int, height: int, where: str = "center", margin: int = 20,
          fallback: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    """Top-left corner for a window: where = center (upper third) | bottom-right | top-right | top-left |
    bottom-left, on the monitor your mouse is on ('center', toasts) or the primary one (HUD corners)."""
    area = work_area() if where in ("center", "toast") else primary_work_area()
    if area is None:
        area = (0, 0) + fallback
    left, top, right, bottom = area
    if where == "center":
        return left + (right - left - width) // 2, top + int((bottom - top) * 0.18)
    if where == "toast":
        return right - width - margin, bottom - height - margin
    x = right - width - margin if "right" in where else left + margin
    y = top + margin if "top" in where else bottom - height - margin
    return x, y


def fullscreen_busy() -> str | None:
    """'game' when a full-screen game (Direct3D) runs, 'fullscreen' for a full-screen app or presentation
    (a film, slides), else None. Uses what Windows itself uses to hold back notifications."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        state = ctypes.c_int(0)
        if ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state)) != 0:
            return None
        # 2 busy (full-screen app), 3 Direct3D full screen (games), 4 presentation mode
        return {3: "game", 2: "fullscreen", 4: "fullscreen"}.get(state.value)
    except Exception:
        return None


# ------------------------------------------------------------------ browser
BROWSERS = {   # ProgId prefix -> (name, app alias)
    "ChromeHTML": ("Google Chrome", "chrome"), "MSEdgeHTM": ("Microsoft Edge", "edge"),
    "FirefoxURL": ("Firefox", "firefox"), "BraveHTML": ("Brave", "brave"), "Opera": ("Opera", "opera"),
    "VivaldiHTM": ("Vivaldi", "vivaldi"), "ArcHTML": ("Arc", "arc"),
}


def default_browser() -> dict | None:
    """{name, alias, exe} of the browser Windows opens links with."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice")
        prog = winreg.QueryValueEx(k, "ProgId")[0]
        name, alias = next(((n, a) for p, (n, a) in BROWSERS.items() if prog.startswith(p)), (prog, None))
        exe = None
        try:
            c = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, prog + r"\shell\open\command")
            cmd = winreg.QueryValueEx(c, "")[0]
            exe = cmd.split('"')[1] if cmd.startswith('"') else cmd.split()[0]
        except OSError:
            pass
        return {"name": name, "alias": alias, "exe": exe, "progid": prog}
    except OSError:
        return None


# ------------------------------------------------------------------ one Jarvis per Windows user
def user_name() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME", "user")


def user_port(name: str | None = None) -> int:
    """Each Windows user gets their own port, so two people signed in at once each run their own Jarvis."""
    return BASE_PORT + zlib.crc32((name or user_name()).lower().encode()) % 2000


# ------------------------------------------------------------------ the first-run check
def windows_language() -> str | None:
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(85)
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, 85):
            return buf.value
    except Exception:
        pass
    return None


def fingerprint() -> str:
    """Changes when Jarvis's data is moved to another PC or user (a backup restored on the laptop)."""
    return f"{platform.node()}|{user_name()}".lower()


def scan(config, apps=None) -> dict:
    """What Jarvis found on this PC, and anything that needs your attention."""
    from .config import HOME, known_folder
    home = Path(HOME)
    folders, notes = {}, []
    for name, path in config.folders.items():
        p = Path(path).expanduser()
        real = known_folder(name)
        info = {"path": str(p), "exists": p.is_dir()}
        low = str(p).lower()
        if "onedrive" in low:
            info["onedrive"] = True
        try:
            p.resolve().relative_to(home.resolve())
        except (ValueError, OSError):
            info["outside_home"] = True
        if real and Path(real) != p:
            info["windows_says"] = str(real)
        folders[name] = info
        if not p.is_dir() and name not in ("Archive", "Screenshots"):
            notes.append(f"{name} isn't at {p}. Set its location in Settings > Folders & apps.")
    od = [n for n, i in folders.items() if i.get("onedrive")]
    if od:
        notes.append(f"{', '.join(od)} {'is' if len(od) == 1 else 'are'} in OneDrive. That's fine; files that are "
                     "only in the cloud are downloaded when I open or move them.")
    br = default_browser()
    p = power()
    mons = monitors()
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M"), "pc": platform.node(), "user": user_name(),
        "windows": platform.platform(), "language": windows_language(), "folders": folders,
        "browser": br, "monitors": mons, "battery": p.on_battery is not None, "on_battery": p.on_battery,
        "saver": p.saver, "port": user_port(), "fingerprint": fingerprint(), "notes": notes,
    }
    if apps is not None:
        try:
            report["apps"] = len(apps.entries())
            for alias in ("chrome", "vscode", "spotify"):
                if not apps.installed(alias, config):
                    report.setdefault("missing", []).append(alias)
        except Exception as e:
            log.info("app scan failed: %s", e)
    if br and "chrome" in report.get("missing", []) and br.get("alias") != "chrome":
        notes.append(f"Chrome isn't installed, so \"open chrome\" opens {br['name']}, your default browser.")
    if len(mons) > 1:
        scales = {m.get("scale") for m in mons if m.get("scale")}
        notes.append(f"{len(mons)} monitors" + (f" at different scaling ({', '.join(f'{s}%' for s in sorted(scales))})"
                                               if len(scales) > 1 else "") +
                     ": the command bar opens on the one your mouse is on.")
    if p.on_battery is not None:
        notes.append("This is a laptop: OCR, meaning search and self-training wait until it's plugged in.")
    return report


def summary(r: dict) -> str:
    """The first-run check in a few lines."""
    lines = [f"Set up for {r['pc']} ({r['user']})."]
    moved = [n for n, i in r["folders"].items() if i.get("onedrive") or i.get("outside_home")]
    ok = sum(1 for i in r["folders"].values() if i["exists"])
    lines.append(f"Folders: {ok} of {len(r['folders'])} found" + (f"; {', '.join(moved)} moved (OneDrive or another "
                                                                     "drive), and that's handled." if moved else "."))
    if r.get("apps") is not None:
        lines.append(f"Apps: {r['apps']} found in the Start menu.")
    if r.get("browser"):
        lines.append(f"Default browser: {r['browser']['name']}.")
    if r.get("monitors"):
        lines.append(f"Monitors: {len(r['monitors'])}.")
    lines += r.get("notes", [])
    return "\n".join(lines)


def load_report(data_dir: Path) -> dict | None:
    try:
        return json.loads((Path(data_dir) / "machine.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_report(data_dir: Path, r: dict) -> None:
    try:
        (Path(data_dir) / "machine.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    except OSError:
        pass


def needs_setup(data_dir: Path) -> bool:
    """First run, or Jarvis's data was moved to another PC or user."""
    r = load_report(data_dir)
    return r is None or r.get("fingerprint") != fingerprint()
