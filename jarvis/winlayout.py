"""Windows on screen (phase 31): list app windows, read and set their exact place, launch and close them.

Uses Win32 GetWindowPlacement / SetWindowPlacement, which remember the normal size even of a maximized window
and work across monitors. Coordinates are the same kind Jarvis uses everywhere (see machine.py).
Off Windows every function returns nothing, so the logic in workspaces.py can be tested with a fake.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

IS_WINDOWS = sys.platform == "win32"
SKIP_EXES = {"applicationframehost.exe", "textinputhost.exe", "shellexperiencehost.exe", "searchhost.exe",
             "startmenuexperiencehost.exe", "lockapp.exe", "systemsettings.exe"}
SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow"}
SW_NORMAL, SW_MINIMIZE, SW_MAXIMIZE = 1, 2, 3


@dataclass
class Win:
    hwnd: int
    exe: str                       # full path of the program
    title: str
    rect: list[int]                # [left, top, right, bottom] when not maximized/minimized
    state: str = "normal"          # normal | max | min
    cls: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def app(self) -> str:
        return os.path.basename(self.exe).lower()


def _api():
    import ctypes
    from ctypes import wintypes
    u, k, d = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.dwmapi

    class WP(ctypes.Structure):
        _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                    ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                    ("rcNormalPosition", wintypes.RECT)]
    return ctypes, wintypes, u, k, d, WP


def _exe_of(hwnd: int) -> str:
    ctypes, wintypes, u, k, _, _ = _api()
    pid = wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = k.OpenProcess(0x1000, False, pid.value)                     # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        k.CloseHandle(h)


def windows(skip_pids: set[int] | None = None) -> list[Win]:
    """The app windows you'd see on the taskbar, top to bottom."""
    if not IS_WINDOWS:
        return []
    ctypes, wintypes, u, k, dwm, WP = _api()
    skip_pids = (skip_pids or set()) | {os.getpid()}
    out: list[Win] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        try:
            if not u.IsWindowVisible(hwnd) or u.GetWindow(hwnd, 4):           # GW_OWNER: dialogs, popups
                return True
            ex = u.GetWindowLongW(hwnd, -20)                                 # GWL_EXSTYLE
            if ex & 0x80 and not ex & 0x40000:                               # tool window, not app window
                return True
            cloaked = wintypes.DWORD()
            dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)    # hidden on another desktop
            if cloaked.value:
                return True
            n = u.GetWindowTextLengthW(hwnd)
            if n == 0:
                return True
            title = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, title, n + 1)
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            if cls.value in SKIP_CLASSES:
                return True
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in skip_pids:
                return True
            exe = _exe_of(hwnd)
            if not exe or os.path.basename(exe).lower() in SKIP_EXES:
                return True
            wp = WP()
            wp.length = ctypes.sizeof(WP)
            u.GetWindowPlacement(hwnd, ctypes.byref(wp))
            r = wp.rcNormalPosition
            state = {SW_MAXIMIZE: "max", SW_MINIMIZE: "min"}.get(wp.showCmd, "normal")
            if u.IsIconic(hwnd):
                state = "min"
            out.append(Win(int(hwnd), exe, title.value, [r.left, r.top, r.right, r.bottom], state, cls.value))
        except Exception:
            pass
        return True
    u.EnumWindows(proc(cb), 0)
    return out


def place(hwnd: int, rect: list[int], state: str = "normal") -> bool:
    """Put a window at rect (its normal size) and maximize / minimize it if asked."""
    if not IS_WINDOWS:
        return False
    ctypes, wintypes, u, _, _, WP = _api()
    wp = WP()
    wp.length = ctypes.sizeof(WP)
    if not u.GetWindowPlacement(hwnd, ctypes.byref(wp)):
        return False
    wp.flags = 0
    wp.showCmd = {"max": SW_MAXIMIZE, "min": SW_MINIMIZE}.get(state, SW_NORMAL)
    wp.rcNormalPosition = wintypes.RECT(*[int(v) for v in rect])
    if state == "max":                    # maximize on the right monitor: restore there first, then maximize
        wp.showCmd = SW_NORMAL
        u.SetWindowPlacement(hwnd, ctypes.byref(wp))
        u.ShowWindow(hwnd, SW_MAXIMIZE)
        return True
    return bool(u.SetWindowPlacement(hwnd, ctypes.byref(wp)))


def foreground() -> Win | None:
    """The window in front right now (None off Windows or for the desktop)."""
    if not IS_WINDOWS:
        return None
    ctypes, wintypes, u, _, _, _ = _api()
    hwnd = u.GetForegroundWindow()
    if not hwnd:
        return None
    n = u.GetWindowTextLengthW(hwnd)
    title = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, title, n + 1)
    return Win(int(hwnd), _exe_of(hwnd), title.value, [0, 0, 0, 0])


def minimize(hwnd: int) -> None:
    if IS_WINDOWS:
        import ctypes
        ctypes.windll.user32.ShowWindow(hwnd, 6)                            # SW_MINIMIZE


def close(hwnd: int) -> None:
    """Ask the app to close this window, as clicking X does (it can still ask to save your work)."""
    if IS_WINDOWS:
        import ctypes
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)             # WM_CLOSE


def launch(exe_or_target: str) -> None:
    if IS_WINDOWS:
        os.startfile(exe_or_target)                                          # noqa: Windows only


def wait_for(app: str, known: set[int], timeout: float = 12.0, poll: float = 0.4) -> Win | None:
    """A new window of this program (app = exe file name), after launching it."""
    end = time.time() + timeout
    while time.time() < end:
        for w in windows():
            if w.app == app and w.hwnd not in known:
                return w
        time.sleep(poll)
    return None
