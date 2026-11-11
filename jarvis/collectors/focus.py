"""Foreground-window collector (Windows, ctypes only).

Logs one 'focus' event whenever the active app or window changes. This stream is the
training data for the routine model in phase 4. Window titles are stored as a coarse
category plus a hash by default so the log stays privacy-friendly.
"""
from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path

from ..store import Store

IS_WINDOWS = sys.platform == "win32"


def _foreground() -> tuple[str, str] | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32       # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32   # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    exe = ""
    if h:
        size = wintypes.DWORD(1024)
        pbuf = ctypes.create_unicode_buffer(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, pbuf, ctypes.byref(size)):
            exe = Path(pbuf.value).name.lower()
        kernel32.CloseHandle(h)
    return exe, buf.value


def title_digest(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8", "ignore")).hexdigest()[:10]


class FocusCollector(threading.Thread):
    def __init__(self, store: Store, poll: float = 1.0, keep_titles: bool = False):
        super().__init__(daemon=True, name="focus-collector")
        self.store, self.poll, self.keep_titles = store, poll, keep_titles
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        if not IS_WINDOWS:
            return
        last = None
        while not self._stop.is_set():
            try:
                fg = _foreground()
                if fg and fg != last:
                    exe, title = fg
                    payload = {"app": exe, "title_hash": title_digest(title)}
                    if self.keep_titles:
                        payload["title"] = title
                    self.store.log_event("focus", payload)
                    last = fg
            except Exception:
                pass
            self._stop.wait(self.poll)
