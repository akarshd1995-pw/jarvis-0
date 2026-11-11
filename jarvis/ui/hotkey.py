"""Global hotkey via Win32 RegisterHotKey (ctypes, no dependencies)."""
from __future__ import annotations

import sys
import threading
from typing import Callable

MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
SPECIAL_VK = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "f1": 0x70, "f2": 0x71, "f3": 0x72,
              "f4": 0x73, "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78,
              "f10": 0x79, "f11": 0x7A, "f12": 0x7B}


def parse_hotkey(spec: str) -> tuple[int, int]:
    mods, vk = MOD_NOREPEAT, None
    for part in spec.lower().split("+"):
        part = part.strip()
        if part in MODS:
            mods |= MODS[part]
        elif part in SPECIAL_VK:
            vk = SPECIAL_VK[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            raise ValueError(f"unsupported key {part!r} in hotkey {spec!r}")
    if vk is None:
        raise ValueError(f"hotkey {spec!r} has no key")
    return mods, vk


class HotkeyThread(threading.Thread):
    def __init__(self, spec: str, on_press: Callable[[], None]):
        super().__init__(daemon=True, name="hotkey")
        self.mods, self.vk = parse_hotkey(spec)
        self.spec, self.on_press = spec, on_press
        self.error: str | None = None

    def run(self):
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if not user32.RegisterHotKey(None, 1, self.mods, self.vk):
            self.error = f"could not register {self.spec} (already in use?)"
            return
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY:
                    self.on_press()
        finally:
            user32.UnregisterHotKey(None, 1)
