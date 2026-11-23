"""One visual language for the command bar, HUD, toasts and settings: deep navy glass, cyan accent.

Our own design. On Windows 11 windows also get rounded corners and a dark title bar through DWM;
older Windows just keeps square corners.
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont

# palette
BG = "#070b11"          # deepest background
SURFACE = "#0d141d"     # panels
RAISED = "#131c28"      # cards, inputs, chips
BORDER = "#1e2a3a"
BORDER_HI = "#2b3b50"
TEXT = "#e8eef5"
MUTED = "#7f8fa3"
FAINT = "#4a5a6e"
ACCENT = "#4fd1ff"      # cyan
ACCENT_DIM = "#1b4f66"
VIOLET = "#a78bfa"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"

STATE = {"idle": ACCENT, "thinking": VIOLET, "success": GREEN, "error": AMBER, "alert": RED}

_fonts: dict[str, str] = {}


def font_family(kind: str = "ui") -> str:
    """Best installed font: Segoe UI Variable on Windows 11, Segoe UI on 10, Inter/Helvetica elsewhere."""
    if kind in _fonts:
        return _fonts[kind]
    prefs = {
        "ui": ["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica"],
        "display": ["Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", "Inter Display", "Inter", "Helvetica"],
        "mono": ["Cascadia Mono", "Cascadia Code", "Consolas", "DejaVu Sans Mono", "Courier"],
    }[kind]
    try:
        have = set(tkfont.families())
    except tk.TclError:
        have = set()
    _fonts[kind] = next((f for f in prefs if f in have), prefs[-1])
    return _fonts[kind]


def F(size: int, weight: str = "normal", kind: str = "ui") -> tuple:
    return (font_family(kind), size, weight)


def blend(a: str, b: str, t: float) -> str:
    x = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    y = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(p + (q - p) * t):02x}" for p, q in zip(x, y))


def round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw) -> int:
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2,
           x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def hwnd_of(win: tk.Misc) -> int:
    import ctypes
    wid = win.winfo_id()
    return ctypes.windll.user32.GetParent(wid) or wid


def win11_style(win: tk.Misc, round_corners: bool = True, dark: bool = True) -> None:
    """Rounded corners + dark title bar on Windows 11 (silently nothing elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = hwnd_of(win)
        dwm = ctypes.windll.dwmapi
        if dark:
            v = ctypes.c_int(1)
            dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(v), ctypes.sizeof(v))     # immersive dark mode
        if round_corners:
            v = ctypes.c_int(2)
            dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(v), ctypes.sizeof(v))     # DWMWCP_ROUND
        v = ctypes.c_int(0x00261c13)                                                  # border colour (BGR)
        dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


def fade(win: tk.Misc, to: float, step: float = 0.12, ms: int = 12, done=None) -> None:
    """Fade the window's opacity to `to`."""
    try:
        cur = float(win.attributes("-alpha"))
    except (tk.TclError, ValueError):
        if done:
            done()
        return
    nxt = cur + step if to > cur else cur - step
    if (to > cur and nxt >= to) or (to <= cur and nxt <= to):
        win.attributes("-alpha", to)
        if done:
            done()
        return
    win.attributes("-alpha", nxt)
    win.after(ms, lambda: fade(win, to, step, ms, done))
