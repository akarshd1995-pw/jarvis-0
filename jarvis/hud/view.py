"""The HUD: a translucent, always-on-top overlay in a screen corner (tkinter Canvas, no extra packages).

  - status orb: rotating rings + breathing core; colour and speed follow Jarvis's state
      idle (cyan, slow) · thinking (violet, fast) · success (green flash) · error (amber flash) · alert (red pulse)
  - ring gauges: CPU, RAM, GPU (with temperature), disk; battery if present
  - running jobs: training progress bars with time left (read from the training logs)
  - next reminders, last reply, clock
By default it is click-through (mouse passes straight through it) and hidden from the taskbar.
Our own design; no film assets.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import textwrap
import time
import tkinter as tk
from typing import Callable

from .metrics import MetricsSampler

BG, PANEL, LINE = "#0b0f14", "#111820", "#1f2a36"
TEXT, MUTED = "#e6edf3", "#7d8b99"
STATE_COLORS = {"idle": "#4fd1ff", "thinking": "#a78bfa", "success": "#4ade80", "error": "#fbbf24",
                "alert": "#f87171"}
STATE_LABELS = {"idle": "Standing by", "thinking": "Working on it", "success": "Done", "error": "Needs attention",
                "alert": "Warning"}
STATE_SPEED = {"idle": 1.2, "thinking": 7.0, "success": 3.0, "error": 2.0, "alert": 4.0}
FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO = "Consolas" if sys.platform == "win32" else "Courier"
W, H, PAD = 340, 470, 16


def _blend(hex_a: str, hex_b: str, t: float) -> str:
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


class HUD:
    def __init__(self, root: tk.Misc, sampler: MetricsSampler, name: str = "Jarvis",
                 reminders: Callable[[], list[tuple[float, str]]] | None = None,
                 corner: str = "top-right", opacity: float = 0.9, click_through: bool = True):
        self.sampler, self.name, self.reminders = sampler, name, reminders or (lambda: [])
        self.state, self.state_until = "idle", 0.0
        self.last_text, self.last_time = "", None
        self.angle, self.visible = 0.0, True
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-alpha", opacity)
        except tk.TclError:
            pass
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = sw - W - 24 if "right" in corner else 24
        y = 24 if "top" in corner else sh - H - 64
        self.win.geometry(f"{W}x{H}+{x}+{y}")
        self.c = tk.Canvas(self.win, width=W, height=H, bg=BG, highlightthickness=0)
        self.c.pack()
        self._build_static()
        if click_through:
            self.win.after(200, self._make_click_through)
        self._tick()
        self._refresh()

    # ------------------------------------------------------------------ public API (call on the UI thread)
    def set_state(self, state: str, hold_s: float = 0.0):
        self.state = state if state in STATE_COLORS else "idle"
        self.state_until = time.time() + hold_s if hold_s else 0.0
        self.c.itemconfigure(self.state_label, text=STATE_LABELS[self.state], fill=STATE_COLORS[self.state])

    def set_last(self, text: str):
        self.last_text, self.last_time = text, dt.datetime.now()
        self._draw_last()

    def toggle(self):
        self.visible = not self.visible
        (self.win.deiconify if self.visible else self.win.withdraw)()

    # ------------------------------------------------------------------ drawing
    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2,
               x1, y2 - r, x1, y1 + r, x1, y1]
        return self.c.create_polygon(pts, smooth=True, **kw)

    def _build_static(self):
        c = self.c
        self._round_rect(2, 2, W - 2, H - 2, 18, fill=PANEL, outline=LINE, width=1)
        self.ox, self.oy, self.orad = 64, 70, 38
        self.rings = [c.create_arc(0, 0, 1, 1, style="arc", width=w, start=0, extent=e)
                      for w, e in ((3, 110), (2, 70), (2, 140))]
        self.core = c.create_oval(0, 0, 1, 1, outline="", fill=STATE_COLORS["idle"])
        self.halo = c.create_oval(0, 0, 1, 1, outline=STATE_COLORS["idle"], width=1)
        c.create_text(122, 48, text=self.name.upper(), anchor="w", fill=TEXT, font=(FONT, 15, "bold"))
        self.state_label = c.create_text(122, 74, text=STATE_LABELS["idle"], anchor="w",
                                         fill=STATE_COLORS["idle"], font=(FONT, 10))
        self.clock = c.create_text(W - PAD, 48, anchor="e", fill=TEXT, font=(MONO, 13))
        self.date = c.create_text(W - PAD, 70, anchor="e", fill=MUTED, font=(FONT, 9))
        c.create_line(PAD, 124, W - PAD, 124, fill=LINE)
        self.gauges = {}
        labels = ["CPU", "RAM", "GPU", "DISK"]
        gap = (W - 2 * PAD) / len(labels)
        for i, lab in enumerate(labels):
            cx, cy, r = PAD + gap * (i + 0.5), 172, 26
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=LINE, width=5)
            arc = c.create_arc(cx - r, cy - r, cx + r, cy + r, style="arc", width=5, start=90, extent=0,
                               outline=STATE_COLORS["idle"])
            val = c.create_text(cx, cy, text="--", fill=TEXT, font=(FONT, 10, "bold"))
            c.create_text(cx, cy + r + 12, text=lab, fill=MUTED, font=(FONT, 8))
            sub = c.create_text(cx, cy + r + 25, text="", fill=MUTED, font=(FONT, 8))
            self.gauges[lab] = (arc, val, sub)
        c.create_line(PAD, 238, W - PAD, 238, fill=LINE)
        c.create_text(PAD, 254, text="JOBS", anchor="w", fill=MUTED, font=(FONT, 8, "bold"))
        self.jobs_items: list[int] = []
        c.create_text(PAD, 318, text="UP NEXT", anchor="w", fill=MUTED, font=(FONT, 8, "bold"))
        self.rem_items: list[int] = []
        c.create_text(PAD, 400, text="LAST", anchor="w", fill=MUTED, font=(FONT, 8, "bold"))
        self.last_item = c.create_text(PAD, 414, anchor="nw", fill=TEXT, font=(FONT, 9), width=W - 2 * PAD)
        self.last_when = c.create_text(W - PAD, 400, anchor="e", fill=MUTED, font=(FONT, 8))

    def _tick(self):
        """Animation frame (~25 fps)."""
        if self.state_until and time.time() > self.state_until:
            self.set_state("idle")
        col = STATE_COLORS[self.state]
        self.angle = (self.angle + STATE_SPEED[self.state]) % 360
        t = time.time()
        breathe = (math.sin(t * (4 if self.state in ("thinking", "alert") else 1.6)) + 1) / 2
        ox, oy, r = self.ox, self.oy, self.orad
        for i, ring in enumerate(self.rings):
            rr = r - i * 8
            self.c.coords(ring, ox - rr, oy - rr, ox + rr, oy + rr)
            direction = -1 if i % 2 else 1
            self.c.itemconfigure(ring, start=(self.angle * direction * (1 + i * 0.35)) % 360,
                                 outline=_blend(PANEL, col, 0.95 - i * 0.25))
        cr = 9 + 3 * breathe
        self.c.coords(self.core, ox - cr, oy - cr, ox + cr, oy + cr)
        self.c.itemconfigure(self.core, fill=_blend(PANEL, col, 0.7 + 0.3 * breathe))
        hr = r + 4 + 4 * breathe
        self.c.coords(self.halo, ox - hr, oy - hr, ox + hr, oy + hr)
        self.c.itemconfigure(self.halo, outline=_blend(PANEL, col, 0.15 + 0.25 * breathe))
        self.win.after(40, self._tick)

    def _gauge(self, name: str, pct: float | None, sub: str = "", warn: bool = False):
        arc, val, subi = self.gauges[name]
        if pct is None:
            self.c.itemconfigure(val, text="--")
            self.c.itemconfigure(arc, extent=0)
        else:
            col = STATE_COLORS["alert"] if warn else STATE_COLORS["error"] if pct >= 85 else STATE_COLORS["idle"]
            self.c.itemconfigure(arc, extent=-3.6 * max(0.0, min(100.0, pct)), outline=col)
            self.c.itemconfigure(val, text=f"{pct:.0f}%")
        self.c.itemconfigure(subi, text=sub)

    def _refresh(self):
        """Data refresh (1 Hz)."""
        now = dt.datetime.now()
        self.c.itemconfigure(self.clock, text=now.strftime("%H:%M"))
        self.c.itemconfigure(self.date, text=now.strftime("%a %d %b"))
        s = self.sampler.snapshot
        self._gauge("CPU", s.cpu)
        self._gauge("RAM", s.ram, f"{s.ram_used_gb:.1f}/{s.ram_total_gb:.0f} GB" if s.ram_used_gb else "")
        hot = s.gpu_temp is not None and s.gpu_temp >= 85
        self._gauge("GPU", s.gpu, f"{s.gpu_temp:.0f}°C" if s.gpu_temp is not None else "", warn=hot)
        disk_sub = f"bat {s.battery:.0f}%{'+' if s.charging else ''}" if s.battery is not None else "C:"
        self._gauge("DISK", s.disk, disk_sub)
        if hot and self.state == "idle":
            self.set_state("alert", hold_s=5)
        self._draw_jobs(s.jobs)
        self._draw_reminders()
        self._draw_last()
        self.win.after(1000, self._refresh)

    def _draw_jobs(self, jobs: list[dict]):
        for i in self.jobs_items:
            self.c.delete(i)
        self.jobs_items = []
        if not jobs:
            self.jobs_items.append(self.c.create_text(PAD, 276, text="Nothing running", anchor="w", fill=MUTED,
                                                      font=(FONT, 9)))
            return
        y = 272
        for j in jobs[:2]:
            x1, x2 = PAD, W - PAD
            label = f"{j['name']}  {j['pct']:.0f}%  ·  ~{j['eta_min']} min left"
            self.jobs_items.append(self.c.create_text(x1, y, text=label, anchor="w", fill=TEXT, font=(FONT, 9)))
            self.jobs_items.append(self.c.create_rectangle(x1, y + 10, x2, y + 14, fill=LINE, outline=""))
            fill_x = x1 + (x2 - x1) * j["pct"] / 100
            self.jobs_items.append(self.c.create_rectangle(x1, y + 10, fill_x, y + 14, fill=STATE_COLORS["idle"],
                                                           outline=""))
            y += 26

    def _draw_reminders(self):
        for i in self.rem_items:
            self.c.delete(i)
        self.rem_items = []
        items = self.reminders()[:3]
        if not items:
            self.rem_items.append(self.c.create_text(PAD, 340, text="No reminders", anchor="w", fill=MUTED,
                                                     font=(FONT, 9)))
            return
        for k, (due, text) in enumerate(items):
            d = dt.datetime.fromtimestamp(due)
            when = d.strftime("%H:%M") if d.date() == dt.date.today() else d.strftime("%a %H:%M")
            y = 338 + k * 16
            self.rem_items.append(self.c.create_text(PAD, y, text=when, anchor="w", fill=STATE_COLORS["idle"],
                                                     font=(MONO, 9)))
            self.rem_items.append(self.c.create_text(PAD + 78, y, text=textwrap.shorten(text, 34, placeholder="…"),
                                                     anchor="w", fill=TEXT, font=(FONT, 9)))

    def _draw_last(self):
        text = self.last_text.split("\n\n")[0] if self.last_text else "Nothing yet"
        self.c.itemconfigure(self.last_item, text=textwrap.shorten(text, 150, placeholder="…"),
                             fill=TEXT if self.last_text else MUTED)
        self.c.itemconfigure(self.last_when, text=self.last_time.strftime("%H:%M") if self.last_time else "")

    def _make_click_through(self):
        """Windows: let the mouse pass through the overlay and keep it out of the taskbar / Alt+Tab."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
            GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = -20, 0x80000, 0x20, 0x80
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
        except Exception:
            pass
