"""The HUD: a translucent, always-on-top overlay in a screen corner (tkinter Canvas, no extra packages).

  - status orb: segmented outer ring, tick ring, counter-rotating arcs and a breathing core; colour and
    speed follow Jarvis's state: idle (cyan) · thinking (violet, fast) · success (green) · error (amber) · alert (red)
  - ring gauges: CPU, RAM, GPU (with temperature), disk; battery if present
  - cards: running jobs (training progress with time left), next reminders, last reply
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

from ..ui import theme as Th
from .metrics import MetricsSampler

BG, PANEL, CARD, LINE = Th.BG, Th.SURFACE, Th.RAISED, Th.BORDER
TEXT, MUTED, FAINT = Th.TEXT, Th.MUTED, Th.FAINT
STATE_COLORS = dict(Th.STATE)
STATE_LABELS = {"idle": "Standing by", "thinking": "Working on it", "success": "Done", "error": "Needs attention",
                "alert": "Warning"}
STATE_SPEED = {"idle": 1.0, "thinking": 6.5, "success": 2.5, "error": 2.0, "alert": 3.5}
W, H, PAD = 340, 500, 16


def _blend(a: str, b: str, t: float) -> str:
    return Th.blend(a, b, t)


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
        self.win.configure(bg=BG)
        try:
            self.win.attributes("-alpha", opacity)
            if sys.platform == "win32":
                self.win.attributes("-transparentcolor", BG)     # only the rounded panel is visible
        except tk.TclError:
            pass
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = sw - W - 24 if "right" in corner else 24
        y = 24 if "top" in corner else sh - H - 64
        self.win.geometry(f"{W}x{H}+{x}+{y}")
        self.c = tk.Canvas(self.win, width=W, height=H, bg=BG, highlightthickness=0)
        self.c.pack()
        self.F = Th.F
        self._build_static()
        if click_through:
            self.win.after(200, self._make_click_through)
        self._tick()
        self._refresh()

    # ------------------------------------------------------------------ public API (call on the UI thread)
    def set_state(self, state: str, hold_s: float = 0.0):
        self.state = state if state in STATE_COLORS else "idle"
        self.state_until = time.time() + hold_s if hold_s else 0.0
        col = STATE_COLORS[self.state]
        self.c.itemconfigure(self.state_label, text=STATE_LABELS[self.state], fill=col)
        self.c.itemconfigure(self.state_dot, fill=col)

    def set_last(self, text: str):
        self.last_text, self.last_time = text, dt.datetime.now()
        self._draw_last()

    def toggle(self):
        self.visible = not self.visible
        (self.win.deiconify if self.visible else self.win.withdraw)()

    # ------------------------------------------------------------------ drawing
    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        return Th.round_rect(self.c, x1, y1, x2, y2, r, **kw)

    def _section(self, y: int, title: str):
        self.c.create_rectangle(PAD, y - 5, PAD + 2, y + 5, fill=STATE_COLORS["idle"], outline="")
        self.c.create_text(PAD + 9, y, text=title, anchor="w", fill=MUTED, font=self.F(8, "bold"))

    def _build_static(self):
        c, F = self.c, self.F
        self._round_rect(1, 1, W - 1, H - 1, 20, fill=PANEL, outline=Th.BORDER_HI, width=1)
        n = 40                                                         # accent glow along the top edge
        for i in range(n):
            t = i / (n - 1)
            col = _blend(PANEL, _blend(Th.ACCENT, Th.VIOLET, t), math.sin(math.pi * t) * 0.9)
            c.create_rectangle(24 + (W - 48) * i / n, 1, 24 + (W - 48) * (i + 1) / n + 1, 3, fill=col, outline="")

        # orb
        self.ox, self.oy, self.orad = 66, 74, 42
        ox, oy = self.ox, self.oy
        self.glow2 = c.create_oval(0, 0, 1, 1, outline="", fill=PANEL)
        self.glow1 = c.create_oval(0, 0, 1, 1, outline="", fill=PANEL)
        self.ticks = []
        for k in range(36):                                           # static tick ring
            a = math.radians(k * 10)
            r1, r2 = self.orad - 1, self.orad - (5 if k % 3 == 0 else 3)
            self.ticks.append(c.create_line(ox + r1 * math.cos(a), oy + r1 * math.sin(a),
                                            ox + r2 * math.cos(a), oy + r2 * math.sin(a), fill=LINE, width=1))
        self.segments = [c.create_arc(0, 0, 1, 1, style="arc", width=2, extent=9) for _ in range(12)]
        self.rings = [c.create_arc(0, 0, 1, 1, style="arc", width=w, start=0, extent=e)
                      for w, e in ((3, 110), (2, 200), (2, 60))]
        self.core = c.create_oval(0, 0, 1, 1, outline="", fill=STATE_COLORS["idle"])

        # header
        c.create_text(126, 52, text="  ".join(self.name.upper()), anchor="w", fill=TEXT, font=F(14, "bold", "display"))
        self.state_dot = c.create_oval(127, 72, 134, 79, outline="", fill=STATE_COLORS["idle"])
        self.state_label = c.create_text(140, 76, text=STATE_LABELS["idle"], anchor="w",
                                         fill=STATE_COLORS["idle"], font=F(10))
        self.clock = c.create_text(W - PAD, 100, anchor="e", fill=TEXT, font=F(18, "normal", "display"))
        self.date = c.create_text(W - PAD, 121, anchor="e", fill=MUTED, font=F(9))
        c.create_text(126, 100, text="SYSTEM", anchor="w", fill=FAINT, font=F(8, "bold"))
        self.summary = c.create_text(126, 117, text="", anchor="w", fill=MUTED, font=F(9))

        # gauges card
        self._round_rect(PAD - 4, 140, W - PAD + 4, 236, 14, fill=CARD, outline=LINE)
        self.gauges = {}
        labels = ["CPU", "RAM", "GPU", "DISK"]
        gap = (W - 2 * PAD) / len(labels)
        for i, lab in enumerate(labels):
            cx, cy, r = PAD + gap * (i + 0.5), 178, 24
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=_blend(CARD, LINE, 1.0), width=6)
            arc = c.create_arc(cx - r, cy - r, cx + r, cy + r, style="arc", width=6, start=90, extent=0,
                               outline=STATE_COLORS["idle"])
            val = c.create_text(cx, cy, text="--", fill=TEXT, font=F(10, "bold"))
            c.create_text(cx, cy + r + 13, text=lab, fill=MUTED, font=F(8, "bold"))
            sub = c.create_text(cx, cy + r + 25, text="", fill=FAINT, font=F(8))
            self.gauges[lab] = (arc, val, sub)

        # cards
        self._section(256, "JOBS")
        self._round_rect(PAD - 4, 266, W - PAD + 4, 330, 12, fill=CARD, outline=LINE)
        self.jobs_items: list[int] = []
        self._section(348, "UP NEXT")
        self._round_rect(PAD - 4, 358, W - PAD + 4, 414, 12, fill=CARD, outline=LINE)
        self.rem_items: list[int] = []
        self._section(432, "LAST")
        self.last_item = c.create_text(PAD + 2, 444, anchor="nw", fill=TEXT, font=F(9), width=W - 2 * PAD - 4)
        self.last_when = c.create_text(W - PAD, 432, anchor="e", fill=FAINT, font=F(8))

    def _tick(self):
        """Animation frame (~30 fps)."""
        if self.state_until and time.time() > self.state_until:
            self.set_state("idle")
        col = STATE_COLORS[self.state]
        self.angle = (self.angle + STATE_SPEED[self.state]) % 360
        t = time.time()
        fast = self.state in ("thinking", "alert")
        breathe = (math.sin(t * (4 if fast else 1.5)) + 1) / 2
        ox, oy, r = self.ox, self.oy, self.orad
        c = self.c
        for g, (rr, a) in zip((self.glow2, self.glow1), ((r + 10 + 4 * breathe, 0.05), (r + 3, 0.09))):
            c.coords(g, ox - rr, oy - rr, ox + rr, oy + rr)
            c.itemconfigure(g, fill=_blend(PANEL, col, a + 0.04 * breathe))
        so = r + 6                                                     # segmented outer ring, slow
        for k, seg in enumerate(self.segments):
            c.coords(seg, ox - so, oy - so, ox + so, oy + so)
            c.itemconfigure(seg, start=(k * 30 - self.angle * 0.35) % 360,
                            outline=_blend(PANEL, col, 0.35 + (0.4 if k % 3 == 0 else 0.0)))
        for i, ring in enumerate(self.rings):
            rr = r - 9 - i * 7
            c.coords(ring, ox - rr, oy - rr, ox + rr, oy + rr)
            direction = -1 if i % 2 else 1
            c.itemconfigure(ring, start=(self.angle * direction * (1 + i * 0.4)) % 360,
                            outline=_blend(PANEL, col, 0.95 - i * 0.28))
        cr = 7 + 3 * breathe
        c.coords(self.core, ox - cr, oy - cr, ox + cr, oy + cr)
        c.itemconfigure(self.core, fill=_blend(PANEL, col, 0.75 + 0.25 * breathe))
        for k, tk_ in enumerate(self.ticks):
            lit = (k * 10 - self.angle * 2) % 360 < 40
            c.itemconfigure(tk_, fill=_blend(LINE, col, 0.6) if lit else LINE)
        self.win.after(33, self._tick)

    def _gauge(self, name: str, pct: float | None, sub: str = "", warn: bool = False):
        arc, val, subi = self.gauges[name]
        if pct is None:
            self.c.itemconfigure(val, text="--", fill=FAINT)
            self.c.itemconfigure(arc, extent=0)
        else:
            col = STATE_COLORS["alert"] if warn else STATE_COLORS["error"] if pct >= 85 else STATE_COLORS["idle"]
            self.c.itemconfigure(arc, extent=-3.6 * max(0.5, min(100.0, pct)), outline=col)
            self.c.itemconfigure(val, text=f"{pct:.0f}%", fill=TEXT)
        self.c.itemconfigure(subi, text=sub)

    def _refresh(self):
        """Data refresh (1 Hz)."""
        now = dt.datetime.now()
        self.c.itemconfigure(self.clock, text=now.strftime("%H:%M"))
        self.c.itemconfigure(self.date, text=now.strftime("%A, %d %b"))
        s = self.sampler.snapshot
        self._gauge("CPU", s.cpu)
        self._gauge("RAM", s.ram, f"{s.ram_used_gb:.1f}/{s.ram_total_gb:.0f} GB" if s.ram_used_gb else "")
        hot = s.gpu_temp is not None and s.gpu_temp >= 85
        self._gauge("GPU", s.gpu, f"{s.gpu_temp:.0f}°C" if s.gpu_temp is not None else "", warn=hot)
        disk_sub = (f"bat {s.battery:.0f}%{'+' if s.charging else ''}" if s.battery is not None
                    else f"{s.disk_free_gb:.0f} GB free" if s.disk_free_gb is not None else "C:")
        self._gauge("DISK", s.disk, disk_sub)
        vals = [v for v in (s.cpu, s.ram, s.gpu) if v is not None]
        busy = max(vals) if vals else None
        self.c.itemconfigure(self.summary, text="—" if busy is None else
                             "Running hot" if hot else "Under load" if busy >= 85 else "Busy" if busy >= 50
                             else "All nominal")
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
            self.jobs_items.append(self.c.create_text(PAD + 8, 298, text="Nothing running", anchor="w", fill=FAINT,
                                                      font=self.F(9)))
            return
        y = 282
        for j in jobs[:2]:
            x1, x2 = PAD + 8, W - PAD - 8
            name = textwrap.shorten(j["name"], 26, placeholder="…")
            self.jobs_items.append(self.c.create_text(x1, y, text=name, anchor="w", fill=TEXT, font=self.F(9)))
            self.jobs_items.append(self.c.create_text(x2, y, text=f"{j['pct']:.0f}%  ·  ~{j['eta_min']} min",
                                                      anchor="e", fill=MUTED, font=self.F(8)))
            self.jobs_items.append(self.c.create_line(x1, y + 13, x2, y + 13, fill=LINE, width=5, capstyle="round"))
            fill_x = x1 + (x2 - x1) * max(0.01, j["pct"] / 100)
            self.jobs_items.append(self.c.create_line(x1, y + 13, fill_x, y + 13, fill=STATE_COLORS["idle"], width=5,
                                                      capstyle="round"))
            y += 26

    def _draw_reminders(self):
        for i in self.rem_items:
            self.c.delete(i)
        self.rem_items = []
        items = self.reminders()[:3]
        if not items:
            self.rem_items.append(self.c.create_text(PAD + 8, 386, text="No reminders", anchor="w", fill=FAINT,
                                                     font=self.F(9)))
            return
        for k, (due, text) in enumerate(items):
            d = dt.datetime.fromtimestamp(due)
            when = d.strftime("%H:%M") if d.date() == dt.date.today() else d.strftime("%a %H:%M")
            y = 372 + k * 15
            self.rem_items.append(self.c.create_text(PAD + 8, y, text=when, anchor="w", fill=STATE_COLORS["idle"],
                                                     font=self.F(9, kind="mono")))
            self.rem_items.append(self.c.create_text(PAD + 84, y, text=textwrap.shorten(text, 32, placeholder="…"),
                                                     anchor="w", fill=TEXT, font=self.F(9)))

    def _draw_last(self):
        text = self.last_text.split("\n\n")[0] if self.last_text else "Nothing yet"
        self.c.itemconfigure(self.last_item, text=textwrap.shorten(text, 140, placeholder="…"),
                             fill=TEXT if self.last_text else FAINT)
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
