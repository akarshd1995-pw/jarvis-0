"""Spotlight-style command bar (tkinter, ships with Python on Windows).

Ctrl+Alt+J (configurable) toggles it. Type, press Enter. Esc hides it. Up/Down walk through history.

  ┌──────────────────────────────────────────────────────────────┐
  │ (orb)  Ask Jarvis anything…                          Enter ⏎ │
  ├──────────────────────────────────────────────────────────────┤
  │ [good morning] [play lofi girl live] [i'm feeling low] …     │  suggestions when idle
  │ MODEL 92%                                            10:42   │  or the reply card
  │ Right away, sir. Moved 12 PDFs to Archive.                   │
  │ move_files(dest=Archive, ext=pdf, src=Downloads)             │
  └──────────────────────────────────────────────────────────────┘

Commands run on a worker thread, so the orb keeps animating and the bar never freezes. Confirmations
appear inside the bar (Enter = yes, Esc = no) instead of a grey Windows dialog; reminders arrive as toasts.
"""
from __future__ import annotations

import datetime as dt
import math
import queue
import random
import re
import threading
import time
import tkinter as tk
import tkinter.font  # noqa: F401  (tk.font)
from tkinter import messagebox
from typing import Callable

from ..safety import Tier
from . import theme as T

W = 760
MAX_H = 560
SUGGESTIONS = [
    "good morning", "brief me", "play lofi girl live", "i'm feeling low", "show pdfs in downloads",
    "play 301 diaries latest video", "take a screenshot", "remind me in 20 mins to stretch", "how was my day",
    "volume 40", "snap this to the left", "is training done?", "search react hooks on stackoverflow",
    "yes theory ki latest video chalao", "open my downloads folder", "tell me a joke",
]


class Orb:
    """Small animated status orb (our own design): breathing core inside two counter-rotating arcs."""

    def __init__(self, parent: tk.Misc, size: int = 40, bg: str = T.SURFACE):
        self.c = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0, bd=0)
        self.size, self.bg, self.state, self.angle = size, bg, "idle", 0.0
        self.halo = self.c.create_oval(0, 0, 1, 1, outline="", fill=bg)
        self.ring1 = self.c.create_arc(0, 0, 1, 1, style="arc", width=2, extent=120)
        self.ring2 = self.c.create_arc(0, 0, 1, 1, style="arc", width=2, extent=70)
        self.core = self.c.create_oval(0, 0, 1, 1, outline="")
        self._tick()

    def set(self, state: str):
        self.state = state if state in T.STATE else "idle"

    def _tick(self):
        s, col = self.size, T.STATE[self.state]
        fast = self.state == "thinking"
        self.angle = (self.angle + (9 if fast else 2.2)) % 360
        b = (math.sin(time.time() * (5 if fast else 1.7)) + 1) / 2
        m = s / 2
        self.c.coords(self.halo, 2, 2, s - 2, s - 2)
        self.c.itemconfigure(self.halo, fill=T.blend(self.bg, col, 0.06 + 0.06 * b))
        r1, r2 = m - 4, m - 9
        self.c.coords(self.ring1, m - r1, m - r1, m + r1, m + r1)
        self.c.coords(self.ring2, m - r2, m - r2, m + r2, m + r2)
        self.c.itemconfigure(self.ring1, start=self.angle, outline=T.blend(self.bg, col, 0.95))
        self.c.itemconfigure(self.ring2, start=-self.angle * 1.4, outline=T.blend(self.bg, col, 0.55))
        cr = 3.5 + 2 * b
        self.c.coords(self.core, m - cr, m - cr, m + cr, m + cr)
        self.c.itemconfigure(self.core, fill=T.blend(self.bg, col, 0.75 + 0.25 * b))
        self.c.after(33, self._tick)


class Chip(tk.Label):
    def __init__(self, parent, text: str, command: Callable[[str], None]):
        super().__init__(parent, text=text, bg=T.RAISED, fg=T.MUTED, font=T.F(10), padx=11, pady=5,
                         cursor="hand2", bd=0, highlightthickness=1, highlightbackground=T.BORDER)
        self.bind("<Enter>", lambda e: self.configure(bg=T.blend(T.RAISED, T.ACCENT, 0.12), fg=T.TEXT,
                                                      highlightbackground=T.ACCENT_DIM))
        self.bind("<Leave>", lambda e: self.configure(bg=T.RAISED, fg=T.MUTED, highlightbackground=T.BORDER))
        self.bind("<Button-1>", lambda e: command(text))


class Toast:
    """Card that slides in at the bottom-right (reminders), fades away on its own or on click."""
    _stack: list["Toast"] = []

    def __init__(self, root: tk.Misc, title: str, body: str, color: str = T.ACCENT, seconds: float = 20):
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        try:
            w.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        outer = tk.Frame(w, bg=T.BORDER_HI)
        outer.pack(fill="both", expand=True)
        card = tk.Frame(outer, bg=T.SURFACE)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Frame(card, bg=color, width=4).pack(side="left", fill="y")
        inner = tk.Frame(card, bg=T.SURFACE, padx=14, pady=12)
        inner.pack(side="left", fill="both", expand=True)
        head = tk.Frame(inner, bg=T.SURFACE)
        head.pack(fill="x")
        tk.Label(head, text=title.upper(), bg=T.SURFACE, fg=color, font=T.F(9, "bold")).pack(side="left")
        tk.Label(head, text=dt.datetime.now().strftime("%H:%M"), bg=T.SURFACE, fg=T.FAINT,
                 font=T.F(9)).pack(side="right")
        tk.Label(inner, text=body, bg=T.SURFACE, fg=T.TEXT, font=T.F(11), wraplength=330, justify="left",
                 anchor="w").pack(fill="x", pady=(6, 8))
        tk.Label(inner, text="Dismiss", bg=T.SURFACE, fg=T.MUTED, font=T.F(9, "underline"),
                 cursor="hand2").pack(anchor="e")
        for wid in (w, outer, card, inner, head) + tuple(inner.winfo_children()):
            wid.bind("<Button-1>", lambda e: self.close())
        w.update_idletasks()
        self.h = w.winfo_reqheight()
        Toast._stack.append(self)
        self._place()
        T.win11_style(w)
        T.fade(w, 0.97)
        w.after(int(seconds * 1000), self.close)

    def _place(self):
        from .. import machine                        # bottom-right of the monitor you're on, above the taskbar
        x, bottom = machine.place(380, 0, "toast", fallback=(self.win.winfo_screenwidth(),
                                                             self.win.winfo_screenheight() - 44))
        y = bottom
        for t in Toast._stack:
            y -= t.h + 10
            if t is self:
                break
        self.win.geometry(f"380x{self.h}+{x}+{y}")

    def close(self):
        if self not in Toast._stack:
            return
        Toast._stack.remove(self)

        def gone():
            try:
                self.win.destroy()
            except tk.TclError:
                pass
            for t in Toast._stack:
                t._place()
        T.fade(self.win, 0.0, done=gone)


class CommandBar:
    def __init__(self, on_submit: Callable[[str], str]):
        self.on_submit = on_submit
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()  # thread-safe inbox
        self.history: list[str] = []
        self.hist_i = 0
        self.busy = False
        self.results_provider: Callable[[], list[dict]] | None = None   # memory search rows (phase 7)
        self._rows: list[dict] = []
        self._pending_confirm = None
        self._ui_thread = threading.current_thread()

        self.root = r = tk.Tk()
        r.title("Jarvis-0")
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.configure(bg=T.BORDER_HI)
        try:
            r.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        self.x = (r.winfo_screenwidth() - W) // 2
        self.y = int(r.winfo_screenheight() * 0.18)

        self.frame = tk.Frame(r, bg=T.SURFACE)
        self.frame.pack(fill="both", expand=True, padx=1, pady=1)
        self.glow = tk.Canvas(self.frame, height=2, bg=T.SURFACE, highlightthickness=0, bd=0)
        self.glow.pack(fill="x")
        self._draw_glow()

        # input row
        row = tk.Frame(self.frame, bg=T.SURFACE)
        row.pack(fill="x", padx=16, pady=(12, 12))
        self.orb = Orb(row, 40)
        self.orb.c.pack(side="left")
        self.hint = tk.Label(row, text="Enter ⏎", bg=T.SURFACE, fg=T.FAINT, font=T.F(9))
        self.hint.pack(side="right")
        holder = tk.Frame(row, bg=T.SURFACE)
        holder.pack(side="left", fill="x", expand=True, padx=(12, 8))
        self.entry = tk.Entry(holder, font=T.F(17), bg=T.SURFACE, fg=T.TEXT, insertbackground=T.ACCENT,
                              relief="flat", bd=0, highlightthickness=0, disabledbackground=T.SURFACE,
                              disabledforeground=T.MUTED, selectbackground=T.ACCENT_DIM, selectforeground=T.TEXT)
        self.entry.pack(fill="x", ipady=4)
        self.placeholder = tk.Label(holder, text="Ask Jarvis anything…", bg=T.SURFACE, fg=T.FAINT, font=T.F(17))
        self.placeholder.place(x=1, rely=0.5, anchor="w")
        self.placeholder.bind("<Button-1>", lambda e: self.entry.focus_set())

        self.sep = tk.Frame(self.frame, bg=T.BORDER, height=1)
        self.sep.pack(fill="x")
        self.body = tk.Frame(self.frame, bg=T.SURFACE)
        self.body.pack(fill="both", expand=True, padx=18, pady=(12, 14))

        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self.hide())
        self.entry.bind("<Up>", lambda e: self._history(-1))
        self.entry.bind("<Down>", lambda e: self._history(1))
        self.entry.bind("<KeyRelease>", lambda e: self._sync_placeholder())
        for n in range(1, 10):
            r.bind(f"<Control-Key-{n}>", lambda e, i=n - 1: self._open_row(i))
        r.bind("<Return>", self._key_yes, add="+")
        r.bind("<Escape>", self._key_no, add="+")

        self._show_suggestions()
        r.after(100, self._pump)
        r.withdraw()
        r.after(50, lambda: T.win11_style(r))

    # ---------------------------------------------------------- thread-safe API
    def post(self, kind: str, payload: object = "") -> None:
        """Called from any thread: toggle | open | reminder | message | call | quit."""
        self.events.put((kind, payload))

    def _pump(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "toggle":
                    self.hide() if self.root.state() == "normal" else self.show()
                elif kind == "open":
                    self.show()
                elif kind == "reminder":
                    Toast(self.root, "Reminder", str(payload), T.ACCENT, seconds=30)
                elif kind == "message":
                    self.show()
                    self._show_reply(str(payload))
                elif kind == "result":
                    self._finish(str(payload))
                elif kind == "confirm_ui":
                    self._show_confirm(*payload)
                elif kind == "call":           # run a function on the UI thread (payload = callable)
                    payload()
                elif kind == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(60, self._pump)

    # ---------------------------------------------------------- window
    def show(self):
        r = self.root
        visible = r.state() == "normal"
        if not visible:                               # phase 44: open on the monitor your mouse is on
            from .. import machine
            pos = machine.place(W, 0, "center", fallback=(r.winfo_screenwidth(), r.winfo_screenheight()))
            self.x, self.y = pos
        r.deiconify()
        self._resize()
        r.lift()
        r.focus_force()
        self.entry.focus_set()
        self.entry.select_range(0, "end")
        if not visible:
            T.fade(r, 0.97)

    def hide(self):
        if self._pending_confirm is not None:
            self._answer(False)
            return
        T.fade(self.root, 0.0, step=0.2, done=self.root.withdraw)

    def _resize(self):
        self.root.update_idletasks()
        h = min(MAX_H, self.frame.winfo_reqheight() + 2)
        self.root.geometry(f"{W}x{h}+{self.x}+{self.y}")

    def _draw_glow(self):
        c = self.glow
        c.delete("all")
        n = 60
        for i in range(n):
            t = i / (n - 1)
            col = T.blend(T.ACCENT, T.VIOLET, t)
            a = math.sin(math.pi * t)                     # bright in the middle, fading at the ends
            c.create_rectangle(W * i / n, 0, W * (i + 1) / n + 1, 2, fill=T.blend(T.SURFACE, col, a), outline="")

    def _sync_placeholder(self):
        if self.entry.get():
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=1, rely=0.5, anchor="w")

    def _clear_body(self):
        for wdg in self.body.winfo_children():
            wdg.destroy()

    # ---------------------------------------------------------- body states
    def _show_suggestions(self):
        self._clear_body()
        tk.Label(self.body, text="TRY", bg=T.SURFACE, fg=T.FAINT, font=T.F(8, "bold")).pack(anchor="w", pady=(0, 6))
        wrap = tk.Frame(self.body, bg=T.SURFACE)
        wrap.pack(fill="x")
        f = tk.font.Font(font=T.F(10))
        rows, used, limit = [tk.Frame(wrap, bg=T.SURFACE)], 0, W - 50
        rows[0].pack(fill="x", pady=3)
        for s in random.sample(SUGGESTIONS, 8):
            wpx = f.measure(s) + 2 * 11 + 2 + 8               # text + padding + border + gap
            if used + wpx > limit:
                if len(rows) == 2:                             # at most two rows
                    break
                rows.append(tk.Frame(wrap, bg=T.SURFACE))
                rows[-1].pack(fill="x", pady=3)
                used = 0
            Chip(rows[-1], s, self._run_suggestion).pack(side="left", padx=(0, 8))
            used += wpx

    def _run_suggestion(self, text: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self._sync_placeholder()
        self._submit()

    def _show_thinking(self):
        self._clear_body()
        lab = tk.Label(self.body, text="Working on it", bg=T.SURFACE, fg=T.VIOLET, font=T.F(11))
        lab.pack(anchor="w")

        def dots(n=0):
            if self.busy and lab.winfo_exists():
                lab.configure(text="Working on it" + "." * (n % 4))
                lab.after(300, lambda: dots(n + 1))
        dots()
        self._resize()

    @staticmethod
    def _split(result: str) -> tuple[str, str, str]:
        """'reply\\n\\n[model 92%] prog · reply: x' -> (reply, badge, technical line)."""
        reply, detail = result, ""
        if "\n\n" in result:
            head, tail = result.rsplit("\n\n", 1)
            if tail.startswith("["):
                reply, detail = head, tail
        badge = ""
        m = re.match(r"\[([^\]]+)\]\s*(.*)", detail, re.S)
        if m:
            badge, detail = m.group(1), m.group(2)
            detail = re.sub(r"\s*·\s*reply:\s*\S+\s*$", "", detail).strip()
        return reply.strip(), badge, detail

    def _show_reply(self, result: str, rows: list[dict] | None = None):
        self._clear_body()
        self._rows = rows or []
        reply, badge, detail = self._split(result)
        head = tk.Frame(self.body, bg=T.SURFACE)
        head.pack(fill="x", pady=(0, 6))
        if badge:
            tk.Label(head, text=badge.upper(), bg=T.blend(T.SURFACE, T.ACCENT, 0.14), fg=T.ACCENT,
                     font=T.F(8, "bold"), padx=7, pady=2).pack(side="left")
        tk.Label(head, text=dt.datetime.now().strftime("%H:%M"), bg=T.SURFACE, fg=T.FAINT,
                 font=T.F(9)).pack(side="right")
        txt = tk.Text(self.body, wrap="word", bg=T.SURFACE, fg=T.TEXT, font=T.F(12), relief="flat", bd=0,
                      highlightthickness=0, padx=0, pady=0, spacing1=2, spacing3=2, cursor="arrow",
                      selectbackground=T.ACCENT_DIM, height=1)
        txt.insert("1.0", reply or "Done.")
        txt.configure(state="disabled")
        txt.pack(fill="x")
        txt.update_idletasks()
        try:
            lines = int(txt.count("1.0", "end", "displaylines")[0])
        except (tk.TclError, TypeError):
            lines = reply.count("\n") + 1 + len(reply) // 80
        txt.configure(height=max(1, min(lines, 14)))
        if self._rows:
            self._draw_rows(self._rows)
        if detail:
            tk.Frame(self.body, bg=T.BORDER, height=1).pack(fill="x", pady=(10, 6))
            tk.Label(self.body, text=detail, bg=T.SURFACE, fg=T.FAINT, font=T.F(9, kind="mono"), justify="left",
                     anchor="w", wraplength=W - 40).pack(fill="x")
        self._resize()

    # ---------------------------------------------------------- memory search results
    ICONS = {"web": "◍", "image": "▣", "video": "▶", "audio": "♪", "document": "▤", "archive": "▦",
             "installer": "⬢", "code": "‹›", "screenshot": "▣"}

    def _icon(self, row: dict) -> str:
        if row["source"] == "note":
            return "✎"
        if row["source"] == "web":
            return self.ICONS["web"]
        if row.get("screenshot"):
            return self.ICONS["screenshot"]
        from .. import filetypes
        for cat, exts in filetypes.CATEGORIES.items():
            if (row.get("ext") or "") in exts:
                return self.ICONS.get(cat, "•")
        return "•"

    @staticmethod
    def _age(ts: float) -> str:
        s = max(0, time.time() - (ts or 0))
        for unit, sec in (("y", 31536000), ("mo", 2592000), ("d", 86400), ("h", 3600), ("m", 60)):
            if s >= sec:
                return f"{int(s // sec)}{unit} ago"
        return "now"

    def _draw_rows(self, rows: list[dict]):
        import os
        box = tk.Frame(self.body, bg=T.SURFACE)
        box.pack(fill="x", pady=(10, 0))
        for i, row in enumerate(rows[:6]):
            line = tk.Frame(box, bg=T.RAISED, padx=10, pady=6, cursor="hand2", highlightthickness=1,
                            highlightbackground=T.BORDER)
            line.pack(fill="x", pady=2)
            icon = tk.Label(line, text=self._icon(row), bg=T.RAISED, fg=T.ACCENT, font=T.F(12), width=2)
            icon.pack(side="left")
            mid = tk.Frame(line, bg=T.RAISED)
            mid.pack(side="left", fill="x", expand=True, padx=(6, 0))
            title = tk.Label(mid, text=row["title"][:80], bg=T.RAISED, fg=T.TEXT, font=T.F(10, "bold"), anchor="w")
            title.pack(fill="x")
            if row["source"] == "note":
                where = "your journal" if row.get("kind") == "journal" else "your notes"
            elif row["source"] == "web":
                where = row["path"].split("//")[-1].split("/")[0]
            else:
                where = os.path.dirname(row["path"])
            sub_text = f"{where}  ·  {self._age(row.get('mtime', 0))}"
            if row.get("snippet"):
                sub_text += "  ·  " + row["snippet"].replace("[", "").replace("]", "")[:70]
            sub = tk.Label(mid, text=sub_text, bg=T.RAISED, fg=T.FAINT, font=T.F(8), anchor="w")
            sub.pack(fill="x")
            key = tk.Label(line, text=f"Ctrl+{i + 1}", bg=T.RAISED, fg=T.FAINT, font=T.F(8))
            key.pack(side="right")
            for w in (line, icon, mid, title, sub, key):
                w.bind("<Button-1>", lambda e, k=i: self._open_row(k))
                w.bind("<Button-3>", lambda e, k=i: self._open_row(k, reveal=True))
                w.bind("<Enter>", lambda e, l=line: l.configure(highlightbackground=T.ACCENT_DIM))
                w.bind("<Leave>", lambda e, l=line: l.configure(highlightbackground=T.BORDER))
        tk.Label(self.body, text="Click to open  ·  right-click to show in folder  ·  Esc closes", bg=T.SURFACE,
                 fg=T.FAINT,
                 font=T.F(8)).pack(anchor="w", pady=(4, 0))

    def _open_row(self, i: int, reveal: bool = False):
        """Open a search result (or show it in its folder). Says what it's doing, then gets out of the way."""
        if i >= len(self._rows):
            return
        row = self._rows[i]
        if row["source"] == "note":                   # a note opens right here, in full
            rows = self._rows
            self._show_reply(row.get("text") or row["title"], rows)
            return
        verb = "Showing" if reveal and row["source"] != "web" else "Opening"
        self._clear_body()
        tk.Label(self.body, text=f"{verb} {row['title'][:70]}…", bg=T.SURFACE, fg=T.ACCENT,
                 font=T.F(11)).pack(anchor="w")
        self._resize()
        self.orb.set("success")
        self.root.update_idletasks()
        try:
            if row["source"] == "web":
                import webbrowser
                webbrowser.open(row["path"])
            elif reveal:
                import subprocess
                subprocess.Popen(["explorer", "/select,", row["path"]])
            else:
                import os
                if not os.path.exists(row["path"]):
                    raise FileNotFoundError("it was moved or deleted since it was indexed")
                os.startfile(row["path"])                 # noqa: Windows
        except (OSError, AttributeError) as e:
            self.orb.set("error")
            rows = self._rows
            self._show_reply(f"I couldn't open {row['title']}: {e}", rows)
            return
        # stay open on top with the same results, so you can open another one (Esc closes the bar)
        rows = self._rows
        self.root.after(400, lambda: self._show_reply(
            f"Opened {row['title']}. Pick another, or press Esc to close.", rows))

    def _show_confirm(self, tier: Tier, summary: str, items: list[str]):
        self._clear_body()
        danger = tier >= Tier.DESTRUCTIVE
        col = T.RED if danger else T.AMBER
        self.orb.set("alert" if danger else "error")
        head = tk.Frame(self.body, bg=T.SURFACE)
        head.pack(fill="x")
        tk.Label(head, text="●", bg=T.SURFACE, fg=col, font=T.F(11)).pack(side="left")
        tk.Label(head, text="  " + ("This can't be undone easily" if danger else "Please confirm").upper(),
                 bg=T.SURFACE, fg=col, font=T.F(9, "bold")).pack(side="left")
        tk.Label(self.body, text=summary, bg=T.SURFACE, fg=T.TEXT, font=T.F(12), justify="left", anchor="w",
                 wraplength=W - 40).pack(fill="x", pady=(8, 6))
        if items:
            box = tk.Frame(self.body, bg=T.RAISED, padx=12, pady=8, highlightthickness=1,
                           highlightbackground=T.BORDER)
            box.pack(fill="x", pady=(0, 10))
            for it in items[:8]:
                tk.Label(box, text="·  " + str(it), bg=T.RAISED, fg=T.MUTED, font=T.F(9, kind="mono"),
                         anchor="w").pack(fill="x")
            if len(items) > 8:
                tk.Label(box, text=f"   …and {len(items) - 8} more", bg=T.RAISED, fg=T.FAINT,
                         font=T.F(9), anchor="w").pack(fill="x")
        btns = tk.Frame(self.body, bg=T.SURFACE)
        btns.pack(anchor="e", pady=(4, 0))
        self._button(btns, "Cancel   Esc", T.RAISED, T.TEXT, lambda: self._answer(False)).pack(side="left", padx=(0, 8))
        self._button(btns, ("Yes, delete" if danger else "Yes, do it") + "   ⏎", col, T.BG,
                     lambda: self._answer(True)).pack(side="left")
        self.show()
        self._resize()

    @staticmethod
    def _button(parent, text, bg, fg, cmd) -> tk.Label:
        b = tk.Label(parent, text=text, bg=bg, fg=fg, font=T.F(10, "bold"), padx=14, pady=6, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.configure(bg=T.blend(bg, "#ffffff", 0.12)))
        b.bind("<Leave>", lambda e: b.configure(bg=bg))
        return b

    # ---------------------------------------------------------- input
    def _history(self, step: int):
        if not self.history or self.busy:
            return
        self.hist_i = max(0, min(len(self.history) - 1, self.hist_i + step))
        self.entry.delete(0, "end")
        self.entry.insert(0, self.history[self.hist_i])
        self._sync_placeholder()

    def _submit(self, _event=None):
        if self._pending_confirm is not None:          # Enter while a confirmation is showing = yes
            self._answer(True)
            return "break"
        if self.busy:
            return "break"
        text = self.entry.get().strip()
        if not text:
            return "break"
        self.history.append(text)
        self.hist_i = len(self.history)
        self.busy = True
        self.orb.set("thinking")
        self.entry.configure(state="disabled")
        self._show_thinking()

        def work():
            try:
                result = self.on_submit(text)
            except Exception as e:                         # never leave the bar stuck in "working"
                result = f"Something went wrong: {e}"
            self.post("result", result)
        threading.Thread(target=work, daemon=True, name="command").start()
        return "break"

    def _finish(self, result: str):
        self.busy = False
        self.entry.configure(state="normal")
        self.entry.delete(0, "end")
        self._sync_placeholder()
        self.entry.focus_set()
        low = result.lower()
        bad = low.startswith(("something went wrong", "sorry", "i couldn't", "couldn't")) or "failed" in low[:80]
        self.orb.set("error" if bad else "success")
        self.root.after(1800, lambda: self.orb.set("idle") if not self.busy else None)
        rows = self.results_provider() if self.results_provider else []
        self._show_reply(result, rows)

    # ---------------------------------------------------------- confirmations (called from the worker thread)
    def confirm(self, tier: Tier, summary: str, items: list[str]) -> bool:
        if threading.current_thread() is self._ui_thread:          # not on a worker: plain dialog
            body = summary + ("\n\n" + "\n".join(f"  - {i}" for i in items[:15]) if items else "")
            return messagebox.askyesno("Confirm", body + "\n\nProceed?", parent=self.root)
        done, answer = threading.Event(), [False]
        self._pending_confirm = (done, answer)
        self.post("confirm_ui", (tier, summary, items))
        done.wait()
        return answer[0]

    def _answer(self, yes: bool):
        if self._pending_confirm is None:
            return
        done, answer = self._pending_confirm
        self._pending_confirm = None
        answer[0] = yes
        self.orb.set("thinking")
        self._show_thinking()
        done.set()

    def _key_yes(self, _e=None):
        if self._pending_confirm is not None:
            self._answer(True)
            return "break"

    def _key_no(self, _e=None):
        if self._pending_confirm is not None:
            self._answer(False)
            return "break"

    def mainloop(self):
        self.root.mainloop()
