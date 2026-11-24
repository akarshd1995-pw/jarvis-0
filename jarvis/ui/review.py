"""Weekly review (phase 39): the commands Jarvis was least sure about, one card at a time.

    "play super machi"            I did: play "sup" on YouTube
    [1  ✓ That's right]   [2  ✗ Wrong]   [3  Skip]

Wrong -> pick what you meant from the model's other readings, or say it more clearly and Jarvis uses
what that means. Every answer becomes a training example for the next improvement run.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from ..nlu.explain import explain
from ..usage import ReviewSession  # noqa: F401  (the logic lives there, testable without a window)
from . import theme as T


class ReviewWindow:
    def __init__(self, parent: tk.Misc, session: ReviewSession, on_done: Callable[[str], None] | None = None):
        self.s, self.on_done = session, on_done
        self.win = w = tk.Toplevel(parent)
        w.title("Jarvis-0 · weekly review")
        w.attributes("-topmost", True)
        w.configure(bg=T.BG)
        w.geometry("640x430")
        w.minsize(560, 380)
        self.head = tk.Label(w, bg=T.BG, fg=T.MUTED, font=T.F(10), anchor="w")
        self.head.pack(fill="x", padx=26, pady=(20, 6))
        self.card = tk.Frame(w, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        self.card.pack(fill="both", expand=True, padx=22, pady=(0, 20))
        for k, fn in (("1", self._right), ("2", self._wrong), ("3", self._skip), ("<Escape>", self.close)):
            w.bind(k if k.startswith("<") else f"<KeyPress-{k}>", lambda e, f=fn: f())
        self.render()

    # ------------------------------------------------------------ drawing
    def _clear(self):
        for c in self.card.winfo_children():
            c.destroy()

    def _btn(self, parent, text, bg, fg, cmd):
        b = tk.Label(parent, text=text, bg=bg, fg=fg, font=T.F(10, "bold"), padx=16, pady=8, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        return b

    def render(self):
        self._clear()
        it = self.s.current
        if it is None:
            return self._finish()
        self.head.configure(text=f"{self.s.i + 1} of {len(self.s.items)}  ·  was this right?")
        tk.Label(self.card, text=f"“{it.text}”", bg=T.SURFACE, fg=T.TEXT, font=T.F(15, "bold", "display"),
                 wraplength=560, justify="left", anchor="w").pack(fill="x", padx=24, pady=(24, 10))
        why = "you undid it right after" if it.doubt >= 0.9 and it.program else \
            "I wasn't sure" if it.program else "I didn't understand it"
        tk.Label(self.card, text=f"I did:  {explain(it.program)}", bg=T.SURFACE, fg=T.ACCENT, font=T.F(11),
                 wraplength=560, justify="left", anchor="w").pack(fill="x", padx=24)
        tk.Label(self.card, text=why, bg=T.SURFACE, fg=T.FAINT, font=T.F(9), anchor="w").pack(fill="x", padx=24,
                                                                                              pady=(4, 0))
        row = tk.Frame(self.card, bg=T.SURFACE)
        row.pack(side="bottom", fill="x", padx=24, pady=22)
        self._btn(row, "1  ✓ That's right", T.GREEN, T.BG, self._right).pack(side="left")
        self._btn(row, "2  ✗ Wrong", T.RAISED, T.TEXT, self._wrong).pack(side="left", padx=10)
        self._btn(row, "3  Skip", T.RAISED, T.MUTED, self._skip).pack(side="left")

    def _wrong(self):
        it = self.s.current
        if it is None:
            return
        self._clear()
        tk.Label(self.card, text=f"“{it.text}” — what did you mean?", bg=T.SURFACE, fg=T.TEXT,
                 font=T.F(12, "bold"), wraplength=560, justify="left", anchor="w").pack(fill="x", padx=24, pady=(20, 10))
        for n, (prog, words) in enumerate(self.s.choices(it), start=1):
            b = self._btn(self.card, f"{n}  {words}", T.RAISED, T.TEXT, lambda p=prog: self._fix(p))
            b.configure(anchor="w", font=T.F(10))
            b.pack(fill="x", padx=24, pady=3)
            self.win.bind(f"<KeyPress-{n}>", lambda e, p=prog: self._fix(p))
        tk.Label(self.card, text="Or say it more clearly:", bg=T.SURFACE, fg=T.MUTED, font=T.F(9),
                 anchor="w").pack(fill="x", padx=24, pady=(12, 2))
        line = tk.Frame(self.card, bg=T.SURFACE)
        line.pack(fill="x", padx=24)
        var = tk.StringVar()
        e = tk.Entry(line, textvariable=var, bg=T.RAISED, fg=T.TEXT, insertbackground=T.ACCENT, relief="flat",
                     font=T.F(11))
        e.pack(side="left", fill="x", expand=True, ipady=6)
        hint = tk.Label(self.card, text="", bg=T.SURFACE, fg=T.ACCENT, font=T.F(10), anchor="w", wraplength=560)
        hint.pack(fill="x", padx=24, pady=(6, 0))
        pending: dict = {}

        def understand(_e=None):
            prog = self.s.clearer(var.get())
            pending["p"] = prog
            hint.configure(text=f"That means: {explain(prog)}.  Press Enter again to use it." if prog else
                           "I still don't get that one. Try different words, or Skip.")
            if prog:
                e.bind("<Return>", lambda _e: self._fix(pending["p"]))
        e.bind("<Return>", understand)
        e.bind("<KeyRelease>", lambda ev: ev.keysym == "Return" or e.bind("<Return>", understand))
        e.focus_set()
        row = tk.Frame(self.card, bg=T.SURFACE)
        row.pack(side="bottom", fill="x", padx=24, pady=18)
        self._btn(row, "Skip", T.RAISED, T.MUTED, self._skip).pack(side="left")
        self._btn(row, "Back", T.RAISED, T.MUTED, self._rebind_and_render).pack(side="left", padx=10)

    def _rebind_and_render(self):
        for k, fn in (("1", self._right), ("2", self._wrong), ("3", self._skip)):
            self.win.bind(f"<KeyPress-{k}>", lambda e, f=fn: f())
        for k in "4":
            self.win.unbind(f"<KeyPress-{k}>")
        self.render()

    def _right(self):
        self.s.right()
        self._rebind_and_render()

    def _fix(self, program):
        if program:
            self.s.fix(program)
        self._rebind_and_render()

    def _skip(self):
        self.s.skip()
        self._rebind_and_render()

    def _finish(self):
        self.head.configure(text="All done")
        msg = self.s.summary()
        tk.Label(self.card, text=msg, bg=T.SURFACE, fg=T.TEXT, font=T.F(12), wraplength=540, justify="left").pack(
            padx=24, pady=30, anchor="w")
        self._btn(self.card, "Close", T.ACCENT, T.BG, self.close).pack(anchor="w", padx=24)
        if self.on_done:
            self.on_done(msg)
            self.on_done = None

    def close(self):
        if self.on_done and self.s.i:
            self.on_done(self.s.summary())
        self.win.destroy()
