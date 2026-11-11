"""Spotlight-style command bar (tkinter, ships with Python on Windows).

Ctrl+Alt+J (configurable) toggles it. Type a command, press Enter.
Esc hides it. Type 'undo' to revert the last command.
"""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox
from typing import Callable

from ..safety import Tier

BG, FG, MUTED, ACCENT = "#1e1e24", "#f2f2f2", "#9a9aa6", "#7f77dd"


class CommandBar:
    def __init__(self, on_submit: Callable[[str], str]):
        self.on_submit = on_submit
        self.events: "queue.Queue[tuple[str, str]]" = queue.Queue()  # thread-safe inbox
        self.history: list[str] = []
        self.hist_i = 0

        self.root = tk.Tk()
        self.root.title("Jarvis-0")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        w = 720
        x = (self.root.winfo_screenwidth() - w) // 2
        self.root.geometry(f"{w}x150+{x}+160")

        self.entry = tk.Entry(self.root, font=("Segoe UI", 16), bg=BG, fg=FG, insertbackground=FG,
                              relief="flat", highlightthickness=1, highlightcolor=ACCENT,
                              highlightbackground="#333")
        self.entry.pack(fill="x", padx=14, pady=(14, 6), ipady=6)
        self.out = tk.Label(self.root, text="Try: move pdf files older than 30 days from Downloads to Archive",
                            font=("Segoe UI", 10), bg=BG, fg=MUTED, justify="left", anchor="nw",
                            wraplength=690)
        self.out.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self.hide())
        self.entry.bind("<Up>", lambda e: self._history(-1))
        self.entry.bind("<Down>", lambda e: self._history(1))
        self.root.after(100, self._pump)
        self.hide()

    # ---------------------------------------------------------- thread-safe API
    def post(self, kind: str, payload: str = "") -> None:
        """Called from any thread: 'toggle' or 'reminder'."""
        self.events.put((kind, payload))

    def _pump(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "toggle":
                    self.hide() if self.root.state() == "normal" else self.show()
                elif kind == "reminder":
                    self.show()
                    messagebox.showinfo("Jarvis reminder", payload, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    # ---------------------------------------------------------- ui
    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.entry.focus_set()
        self.entry.select_range(0, "end")

    def hide(self):
        self.root.withdraw()

    def _history(self, step: int):
        if not self.history:
            return
        self.hist_i = max(0, min(len(self.history) - 1, self.hist_i + step))
        self.entry.delete(0, "end")
        self.entry.insert(0, self.history[self.hist_i])

    def _submit(self, _event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.history.append(text)
        self.hist_i = len(self.history)
        self.out.configure(text="...", fg=MUTED)
        self.root.update_idletasks()
        result = self.on_submit(text)
        self.out.configure(text=result, fg=FG)
        self.entry.delete(0, "end")

    def confirm(self, tier: Tier, summary: str, items: list[str]) -> bool:
        body = summary
        if items:
            shown = "\n".join(f"  - {i}" for i in items[:15])
            more = f"\n  ...and {len(items) - 15} more" if len(items) > 15 else ""
            body += f"\n\n{shown}{more}"
        title = "Confirm (destructive)" if tier >= Tier.DESTRUCTIVE else "Confirm"
        icon = "warning" if tier >= Tier.DESTRUCTIVE else "question"
        return messagebox.askyesno(title, body + "\n\nProceed?", icon=icon, parent=self.root)

    def mainloop(self):
        self.root.mainloop()
