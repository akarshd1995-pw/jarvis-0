"""Settings window (tkinter). Opened from the tray icon or by typing /settings."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk

from .. import autostart
from ..config import Config


class SettingsWindow:
    def __init__(self, parent: tk.Misc, config: Config):
        self.config = config
        self.win = tk.Toplevel(parent)
        self.win.title("Jarvis-0 settings")
        self.win.attributes("-topmost", True)
        self.win.geometry("580x860")
        pad = dict(padx=10, pady=4, sticky="w")
        f = ttk.Frame(self.win, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Hotkey").grid(row=0, column=0, **pad)
        self.hotkey = tk.StringVar(value=config.hotkey)
        ttk.Entry(f, textvariable=self.hotkey).grid(row=0, column=1, sticky="ew", padx=10)

        ttk.Label(f, text="Min. model confidence").grid(row=1, column=0, **pad)
        self.conf = tk.DoubleVar(value=config.min_confidence)
        row = ttk.Frame(f)
        row.grid(row=1, column=1, sticky="ew", padx=10)
        self.conf_label = ttk.Label(row, width=5)
        ttk.Scale(row, from_=0.3, to=0.95, variable=self.conf, command=lambda _v: self._conf_text()).pack(
            side="left", fill="x", expand=True)
        self.conf_label.pack(side="left")
        self._conf_text()

        ttk.Label(f, text="Model backend").grid(row=2, column=0, **pad)
        self.backend = tk.StringVar(value=config.model_backend)
        ttk.Combobox(f, textvariable=self.backend, values=["auto", "onnx", "torch"], state="readonly",
                     width=10).grid(row=2, column=1, **pad)

        ttk.Label(f, text="Ask before (comma-separated)").grid(row=13, column=0, **pad)
        self.confirm = tk.StringVar(value=", ".join(config.confirm_actions))
        ttk.Entry(f, textvariable=self.confirm).grid(row=13, column=1, sticky="ew", padx=10)
        ttk.Label(f, text="e.g. move_files, copy_files, system, undo. Deleting always asks.",
                  foreground="gray").grid(row=14, column=0, columnspan=2, **pad)

        ttk.Separator(f).grid(row=16, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(f, text="Assistant name").grid(row=17, column=0, **pad)
        self.p_name = tk.StringVar(value=config.persona_name)
        ttk.Entry(f, textvariable=self.p_name).grid(row=17, column=1, sticky="ew", padx=10)
        ttk.Label(f, text="Call me (sir, ma'am, a name, or blank)").grid(row=18, column=0, **pad)
        self.p_addr = tk.StringVar(value=config.persona_address)
        ttk.Entry(f, textvariable=self.p_addr).grid(row=18, column=1, sticky="ew", padx=10)
        ttk.Label(f, text="Tone").grid(row=19, column=0, **pad)
        self.p_tone = tk.StringVar(value=config.persona_tone)
        ttk.Combobox(f, textvariable=self.p_tone, values=["butler", "friendly", "minimal"], state="readonly",
                     width=10).grid(row=19, column=1, **pad)
        self.details = tk.BooleanVar(value=config.show_details)
        ttk.Checkbutton(f, text="Show the technical command under each reply", variable=self.details).grid(
            row=20, column=0, columnspan=2, **pad)

        self.encrypt = tk.BooleanVar(value=config.encrypt_logs)
        ttk.Checkbutton(f, text="Encrypt activity log (Windows DPAPI)", variable=self.encrypt).grid(
            row=3, column=0, columnspan=2, **pad)
        self.titles = tk.BooleanVar(value=config.keep_window_titles)
        ttk.Checkbutton(f, text="Store full window titles (off = app name only)", variable=self.titles).grid(
            row=4, column=0, columnspan=2, **pad)
        self.auto = tk.BooleanVar(value=autostart.is_enabled())
        ttk.Checkbutton(f, text="Start Jarvis when I log in to Windows", variable=self.auto).grid(
            row=5, column=0, columnspan=2, **pad)

        ttk.Label(f, text="Folder aliases (JSON)").grid(row=6, column=0, columnspan=2, **pad)
        self.folders = tk.Text(f, height=8, font=("Consolas", 9))
        self.folders.insert("1.0", json.dumps(config.folders, indent=2))
        self.folders.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=10)

        ttk.Label(f, text="App aliases (JSON)").grid(row=8, column=0, columnspan=2, **pad)
        self.apps = tk.Text(f, height=6, font=("Consolas", 9))
        self.apps.insert("1.0", json.dumps(config.apps, indent=2))
        self.apps.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=10)

        ttk.Label(f, text="Allowed folders, one per line (Jarvis can't touch anything else)").grid(
            row=10, column=0, columnspan=2, **pad)
        self.roots = tk.Text(f, height=3, font=("Consolas", 9))
        self.roots.insert("1.0", "\n".join(config.allowed_roots))
        self.roots.grid(row=11, column=0, columnspan=2, sticky="nsew", padx=10)

        btns = ttk.Frame(f)
        btns.grid(row=21, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="Save", command=self.save).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.win.destroy).pack(side="left", padx=6)

    def _conf_text(self):
        self.conf_label.configure(text=f"{self.conf.get():.2f}")

    def save(self):
        try:
            folders = json.loads(self.folders.get("1.0", "end"))
            apps = json.loads(self.apps.get("1.0", "end"))
            assert isinstance(folders, dict) and isinstance(apps, dict)
        except (ValueError, AssertionError) as e:
            messagebox.showerror("Invalid JSON", f"Folder/app aliases must be JSON objects.\n{e}", parent=self.win)
            return
        roots = [r.strip() for r in self.roots.get("1.0", "end").splitlines() if r.strip()]
        if not roots:
            messagebox.showerror("Allowed folders", "Keep at least one allowed folder.", parent=self.win)
            return
        c = self.config
        c.hotkey, c.min_confidence, c.model_backend = self.hotkey.get().strip(), round(self.conf.get(), 2), self.backend.get()
        c.encrypt_logs, c.keep_window_titles = self.encrypt.get(), self.titles.get()
        c.folders, c.apps, c.allowed_roots = folders, apps, roots
        c.confirm_actions = [a.strip() for a in self.confirm.get().split(",") if a.strip()]
        c.persona_name = self.p_name.get().strip() or "Jarvis"
        c.persona_address, c.persona_tone = self.p_addr.get().strip(), self.p_tone.get()
        c.show_details = self.details.get()
        path = c.save()
        try:
            (autostart.enable if self.auto.get() else autostart.disable)()
        except OSError as e:
            messagebox.showwarning("Autostart", f"Couldn't change autostart: {e}", parent=self.win)
        messagebox.showinfo("Saved", f"Saved to {path}.\nRestart Jarvis for hotkey, model and persona changes.",
                            parent=self.win)
        self.win.destroy()
