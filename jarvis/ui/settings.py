"""Settings window (tkinter). Opened from the tray icon or by typing /settings."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk

from .. import autostart
from ..config import Config
from . import theme as T


class Toggle(tk.Canvas):
    """Pill-shaped on/off switch bound to a BooleanVar."""

    def __init__(self, parent, variable: tk.BooleanVar, bg: str = T.SURFACE):
        super().__init__(parent, width=40, height=22, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.var = variable
        self.bind("<Button-1>", lambda e: (self.var.set(not self.var.get()), self.draw()))
        self.draw()

    def draw(self):
        self.delete("all")
        on = bool(self.var.get())
        T.round_rect(self, 1, 1, 39, 21, 10, fill=T.ACCENT if on else T.BORDER_HI, outline="")
        x = 28 if on else 12
        self.create_oval(x - 7, 4, x + 7, 18, fill=T.BG if on else T.MUTED, outline="")


class SettingsWindow:
    PAGES = ["General", "Persona", "HUD & alerts", "Briefings", "Places & maps", "Folders & apps", "Privacy"]

    def __init__(self, parent: tk.Misc, config: Config):
        self.config = config
        self.win = w = tk.Toplevel(parent)
        w.title("Jarvis-0 settings")
        w.attributes("-topmost", True)
        w.configure(bg=T.BG)
        w.geometry("820x600")
        w.minsize(760, 520)
        self._style()

        side = tk.Frame(w, bg=T.BG, width=200)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="  ".join(config.persona_name.upper()), bg=T.BG, fg=T.TEXT,
                 font=T.F(13, "bold", "display")).pack(anchor="w", padx=22, pady=(24, 2))
        tk.Label(side, text="Settings", bg=T.BG, fg=T.MUTED, font=T.F(10)).pack(anchor="w", padx=22, pady=(0, 18))
        self.nav: dict[str, tk.Label] = {}
        for name in self.PAGES:
            lab = tk.Label(side, text=name, bg=T.BG, fg=T.MUTED, font=T.F(10), anchor="w", padx=22, pady=8,
                           cursor="hand2")
            lab.pack(fill="x")
            lab.bind("<Button-1>", lambda e, n=name: self.show(n))
            self.nav[name] = lab
        tk.Label(side, text="Changes apply after Save.\nHotkey, model and persona\nneed a restart.", bg=T.BG,
                 fg=T.FAINT, font=T.F(8), justify="left").pack(side="bottom", anchor="w", padx=22, pady=20)

        main = tk.Frame(w, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        main.pack(side="left", fill="both", expand=True)
        foot = tk.Frame(main, bg=T.SURFACE)
        foot.pack(side="bottom", fill="x", padx=24, pady=16)
        self._button(foot, "Save", T.ACCENT, T.BG, self.save).pack(side="right")
        self._button(foot, "Cancel", T.RAISED, T.TEXT, w.destroy).pack(side="right", padx=(0, 10))
        tk.Frame(main, bg=T.BORDER, height=1).pack(side="bottom", fill="x")
        self.content = tk.Frame(main, bg=T.SURFACE)
        self.content.pack(fill="both", expand=True, padx=28, pady=(22, 8))

        self._vars(config)
        self.pages = {n: self._build_page(n) for n in self.PAGES}
        self.show("General")
        w.after(30, lambda: T.win11_style(w))

    # ------------------------------------------------------------ look
    def _style(self):
        st = ttk.Style(self.win)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TCombobox", fieldbackground=T.RAISED, background=T.RAISED, foreground=T.TEXT,
                     arrowcolor=T.MUTED, bordercolor=T.BORDER_HI, lightcolor=T.RAISED, darkcolor=T.RAISED,
                     selectbackground=T.RAISED, selectforeground=T.TEXT, padding=5)
        st.map("TCombobox", fieldbackground=[("readonly", T.RAISED)], foreground=[("readonly", T.TEXT)])
        st.configure("Horizontal.TScale", background=T.SURFACE, troughcolor=T.BORDER, bordercolor=T.SURFACE,
                     lightcolor=T.ACCENT, darkcolor=T.ACCENT)
        self.win.option_add("*TCombobox*Listbox.background", T.RAISED)
        self.win.option_add("*TCombobox*Listbox.foreground", T.TEXT)
        self.win.option_add("*TCombobox*Listbox.selectBackground", T.ACCENT_DIM)
        self.win.option_add("*TCombobox*Listbox.font", T.F(10))

    @staticmethod
    def _button(parent, text, bg, fg, cmd) -> tk.Label:
        b = tk.Label(parent, text=text, bg=bg, fg=fg, font=T.F(10, "bold"), padx=18, pady=7, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.configure(bg=T.blend(bg, "#ffffff", 0.12)))
        b.bind("<Leave>", lambda e: b.configure(bg=bg))
        return b

    def show(self, name: str):
        for n, frame in self.pages.items():
            frame.pack_forget()
            self.nav[n].configure(bg=T.BG, fg=T.MUTED, font=T.F(10))
        self.pages[name].pack(fill="both", expand=True)
        self.nav[name].configure(bg=T.SURFACE, fg=T.ACCENT, font=T.F(10, "bold"))

    # ------------------------------------------------------------ widgets
    def _entry(self, parent, var, width=None):
        e = tk.Entry(parent, textvariable=var, bg=T.RAISED, fg=T.TEXT, insertbackground=T.ACCENT, relief="flat",
                     font=T.F(10), highlightthickness=1, highlightbackground=T.BORDER_HI, highlightcolor=T.ACCENT,
                     selectbackground=T.ACCENT_DIM, **({"width": width} if width else {}))
        return e

    def _text(self, parent, value: str, height: int):
        t = tk.Text(parent, height=height, bg=T.RAISED, fg=T.TEXT, insertbackground=T.ACCENT, relief="flat",
                    font=T.F(9, kind="mono"), highlightthickness=1, highlightbackground=T.BORDER_HI,
                    highlightcolor=T.ACCENT, selectbackground=T.ACCENT_DIM, padx=8, pady=6)
        t.insert("1.0", value)
        return t

    def _combo(self, parent, var, values, width=14):
        return ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width, font=T.F(10))

    def _row(self, page, label: str, widget_fn, hint: str = "", stretch: bool = False):
        row = tk.Frame(page, bg=T.SURFACE)
        row.pack(fill="x", pady=7)
        left = tk.Frame(row, bg=T.SURFACE)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=label, bg=T.SURFACE, fg=T.TEXT, font=T.F(10), anchor="w").pack(anchor="w")
        if hint:
            tk.Label(left, text=hint, bg=T.SURFACE, fg=T.FAINT, font=T.F(8), anchor="w", justify="left",
                     wraplength=360).pack(anchor="w")
        wdg = widget_fn(row)
        wdg.pack(side="right", fill="x" if stretch else None, expand=stretch, padx=(16, 0))
        return wdg

    def _title(self, page, text: str, sub: str = ""):
        tk.Label(page, text=text, bg=T.SURFACE, fg=T.TEXT, font=T.F(15, "bold", "display")).pack(anchor="w")
        if sub:
            tk.Label(page, text=sub, bg=T.SURFACE, fg=T.MUTED, font=T.F(9)).pack(anchor="w", pady=(2, 0))
        tk.Frame(page, bg=T.BORDER, height=1).pack(fill="x", pady=(12, 8))

    # ------------------------------------------------------------ state
    def _vars(self, c: Config):
        self.hotkey = tk.StringVar(value=c.hotkey)
        self.conf = tk.DoubleVar(value=c.min_confidence)
        self.backend = tk.StringVar(value=c.model_backend)
        self.confirm = tk.StringVar(value=", ".join(c.confirm_actions))
        self.details = tk.BooleanVar(value=c.show_details)
        self.p_name = tk.StringVar(value=c.persona_name)
        self.p_addr = tk.StringVar(value=c.persona_address)
        self.p_tone = tk.StringVar(value=c.persona_tone)
        self.p_model = tk.BooleanVar(value=c.persona_model)
        self.hud_on = tk.BooleanVar(value=c.hud_enabled)
        self.hud_corner = tk.StringVar(value=c.hud_corner)
        self.hud_alpha = tk.DoubleVar(value=c.hud_opacity)
        self.hud_ct = tk.BooleanVar(value=c.hud_click_through)
        self.mon_on = tk.BooleanVar(value=c.monitor_enabled)
        self.a_temp = tk.StringVar(value=f"{c.alert_gpu_temp:g}")
        self.a_disk = tk.StringVar(value=f"{c.alert_disk_free_gb:g}")
        self.a_cool = tk.StringVar(value=f"{c.alert_cooldown_min:g}")
        self.b_auto = tk.BooleanVar(value=c.briefing_auto)
        self.b_morning = tk.StringVar(value=c.briefing_morning)
        self.b_evening = tk.StringVar(value=c.briefing_evening)
        self.encrypt = tk.BooleanVar(value=c.encrypt_logs)
        self.titles = tk.BooleanVar(value=c.keep_window_titles)
        self.auto = tk.BooleanVar(value=autostart.is_enabled())
        self.home_addr = tk.StringVar(value=(c.places or {}).get("home", ""))
        self.work_addr = tk.StringVar(value=(c.places or {}).get("work", ""))
        from ..crypto import unprotect
        self.maps_key = tk.StringVar(value=unprotect(c.maps_api_key) if c.maps_api_key else "")
        self.maps_osm = tk.BooleanVar(value=c.maps_osm)
        self.mem_on = tk.BooleanVar(value=c.memory_enabled)
        self.mem_web = tk.BooleanVar(value=c.memory_browser)
        self.mem_folders = tk.StringVar(value=", ".join(c.memory_folders))

    def _build_page(self, name: str) -> tk.Frame:
        pg = tk.Frame(self.content, bg=T.SURFACE)
        tog = lambda var: (lambda parent: Toggle(parent, var))          # noqa: E731
        if name == "General":
            self._title(pg, "General", "How you call Jarvis and how sure it must be before acting.")
            self._row(pg, "Hotkey", lambda p: self._entry(p, self.hotkey, 18), "Opens the command bar, e.g. ctrl+alt+j")

            def conf(p):
                f = tk.Frame(p, bg=T.SURFACE)
                self.conf_label = tk.Label(f, width=5, bg=T.SURFACE, fg=T.ACCENT, font=T.F(10, "bold"))
                ttk.Scale(f, from_=0.3, to=0.95, variable=self.conf, length=180,
                          command=lambda _v: self._conf_text()).pack(side="left")
                self.conf_label.pack(side="left", padx=(8, 0))
                self._conf_text()
                return f
            self._row(pg, "Minimum model confidence", conf, "Below this, the trained model's answer isn't run.")
            self._row(pg, "Model backend", lambda p: self._combo(p, self.backend, ["auto", "onnx", "torch"]))
            self._row(pg, "Ask before", lambda p: self._entry(p, self.confirm, 30),
                      "Comma-separated, e.g. move_files, undo. Deleting and shutdown always ask.")
            self._row(pg, "Show the technical command under replies", tog(self.details))
            self._row(pg, "Start Jarvis when I log in to Windows", tog(self.auto))
        elif name == "Persona":
            self._title(pg, "Persona", "Who Jarvis is and how it talks to you.")
            self._row(pg, "Assistant name", lambda p: self._entry(p, self.p_name, 22))
            self._row(pg, "Call me", lambda p: self._entry(p, self.p_addr, 22), "sir, ma'am, your name, or blank")
            self._row(pg, "Tone", lambda p: self._combo(p, self.p_tone, ["butler", "friendly", "minimal"]))
            self._row(pg, "Use my trained reply model", tog(self.p_model),
                      "Fact-checked; falls back to templates when it's unsure.")
        elif name == "HUD & alerts":
            self._title(pg, "HUD & alerts", "The overlay with the orb and gauges, and what it warns you about.")
            self._row(pg, "Show the HUD", tog(self.hud_on), "Toggle any time with Ctrl+Alt+H")
            self._row(pg, "Corner", lambda p: self._combo(p, self.hud_corner,
                                                          ["top-right", "top-left", "bottom-right", "bottom-left"]))
            self._row(pg, "Opacity", lambda p: ttk.Scale(p, from_=0.4, to=1.0, variable=self.hud_alpha, length=180))
            self._row(pg, "Clicks pass through the HUD", tog(self.hud_ct))
            self._row(pg, "Monitoring alerts", tog(self.mon_on))
            self._row(pg, "Alert when GPU reaches (°C)", lambda p: self._entry(p, self.a_temp, 8))
            self._row(pg, "Alert when C: has less free (GB)", lambda p: self._entry(p, self.a_disk, 8))
            self._row(pg, "Repeat the same alert after (min)", lambda p: self._entry(p, self.a_cool, 8))
        elif name == "Briefings":
            self._title(pg, "Briefings", "Morning and evening summaries. You can always ask: \"brief me\".")
            self._row(pg, "Briefings on their own", tog(self.b_auto), "Once a day each, only when you're at the PC.")
            self._row(pg, "Morning briefing from", lambda p: self._entry(p, self.b_morning, 8),
                      "The first time you're at the PC after this time (until noon). HH:MM")
            self._row(pg, "Evening briefing at", lambda p: self._entry(p, self.b_evening, 8), "HH:MM")
        elif name == "Places & maps":
            self._title(pg, "Places & maps", "\"how long to get home\", \"fastest route to the airport\".")
            self._row(pg, "Home", lambda p: self._entry(p, self.home_addr, 34), "An address or area, e.g. HSR Layout, Bengaluru")
            self._row(pg, "Work", lambda p: self._entry(p, self.work_addr, 34), "Used for \"work\" / \"office\"")

            def key_entry(p):
                e = self._entry(p, self.maps_key, 34)
                e.configure(show="•")
                return e
            self._row(pg, "Google Maps API key (optional)", key_entry,
                      "For live-traffic times. Create one in Google Cloud with the Routes API turned on; it stays on this PC.")
            self._row(pg, "Without a key, estimate with OpenStreetMap", tog(self.maps_osm),
                      "Normal-traffic times (car, bike, walking). Google Maps always opens with live traffic.")
        elif name == "Folders & apps":
            self._title(pg, "Folders & apps", "Names you can use in commands, and where Jarvis may touch files.")
            tk.Label(pg, text="Folder aliases (JSON)", bg=T.SURFACE, fg=T.TEXT, font=T.F(10)).pack(anchor="w")
            self.folders = self._text(pg, json.dumps(self.config.folders, indent=2), 6)
            self.folders.pack(fill="x", pady=(4, 10))
            tk.Label(pg, text="App aliases (JSON)", bg=T.SURFACE, fg=T.TEXT, font=T.F(10)).pack(anchor="w")
            self.apps = self._text(pg, json.dumps(self.config.apps, indent=2), 5)
            self.apps.pack(fill="x", pady=(4, 10))
            tk.Label(pg, text="Allowed folders, one per line (Jarvis can't touch anything else)", bg=T.SURFACE,
                     fg=T.TEXT, font=T.F(10)).pack(anchor="w")
            self.roots = self._text(pg, "\n".join(self.config.allowed_roots), 3)
            self.roots.pack(fill="x", pady=(4, 0))
        elif name == "Privacy":
            self._title(pg, "Privacy", "Everything stays on this PC.")
            self._row(pg, "Encrypt the activity log", tog(self.encrypt), "Windows DPAPI: only your account can read it.")
            self._row(pg, "Store full window titles", tog(self.titles), "Off: only app names (used for briefings).")
            self._row(pg, "Index my files for search", tog(self.mem_on),
                      "\"find the invoice pdf\". Names and text of your files, kept in a local index.")
            self._row(pg, "Folders to index", lambda p: self._entry(p, self.mem_folders, 30),
                      "Comma-separated names or paths, e.g. Downloads, Documents, D:/work")
            self._row(pg, "Include browser history", tog(self.mem_web),
                      "Titles and addresses from Chrome, Edge, Brave and Firefox. Never leaves this PC.")
        return pg

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
        c.persona_model = self.p_model.get()
        c.hud_enabled, c.hud_corner = self.hud_on.get(), self.hud_corner.get()
        c.hud_opacity, c.hud_click_through = round(self.hud_alpha.get(), 2), self.hud_ct.get()
        try:
            c.alert_gpu_temp, c.alert_disk_free_gb = float(self.a_temp.get()), float(self.a_disk.get())
            c.alert_cooldown_min = float(self.a_cool.get())
        except ValueError:
            messagebox.showerror("Alerts", "Alert thresholds must be numbers.", parent=self.win)
            return
        c.monitor_enabled = self.mon_on.get()
        import re as _re
        for label, var in (("Morning briefing", self.b_morning), ("Evening briefing", self.b_evening)):
            if not _re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", var.get().strip()):
                messagebox.showerror("Briefings", f"{label} time must look like 07:30.", parent=self.win)
                return
        c.places = {**(c.places or {}), "home": self.home_addr.get().strip(), "work": self.work_addr.get().strip()}
        from ..crypto import protect
        key = self.maps_key.get().strip()
        c.maps_api_key = protect(key) if key else ""        # encrypted to your Windows account
        c.maps_osm = self.maps_osm.get()
        c.memory_enabled, c.memory_browser = self.mem_on.get(), self.mem_web.get()
        c.memory_folders = [x.strip() for x in self.mem_folders.get().split(",") if x.strip()] or c.memory_folders
        c.briefing_auto = self.b_auto.get()
        c.briefing_morning, c.briefing_evening = self.b_morning.get().strip(), self.b_evening.get().strip()
        path = c.save()
        try:
            (autostart.enable if self.auto.get() else autostart.disable)()
        except OSError as e:
            messagebox.showwarning("Autostart", f"Couldn't change autostart: {e}", parent=self.win)
        messagebox.showinfo("Saved", f"Saved to {path}.\nRestart Jarvis for hotkey, model and persona changes.",
                            parent=self.win)
        self.win.destroy()
