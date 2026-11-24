"""Workspaces (phase 31): one command to open your apps and put every window where it belongs.

    "start coding mode"            open what's missing, then arrange the windows (and the music, if it has some)
    "save this layout as work"     remember every open window: which app, which monitor, where, how big, maximized
    "close coding mode"            close that workspace's windows (asks first; apps can still ask to save)
    "list my workspaces" / "delete the work workspace"

Built-in starters (coding, study) tile their apps sensibly until you save your own layout under that name.
Saved in workspaces.json in Jarvis's data folder. "undo" after start puts the windows back where they were;
after save or delete it brings the previous version back.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

PRESETS = {
    "coding": {"apps": ["vscode", "chrome", "terminal"], "music": "lofi hip hop radio beats to code to"},
    "study": {"apps": ["chrome", "notepad"], "music": "lofi study music"},
}
ALIASES = {"code": "coding", "dev": "coding", "programming": "coding", "development": "coding",
           "studying": "study", "padhai": "study"}
# what each built-in app name runs as, to find its windows
APP_EXES = {"vscode": {"code.exe"}, "chrome": {"chrome.exe"}, "edge": {"msedge.exe"}, "firefox": {"firefox.exe"},
            "terminal": {"windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe", "cmd.exe"},
            "notepad": {"notepad.exe"}, "spotify": {"spotify.exe"}, "explorer": {"explorer.exe"},
            "calculator": {"calculatorapp.exe", "calc.exe"}, "brave": {"brave.exe"}}
NICE = {"code.exe": "VS Code", "chrome.exe": "Chrome", "msedge.exe": "Edge", "firefox.exe": "Firefox",
        "windowsterminal.exe": "Terminal", "notepad.exe": "Notepad", "spotify.exe": "Spotify",
        "explorer.exe": "File Explorer", "brave.exe": "Brave", "winword.exe": "Word", "excel.exe": "Excel",
        "powerpnt.exe": "PowerPoint", "obsidian.exe": "Obsidian", "slack.exe": "Slack", "discord.exe": "Discord"}


def norm(name: str) -> str:
    """'Coding Mode' / 'my coding workspace' -> 'coding'."""
    n = re.sub(r"\b(?:my|the|mode|workspace|layout|setup|set up|wala|vala)\b", " ", name.lower())
    n = re.sub(r"[^a-z0-9 _-]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)


def nice_app(exe_or_alias: str) -> str:
    base = os.path.basename(exe_or_alias).lower()
    if base in NICE:
        return NICE[base]
    if base.endswith(".exe"):
        return base[:-4].replace("_", " ").title()
    return {"vscode": "VS Code", "chrome": "Chrome", "terminal": "Terminal"}.get(base, base.title())


class WorkspaceNotFound(Exception):
    def __init__(self, name: str, known: list[str]):
        self.name, self.known = name, known
        super().__init__(f"no workspace called {name}")


@dataclass
class Report:
    name: str
    arranged: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    music: str | None = None
    before: list[list] = field(default_factory=list)     # [[hwnd, rect, state]] for undo


class RealBackend:
    """Windows itself (winlayout.py) plus Jarvis's own open_app / play_media."""

    def __init__(self, ctx=None):
        self.ctx = ctx

    def windows(self):
        from . import winlayout
        return winlayout.windows()

    def place(self, hwnd, rect, state):
        from . import winlayout
        return winlayout.place(hwnd, rect, state)

    def close(self, hwnd):
        from . import winlayout
        winlayout.close(hwnd)

    def launch(self, exe):
        from . import winlayout
        winlayout.launch(exe)

    def open_app(self, alias):
        from .executor import REGISTRY
        REGISTRY["open_app"]({"name": alias}, self.ctx, False)

    def play(self, query):
        from .dsl.grammar import Ident
        from .executor import REGISTRY
        REGISTRY["play_media"]({"query": query, "site": Ident("youtube")}, self.ctx, False)

    def monitors(self):
        from . import machine
        mons = machine.monitors()
        areas = []
        for m in sorted(mons, key=lambda m: (not m["primary"], m["left"])):
            a = machine.work_area((m["left"] + m["right"]) // 2, (m["top"] + m["bottom"]) // 2)
            areas.append(list(a) if a else [m["left"], m["top"], m["right"], m["bottom"]])
        return areas or [[0, 0, 1920, 1040]]

    def wait_new(self, known: set[int], app: str | None = None, timeout: float = 12.0):
        end = time.time() + timeout
        while time.time() < end:
            for w in self.windows():
                if w.hwnd not in known and (app is None or w.app == app):
                    return w
            time.sleep(0.4)
        return None


class Workspaces:
    def __init__(self, path: Path, backend=None):
        self.path = Path(path)
        self.backend = backend or RealBackend()

    # -------------------------------------------------------------- storage
    def saved(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def names(self) -> list[str]:
        return sorted(set(PRESETS) | set(self.saved()))

    def get(self, name: str) -> tuple[str, dict]:
        key = norm(name)
        s = self.saved()
        if key in s:
            return key, s[key]
        if key in PRESETS:
            return key, dict(PRESETS[key], preset=True)
        raise WorkspaceNotFound(key, self.names())

    def set_definition(self, key: str, definition: dict | None) -> None:
        data = self.saved()
        if definition is None:
            data.pop(key, None)
        else:
            data[key] = definition
        self._write(data)

    # -------------------------------------------------------------- save
    def save(self, name: str) -> tuple[str, dict, dict | None]:
        """Remember every open window. Returns (name, new definition, previous definition or None)."""
        key = norm(name)
        if not key:
            raise ValueError("give the layout a name, like \"save this layout as work\"")
        wins = self.backend.windows()
        if not wins:
            raise ValueError("there are no app windows open to save")
        mons = self.backend.monitors()
        entries = []
        for w in wins:
            cx, cy = (w.rect[0] + w.rect[2]) // 2, (w.rect[1] + w.rect[3]) // 2
            mon = next((i for i, a in enumerate(mons) if a[0] <= cx < a[2] and a[1] <= cy < a[3]), 0)
            entries.append({"app": w.app, "exe": w.exe, "title": w.title[:120], "rect": list(w.rect),
                            "state": w.state, "monitor": mon})
        previous = self.saved().get(key)
        definition = {"windows": entries, "music": (previous or PRESETS.get(key, {})).get("music"),
                      "saved": time.time()}
        self.set_definition(key, definition)
        return key, definition, previous

    # -------------------------------------------------------------- start
    def start(self, name: str, music: bool = True) -> Report:
        key, d = self.get(name)
        rep = Report(key)
        current = self.backend.windows()
        rep.before = [[w.hwnd, list(w.rect), w.state] for w in current]
        used: set[int] = set()
        known = {w.hwnd for w in current}
        if d.get("windows"):
            for e in d["windows"]:
                w = next((w for w in current if w.hwnd not in used and w.app == e["app"] and w.title == e.get("title")),
                         None) or next((w for w in current if w.hwnd not in used and w.app == e["app"]), None)
                if w is None and e.get("exe") and os.path.exists(e["exe"]):
                    try:
                        self.backend.launch(e["exe"])
                        w = self.backend.wait_new(known | used, e["app"])
                    except OSError:
                        w = None
                    if w is not None:
                        rep.opened.append(nice_app(e["app"]))
                if w is None:
                    rep.missing.append(nice_app(e["app"]))
                    continue
                used.add(w.hwnd)
                known.add(w.hwnd)
                self.backend.place(w.hwnd, e["rect"], e.get("state", "normal"))
                rep.arranged.append(nice_app(e["app"]))
        else:                                           # a built-in starter: open its apps, then tile them
            chosen = []
            for alias in d.get("apps", []):
                exes = APP_EXES.get(alias, {alias.lower() + ".exe"})
                w = next((w for w in current if w.hwnd not in used and w.app in exes), None)
                if w is None:
                    try:
                        self.backend.open_app(alias)
                        w = self.backend.wait_new(known | used)
                    except Exception:
                        w = None
                    if w is not None:
                        rep.opened.append(nice_app(alias))
                if w is None:
                    rep.missing.append(nice_app(alias))
                    continue
                used.add(w.hwnd)
                known.add(w.hwnd)
                chosen.append(w)
            for w, (rect, state) in zip(chosen, tile(len(chosen), self.backend.monitors())):
                self.backend.place(w.hwnd, rect, state)
                rep.arranged.append(nice_app(w.app))
        if music and d.get("music"):
            try:
                self.backend.play(d["music"])
                rep.music = d["music"]
            except Exception:
                pass
        return rep

    # -------------------------------------------------------------- close / undo
    def windows_of(self, name: str):
        key, d = self.get(name)
        apps = {e["app"] for e in d.get("windows", [])}
        for alias in d.get("apps", []):
            apps |= APP_EXES.get(alias, {alias.lower() + ".exe"})
        return key, [w for w in self.backend.windows() if w.app in apps]

    def close(self, name: str) -> tuple[str, list[str]]:
        key, wins = self.windows_of(name)
        for w in wins:
            self.backend.close(w.hwnd)
        return key, [nice_app(w.app) for w in wins]

    def restore_placements(self, placements: list[list]) -> int:
        """Undo a start: windows that still exist go back where they were."""
        alive = {w.hwnd for w in self.backend.windows()}
        n = 0
        for hwnd, rect, state in placements:
            if hwnd in alive:
                self.backend.place(hwnd, rect, state)
                n += 1
        return n


def tile(n: int, monitors: list[list[int]]) -> list[tuple[list[int], str]]:
    """Sensible places for n windows: one per monitor maximized; on one monitor, the first gets 60% on the left
    and the rest share the right."""
    if n <= 0:
        return []
    m = len(monitors)
    if m >= 2:
        out = []
        per = [[] for _ in range(m)]
        for i in range(n):
            per[min(i, m - 1)].append(i)
        for mi, idxs in enumerate(per):
            out += _split(monitors[mi], len(idxs))
        return out
    return _split(monitors[0], n)


def _split(area: list[int], n: int) -> list[tuple[list[int], str]]:
    l, t, r, b = area
    if n == 0:
        return []
    if n == 1:
        return [([l, t, r, b], "max")]
    main = l + int((r - l) * 0.6)
    out = [([l, t, main, b], "normal")]
    h = (b - t) // (n - 1)
    for i in range(n - 1):
        out.append(([main, t + i * h, r, t + (i + 1) * h if i < n - 2 else b], "normal"))
    return out
