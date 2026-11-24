"""Finding apps on any PC (phase 44): "open obsidian", "open whatsapp", "open chrome" on a PC without Chrome.

Index = everything in your Start menu: classic programs (.lnk shortcuts) and Store apps (Get-StartApps), cached in
apps.json and refreshed every 3 days or when you ask for an app it doesn't know. open_app tries, in order:
    your own alias (Settings > Folders & apps) -> a program Windows knows by name (App Paths / PATH)
    -> the Start-menu index (exact name, then close matches) -> for a browser, your default browser.
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"
REFRESH_S = 3 * 86400
MISS_RESCAN_S = 600

# Built-in aliases -> what the app is called in the Start menu
DISPLAY = {"vscode": ["Visual Studio Code", "VS Code"], "cursor": ["Cursor"], "claude": ["Claude"], "chrome": ["Google Chrome"], "edge": ["Microsoft Edge"],
           "notepad": ["Notepad"], "explorer": ["File Explorer"], "terminal": ["Terminal", "Windows Terminal",
                                                                                "Command Prompt", "Windows PowerShell"],
           "spotify": ["Spotify"], "calculator": ["Calculator"], "settings": ["Settings"],
           "firefox": ["Firefox", "Mozilla Firefox"], "brave": ["Brave"], "opera": ["Opera", "Opera GX Browser"]}
BROWSER_ALIASES = ("chrome", "edge", "firefox", "brave", "opera", "browser", "vivaldi", "arc")
_JUNK = re.compile(r"\b(uninstall|readme|read me|help|documentation|release notes|license|website|manual|"
                   r"changelog|support|faq|what'?s new)\b", re.I)


class AppNotFound(Exception):
    def __init__(self, name: str, suggestions: list[str]):
        self.name, self.suggestions = name, suggestions
        hint = f" Did you mean {' or '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"no app called {name} on this PC.{hint}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _start_menu_dirs() -> list[Path]:
    out = []
    for env in ("PROGRAMDATA", "APPDATA"):
        base = os.environ.get(env)
        if base:
            out.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return out


def scan_shortcuts(dirs: list[Path] | None = None) -> list[dict]:
    out = []
    for d in dirs if dirs is not None else _start_menu_dirs():
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".lnk", ".url", ".appref-ms") and not _JUNK.search(p.stem):
                out.append({"name": p.stem, "target": str(p), "kind": "shortcut"})
    return out


def scan_store_apps() -> list[dict]:
    """Every Start-menu app including Store apps (WhatsApp, Netflix, Spotify from the Store), via Get-StartApps."""
    if not IS_WINDOWS:
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"],
                             capture_output=True, text=True, timeout=20, creationflags=0x08000000).stdout
        rows = json.loads(out or "[]")
        rows = [rows] if isinstance(rows, dict) else rows
        return [{"name": r["Name"], "target": r["AppID"], "kind": "appid"} for r in rows
                if r.get("Name") and r.get("AppID") and not _JUNK.search(r["Name"])]
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log.info("Get-StartApps unavailable: %s", e)
        return []


def app_paths_has(exe: str) -> bool:
    """Windows can start this program by name (registered in App Paths, or on PATH)."""
    exe = exe if exe.lower().endswith(".exe") else exe + ".exe"
    if shutil.which(exe):
        return True
    if not IS_WINDOWS:
        return False
    import winreg
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            winreg.CloseKey(winreg.OpenKey(hive, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}"))
            return True
        except OSError:
            continue
    return False


class AppIndex:
    def __init__(self, cache: Path, scanners=None):
        self.cache = Path(cache)
        self.scanners = scanners or (scan_store_apps, scan_shortcuts)
        self._entries: list[dict] | None = None
        self._ts = 0.0
        self._last_miss_scan = 0.0

    def refresh(self) -> list[dict]:
        seen, out = set(), []
        for scan in self.scanners:
            for e in scan():
                k = _norm(e["name"])
                if k and k not in seen:                # Store/AppID entries first: they launch reliably
                    seen.add(k)
                    out.append(e)
        self._entries, self._ts = out, time.time()
        try:
            self.cache.write_text(json.dumps({"ts": self._ts, "apps": out}), encoding="utf-8")
        except OSError:
            pass
        return out

    def entries(self) -> list[dict]:
        if self._entries is None:
            try:
                d = json.loads(self.cache.read_text(encoding="utf-8"))
                self._entries, self._ts = d["apps"], d["ts"]
            except (OSError, ValueError, KeyError):
                pass
        if self._entries is None or time.time() - self._ts > REFRESH_S:
            self.refresh()
        return self._entries or []

    def find(self, name: str, rescan: bool = True) -> dict | None:
        q = _norm(name)
        if not q:
            return None
        hit = self._match(q, self.entries())
        if hit is None and rescan and time.time() - self._last_miss_scan > MISS_RESCAN_S:
            self._last_miss_scan = time.time()                 # just installed? look again (once per 10 min)
            hit = self._match(q, self.refresh())
        return hit

    @staticmethod
    def _match(q: str, entries: list[dict]) -> dict | None:
        best, best_score = None, 0.0
        qw = q.split()
        for e in entries:
            n = _norm(e["name"])
            nw = n.split()
            if n == q:
                score = 100
            elif nw[: len(qw)] == qw:                                       # "visual studio" -> Visual Studio Code
                score = 90 - len(nw) * 0.5
            elif all(w in nw for w in qw):                                  # "word" -> Microsoft Word
                score = 80 - len(nw) * 0.5
            elif len(q) >= 4 and q.replace(" ", "") in n.replace(" ", ""):  # "whatsapp" -> WhatsApp Beta
                score = 72 - len(n) * 0.05
            else:
                r = difflib.SequenceMatcher(None, q, n).ratio()               # typos: "obsdian"
                score = r * 70 if r >= 0.84 else 0
            if score > best_score:
                best, best_score = e, score
        return best

    def suggest(self, name: str, n: int = 2) -> list[str]:
        names = [e["name"] for e in self.entries()]
        return difflib.get_close_matches(name, names, n=n, cutoff=0.55)

    def installed(self, alias: str, config) -> bool:
        target = (getattr(config, "apps", {}) or {}).get(alias, alias)
        if ":" in target[:12] and not re.match(r"^[a-zA-Z]:\\", target):
            return True if alias != "spotify" else bool(self.find("spotify", rescan=False))
        if app_paths_has(target):
            return True
        return any(self.find(d, rescan=False) for d in DISPLAY.get(alias, [alias]))


def launch(entry: dict) -> None:
    if entry["kind"] == "appid":
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{entry['target']}"])
    else:
        os.startfile(entry["target"])                        # noqa: Windows only


def resolve(name: str, config, index: AppIndex | None, browser: dict | None = None) -> tuple[str, dict | None, str]:
    """How to open this app: (how, entry_or_None, label). how = startfile | index | browser.
    Raises AppNotFound with suggestions when nothing fits."""
    alias = name.lower().strip()
    target = (config.apps or {}).get(alias)
    if target:
        is_uri = ":" in target[:15] and not re.match(r"^[a-zA-Z]:\\", target)
        if is_uri and alias != "spotify":
            return "startfile", {"target": target}, name
        if os.path.exists(os.path.expandvars(target)) or (not is_uri and app_paths_has(target)):
            return "startfile", {"target": os.path.expandvars(target)}, name
    if index is not None:
        for display in DISPLAY.get(alias, []) + [name]:
            hit = index.find(display, rescan=display == name)
            if hit:
                return "index", hit, hit["name"]
    if target and ":" in target[:15]:                           # spotify: URI when the app isn't indexed
        return "startfile", {"target": target}, name
    if alias in BROWSER_ALIASES and browser and (browser.get("exe") or browser.get("alias")):
        return "browser", {"target": browser.get("exe") or browser["alias"]}, browser["name"]
    if not IS_WINDOWS and target:                               # tests / other systems: trust the alias
        return "startfile", {"target": target}, name
    raise AppNotFound(name, index.suggest(name) if index is not None else [])


_INDEX: dict = {}


def get_index(data_dir: Path) -> AppIndex:
    key = str(data_dir)
    if key not in _INDEX:
        _INDEX[key] = AppIndex(Path(data_dir) / "apps.json")
    return _INDEX[key]
