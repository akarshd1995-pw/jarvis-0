"""Context awareness (phase 38): what "this", "these" and "here" point to right now.

The command model writes two references and Jarvis fills them in when the command runs:

    this   files:  the files selected in the File Explorer window you came from
                   -> else the results Jarvis just showed you (a search or a file list, last 10 minutes)
                   -> else files you copied (Ctrl+C in Explorer)
           text:   the text or link you copied ("google this", "open this link", "directions to this")
    here   the folder open in the File Explorer window you came from

    open_result(n=2)   "open the second one": the second of the results Jarvis just showed you

Reading Explorer goes through the Windows Shell (one short PowerShell call, only when a command says this/here);
the clipboard through the Win32 API. Nothing is stored.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dsl.grammar import Ident, Program

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"
RESULTS_TTL = 600                      # "these" can mean the results Jarvis showed you, for 10 minutes

FILE_ACTIONS = ("move_files", "copy_files", "delete_files", "list_files")
TEXT_ARGS = {"search_web": "query", "play_media": "query", "search_memory": "query", "open_url": "url",
             "remind": "text", "directions": "destination"}


class ContextError(Exception):
    """A reference that can't be filled: says what to do instead."""


@dataclass
class Explorer:
    folder: str | None = None
    selected: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ Windows readers
_PS_EXPLORER = r"""
$ErrorActionPreference = 'SilentlyContinue'
$h = %d
$title = '%s'
$hits = @()
foreach ($w in (New-Object -ComObject Shell.Application).Windows()) {
  if ($w.HWND -ne $h) { continue }
  $f = $null; $sel = @()
  try { $f = $w.Document.Folder.Self.Path } catch {}
  try { $sel = @($w.Document.SelectedItems() | ForEach-Object { $_.Path }) } catch {}
  $hits += @{folder=$f; selected=$sel; name=$w.LocationName}
}
# Windows 11 tabs share one window: the active tab is the one whose name is in the window title
$pick = $hits | Where-Object { $title -and $_.name -and $title.StartsWith($_.name) } | Select-Object -First 1
if (-not $pick) { $pick = $hits | Where-Object { $_.selected.Count -gt 0 } | Select-Object -First 1 }
if (-not $pick) { $pick = $hits | Select-Object -First 1 }
if ($pick) { @{folder=$pick.folder; selected=@($pick.selected); found=$true} | ConvertTo-Json -Compress }
else { '{"found": false}' }
"""


def explorer_state(hwnd: int | None) -> Explorer | None:
    """Folder and selection of the File Explorer window with this handle (None if it isn't Explorer)."""
    if not IS_WINDOWS or not hwnd:
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(int(hwnd), buf, 512)
        title = buf.value.replace("'", "''")
    except Exception:
        title = ""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              _PS_EXPLORER % (int(hwnd), title)],
                             capture_output=True, text=True, timeout=6,
                             creationflags=0x08000000).stdout.strip()     # CREATE_NO_WINDOW
        d = json.loads(out or "{}")
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log.info("couldn't read File Explorer: %s", e)
        return None
    if not d.get("found"):
        return None
    sel = d.get("selected") or []
    if isinstance(sel, str):
        sel = [sel]
    return Explorer(d.get("folder"), [s for s in sel if s])


def desktop_if_foreground(hwnd: int | None) -> str | None:
    """The desktop counts as 'here' when you came from it."""
    if not IS_WINDOWS or not hwnd:
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(64)
        ctypes.windll.user32.GetClassNameW(int(hwnd), buf, 64)
        if buf.value in ("Progman", "WorkerW"):
            from .config import known_folder
            return str(known_folder("Desktop") or Path.home() / "Desktop")
    except Exception:
        pass
    return None


def clipboard() -> tuple[list[str], str]:
    """(files you copied in Explorer, text you copied). Win32 API, no extra packages."""
    if not IS_WINDOWS:
        return [], ""
    import ctypes
    from ctypes import wintypes
    u32, k32, sh = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.shell32
    u32.GetClipboardData.restype = wintypes.HANDLE
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    sh.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_wchar_p, wintypes.UINT]
    files, text = [], ""
    for _ in range(5):
        if u32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return [], ""
    try:
        h = u32.GetClipboardData(15)                         # CF_HDROP: files copied in Explorer
        if h:
            n = sh.DragQueryFileW(h, 0xFFFFFFFF, None, 0)
            for i in range(n):
                size = sh.DragQueryFileW(h, i, None, 0) + 1
                buf = ctypes.create_unicode_buffer(size)
                sh.DragQueryFileW(h, i, buf, size)
                files.append(buf.value)
        h = u32.GetClipboardData(13)                         # CF_UNICODETEXT
        if h:
            p = k32.GlobalLock(h)
            if p:
                try:
                    text = ctypes.wstring_at(p)
                finally:
                    k32.GlobalUnlock(h)
    finally:
        u32.CloseClipboard()
    return files, text.strip()


# ------------------------------------------------------------------ filling the references
class ScreenContext:
    """Everything "this" and "here" can mean. Readers are injectable for tests."""

    def __init__(self, hwnd: Callable[[], int | None] = lambda: None,
                 explorer: Callable[[int | None], Explorer | None] = explorer_state,
                 desktop: Callable[[int | None], str | None] = desktop_if_foreground,
                 clip: Callable[[], tuple[list[str], str]] = clipboard):
        self._hwnd, self._explorer, self._desktop, self._clip = hwnd, explorer, desktop, clip
        self.results: list[dict] = []
        self.results_ts = 0.0
        self._cache: dict = {}

    def set_results(self, rows: list[dict], now: float | None = None) -> None:
        """Rows Jarvis just showed: [{title, path, source}]."""
        self.results, self.results_ts = list(rows), time.time() if now is None else now

    def fresh_results(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        return self.results if self.results and now - self.results_ts <= RESULTS_TTL else []

    # cached per command: Explorer is asked at most once
    def _ex(self) -> Explorer | None:
        if "ex" not in self._cache:
            self._cache["ex"] = self._explorer(self._hwnd())
        return self._cache["ex"]

    def _cb(self) -> tuple[list[str], str]:
        if "cb" not in self._cache:
            self._cache["cb"] = self._clip()
        return self._cache["cb"]

    def this_files(self) -> tuple[list[str], str]:
        """(paths, where they came from)."""
        ex = self._ex()
        if ex and ex.selected:
            return ex.selected, "selected"
        rows = [r for r in self.fresh_results() if r.get("source") != "web" and r.get("path")]
        if rows:
            return [r["path"] for r in rows], "results"
        files, _ = self._cb()
        if files:
            return files, "copied"
        if ex is not None:
            raise ContextError("nothing is selected in File Explorer. Select the files first, then ask again")
        raise ContextError("I can't tell what \"this\" is. Select files in File Explorer (or copy them), "
                           "then open me and ask again")

    def here(self) -> str:
        ex = self._ex()
        if ex and ex.folder and os.path.isdir(ex.folder):
            return ex.folder
        d = self._desktop(self._hwnd())
        if d:
            return d
        raise ContextError("I don't know which folder \"here\" is. Open it in File Explorer, then open me "
                           "from that window and ask again")

    def this_text(self) -> str:
        _, text = self._cb()
        text = " ".join((text or "").split())
        if text:
            return text[:500]
        ex = self._ex()
        if ex and ex.selected:
            return Path(ex.selected[0]).stem
        raise ContextError("I don't see anything copied. Copy the text or link first (Ctrl+C), then ask again")

    def fill(self, prog: Program) -> Program:
        """The program with this/here replaced by real paths and text. Raises ContextError if it can't."""
        self._cache = {}
        for call in prog.calls:
            a = call.args
            if call.action in FILE_ACTIONS:
                src = a.get("src")
                if isinstance(src, Ident) and src.name == "this":
                    files, how = self.this_files()
                    a["_files"], a["_from"] = files, how
                elif isinstance(src, Ident) and src.name == "here":
                    a["src"] = self.here()
                dest = a.get("dest")
                if isinstance(dest, Ident) and dest.name == "here":
                    a["dest"] = self.here()
            key = TEXT_ARGS.get(call.action)
            if key and isinstance(a.get(key), Ident) and a[key].name == "this":
                a[key] = self.this_text()
                if call.action == "open_url" and not a[key].lower().startswith(("http://", "https://", "www.")) \
                        and "." not in a[key].split()[0]:
                    raise ContextError("what you copied isn't a link")
        return prog


def uses_refs(prog: Program | None) -> bool:
    if prog is None:
        return False
    for c in prog.calls:
        if c.action == "open_result":
            return True
        for v in c.args.values():
            if isinstance(v, Ident) and v.name in ("this", "here") and \
                    (c.action in FILE_ACTIONS or TEXT_ARGS.get(c.action)):
                return True
    return False
