"""Browser history (Chrome, Edge, Brave; Firefox if present), read from the browsers' own local files.

The History database is locked while the browser runs, so we read a temporary copy. Only titles, URLs and
visit times are used; nothing leaves this PC. Turn it off in Settings > Privacy.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

CHROMIUM_EPOCH = 11644473600  # seconds between 1601-01-01 and 1970-01-01


def chromium_profiles() -> list[tuple[str, Path]]:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    roots = {"Chrome": local / "Google" / "Chrome" / "User Data", "Edge": local / "Microsoft" / "Edge" / "User Data",
             "Brave": local / "BraveSoftware" / "Brave-Browser" / "User Data"}
    out = []
    for name, root in roots.items():
        if not root.is_dir():
            continue
        for prof in [root / "Default"] + sorted(root.glob("Profile *")):
            if (prof / "History").is_file():
                out.append((name, prof / "History"))
    return out


def firefox_profiles() -> list[tuple[str, Path]]:
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    root = appdata / "Mozilla" / "Firefox" / "Profiles"
    return [("Firefox", p / "places.sqlite") for p in root.glob("*") if (p / "places.sqlite").is_file()] \
        if root.is_dir() else []


def _copy_query(db: Path, sql: str, args: tuple) -> list[tuple]:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "h.sqlite"
        shutil.copy2(db, dst)
        con = sqlite3.connect(str(dst))
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()


def history(days: int = 120, limit: int = 20000) -> list[dict]:
    """[{url, title, ts, browser}] most recent first, one row per URL."""
    since = time.time() - days * 86400
    rows: dict[str, dict] = {}
    for browser, db in chromium_profiles():
        try:
            got = _copy_query(db, "SELECT url, title, last_visit_time FROM urls WHERE last_visit_time > ? "
                                  "ORDER BY last_visit_time DESC LIMIT ?",
                              (int((since + CHROMIUM_EPOCH) * 1_000_000), limit))
        except (OSError, sqlite3.Error):
            continue
        for url, title, t in got:
            ts = t / 1_000_000 - CHROMIUM_EPOCH
            if url not in rows or rows[url]["ts"] < ts:
                rows[url] = {"url": url, "title": title or "", "ts": ts, "browser": browser}
    for browser, db in firefox_profiles():
        try:
            got = _copy_query(db, "SELECT url, title, last_visit_date FROM moz_places WHERE last_visit_date > ? "
                                  "ORDER BY last_visit_date DESC LIMIT ?", (int(since * 1_000_000), limit))
        except (OSError, sqlite3.Error):
            continue
        for url, title, t in got:
            ts = (t or 0) / 1_000_000
            if url not in rows or rows[url]["ts"] < ts:
                rows[url] = {"url": url, "title": title or "", "ts": ts, "browser": browser}
    skip = ("chrome://", "edge://", "about:", "chrome-extension://", "file://", "data:")
    out = [r for r in rows.values() if not r["url"].startswith(skip)]
    return sorted(out, key=lambda r: -r["ts"])[:limit]
