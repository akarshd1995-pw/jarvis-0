"""App launching, reminders and Windows system actions (ctypes only, no extra deps)."""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
import time
import webbrowser
from urllib.parse import quote, quote_plus

from ..dsl.grammar import ClockTime, Duration
from ..safety import check_path
from .base import ActionResult, Ctx

IS_WINDOWS = sys.platform == "win32"


def _startfile(target: str) -> None:
    if IS_WINDOWS:
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", target])


def open_app(args: dict, ctx: Ctx, dry_run: bool, index=None, browser=None) -> ActionResult:
    """Your alias, a program Windows knows by name, the Start menu (Store apps too), or your default browser."""
    from .. import apps, machine
    name = str(args["name"])
    if index is None:
        index = apps.get_index(ctx.config.data_dir) if sys.platform == "win32" else None
    if browser is None and name.lower() in apps.BROWSER_ALIASES:
        browser = machine.default_browser()
    how, entry, label = apps.resolve(name, ctx.config, index, browser)
    if not dry_run:
        if how == "index":
            apps.launch(entry)
        else:
            _startfile(entry["target"])
    data = {"what": "app", "name": name}
    if how == "browser" and name.lower() != "browser":
        data.update(instead=label)
    return ActionResult(f"Opening {label}", kind="open", data=data)


def open_url(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not dry_run:
        webbrowser.open(url)
    return ActionResult(f"Opening {url}", kind="open", data={"what": "url", "name": args["url"]})


def open_folder(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    p = check_path(ctx.config.resolve_folder(str(args["path"])), ctx.config.allowed_roots)
    if not dry_run:
        _startfile(str(p))
    return ActionResult(f"Opening {p}", kind="open", data={"what": "folder", "name": str(args["path"])})


def _due_timestamp(args: dict, now: float) -> float:
    if "after" in args:
        d: Duration = args["after"]
        return now + d.seconds()
    t: ClockTime = args["at"]
    now_dt = dt.datetime.fromtimestamp(now)
    due = now_dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if due <= now_dt:
        due += dt.timedelta(days=1)
    return due.timestamp()


def remind(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    due = _due_timestamp(args, time.time())
    when = dt.datetime.fromtimestamp(due).strftime("%a %H:%M")
    if not dry_run:
        ctx.store.add_reminder(due, args["text"])
    return ActionResult(f"Reminder set for {when}: {args['text']}", kind="remind",
                        data={"text": args["text"], "due": due, "relative": "after" in args})


# ------------------------------------------------------------ system controls
VK_VOLUME_MUTE, VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAD, 0xAE, 0xAF
KEYEVENTF_KEYUP = 0x0002


def _press(vk: int, times: int = 1) -> None:
    import ctypes
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    for _ in range(times):
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def system(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    if dry_run:
        return ActionResult(f"Will run system action: {action}", kind="system", data={"action": action, "dry": True})
    if not IS_WINDOWS:
        return ActionResult(f"(system action {action} only supported on Windows)", kind="system",
                            data={"action": action, "unsupported_os": True})
    import ctypes
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    if action == "lock":
        user32.LockWorkStation()
    elif action == "mute":
        _press(VK_VOLUME_MUTE)
    elif action == "volume_up":
        _press(VK_VOLUME_UP, 5)
    elif action == "volume_down":
        _press(VK_VOLUME_DOWN, 5)
    elif action == "sleep_display":
        HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER = 0xFFFF, 0x0112, 0xF170
        user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
    return ActionResult(f"Done: {action}", kind="system", data={"action": action})


def search_memory(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    """Phase 7: search the local index of your files, screenshots and browser history."""
    from ..memory import index as MI
    idx = MI.get(ctx.config.data_dir)
    query = str(args.get("query", ""))
    ext = str(args["ext"]) if "ext" in args else None
    where = str(args["where"]) if "where" in args else None
    newer = args["newer_than"].seconds() if "newer_than" in args else None
    older = args["older_than"].seconds() if "older_than" in args else None
    from ..memory.meaning import get_embedder
    from pathlib import Path as _P
    emb = get_embedder(_P(__file__).resolve().parents[2] / "checkpoints") if ctx.config.memory_meaning else None
    hits = idx.search(query, ext=ext, newer_than_s=newer, older_than_s=older, where=where, embedder=emb)
    results = [{"title": h.title, "path": h.path, "source": h.source, "ext": h.ext, "mtime": h.mtime,
                "snippet": h.snippet, "screenshot": h.screenshot} for h in hits]
    return ActionResult(f"{len(hits)} match(es) for {query!r}", [h.title for h in hits], kind="memory",
                        data={"query": query, "ext": ext, "where": where, "count": len(hits), "results": results,
                              "indexed": idx.count()})


def unknown(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    return ActionResult("I can't do that yet. I can manage files (move/copy/delete/list), open apps, "
                        "websites and folders, set reminders, play and search, and lock/mute/change volume.",
                        kind="unknown")


SEARCH_URLS = {
    "google": "https://www.google.com/search?q={qp}",
    "youtube": "https://www.youtube.com/results?search_query={qp}",
    "amazon": "https://www.amazon.in/s?k={qp}",
    "flipkart": "https://www.flipkart.com/search?q={qp}",
    "swiggy": "https://www.swiggy.com/search?query={qp}",
    "zomato": "https://www.google.com/search?q={qp}+site%3Azomato.com",
    "github": "https://github.com/search?q={qp}",
    "stackoverflow": "https://stackoverflow.com/search?q={qp}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={qp}",
    "spotify": "https://open.spotify.com/search/{qpath}",
    "maps": "https://www.google.com/maps/search/{qp}",
    "reddit": "https://www.reddit.com/search/?q={qp}",
    "linkedin": "https://www.linkedin.com/search/results/all/?keywords={qp}",
}


def search_url(query: str, site: str = "google") -> str:
    return SEARCH_URLS[site].format(qp=quote_plus(query), qpath=quote(query))


def search_web(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    site = str(args.get("site", "google"))
    url = search_url(args["query"], site)
    if not dry_run:
        webbrowser.open(url)
    return ActionResult(f'Searching {site} for "{args["query"]}"', kind="search",
                        data={"query": args["query"], "site": site})


def unsupported(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    return ActionResult(f'Skipped "{args["request"]}": I can\'t do that yet.', kind="unsupported",
                        data={"request": args["request"]})


# ------------------------------------------------------------ play media
from . import youtube as yt  # noqa: E402


def first_youtube_video(query: str, fetch=None, timeout: float = 5.0) -> str | None:
    """Kept for compatibility: the top video for a query."""
    return yt.top_video(query, fetch)


def play_media(args: dict, ctx: Ctx, dry_run: bool, fetch=None) -> ActionResult:
    """mode: top (default) | latest (newest upload of a channel) | playlist | live | channel (open the channel)."""
    query, site = args["query"], str(args.get("site", "youtube"))
    mode = str(args.get("mode", "top"))
    data = {"query": query, "site": site, "mode": mode}
    if site == "spotify":
        if not dry_run:
            _startfile(f"spotify:search:{quote(query)}")   # opens the Spotify app on the search
        return ActionResult(f'Opening "{query}" in Spotify', kind="play", data=data)
    if dry_run:
        return ActionResult(f'Will play "{query}" on YouTube ({mode})', kind="play", data=data)
    url = None
    if mode == "latest":
        found = yt.latest_video(query, fetch)
        if found:
            vid, title, channel = found
            url = f"https://www.youtube.com/watch?v={vid}"
            data.update(title=title, channel=channel)
    elif mode == "playlist":
        pid = yt.playlist(query, fetch)
        url = f"https://www.youtube.com/playlist?list={pid}&play=1" if pid else None
        if pid:
            vid = yt.top_video(query + " playlist", fetch)
            url = f"https://www.youtube.com/watch?v={vid}&list={pid}" if vid else url
    elif mode == "live":
        vid = yt.live_video(query, fetch)
        url = f"https://www.youtube.com/watch?v={vid}" if vid else None
    elif mode == "channel":
        url = yt.channel_url(query, fetch)
    else:
        status, vid, title = yt.best_video(query, fetch)
        if status == "ok":
            url = f"https://www.youtube.com/watch?v={vid}"
            if title:
                data["title"] = title
        else:
            # Nothing that matches: never play a random video. Show the results so the user can pick or fix the spelling.
            webbrowser.open(yt.results_url(query))
            msg = {"offline": f'Couldn\'t reach YouTube, opened the results page for "{query}"',
                   "none": f'YouTube has no videos for "{query}"',
                   "nomatch": f'No video matches "{query}", showing YouTube results'}[status]
            return ActionResult(msg, kind="play", data={**data, "fallback": True, "reason": status,
                                                         "closest": title or ""})
    if url:
        webbrowser.open(url)
        return ActionResult(f'Playing "{query}" on YouTube ({mode})', kind="play", data=data)
    webbrowser.open(yt.results_url(query))
    return ActionResult(f'Couldn\'t find that automatically, opened YouTube results for "{query}"', kind="play",
                        data={**data, "fallback": True})
