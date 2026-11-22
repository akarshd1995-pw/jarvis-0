"""App launching, reminders and Windows system actions (ctypes only, no extra deps)."""
from __future__ import annotations

import datetime as dt
import os
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


def open_app(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    name = str(args["name"])
    target = ctx.config.apps.get(name.lower(), name)
    if not dry_run:
        _startfile(target)
    return ActionResult(f"Opening {name}")


def open_url(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not dry_run:
        webbrowser.open(url)
    return ActionResult(f"Opening {url}")


def open_folder(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    p = check_path(ctx.config.resolve_folder(str(args["path"])), ctx.config.allowed_roots)
    if not dry_run:
        _startfile(str(p))
    return ActionResult(f"Opening {p}")


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
    return ActionResult(f"Reminder set for {when}: {args['text']}")


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
        return ActionResult(f"Will run system action: {action}")
    if not IS_WINDOWS:
        return ActionResult(f"(system action {action} only supported on Windows)")
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
    return ActionResult(f"Done: {action}")


def search_memory(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    return ActionResult("Memory search arrives in phase 3 (own embeddings + HNSW).")


def unknown(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    return ActionResult("I can't do that yet. I can manage files (move/copy/delete/list), open apps, "
                        "websites and folders, set reminders, and lock/mute/change volume.")


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
    return ActionResult(f'Searching {site} for "{args["query"]}"')


def unsupported(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    return ActionResult(f'Skipped "{args["request"]}": I can\'t do that yet.')
