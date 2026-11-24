"""A program in plain words, for the review window: play_media(query="super machi") -> play "super machi" on YouTube."""
from __future__ import annotations

from ..dsl.grammar import DSLError, parse


def _v(x) -> str:
    return str(getattr(x, "name", x))


def _filters(a: dict) -> str:
    out = []
    if a.get("ext"):
        out.append(f"{_v(a['ext'])} files")
    if a.get("name_contains"):
        out.append(f'named "{a["name_contains"]}"')
    if a.get("older_than"):
        out.append(f"older than {_v(a['older_than'])}")
    if a.get("newer_than"):
        out.append(f"from the last {_v(a['newer_than'])}")
    return " ".join(out) or "all files"


_SITE = {"youtube": "YouTube", "spotify": "Spotify", "google": "Google", "github": "GitHub",
         "stackoverflow": "Stack Overflow", "linkedin": "LinkedIn"}
_MODE = {"driving": "by car", "two_wheeler": "by two-wheeler", "walking": "on foot", "transit": "by public transport",
         "bicycling": "by cycle"}


def _call(c) -> str:
    a, n = c.args, c.action
    site = _SITE.get(_v(a.get("site", "")), _v(a.get("site", "")).capitalize())
    if n == "play_media":
        mode = _v(a.get("mode", ""))
        what = {"latest": f'the latest video from "{a["query"]}"', "playlist": f'a "{a["query"]}" playlist',
                "live": f'"{a["query"]}" live', "channel": f'the "{a["query"]}" channel'}.get(mode, f'"{a["query"]}"')
        return f"play {what} on {site or 'YouTube'}"
    if n == "search_web":
        return f'search {site or "Google"} for "{a["query"]}"'
    if n == "open_app":
        return f'open {a["name"]}'
    if n == "open_url":
        return f'open {a["url"]}'
    if n == "open_folder":
        return f'open the {a["path"]} folder'
    if n in ("list_files", "delete_files"):
        verb = "list" if n == "list_files" else "delete"
        return f"{verb} {_filters(a)} in {a.get('src', 'Downloads')}"
    if n in ("move_files", "copy_files"):
        to = f'a new folder "{a["new_folder"]}"' if a.get("new_folder") else a.get("dest", "?")
        return f"{n.split('_')[0]} {_filters(a)} from {a.get('src', 'Downloads')} to {to}"
    if n == "remind":
        when = f" at {_v(a['at'])}" if a.get("at") else f" in {_v(a['after'])}" if a.get("after") else ""
        return f'remind you "{a["text"]}"{when}'
    if n == "search_memory":
        where = {"web": " in your browser history", "screenshots": " in your screenshots"}.get(_v(a.get("where", "")), "")
        return f'search your files{where} for "{a["query"]}"'
    if n == "directions":
        frm = f' from {a["origin"]}' if a.get("origin") else ""
        return f'directions to {a["destination"]}{frm} {_MODE.get(_v(a.get("mode", "")), "")}'.rstrip()
    if n == "set_volume":
        return f"set the volume to {_v(a['level'])}%"
    if n == "brightness":
        return f"set brightness to {_v(a['level'])}%" if a.get("level") is not None else f"brightness {_v(a.get('action'))}"
    if n == "radio":
        return f"turn {_v(a['kind'])} {_v(a['state'])}"
    if n == "mood":
        return f"you're feeling {_v(a['kind'])}" + (" (play music)" if _v(a.get("music", "")) == "yes" else "")
    if n == "unknown":
        return "nothing (didn't understand)"
    if n == "unsupported":
        return f'"{a.get("request", "")}" (something I can\'t do)'
    if n == "undo":
        return "undo the last action"
    act = _v(a.get("action", a.get("kind", a.get("mode", ""))))
    return f"{n.replace('_', ' ')}: {act.replace('_', ' ')}" if act else n.replace("_", " ")


def explain(program: str | None) -> str:
    if not program:
        return "nothing (didn't understand)"
    try:
        prog = parse(program)
    except DSLError:
        return program
    return ", then ".join(_call(c) for c in prog.calls)
