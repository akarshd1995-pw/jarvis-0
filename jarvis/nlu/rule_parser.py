"""Bootstrap parser: a few regex rules so Jarvis is usable on day 1, before the model is trained.

Order of resolution in the daemon:
  1. input that is already valid DSL       -> run directly (power-user mode)
  2. trained intent model (if checkpoint)  -> model output, validated
  3. these rules                           -> fallback
"""
from __future__ import annotations

import re

from ..dsl.grammar import DSLError, Program, parse

_FOLDER = r"(downloads|documents|desktop|pictures|music|videos|archive)"
_UNITS = {"min": "m", "minute": "m", "minutes": "m", "mins": "m", "m": "m",
          "hour": "h", "hours": "h", "hr": "h", "hrs": "h", "h": "h",
          "day": "d", "days": "d", "d": "d", "week": "w", "weeks": "w", "w": "w",
          "month": "month", "months": "month"}


def _dur(n: str, unit: str) -> str:
    u = _UNITS[unit.lower()]
    if u == "month":
        return f"{int(n) * 30}d"
    return f"{int(n)}{u}"


def _time(h: str, m: str | None, ampm: str | None) -> str:
    hour, minute = int(h), int(m or 0)
    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute:02d}"


KNOWN_EXT = {"pdf", "docx", "doc", "xlsx", "xls", "csv", "pptx", "txt", "md", "png", "jpg", "jpeg",
             "gif", "webp", "svg", "mp4", "mkv", "mov", "mp3", "wav", "zip", "rar", "7z", "exe",
             "msi", "iso", "py", "json", "html"}


def _ext(text: str) -> str | None:
    for word in re.findall(r"\.?[A-Za-z0-9]+", text):
        w = word.lower().lstrip(".")
        if w in KNOWN_EXT:
            return w
        if w.endswith("s") and w[:-1] in KNOWN_EXT:
            return w[:-1]
    return None


def _filters(text: str) -> str:
    parts = []
    ext = _ext(text)
    if ext:
        parts.append(f"ext={ext}")
    m = re.search(r"older than (\d+) (\w+)", text)
    if m and m.group(2).lower() in _UNITS:
        parts.append(f"older_than={_dur(m.group(1), m.group(2))}")
    return ", ".join(parts)


RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^undo\b", re.I), lambda m, t: "undo()"),
    (re.compile(r"^(?:open|launch|start) (?:the )?(?:website |site )?((?:https?://)?[\w\-]+\.[\w.\-/]+)$", re.I),
     lambda m, t: f'open_url(url="{m.group(1)}")'),
    (re.compile(rf"^(?:open|show) (?:my )?{_FOLDER}(?: folder)?$", re.I),
     lambda m, t: f"open_folder(path={m.group(1).capitalize()})"),
    (re.compile(r"^(?:open|launch|start) (?:the )?([\w\-]+)(?: app)?$", re.I),
     lambda m, t: f"open_app(name={m.group(1).lower()})"),
    (re.compile(r"^remind me (?:to )?(.+?) in (\d+) (\w+)$", re.I),
     lambda m, t: f'remind(text="{m.group(1)}", after={_dur(m.group(2), m.group(3))})'),
    (re.compile(r"^remind me (?:to )?(.+?) at (\d{1,2})(?::(\d{2}))? ?(am|pm)?$", re.I),
     lambda m, t: f'remind(text="{m.group(1)}", at={_time(m.group(2), m.group(3), m.group(4))})'),
    (re.compile(rf"^(move|copy) .*?from {_FOLDER} to {_FOLDER}", re.I),
     lambda m, t: "{}_files(src={}, dest={}{})".format(
         m.group(1).lower(), m.group(2).capitalize(), m.group(3).capitalize(),
         (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(?:delete|remove|clean ?up) .*?(?:from|in) {_FOLDER}", re.I),
     lambda m, t: "delete_files(src={}{})".format(
         m.group(1).capitalize(), (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(?:list|show) .*?(?:in|from) {_FOLDER}", re.I),
     lambda m, t: "list_files(src={}{})".format(
         m.group(1).capitalize(), (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(r"^lock(?: the)?(?: screen| pc| computer)?$", re.I), lambda m, t: "system(action=lock)"),
    (re.compile(r"^(?:mute|unmute)(?: the)?(?: sound| audio| volume)?$", re.I), lambda m, t: "system(action=mute)"),
    (re.compile(r"^(?:volume up|louder|increase volume)$", re.I), lambda m, t: "system(action=volume_up)"),
    (re.compile(r"^(?:volume down|quieter|decrease volume)$", re.I), lambda m, t: "system(action=volume_down)"),
]


def rule_parse(text: str) -> Program | None:
    text = text.strip().rstrip(".!")
    # Compound commands: split on ' and then ' / ' then ' / ' and '
    pieces = re.split(r"\s+(?:and then|then|and)\s+(?=(?:open|launch|start|remind|move|copy|delete|remove|list|show|lock|mute)\b)",
                      text, flags=re.I)
    dsl = []
    for piece in pieces:
        for pattern, build in RULES:
            m = pattern.search(piece.strip())
            if m:
                dsl.append(build(m, piece))
                break
        else:
            return None
    try:
        return parse(" ; ".join(dsl))
    except DSLError:
        return None
