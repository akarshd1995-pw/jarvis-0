"""Persona layer: turns structured action results into in-character replies.

Deterministic templates (not a generative model) on purpose: replies must be factually exact
(counts, folders, what was skipped) and safe. Free-form conversation comes later with Jarvis-LM (phase 27).
"""
from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass

from ..actions.base import ActionResult
from ..safety import SafetyError
from .templates import (ACTION_PHRASES, ADDRESS_RATE, OP_VERBS, SITE_NAMES, SYSTEM_PHRASES, T, TONES)

_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
_NOUN_EXT = {"pdf": "PDF", "exe": "installer", "msi": "installer", "apk": "APK", "iso": "ISO image"}
_UNITS = {"m": "minute", "h": "hour", "d": "day", "w": "week"}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def humanize_duration(d: str | None) -> str:
    if not d:
        return ""
    n, u = int(d[:-1]), d[-1]
    if u == "d" and n % 30 == 0 and n >= 30:
        n, word = n // 30, "month"
    else:
        word = _UNITS.get(u, u)
    return f"{n} {word}{'s' if n != 1 else ''}"


def files_word(count: int, ext: str | None) -> str:
    if not ext:
        return "file" if count == 1 else "files"
    noun = _NOUN_EXT.get(ext)
    if noun:
        return noun if count == 1 else noun + "s"
    return f"{ext.upper()} file" + ("" if count == 1 else "s")


def folder_name(v: str | None) -> str:
    if not v:
        return ""
    v = str(v)
    if "/" in v or "\\" in v:
        parts = [p for p in re.split(r"[\\/]", v) if p]
        return v if len(parts) <= 3 else "the " + parts[-1] + " folder"
    return v


@dataclass
class Persona:
    name: str = "Jarvis"
    address: str = "sir"           # "", "sir", "ma'am", a first name...
    tone: str = "butler"           # butler | friendly | minimal
    seed: int | None = None

    def __post_init__(self):
        if self.tone not in TONES:
            self.tone = "butler"
        self.rng = random.Random(self.seed)
        self._last: dict[str, str] = {}

    # ------------------------------------------------------------------ core
    def _num(self, n: int, start: bool) -> str:
        if self.tone != "minimal" and 0 <= n < len(_WORDS):
            w = _WORDS[n]
            return _cap(w) if start else w
        return str(n)

    def say(self, key: str, **f) -> str:
        options = T[self.tone].get(key) or T["butler"][key]
        if len(options) > 1 and key in self._last:               # avoid repeating the same line twice in a row
            options = [o for o in options if o != self._last[key]] or options
        tpl = self.rng.choice(options)
        self._last[key] = tpl
        use_addr = self.address and self.rng.random() < ADDRESS_RATE.get(self.tone, 0)
        f.setdefault("A", f", {self.address}" if use_addr else "")
        for k in list(f):
            if isinstance(f[k], str) and k[:1].islower():
                f.setdefault(_cap(k), _cap(f[k]))
        out = tpl.format_map(_Defaults(f))
        return _cap(re.sub(r"\s+([.,?:])", r"\1", out).strip())

    # ------------------------------------------------------------------ results
    def _file_fields(self, d: dict) -> dict:
        count, ext = d.get("count", 0), d.get("ext")
        filt = ""
        if d.get("older_than"):
            filt = f" older than {humanize_duration(d['older_than'])}"
        elif d.get("newer_than"):
            filt = f" from the last {humanize_duration(d['newer_than'])}"
        if d.get("name_contains"):
            filt += f" with {d['name_contains']} in the name"
        return {"count": self._num(count, False), "Count": self._num(count, True), "files": files_word(count, ext),
                "files_pl": files_word(2, ext), "filt": filt, "src": folder_name(d.get("src")),
                "dest": folder_name(d.get("dest")), "opverb": OP_VERBS.get(d.get("op", ""), "do")}

    def result(self, r: ActionResult) -> str:
        d, k = r.data, r.kind
        if k == "files":
            f = self._file_fields(d)
            if d.get("count", 0) == 0:
                return self.say("none", **f)
            return self.say(d["op"], **f)
        if k == "list":
            f = self._file_fields(d)
            if not r.items:
                return self.say("list_none", **f)
            shown = ", ".join(r.items[:5]) + (f", and {len(r.items) - 5} more" if len(r.items) > 5 else "")
            return self.say("list", items=shown, **f)
        if k == "open":
            name = d["name"]
            if d["what"] == "folder":
                name = folder_name(name)
            return self.say(f"open_{d['what']}", name=name)
        if k == "remind":
            return self.say("remind", text=d["text"], when=self.when(d["due"], d.get("relative", False)))
        if k == "system":
            key = "system_unsupported_os" if d.get("unsupported_os") else f"system_{d['action']}"
            return self.say(key)
        if k == "search":
            return self.say("search", site=SITE_NAMES.get(d["site"], d["site"]), query=d["query"])
        if k == "play":
            key = "play_fallback" if d.get("fallback") else ("play_spotify" if d["site"] == "spotify" else "play")
            return self.say(key, query=d["query"])
        if k == "memory":
            return self.say("memory", query=d["query"])
        if k == "unsupported":
            return self.say("unsupported", request=_lower_first(d["request"]))
        if k == "unknown":
            return self.say("unknown")
        if k == "undo":
            if d.get("cancelled"):
                return self.say("undo_cancelled")
            if not d.get("count"):
                return self.say("undo_none")
            return self.say("undo", count=self._num(d["count"], False), Count=self._num(d["count"], True))
        return r.summary

    def confirm_question(self, preview: ActionResult) -> str:
        d = preview.data
        if preview.kind == "files":
            return self.say(f"confirm_{d['op']}", **self._file_fields(d))
        if preview.kind == "system":
            return self.say("confirm_system", action_phrase=SYSTEM_PHRASES.get(d["action"], d["action"]))
        return preview.summary

    def error(self, action: str, e: Exception) -> str:
        msg = str(e)
        if isinstance(e, FileNotFoundError):
            return self.say("err_not_found", name=_basename(msg.split(": ", 1)[-1]))
        if isinstance(e, SafetyError) and "outside" in msg:
            return self.say("err_outside", name=_basename(msg.split(" is outside")[0]))
        if isinstance(e, SafetyError) and "cap is" in msg:
            return self.say("err_too_many")
        return self.say("err_other", action_phrase=ACTION_PHRASES.get(action, action), error=msg)

    def report(self, report) -> str:
        parts = [self.result(r) for r in report.results]
        if report.declined:
            parts.append(self.say("declined"))
        parts += [self.error(a, e) for a, e in report.failures]
        return " ".join(p for p in parts if p) or self.say("unparsed")

    # ------------------------------------------------------------------ misc
    def when(self, due: float, relative: bool, now: float | None = None) -> str:
        now_dt = dt.datetime.fromtimestamp(now) if now else dt.datetime.now()
        due_dt = dt.datetime.fromtimestamp(due)
        if relative:
            mins = max(1, round((due_dt - now_dt).total_seconds() / 60))
            if mins < 60:
                return f"in {self._num(mins, False)} minute{'s' if mins != 1 else ''}"
            hours = mins / 60
            if hours == int(hours):
                h = int(hours)
                return f"in {self._num(h, False)} hour{'s' if h != 1 else ''}"
        clock = due_dt.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
        day = "today" if due_dt.date() == now_dt.date() else \
            "tomorrow" if due_dt.date() == (now_dt + dt.timedelta(days=1)).date() else due_dt.strftime("on %A")
        return f"at {clock} {day}"

    def reminder_due(self, text: str) -> str:
        return self.say("reminder_due", text=text)

    def unparsed(self) -> str:
        return self.say("unparsed")


def _basename(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", path.strip()) if p]
    return parts[-1] if parts else path


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


class _Defaults(dict):
    def __missing__(self, key):
        return ""
