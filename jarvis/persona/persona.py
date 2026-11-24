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
_NOUN_EXT = {"pdf": "PDF", "exe": "installer", "msi": "installer", "apk": "APK", "iso": "ISO image",
             "video": "video", "image": "image", "audio": "audio file", "document": "document", "archive": "archive",
             "installer": "installer", "code": "code file"}
_UNITS = {"m": "minute", "h": "hour", "d": "day", "w": "week"}
APP_DISPLAY = {"vscode": "VS Code", "explorer": "File Explorer", "terminal": "Terminal", "chrome": "Chrome",
               "edge": "Edge", "notepad": "Notepad", "spotify": "Spotify", "calculator": "Calculator",
               "settings": "Settings"}


def app_name(v: str) -> str:
    return APP_DISPLAY.get(str(v).lower(), str(v))


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
    model: object | None = None     # jarvis.nlg.ReplyModel (trained reply generator), optional

    def __post_init__(self):
        if self.tone not in TONES:
            self.tone = "butler"
        self.rng = random.Random(self.seed)
        self._last: dict[str, str] = {}
        self.last_source = "template"

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
        use_addr = bool(self.address) and self.rng.random() < ADDRESS_RATE.get(self.tone, 0)
        return render_template(tpl, f, self.address if use_addr else "")

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
                "dest": (f'a new folder "{d.get("dest")}" in {folder_name(d.get("src"))}' if d.get("new_folder")
                         else folder_name(d.get("dest"))), "opverb": OP_VERBS.get(d.get("op", ""), "do")}

    def plan(self, r: ActionResult) -> tuple[str, dict]:
        """Facts for a result: (template key, fields). Shared by templates and the reply model."""
        d, k = r.data, r.kind
        if k == "files":
            f = self._file_fields(d)
            return ("none" if d.get("count", 0) == 0 else d["op"]), f
        if k == "list":
            f = self._file_fields(d)
            if not r.items:
                return "list_none", f
            shown = ", ".join(r.items[:5]) + (f", and {len(r.items) - 5} more" if len(r.items) > 5 else "")
            return "list", {**f, "items": shown}
        if k == "open":
            name = folder_name(d["name"]) if d["what"] == "folder" else app_name(d["name"]) if d["what"] == "app" \
                else d["name"]
            return f"open_{d['what']}", {"name": name}
        if k == "remind":
            return "remind", {"text": d["text"], "when": self.when(d["due"], d.get("relative", False))}
        if k == "system":
            return ("system_unsupported_os" if d.get("unsupported_os") else f"system_{d['action']}"), {}
        if k == "search":
            return "search", {"site": SITE_NAMES.get(d["site"], d["site"]), "query": d["query"]}
        if k == "play":
            mode = d.get("mode", "top")
            if d.get("fallback"):
                reason = d.get("reason")
                if reason == "nomatch" and d.get("closest"):
                    return "play_nomatch", {"query": d["query"], "closest": d["closest"]}
                if reason in ("offline", "none"):
                    return f"play_{reason}", {"query": d["query"]}
                return "play_fallback", {"query": d["query"]}
            if d["site"] == "spotify":
                return "play_spotify", {"query": d["query"]}
            if mode == "latest" and d.get("title"):
                return "play_latest", {"channel": d.get("channel") or d["query"], "title": d["title"]}
            if mode in ("playlist", "live", "channel"):
                return f"play_{mode}", {"query": d["query"]}
            return "play", {"query": d["query"]}
        if k == "memory":
            return self._plan_memory(d)
        if k == "unsupported":
            return "unsupported", {"request": _lower_first(d["request"])}
        if k == "unknown":
            return "unknown", {}
        if k == "window":
            return f"window_{d['action']}", {"target": app_name(d["app"]) if d.get("app") else "the window"}
        if k == "desktop":
            return f"desktop_{d['action']}", {}
        if k == "media":
            return f"media_{d['action']}", {}
        if k == "brightness":
            key = "brightness_set" if d.get("action") == "set" else f"brightness_{d['action']}"
            return key, {"level": str(d.get("level", ""))}
        if k == "volume_set":
            return "volume_set", {"level": str(d["level"])}
        if k == "radio":
            return f"radio_{d['state']}", {"device": "Wi-Fi" if d["kind"] == "wifi" else "Bluetooth"}
        if k == "screenshot":
            return f"screenshot_{d['mode']}", {}
        if k == "record":
            return f"record_{d['action']}", {}
        if k == "power":
            return f"power_{d['action']}", {"delay": self.delay(d.get("seconds", 30))}
        if k == "training":
            return self._plan_training(d)
        if k == "directions":
            return self._plan_directions(d)
        if k == "undo":
            if d.get("cancelled"):
                return "undo_cancelled", {}
            if not d.get("count"):
                return "undo_none", {}
            return "undo", {"count": self._num(d["count"], False), "Count": self._num(d["count"], True)}
        return "", {"summary": r.summary}

    def plan_confirm(self, preview: ActionResult) -> tuple[str, dict]:
        d = preview.data
        if preview.kind == "files":
            return f"confirm_{d['op']}", self._file_fields(d)
        if preview.kind == "system":
            return "confirm_system", {"action_phrase": SYSTEM_PHRASES.get(d["action"], d["action"])}
        if preview.kind == "window":
            return f"confirm_window_{d['action']}", {"target": app_name(d["app"]) if d.get("app") else "the window"}
        if preview.kind == "radio":
            return f"confirm_radio_{d['state']}", {"device": "Wi-Fi" if d["kind"] == "wifi" else "Bluetooth"}
        if preview.kind == "power":
            return f"confirm_power_{d['action']}", {"delay": self.delay(d.get("seconds", 30))}
        return "", {"summary": preview.summary}

    def plan_error(self, action: str, e: Exception) -> tuple[str, dict]:
        msg = str(e)
        if isinstance(e, FileNotFoundError):
            return "err_not_found", {"name": _basename(msg.split(": ", 1)[-1])}
        if isinstance(e, SafetyError) and "outside" in msg:
            return "err_outside", {"name": _basename(msg.split(" is outside")[0])}
        if isinstance(e, SafetyError) and "cap is" in msg:
            return "err_too_many", {}
        if type(e).__name__ == "NoWindowFound":
            name = msg.replace("no open window for ", "") if msg.startswith("no open window for ") else ""
            if name:
                return "err_no_window", {"name": name}
        if type(e).__name__ == "NotOnWindows":
            return "err_not_windows", {}
        return "err_other", {"action_phrase": ACTION_PHRASES.get(action, action), "error": msg}

    def speak(self, key: str, fields: dict) -> str:
        """Render facts as a sentence: the trained reply model if loaded (and its output passes the
        fact check), otherwise the templates."""
        if not key:
            return fields.get("summary", "")
        if self.model is not None and getattr(self.model, "knows_key", lambda k: True)(key):
            out = self.model.generate(key, fields, self.tone, self.address)
            if out:
                self.last_source = "reply-model"
                return out
        self.last_source = "template"
        return self.say(key, **fields)

    def result(self, r: ActionResult) -> str:
        if r.kind in ("briefing", "report"):         # already written in full (briefings, health check)
            self.last_source = "briefing"
            return r.data["text"]
        failed = r.data.get("failed") if r.kind in ("files", "undo") else None
        if failed and not r.data.get("count"):        # nothing worked: say so, never "no files matched"
            what = {"move": "move", "copy": "copy", "delete": "delete"}.get(r.data.get("op"), "undo")
            return f"I couldn't {what} any of them{self._addr()}. " + self.failures_line(failed)
        text = self.speak(*self.plan(r))
        if failed:
            text += " " + self.failures_line(failed, r.data.get("requested"))
        return text

    def _addr(self) -> str:
        return f", {self.address}" if self.address and self.tone == "butler" else ""

    def failures_line(self, failed: list, requested: int | None = None) -> str:
        """'2 couldn't be moved: report.xlsx (it's open in another app), clip.mp4 (there isn't enough disk space).'"""
        shown = "; ".join(f"{n} ({why})" for n, why in failed[:3])
        more = f"; and {len(failed) - 3} more" if len(failed) > 3 else ""
        n = len(failed)
        head = f"{n} file{'s' * (n != 1)} didn't go through" if requested else "Details"
        return f"{head}: {shown}{more}. Everything else is done and the originals of the failed ones are untouched."

    def confirm_question(self, preview: ActionResult) -> str:
        return self.speak(*self.plan_confirm(preview))

    def error(self, action: str, e: Exception) -> str:
        return self.speak(*self.plan_error(action, e))

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

    @staticmethod
    def _dur(seconds: int) -> str:
        m = max(1, round(seconds / 60))
        if m < 60:
            return f"{m} minute{'s' * (m != 1)}"
        h, m = divmod(m, 60)
        return f"{h} hour{'s' * (h != 1)}" + (f" {m} minute{'s' * (m != 1)}" if m else "")

    def _plan_directions(self, d: dict) -> tuple[str, dict]:
        dest = d["destination"]
        mode = {"driving": "by car", "two_wheeler": "on a two-wheeler", "walking": "on foot",
                "transit": "by public transport", "bicycling": "by cycle"}.get(d.get("mode"), "")
        if d.get("missing_place"):
            return "directions_missing_place", {"place": d["missing_place"], "dest": dest}
        if d.get("seconds"):
            via = f" via {d['via']}" if d.get("via") else ""
            km = f"{d['meters'] / 1000:.0f} km" if d.get("meters") else ""
            if d.get("source") == "Google Maps":
                cond = " in current traffic" if d.get("traffic") else ""
                usual = d.get("usual_seconds")
                if usual and d["seconds"] - usual >= 300:
                    cond += f", about {self._dur(d['seconds'] - usual)} more than usual"
            else:
                cond = " in normal traffic (OpenStreetMap estimate; Maps has live traffic)"
            return "directions_time", {"dest": dest, "time": self._dur(d["seconds"]), "via": via, "km": km,
                                       "cond": cond, "mode": mode}
        if d.get("no_origin"):
            return "directions_noorigin", {"dest": dest}
        return "directions_open", {"dest": dest, "mode": mode}

    def _plan_memory(self, d: dict) -> tuple[str, dict]:
        what = d.get("query") or ("your latest screenshot" if d.get("where") == "screenshots" else
                                  f"your latest {files_word(2, d['ext'])}" if d.get("ext") else "that")
        if not d.get("count"):
            if not sum((d.get("indexed") or {}).values()):
                return "memory_indexing", {}
            return "memory_none", {"query": what}
        top = d["results"][0]
        if top["source"] == "web":
            place = "your browser history"
        else:
            import os
            place = os.path.basename(os.path.dirname(top["path"])) or "your files"
        more = d["count"] - 1
        return ("memory_found" if more else "memory_found_one"), {
            "title": top["title"], "place": place, "age": self.ago(top["mtime"]), "more": self._num(more, False),
            "More": self._num(more, True),
            "query": what}

    def ago(self, ts: float) -> str:
        import time as _t
        s = max(0, _t.time() - (ts or 0))
        if s < 3600:
            return "just now" if s < 120 else f"{int(s // 60)} minutes ago"
        if s < 86400:
            return f"{int(s // 3600)} hour{'s' if s >= 7200 else ''} ago"
        days = int(s // 86400)
        if days == 1:
            return "yesterday"
        if days < 30:
            return f"{days} days ago"
        if days < 365:
            m = days // 30
            return f"{m} month{'s' if m > 1 else ''} ago"
        return f"{days // 365} year{'s' if days >= 730 else ''} ago"

    def _plan_training(self, d: dict) -> tuple[str, dict]:
        if d["action"] == "improve":
            key = "running" if d.get("running") else "nomodel" if d.get("nomodel") else "started"
            return f"training_improve_{key}", {}
        if d["action"] == "review":
            if d.get("off"):
                return "training_review_off", {}
            if not d.get("n"):
                return "training_review_none", {}
            if d.get("opened"):
                return "training_review_open", {"n": d["n"]}
            return "training_review_text", {"n": d["n"], "list": d.get("text", "")}
        if d["action"] == "rollback":
            return ("training_rollback", {"version": d["version"]}) if d.get("ok") else \
                ("training_rollback_none", {"msg": d.get("msg", "")})
        if d["action"] == "pause":
            return ("training_pause_none", {}) if d.get("none") else ("training_paused", {"name": d.get("name", "")})
        if d["action"] == "resume":
            return ("training_resume_none", {}) if d.get("none") else ("training_resumed", {})
        jobs = d.get("jobs") or []
        running = [j for j in jobs if j["state"] == "running"]
        if running:
            j = running[0]
            f = {"name": j["name"], "pct": f"{self._num(round(j.get('pct', 0)), False)} percent",
                 "eta": self.minutes(j.get("eta_min", 0))}
            return ("training_status_paused" if d.get("paused") else "training_status"), f
        done = [j for j in jobs if j["state"] == "finished"]
        if done:
            return "training_status_done", {"name": done[0]["name"], "best": _pct(done[0].get("best"))}
        return "training_status_none", {}

    def minutes(self, m: int) -> str:
        if m < 60:
            return f"{self._num(m, False)} minute{'s' if m != 1 else ''}"
        h, r = divmod(m, 60)
        return f"{self._num(h, False)} hour{'s' if h != 1 else ''}" + (f" {r} minutes" if r >= 5 else "")

    def alert(self, a) -> str:
        """In-character text for a monitoring alert (jarvis.monitor.Alert)."""
        d = a.data
        if a.kind == "gpu_hot":
            return self.speak("alert_gpu_hot_training" if d.get("training") else "alert_gpu_hot", {"temp": str(d["temp"])})
        if a.kind == "disk_low":
            return self.speak("alert_disk_low", {"free_gb": f"{d['free_gb']:g}"})
        if a.kind in ("battery_low", "ram_high"):
            return self.speak(f"alert_{a.kind}", {"pct": str(d["pct"])})
        if a.kind == "training_done":
            return self.speak("alert_training_done", {"name": d["name"].replace("Training ", ""), "best": _pct(d.get("best"))})
        if a.kind == "training_crashed":
            return self.speak("alert_training_crashed", {"name": d["name"].replace("Training ", ""), "error": d.get("error", "")})
        if a.kind == "training_stalled":
            return self.speak("alert_training_stalled", {"name": d["name"].replace("Training ", ""),
                                                         "minutes": str(d["minutes"])})
        if a.kind == "unusual":
            label = {"cpu": "CPU load", "ram": "memory use", "gpu": "GPU load", "gpu_temp": "GPU temperature"}[d["metric"]]
            return self.speak("alert_unusual", {"metric": label, "value": str(d["value"]), "typical": str(d["typical"])})
        return a.kind

    def delay(self, seconds: int) -> str:
        if seconds < 60:
            return f"{self._num(seconds, False)} seconds"
        mins = round(seconds / 60)
        if mins < 60:
            return f"{self._num(mins, False)} minute{'s' if mins != 1 else ''}"
        hours = round(mins / 60)
        return f"{self._num(hours, False)} hour{'s' if hours != 1 else ''}"

    def reminder_due(self, text: str) -> str:
        return self.speak("reminder_due", {"text": text})

    def unparsed(self) -> str:
        return self.speak("unparsed", {})


def render_template(tpl: str, fields: dict, address: str = "") -> str:
    """Fill a phrasing with facts. address="" leaves the {A} slot empty."""
    f = dict(fields)
    f.setdefault("A", f", {address}" if address else "")
    for k in list(f):
        if isinstance(f[k], str) and k[:1].islower():
            f.setdefault(_cap(k), _cap(f[k]))
    out = tpl.format_map(_Defaults(f))
    return _cap(re.sub(r"\s+([.,?:])", r"\1", out).strip())


def _pct(x) -> str:
    if x is None:
        return "unknown"
    return f"{x * 100:.1f} percent" if x <= 1 else f"{x:.1f} percent"


def _basename(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", path.strip()) if p]
    return parts[-1] if parts else path


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


class _Defaults(dict):
    def __missing__(self, key):
        return ""
