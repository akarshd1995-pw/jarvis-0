"""Wellbeing (phase 34): small nudges that look after you, never in the way.

  eye breaks     every 20 minutes of screen time: look at something 20 feet away for 20 seconds (20-20-20)
  stretch        after an hour without a real break: stand up. Five minutes away from the keyboard is a break
                 and starts the count again
  screen time    "limit youtube to an hour a day": a heads-up near the limit, a note when it's reached, and (only
                 if you turn it on) the app is minimized after that. Apps by program (steam.exe) or by a word in
                 the window title (youtube), like focus blocking
  wind-down      learns when you usually stop for the day from your activity log; on a night you're well past
                 that, one gentle nudge
None of it interrupts a focus session or a game: those reminders are simply skipped.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Callable

EYES_S, STRETCH_S, BREAK_S, ACTIVE_IDLE_S = 20 * 60, 60 * 60, 5 * 60, 90
WARN_AT = 0.8
CURRENT: "tuple | None" = None           # (engine, config) of the running Jarvis: undo of a limit reaches it
DAY_START_HOUR = 5                      # a "day" for screen time runs 5 am to 5 am: 1 am still counts as tonight


def day_key(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    if d.hour < DAY_START_HOUR:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def matches(key: str, title: str, exe: str) -> bool:
    k = key.lower().strip()
    e = os.path.basename((exe or "").replace("\\", "/")).lower()
    if k.endswith(".exe"):
        return e == k
    if e.endswith(".exe") and e[:-4] == k:                # "steam" also means steam.exe
        return True
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", (title or "").lower()))


@dataclass
class Wellbeing:
    path: Path                                            # screentime.json: today's minutes per limited app
    limits: dict[str, int] = field(default_factory=dict)  # app -> minutes a day
    eyes: bool = True
    stretch: bool = True
    winddown: bool = True
    enforce: bool = False
    clock: Callable[[], float] = time.time
    active_s: float = 0.0                                 # screen time since the last real break
    since_eyes: float = 0.0
    snoozed_until: float = 0.0
    used: dict = field(default_factory=dict)              # {day: {app: seconds}}
    warned: dict = field(default_factory=dict)            # {day: {app: "warn"|"limit"}}
    wound_down: str = ""                                  # the day we last nudged about the hour
    _last: float | None = None
    _away_s: float = 0.0

    def __post_init__(self):
        try:
            d = json.loads(Path(self.path).read_text(encoding="utf-8"))
            self.used, self.warned, self.wound_down = d.get("used", {}), d.get("warned", {}), d.get("wound_down", "")
        except (OSError, ValueError):
            pass

    def save(self):
        today = day_key(self.clock())
        keep = {k: v for k, v in self.used.items() if k >= day_key(self.clock() - 14 * 86400)}
        data = {"used": keep, "warned": {today: self.warned.get(today, {})}, "wound_down": self.wound_down}
        try:
            Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    # -------------------------------------------------------------- the clock
    def tick(self, idle_s: float | None, title: str, exe: str, quiet: bool = False,
             usual_end: float | None = None) -> list[tuple[str, dict]]:
        """Call every few seconds. Returns nudges: ("eyes"|"stretch"|"limit_warn"|"limit_hit"|"winddown", data);
        ("minimize", {}) means: the app in front is over its limit and enforcing is on."""
        now = self.clock()
        dt = 0.0 if self._last is None else min(max(0.0, now - self._last), 60.0)
        self._last = now
        out: list[tuple[str, dict]] = []
        active = idle_s is not None and idle_s < ACTIVE_IDLE_S
        if not active:
            self._away_s = max(self._away_s + dt, idle_s or 0.0)
            if self._away_s >= BREAK_S:                   # a real break: start counting again
                self.active_s = self.since_eyes = 0.0
            return out
        self._away_s = 0.0
        self.active_s += dt
        self.since_eyes += dt
        # screen time per limited app
        today = day_key(now)
        day = self.used.setdefault(today, {})
        for app, minutes in self.limits.items():
            if not matches(app, title, exe):
                continue
            day[app] = day.get(app, 0.0) + dt
            used_min = day[app] / 60
            said = self.warned.setdefault(today, {}).get(app)
            if used_min >= minutes and said != "limit":
                self.warned[today][app] = "limit"
                out.append(("limit_hit", {"app": app, "minutes": minutes}))
            elif used_min >= minutes * WARN_AT and said is None:
                self.warned[today][app] = "warn"
                out.append(("limit_warn", {"app": app, "minutes": minutes, "left": max(1, round(minutes - used_min))}))
            if used_min >= minutes and self.enforce:
                out.append(("minimize", {"app": app}))
        if quiet or now < self.snoozed_until:
            return out                                    # focus / gaming / snoozed: no break nudges
        if self.stretch and self.active_s >= STRETCH_S:
            out.append(("stretch", {"minutes": round(self.active_s / 60)}))
            self.active_s = 0.0
            self.since_eyes = 0.0
        elif self.eyes and self.since_eyes >= EYES_S:
            out.append(("eyes", {}))
            self.since_eyes = 0.0
        if self.winddown and usual_end is not None and self.wound_down != today:
            late = minutes_after_midnight_ish(now)
            if late >= usual_end + 45 and late >= 22 * 60 + 30:
                self.wound_down = today
                out.append(("winddown", {"now": datetime.fromtimestamp(now).strftime("%H:%M").lstrip("0"),
                                         "usual": fmt_clock(usual_end)}))
        return out

    def snooze(self, minutes: int):
        self.snoozed_until = self.clock() + minutes * 60

    def usage_today(self) -> dict[str, float]:
        return {a: s / 60 for a, s in self.used.get(day_key(self.clock()), {}).items()}


def minutes_after_midnight_ish(ts: float) -> float:
    """Minutes since midnight, with the early hours counted as the previous evening (1:30 am -> 25:30)."""
    d = datetime.fromtimestamp(ts)
    m = d.hour * 60 + d.minute
    return m + 24 * 60 if d.hour < DAY_START_HOUR else m


def fmt_clock(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return datetime(2000, 1, 1, m // 60, m % 60).strftime("%I:%M %p").lstrip("0").replace(":00", "").lower()


def usual_end(event_times: list[float], now: float | None = None, min_days: int = 5) -> float | None:
    """When you usually stop: the median of each day's last activity over the past two weeks (not tonight)."""
    now = time.time() if now is None else now
    today = day_key(now)
    last: dict[str, float] = {}
    for ts in event_times:
        if now - ts > 15 * 86400:
            continue
        k = day_key(ts)
        if k == today:
            continue
        last[k] = max(last.get(k, 0), minutes_after_midnight_ish(ts))
    if len(last) < min_days:
        return None                                       # not enough history to know your hours yet
    return median(last.values())


def parse_app_minutes(text: str) -> int | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|min|mins|minutes?)\b", text.lower())
    if m:
        n = float(m.group(1))
        return round(n * 60 if m.group(2).startswith("h") else n)
    if re.search(r"\b(?:an|one) hour\b", text.lower()):
        return 60
    if re.search(r"\bhalf an hour\b", text.lower()):
        return 30
    return None
