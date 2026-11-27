"""Focus & gaming mode (phase 32).

Focus:   "focus for 45 minutes" / "start a pomodoro" / "stop focus" / "how long is left"
  * a countdown on the HUD; pomodoro = 25 min work, 5 min break, every 4th break 15 min, until you stop
  * do not disturb: Jarvis's own briefings, model updates and other notices wait until the end (or a break)
    and arrive as one short summary; reminders still come through, unless you asked for "no interruptions at all"
  * optional gentle blocking: when a blocked app or site (by window title) comes to the front, Jarvis minimizes
    it and tells you when it's allowed again. No admin rights, no system changes; it stops when focus ends.
Gaming:  turns on by itself when a full-screen game starts (or "gaming mode on"): quiet, HUD hidden, training
         paused (phase 41); when the game ends, a summary of what came in meanwhile.
Focused minutes are logged, so the evening briefing can say how long you focused today.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

WORK, SHORT, LONG, EVERY = 25, 5, 15, 4
CURRENT: "Engine | None" = None          # the running Jarvis's engine (undo reaches it through here)
DEFAULT_SITES = ["youtube", "instagram", "netflix", "reddit", "facebook", "twitter", "x.com", "prime video",
                 "hotstar", "twitch"]
DEFAULT_APPS = ["steam.exe", "epicgameslauncher.exe", "discord.exe"]
NOTICE_KINDS = ("briefing", "toast", "alert", "message")       # held back while quiet; "reminder" only if total


@dataclass
class Session:
    mode: str                     # single | pomodoro
    work_min: int
    block: bool
    total: bool
    started: float
    phase: str = "work"           # work | break
    phase_end: float = 0.0
    cycle: int = 1
    focused_s: float = 0.0
    phase_start: float = 0.0


@dataclass
class Gaming:
    since: float
    auto: bool


@dataclass
class Engine:
    clock: Callable[[], float] = time.time
    log: Callable[[str, dict], None] = lambda kind, data: None      # store.log_event
    session: Session | None = None
    gaming: Gaming | None = None
    held: list[tuple[str, str]] = field(default_factory=list)
    sites: list[str] = field(default_factory=lambda: list(DEFAULT_SITES))
    apps: list[str] = field(default_factory=lambda: list(DEFAULT_APPS))

    # -------------------------------------------------------------- focus
    def start(self, minutes: int | None = None, mode: str = "single", block: bool = False,
              total: bool = False) -> Session:
        now = self.clock()
        if self.session:
            self.stop("restarted")
        work = int(minutes or WORK)
        work = max(1, min(work, 600))
        self.session = Session(mode, work, block, total, now, phase_end=now + work * 60, phase_start=now)
        return self.session

    def stop(self, why: str = "stopped") -> dict | None:
        s = self.session
        if s is None:
            return None
        now = self.clock()
        if s.phase == "work":
            s.focused_s += max(0.0, now - s.phase_start)
        self.session = None
        summary = {"minutes": round(s.focused_s / 60), "cycles": s.cycle if s.mode == "pomodoro" else 1,
                   "mode": s.mode, "why": why, "held": self.release()}
        if s.focused_s >= 60:
            self.log("focus_session", {"minutes": round(s.focused_s / 60, 1), "mode": s.mode})   # not "focus":
            # that event name already means "which app had the window focus" (briefing.py)
        return summary

    def tick(self) -> list[tuple[str, dict]]:
        """Advance the clock: [("break", {...}), ("work", {...}), ("done", summary)]."""
        s = self.session
        if s is None or self.clock() < s.phase_end:
            return []
        now = self.clock()
        if s.phase == "work":
            s.focused_s += s.phase_end - s.phase_start
            if s.mode == "single":
                s.phase = "done"                            # counted above: stop() mustn't count it again
                summary = self.stop("done")
                return [("done", summary)]
            long = s.cycle % EVERY == 0
            mins = LONG if long else SHORT
            s.phase, s.phase_start, s.phase_end = "break", now, now + mins * 60
            return [("break", {"minutes": mins, "cycle": s.cycle, "long": long, "held": self.release()})]
        s.cycle += 1
        s.phase, s.phase_start, s.phase_end = "work", now, now + s.work_min * 60
        return [("work", {"minutes": s.work_min, "cycle": s.cycle})]

    def status(self) -> dict | None:
        s = self.session
        if s is None:
            return None
        left = max(0, int(s.phase_end - self.clock()))
        focused = s.focused_s + (self.clock() - s.phase_start if s.phase == "work" else 0)
        return {"phase": s.phase, "left_s": left, "mode": s.mode, "cycle": s.cycle, "block": s.block,
                "focused_min": int(focused // 60), "until": s.phase_end}

    def countdown(self) -> str | None:
        st = self.status()
        if st is None:
            return "GAMING" if self.gaming else None
        m, sec = divmod(st["left_s"], 60)
        return f"{'FOCUS' if st['phase'] == 'work' else 'BREAK'} {m:02d}:{sec:02d}"

    # -------------------------------------------------------------- do not disturb
    def quiet(self, kind: str) -> bool:
        if self.gaming is not None:                    # games: notices wait; reminders you set still come through
            return kind in NOTICE_KINDS
        s = self.session
        if s is None or s.phase != "work":
            return False
        return kind in NOTICE_KINDS or (kind == "reminder" and s.total)

    def hold(self, kind: str, text: str) -> bool:
        """True if the notice was held back for later (don't show it now)."""
        if self.quiet(kind):
            self.held.append((kind, text))
            return True
        return False

    def release(self) -> list[tuple[str, str]]:
        out, self.held = self.held, []
        return out

    # -------------------------------------------------------------- gentle blocking
    def blocked(self, title: str, exe: str) -> str | None:
        """The name of the blocked app/site in front, if focus is blocking right now."""
        s = self.session
        if s is None or not s.block or s.phase != "work":
            return None
        t, e = (title or "").lower(), (exe or "").lower().rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        for a in self.apps:
            if e == a.lower():
                return a[:-4].title() if a.lower().endswith(".exe") else a
        for site in self.sites:
            s_ = site.lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(s_)}(?![a-z0-9])", t):
                return site.title() if "." not in site else site
        return None

    # -------------------------------------------------------------- gaming
    def gaming_on(self, auto: bool = False) -> bool:
        if self.gaming is not None:
            return False
        self.gaming = Gaming(self.clock(), auto)
        return True

    def gaming_off(self) -> dict | None:
        g = self.gaming
        if g is None:
            return None
        self.gaming = None
        return {"minutes": round((self.clock() - g.since) / 60), "auto": g.auto, "held": self.release()}


def held_summary(held: list[tuple[str, str]]) -> str:
    """'While you were focused: 1 briefing, 2 notices.' plus the first lines."""
    if not held:
        return ""
    names = {"briefing": "briefing", "toast": "notice", "alert": "alert", "message": "message", "reminder": "reminder"}
    counts: dict[str, int] = {}
    for k, _ in held:
        counts[k] = counts.get(k, 0) + 1
    parts = [f"{n} {names.get(k, k)}{'s' * (n != 1)}" for k, n in counts.items()]
    first = "; ".join(t.split("\n")[0][:90] for _, t in held[:3])
    return f"{', '.join(parts)} came in: {first}" + ("…" if len(held) > 3 else "")


def fmt_left(seconds: int) -> str:
    m = max(0, round(seconds / 60))
    if m < 60:
        return f"{m} minute{'s' * (m != 1)}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' * (h != 1)}" + (f" {m} minute{'s' * (m != 1)}" if m else "")
