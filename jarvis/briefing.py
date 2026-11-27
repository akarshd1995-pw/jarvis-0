"""Briefings (phase 24): morning and evening summaries, in character.

    Good morning, sir. It's Tuesday, 8:42 am. You have 2 reminders today: call mom at 7:00 pm and pay rent
    at 9:00 pm. All systems are healthy: CPU 6%, GPU 41°C, 212 GB free. Last night's training finished:
    99.8% on familiar phrasing, 99.7% on new phrasing. Downloads has 143 files (6.2 GB); 87 are over a month old.

Facts are gathered first (gather), then written in the persona's tone (render), so the same facts can be
shown on the HUD now and read aloud once voice lands. Nothing leaves the machine.

Triggers: "brief me", "good morning", "how was my day", "evening summary", and automatically: the morning
briefing the first time you're at the PC after `briefing_morning` (default 07:00), the evening one at
`briefing_evening` (default 21:00) if you're active. Each at most once a day.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# exe name (lowercase, no .exe) -> how to say it
APP_NAMES = {
    "code": "VS Code", "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "brave": "Brave",
    "spotify": "Spotify", "explorer": "File Explorer", "windowsterminal": "Terminal", "wt": "Terminal",
    "cmd": "Command Prompt", "powershell": "PowerShell", "pwsh": "PowerShell", "notepad": "Notepad",
    "winword": "Word", "excel": "Excel", "powerpnt": "PowerPoint", "outlook": "Outlook", "teams": "Teams",
    "ms-teams": "Teams", "slack": "Slack", "discord": "Discord", "whatsapp": "WhatsApp", "telegram": "Telegram",
    "vlc": "VLC", "obs64": "OBS", "pycharm64": "PyCharm", "idea64": "IntelliJ", "cursor": "Cursor",
    "unity": "Unity", "unity hub": "Unity Hub", "blender": "Blender", "photoshop": "Photoshop", "figma": "Figma",
    "steam": "Steam", "zoom": "Zoom", "notion": "Notion", "obsidian": "Obsidian", "postman": "Postman",
}
# Windows plumbing that isn't "work"
_IGNORE = {"lockapp", "searchhost", "searchapp", "shellexperiencehost", "startmenuexperiencehost", "applicationframehost",
           "textinputhost", "systemsettings", "taskmgr", "python", "pythonw", "dwm", "sihost", "rundll32", ""}
LAST_CAP_S = 30 * 60        # the stretch still open when the briefing runs
IDLE_CAP_S = 4 * 3600        # one uninterrupted stretch longer than this is treated as away (lock screen is ignored anyway)


def app_label(exe: str) -> str:
    k = re.sub(r"\.exe$", "", str(exe).strip().lower())
    return APP_NAMES.get(k, k[:1].upper() + k[1:])


def _is_noise(exe: str) -> bool:
    return re.sub(r"\.exe$", "", str(exe).strip().lower()) in _IGNORE


@dataclass
class Facts:
    kind: str                                  # morning | evening
    now: datetime
    reminders: list[tuple[datetime, str]] = field(default_factory=list)
    reminders_label: str = "today"             # today | tomorrow
    health: dict = field(default_factory=dict)  # cpu, ram, gpu_temp, disk_free_gb, battery, charging
    issues: list[str] = field(default_factory=list)
    training: list[dict] = field(default_factory=list)   # [{name, state, best, pct, eta, iid, ood, ago_h}]
    clutter: dict = field(default_factory=dict)  # files, gb, old, old_gb, new_today
    apps: list[tuple[str, int]] = field(default_factory=list)   # [(label, minutes)]
    improve: str = ""                          # what the self-improving loop did overnight
    learning: list[str] = field(default_factory=list)   # phase 39: weekly review nudge, drift alert
    notes: list[tuple[str, str]] = field(default_factory=list)   # phase 30: today's (kind, text), oldest first
    focus_min: float = 0.0                                        # phase 32: minutes of focus sessions today
    tasks: list[str] = field(default_factory=list)                # phase 33: due today / overdue / still open
    active_min: int = 0
    commands: int = 0
    alerts: int = 0


# ------------------------------------------------------------------ gathering (pure where possible, unit-tested)

def focus_minutes(events: list[tuple[float, str, dict]], until: float) -> dict[str, float]:
    """Seconds per app from focus-change events: each stretch runs until the next change (capped for idle)."""
    out: dict[str, float] = {}
    ev = sorted((ts, p.get("app", "")) for ts, k, p in events if k == "focus")
    ends = [e[0] for e in ev[1:]] + [until]
    for i, ((ts, app), nxt) in enumerate(zip(ev, ends)):
        if _is_noise(app):
            continue
        cap = IDLE_CAP_S if i < len(ev) - 1 else LAST_CAP_S      # nothing confirms you're still there
        out[app_label(app)] = out.get(app_label(app), 0.0) + min(max(0.0, nxt - ts), cap)
    return out


def folder_clutter(folder: Path, now: float, old_days: int = 30, today_start: float | None = None) -> dict:
    files = old = new = 0
    size = old_size = 0
    try:
        for p in folder.iterdir():
            try:
                if not p.is_file():
                    continue
                st = p.stat()
            except OSError:
                continue
            files += 1
            size += st.st_size
            if now - st.st_mtime > old_days * 86400:
                old += 1
                old_size += st.st_size
            if today_start is not None and st.st_mtime >= today_start:
                new += 1
    except OSError:
        return {}
    return {"files": files, "gb": size / 2**30, "old": old, "old_gb": old_size / 2**30, "new_today": new}


def eval_summary(path: Path) -> dict:
    """Exact-match scores from checkpoints/eval_report.txt: {'iid': 0.998, 'ood': 0.9975}."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out = {}
    for split, score in re.findall(r"split=test_(iid|ood).*?exact match\s*:\s*([\d.]+)", text, re.S):
        out[split] = float(score)
    return out


def training_facts(logs: dict[str, Path], eval_report: Path | None, now: float, recent_h: float = 18) -> list[dict]:
    from .hud.metrics import job_status
    out = []
    for name, path in logs.items():
        st = job_status(path)
        if not st:
            continue
        ago_h = st.get("age_s", 0) / 3600
        if st["state"] != "running" and ago_h > recent_h:
            continue                                     # old news
        if st["state"] == "running" and st.get("age_s", 0) > 600:
            st = {**st, "state": "stalled"}
        item = {"name": name, "state": st["state"], "ago_h": ago_h, "pct": st.get("pct"), "eta": st.get("eta_min"),
                "best": st.get("best")}
        if st["state"] == "finished" and "command" in name.lower() and eval_report is not None:
            item.update(eval_summary(eval_report))
        out.append(item)
    return out


def health_facts(snap) -> tuple[dict, list[str]]:
    h = {k: getattr(snap, k, None) for k in ("cpu", "ram", "gpu_temp", "disk_free_gb", "battery", "charging", "gpu")}
    issues = []
    if h["gpu_temp"] is not None and h["gpu_temp"] >= 83:
        issues.append(f"the GPU is running hot at {h['gpu_temp']:.0f}°C")
    if h["disk_free_gb"] is not None and h["disk_free_gb"] < 15:
        issues.append(f"only {h['disk_free_gb']:.0f} GB free on C:")
    if h["ram"] is not None and h["ram"] >= 90:
        issues.append(f"memory is at {h['ram']:.0f}%")
    if h["battery"] is not None and h["battery"] < 25 and not h["charging"]:
        issues.append(f"battery is at {h['battery']:.0f}% and not charging")
    return h, issues


def gather(kind: str, store, snapshot, logs: dict[str, Path], eval_report: Path | None, downloads: Path | None,
           now: float | None = None) -> Facts:
    now = time.time() if now is None else now
    dt = datetime.fromtimestamp(now)
    f = Facts(kind=kind, now=dt)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = (midnight + timedelta(hours=4)).timestamp()
    if day_start > now:                                  # 1 am still counts as "yesterday's" day
        day_start -= 86400
    # reminders: rest of today (morning) / tomorrow (evening)
    if kind == "morning":
        lo, hi = now, (midnight + timedelta(days=1)).timestamp()
    else:
        lo, hi = (midnight + timedelta(days=1)).timestamp(), (midnight + timedelta(days=2)).timestamp()
        f.reminders_label = "tomorrow"
    for due, text in store.upcoming_reminders(lo - 1, limit=20):
        if lo <= due < hi:
            f.reminders.append((datetime.fromtimestamp(due), text))
    f.reminders = f.reminders[:5]
    if snapshot is not None:
        f.health, f.issues = health_facts(snapshot)
    f.training = training_facts({k: v for k, v in logs.items() if "improve" not in k.lower()}, eval_report, now)
    try:
        rp = Path(eval_report).parent / "improve_report.json" if eval_report else None
        if rp and rp.exists() and now - rp.stat().st_mtime < 18 * 3600:
            f.improve = improve_line(json.loads(rp.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    if downloads is not None:
        f.clutter = folder_clutter(downloads, now, today_start=day_start)
    if kind == "evening" and hasattr(store, "notes"):
        f.notes = [(k, t) for _, _, k, t in reversed(store.notes(since=day_start))]
    if hasattr(store, "tasks"):
        from .actions.tasks import briefing_lines
        f.tasks = briefing_lines(store, kind, now)
    events = store.events_since(day_start)
    if kind == "evening":
        f.focus_min = sum(float(p.get("minutes", 0)) for _, k, p in events if k == "focus_session")
    secs = focus_minutes(events, now)
    f.active_min = int(sum(secs.values()) // 60)
    f.apps = [(a, int(s // 60)) for a, s in sorted(secs.items(), key=lambda x: -x[1]) if s >= 120][:4]
    f.commands = sum(1 for _, k, _ in events if k == "command_run")
    f.alerts = sum(1 for _, k, _ in events if k == "alert")
    return f


def notes_lines(notes: list[tuple[str, str]]) -> list[str]:
    """Evening: what you noted today (a few of them) and whether you wrote in your journal."""
    out = []
    plain = [t for k, t in notes if k == "note"]
    journal = [t for k, t in notes if k == "journal"]
    if plain:
        short = ["\u201c" + (t if len(t) <= 60 else t[:59].rstrip() + "\u2026") + "\u201d" for t in plain[:3]]
        more = f", and {len(plain) - 3} more" if len(plain) > 3 else ""
        out.append(f"You noted {len(plain)} thing{'s' * (len(plain) != 1)} today: {'; '.join(short)}{more}.")
    if journal:
        out.append(f"You wrote {len(journal)} journal entr{'ies' if len(journal) != 1 else 'y'} today.")
    return out


def improve_line(r: dict) -> str:
    """'Overnight I fine-tuned on 7 of your corrections: 99.84% vs 99.80% before, so I switched to v004.'"""
    sc = r.get("scores") or {}
    live, cand = sc.get("live", {}), sc.get("candidate", {})
    why = []
    if r.get("corrections"):
        why.append(f"{r['corrections']} correction{'s' * (r['corrections'] != 1)} of yours")
    if any("skills" in x for x in r.get("reasons", [])):
        why.append("the newest skills")
    on = " and ".join(why) or "fresh examples"
    nums = ""
    if "unseen" in live and "unseen" in cand:
        nums = f" ({cand['unseen']['exact']:.2%} on new phrasing, was {live['unseen']['exact']:.2%})"
    if r.get("promoted"):
        from .usage import report_card_line
        card = report_card_line(r.get("real"))
        return (f"I fine-tuned myself on {on}{nums} and switched to the better model, {r.get('version', 'a new version')}."
                + (f" {card}" if card else ""))
    return f"I tried fine-tuning on {on}, but {r.get('reason', 'it was not better')}, so I kept the current model."


# ------------------------------------------------------------------ writing it in character

def _clock(d: datetime) -> str:
    return d.strftime("%I:%M %p").lstrip("0").lower()


def _dur(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m} min" if h and m else f"{h} h" if h else f"{m} min"


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _pct(x: float) -> str:
    return f"{100 * x:.2f}".rstrip("0").rstrip(".") + "%"


def _size(gb: float) -> str:
    return f"{gb:.1f} GB" if gb >= 1 else f"{max(1, round(gb * 1024))} MB"


def render(f: Facts, name: str = "Jarvis", address: str = "sir", tone: str = "butler") -> str:
    butler, minimal = tone == "butler", tone == "minimal"
    addr = f", {address}" if address and butler else ""
    part = "morning" if f.now.hour < 12 else "afternoon" if f.now.hour < 17 else "evening"
    lines: list[str] = []
    if minimal:
        lines.append(f"{f.now.strftime('%a')} {_clock(f.now)}.")
    elif f.kind == "morning":
        lines.append(f"Good {part}{addr}. It's {f.now.strftime('%A')}, {_clock(f.now)}.")
    else:
        lines.append(f"Good {part}{addr}. Here's your day.")

    # what you worked on (evening)
    if f.kind == "evening":
        if f.apps:
            spent = _list([f"{a} ({_dur(m)})" for a, m in f.apps])
            lines.append(f"Active {_dur(f.active_min)}: {spent}." if minimal else
                         f"You were at the PC for about {_dur(f.active_min)}, mostly in {spent}.")
        bits = []
        if f.commands:
            bits.append(f"ran {f.commands} command{'s' * (f.commands != 1)} for you")
        if f.alerts:
            bits.append(f"raised {f.alerts} alert{'s' * (f.alerts != 1)}")
        if bits and not minimal:
            lines.append("I " + _list(bits) + ".")

    # reminders
    if f.reminders:
        rs = _list([f"{t} at {_clock(d)}" for d, t in f.reminders])
        n = len(f.reminders)
        lines.append(f"{f.reminders_label.capitalize()}: {rs}." if minimal else
                     f"You have {n} reminder{'s' * (n != 1)} {f.reminders_label}: {rs}.")
    elif not minimal:
        lines.append(f"Nothing on your reminders {f.reminders_label}.")

    # system health
    h = f.health
    if f.issues:
        lines.append(("Heads up: " if butler else "Note: ") + _list(f.issues) + ".")
    elif h:
        vitals = []
        if h.get("cpu") is not None:
            vitals.append(f"CPU {h['cpu']:.0f}%")
        if h.get("gpu_temp") is not None:
            vitals.append(f"GPU {h['gpu_temp']:.0f}°C")
        if h.get("disk_free_gb") is not None:
            vitals.append(f"{h['disk_free_gb']:.0f} GB free")
        if vitals:
            lines.append(("All systems healthy: " if not minimal else "Systems OK: ") + ", ".join(vitals) + ".")

    # training
    for t in f.training:
        who = "the reply model" if "reply" in t["name"].lower() else "training"
        if t["state"] == "running":
            prog = f"{t['pct']:.0f}% through" if t.get("pct") is not None else "running"
            eta = f", about {_dur(int(t['eta']))} left" if t.get("eta") else ""
            lines.append(f"{who.capitalize()} is {prog}{eta}.")
        elif t["state"] == "finished":
            when = "Last night's" if f.kind == "morning" and t["ago_h"] > 3 else "The latest"
            if "iid" in t and "ood" in t:
                lines.append(f"{when} {who} run finished: {_pct(t['iid'])} on familiar phrasing, "
                             f"{_pct(t['ood'])} on new phrasing.")
            elif t.get("best") is not None:
                lines.append(f"{when} {who} run finished with a best score of {_pct(t['best'])}.")
        elif t["state"] == "crashed":
            lines.append(f"{who.capitalize()} crashed{addr}. The log has the error.")
        elif t["state"] == "stalled":
            lines.append(f"{who.capitalize()} hasn't written to its log in a while; it may be stuck or paused.")

    if f.improve:
        lines.append(f.improve)
    lines.extend(f.learning)
    lines.extend(f.tasks)
    lines.extend(notes_lines(f.notes))
    if f.focus_min >= 5:
        from .focus import fmt_left
        lines.append(f"You focused for {fmt_left(int(f.focus_min * 60))} today.")

    # Downloads clutter
    c = f.clutter
    if c:
        if f.kind == "evening" and c.get("new_today"):
            lines.append(f"{c['new_today']} new file{'s' * (c['new_today'] != 1)} landed in Downloads today.")
        if c.get("old", 0) >= 20:
            tail = "" if minimal else " Say \"show files older than 30 days in downloads\" to review them."
            lines.append(f"Downloads has {c['files']} files ({_size(c['gb'])}); {c['old']} are over a month old."
                         + tail)

    if f.kind == "evening" and butler:
        lines.append("Anything else before you call it a day?")
    return " ".join(lines)


def short(text: str, limit: int = 140) -> str:
    """The first informative sentence or two (greetings skipped), for the HUD line and the notification."""
    out = ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text)
                 if not re.match(r"^(Good (morning|afternoon|evening)|Here's your day|It's \w+day|\w{3} \d)", s)]
    for s in sentences:
        if len(out) + len(s) > limit and out:
            break
        out = (out + " " + s).strip()
    return out


# ------------------------------------------------------------------ automatic delivery, once a day each

class Scheduler:
    def __init__(self, state_path: Path, morning: str = "07:00", evening: str = "21:00", enabled: bool = True):
        self.path, self.enabled = state_path, enabled
        self.morning, self.evening = self._hm(morning), self._hm(evening)
        try:
            self.state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            self.state = {}

    @staticmethod
    def _hm(s: str) -> tuple[int, int]:
        try:
            h, m = (int(x) for x in str(s).split(":"))
            return h, m
        except ValueError:
            return 7, 0

    def mark(self, kind: str, now: datetime) -> None:
        self.state[kind] = now.strftime("%Y-%m-%d")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state))
        except OSError:
            pass

    def given_today(self, kind: str, now: datetime) -> bool:
        return self.state.get(kind) == now.strftime("%Y-%m-%d")

    def due(self, now: datetime, user_active: bool) -> str | None:
        """'morning' the first time you're active after the morning time (until noon); 'evening' after the
        evening time if you're active (until 2 am)."""
        if not self.enabled or not user_active:
            return None
        t = (now.hour, now.minute)
        if self.morning <= t < (12, 0) and not self.given_today("morning", now):
            return "morning"
        if t >= self.evening and not self.given_today("evening", now):
            return "evening"
        return None
