"""Deadlines in plain words (phase 33): "friday", "tomorrow 5 pm", "5th december", "next week", "in 3 days".

The command model copies the date phrase exactly as you said it (due="friday"); this turns it into a time.
parse_due(text, now) -> (timestamp, has_time) or None. A date without a time is due at the end of that day.
Indian style: "5/12" is 5 December. "kal" means tomorrow here (a deadline is never in the past).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

DAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3,
        "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
        "somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "shukravar": 4, "shanivar": 5, "ravivar": 6}
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
PARTS = {"morning": 9, "noon": 12, "afternoon": 15, "evening": 18, "tonight": 20, "night": 21, "subah": 9,
         "dopahar": 14, "shaam": 18, "raat": 21}
END_OF_DAY = (23, 59)


def _time_of(text: str) -> tuple[int, int] | None:
    t = text.lower()
    m = re.search(r"\b(\d{1,2})(?::|\.)(\d{2})\s*(am|pm)?\b", t) or re.search(r"\b(\d{1,2})\s*(am|pm)\b", t) \
        or re.search(r"\b(\d{1,2})\s*baje\b", t)
    if m:
        g = m.groups()
        h = int(g[0])
        mi = int(g[1]) if len(g) > 2 and g[1] and g[1].isdigit() else 0
        ap = g[-1] if g[-1] in ("am", "pm") else None
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if ap is None and "baje" in t and h < 8:
            h += 12                                          # "5 baje" in a to-do is 5 pm
        if 0 <= h < 24 and 0 <= mi < 60:
            return h, mi
    for word, h in PARTS.items():
        if re.search(rf"\b{word}\b", t):
            return h, 0
    return None


def parse_due(text: str | None, now: float | datetime | None = None) -> tuple[float, bool] | None:
    if not text or not str(text).strip():
        return None
    base = now if isinstance(now, datetime) else datetime.fromtimestamp(now) if now else datetime.now()
    t = " " + re.sub(r"[,]", " ", str(text).lower()).strip() + " "
    t = re.sub(r"\b(?:by|before|on|due|till|until|tak|ko|this coming)\b", " ", t)
    day = None
    m = re.search(r"\bin (\d+|a|an|one|two|three) (day|days|week|weeks|month|months)\b", t)
    if m:
        n = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))
        unit = m.group(2)
        day = base + timedelta(days=n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1))
    elif re.search(r"\b(?:day after tomorrow|parso|parson)\b", t):
        day = base + timedelta(days=2)
    elif re.search(r"\b(?:tomorrow|tmrw|tmr|kal)\b", t):
        day = base + timedelta(days=1)
    elif re.search(r"\b(?:today|tonight|aaj|abhi)\b", t):
        day = base
    elif re.search(r"\bend of (?:the )?month\b|\bmonth end\b", t):
        nxt = (base.replace(day=28) + timedelta(days=4)).replace(day=1)
        day = nxt - timedelta(days=1)
    elif re.search(r"\bend of (?:the )?week\b|\bweekend\b", t):
        day = base + timedelta(days=(5 - base.weekday()) % 7 or (7 if base.weekday() == 5 else 0))
    elif re.search(r"\bnext week\b", t):
        day = base + timedelta(days=7 - base.weekday())                     # next Monday
    else:
        for name, wd in sorted(DAYS.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{name}\b", t):
                if re.search(rf"\bnext {name}\b", t):        # that day in next week (Mon-Sun)
                    day = base + timedelta(days=7 - base.weekday() + wd)
                else:
                    day = base + timedelta(days=(wd - base.weekday()) % 7)
                break
    if day is None:
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\b", t) or \
            re.search(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
        if m:
            a, b = m.groups()
            d, mon = (a, b) if a.isdigit() else (b, a)
            if mon in MONTHS:
                day = _date_ahead(base, int(d), MONTHS[mon])
        if day is None:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", t)           # "the 1st": this month, or next
            if m and not re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s*(?:am|pm|baje)\b", t):
                d = int(m.group(1))
                for k in range(3):
                    mon = (base.month - 1 + k) % 12 + 1
                    y = base.year + (base.month - 1 + k) // 12
                    try:
                        cand = base.replace(year=y, month=mon, day=d)
                    except ValueError:
                        continue
                    if cand.date() >= base.date():
                        day = cand
                        break
        if day is None:
            m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
            if m:
                d, mon, y = int(m.group(1)), int(m.group(2)), m.group(3)
                if 1 <= mon <= 12:
                    day = _date_ahead(base, d, mon, int(y) + (2000 if y and len(y) == 2 else 0) if y else None)
    tm = _time_of(t)
    if day is None:
        if tm is None:
            return None
        day = base                                               # "at 5 pm": today, or tomorrow if that's past
        if (tm[0], tm[1]) <= (base.hour, base.minute):
            day = base + timedelta(days=1)
    if day is None:
        return None
    h, mi = tm if tm else END_OF_DAY
    try:
        due = day.replace(hour=h, minute=mi, second=0, microsecond=0)
    except ValueError:
        return None
    return due.timestamp(), tm is not None


def _date_ahead(base: datetime, d: int, mon: int, year: int | None = None) -> datetime | None:
    try:
        if year:
            return base.replace(year=year, month=mon, day=d)
        cand = base.replace(month=mon, day=d)
        if cand.date() < base.date():
            cand = cand.replace(year=base.year + 1)
        return cand
    except ValueError:
        return None


def nice(ts: float, has_time: bool, now: float | None = None) -> str:
    """'today', 'tomorrow 5 pm', 'Friday', 'Fri 5 Dec'."""
    n = datetime.fromtimestamp(now) if now else datetime.now()
    d = datetime.fromtimestamp(ts)
    delta = (d.date() - n.date()).days
    if delta == 0:
        day = "today"
    elif delta == 1:
        day = "tomorrow"
    elif delta == -1:
        day = "yesterday"
    elif 1 < delta < 7:
        day = d.strftime("%A")
    else:
        day = d.strftime("%a %d %b").replace(" 0", " ")
    if has_time:
        hm = d.strftime("%I:%M %p").lstrip("0").replace(":00", "").lower()
        return f"{day} {hm}"
    return day


def next_occurrence(ts: float, every: str) -> float:
    """The next deadline of a recurring task after this one."""
    d = datetime.fromtimestamp(ts)
    if every == "daily":
        d += timedelta(days=1)
    elif every == "weekdays":
        d += timedelta(days=3 if d.weekday() == 4 else 2 if d.weekday() == 5 else 1)
    elif every == "weekly":
        d += timedelta(days=7)
    elif every == "monthly":
        m = d.month % 12 + 1
        y = d.year + (d.month == 12)
        for day in (d.day, 30, 29, 28):
            try:
                d = d.replace(year=y, month=m, day=day)
                break
            except ValueError:
                continue
    return d.timestamp()
