"""Notes and journal (phase 30): finding them again.

"what did I note about the bank" -> the notes whose words match best (simple stemming, so 'cards' finds 'card'),
newest first among equals. Everything stays encrypted in Jarvis's database; nothing is copied to the file index.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta

STOP = {"the", "a", "an", "about", "my", "me", "i", "to", "of", "on", "for", "in", "and", "or", "is", "was", "that",
        "this", "it", "what", "did", "note", "notes", "ka", "ki", "ke", "me", "wala", "wali", "baare"}


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def score(query: str, text: str) -> float:
    q = words(query)
    if not q:
        return 1.0
    t = set(words(text))
    hits = sum(1 for w in q if w in t or any(x.startswith(w) and len(w) >= 3 for x in t))
    s = hits / len(q)
    if query.lower().strip() and query.lower().strip() in text.lower():
        s += 0.5                                                  # the exact phrase
    return s


def window(when: str | None, now: float | None = None) -> tuple[float | None, float | None]:
    """today / yesterday / week / month -> (since, until) timestamps. Days start at 4 am like the briefing."""
    if not when:
        return None, None
    now_dt = datetime.fromtimestamp(time.time() if now is None else now)
    start = now_dt.replace(hour=4, minute=0, second=0, microsecond=0)
    if start > now_dt:
        start -= timedelta(days=1)
    if when == "today":
        return start.timestamp(), None
    if when == "yesterday":
        return (start - timedelta(days=1)).timestamp(), start.timestamp()
    if when == "week":
        return (now_dt - timedelta(days=7)).timestamp(), None
    if when == "month":
        return (now_dt - timedelta(days=31)).timestamp(), None
    return None, None


def find(store, query: str | None = None, kind: str | None = None, when: str | None = None,
         limit: int = 20, now: float | None = None) -> list[tuple[int, float, str, str]]:
    since, until = window(when, now)
    rows = store.notes(kind=kind, since=since, until=until)
    if not query:
        return rows[:limit]
    scored = [(score(query, r[3]), r) for r in rows]
    best = [(s, r) for s, r in scored if s >= 0.5]
    best.sort(key=lambda x: (-x[0], -x[1][1]))
    return [r for _, r in best[:limit]]
