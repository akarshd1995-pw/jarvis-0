"""Real-usage learning (phase 39): learn from how you actually talk, not only from generated data.

Every command you run is kept here, on this PC only, encrypted to your Windows account like the activity log:
the text, what Jarvis understood, how sure the model was and its runner-up reading. From that:

  * doubt score  - how unsure Jarvis was (low confidence, two close readings, rules had to step in, you undid it)
  * review       - each week the most doubtful commands (about 10, ~2 minutes) are offered for a one-click
                   "right / wrong, I meant ..." review. Answers become training examples for the next /improve.
  * drift        - if this week's commands are noticeably more doubtful than your usual, the briefing says so
  * report card  - training/loop.py scores every new model on your real commands (real_texts) before switching

Turn it off in Settings > Privacy ("Learn from my commands"); /usage forget wipes it.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .crypto import protect, unprotect

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    text TEXT NOT NULL,          -- encrypted
    program TEXT,                -- encrypted; what Jarvis ran (NULL = not understood)
    source TEXT NOT NULL,        -- model 93% | rules (model unsure) | dsl | ...
    conf REAL,                   -- model confidence, if the model was asked
    alts TEXT,                   -- encrypted JSON [[conf, program], ...] the model's readings
    doubt REAL NOT NULL,         -- 0 sure .. 1 lost
    status TEXT NOT NULL DEFAULT 'new',   -- new | right | fixed | skipped | undone
    label TEXT                   -- encrypted; the right program once reviewed
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
"""

WEEK = 7 * 86400
KEEP_DAYS = 180
REVIEW_MIN_DOUBT = 0.35
DRIFT_MIN_COMMANDS = 15


def _pct(source: str) -> float | None:
    m = re.search(r"model (\d+)%", source or "")
    return int(m.group(1)) / 100 if m else None


def doubt_score(source: str, program: str | None, conf: float | None, alts: list) -> float:
    """How unsure Jarvis was, 0..1. Pure, unit-tested."""
    source = source or ""
    if source == "dsl":
        return 0.0                                    # you typed the exact program
    if program is None or source == "unparsed":
        return 1.0
    if source == "smalltalk":
        return 0.25
    if program == "unknown()":
        return 0.7
    d = 0.0
    if "model unsure" in source:
        d = 0.8                                       # the model wasn't confident; a rule guessed
    elif "cut the name short" in source or "name check" in source:
        d = 0.6                                       # the model's reading had to be repaired
    elif "not trained on this yet" in source:
        d = 0.15                                      # precise rule, model just hasn't learned it yet
    elif source == "rules":
        d = 0.5                                       # model said it couldn't, a rule did
    conf = conf if conf is not None else _pct(source)
    if conf is not None and source.startswith("model"):
        d = max(d, min(1.0, (1 - conf) * 1.6))        # 90% -> 0.16, 75% -> 0.4, 60% -> 0.64
    if alts and len(alts) > 1 and source.startswith("model"):
        (c1, p1), (c2, p2) = alts[0], alts[1]
        if p1 != p2 and c1 > 0 and c2 / c1 > 0.75:    # two readings nearly as likely: torn
            d = max(d, 0.5 + 0.4 * (c2 / c1 - 0.75) / 0.25)
    return round(min(1.0, d), 3)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


@dataclass
class Item:
    id: int
    ts: float
    text: str
    program: str | None
    source: str
    doubt: float
    alts: list


class UsageLog:
    def __init__(self, path: Path | str, encrypt: bool = True, enabled: bool = True,
                 corrections: Path | None = None):
        self.encrypt, self.enabled = encrypt, enabled
        self.corrections = Path(corrections) if corrections else None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.last_id: int | None = None

    # ------------------------------------------------------------ writing
    def _enc(self, s):
        return None if s is None else protect(s, self.encrypt)

    @staticmethod
    def _dec(s):
        return None if s is None else unprotect(s)

    def record(self, text: str, program: str | None, source: str, detail: dict | None = None,
               ts: float | None = None) -> int | None:
        """One command you ran. Returns its id (or None when learning from usage is off)."""
        if not self.enabled or not text.strip() or text.strip().startswith("/"):
            return None
        detail = detail or {}
        alts = [[round(float(c), 4), str(p)] for c, p in (detail.get("alts") or [])][:4]
        conf = detail.get("conf")
        d = doubt_score(source, program, conf, alts)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO usage(ts, text, program, source, conf, alts, doubt) VALUES (?,?,?,?,?,?,?)",
                (ts or time.time(), self._enc(text.strip()), self._enc(program), source,
                 None if conf is None else float(conf), self._enc(json.dumps(alts)), d))
            self._db.commit()
            self.last_id = cur.lastrowid
        return self.last_id

    def mark_undone(self, within_s: float = 120, now: float | None = None) -> bool:
        """You said "undo" right after a command: a strong sign it was misunderstood."""
        now = now or time.time()
        with self._lock:
            row = self._db.execute("SELECT id FROM usage WHERE ts >= ? AND program IS NOT NULL AND program != '' "
                                   "AND status = 'new' ORDER BY ts DESC LIMIT 1", (now - within_s,)).fetchone()
            if not row:
                return False
            self._db.execute("UPDATE usage SET status = 'undone', doubt = MAX(doubt, 0.9) WHERE id = ?", (row[0],))
            self._db.commit()
        return True

    def _label(self, item_id: int, status: str, program: str | None) -> Item | None:
        it = self.get(item_id)
        if it is None:
            return None
        with self._lock:
            self._db.execute("UPDATE usage SET status = ?, label = ? WHERE id = ?",
                             (status, self._enc(program), item_id))
            self._db.commit()
        if program and self.corrections is not None:          # becomes training data for the next /improve
            self.corrections.parent.mkdir(parents=True, exist_ok=True)
            with open(self.corrections, "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": it.text, "program": program, "from": "review"}, ensure_ascii=False) + "\n")
        return it

    def confirm(self, item_id: int) -> Item | None:
        it = self.get(item_id)
        if it is None or not it.program:
            return None
        return self._label(item_id, "right", it.program)

    def fix(self, item_id: int, program: str) -> Item | None:
        from .dsl.grammar import canonical
        return self._label(item_id, "fixed", canonical(program))

    def skip(self, item_id: int) -> None:
        self._label(item_id, "skipped", None)

    def mark_last(self, status: str, program: str | None = None) -> None:
        """/good and /fix settle the last command's review; /wrong (status 'new') puts it first in the queue."""
        if self.last_id is None:
            return
        with self._lock:
            if status == "new":
                self._db.execute("UPDATE usage SET doubt = 1.0 WHERE id = ?", (self.last_id,))
            else:
                self._db.execute("UPDATE usage SET status = ?, label = ? WHERE id = ?",
                                 (status, self._enc(program), self.last_id))
            self._db.commit()

    def forget(self) -> int:
        with self._lock:
            n = self._db.execute("DELETE FROM usage").rowcount
            self._db.commit()
        return n

    def prune(self, now: float | None = None) -> int:
        now = now or time.time()
        with self._lock:
            n = self._db.execute("DELETE FROM usage WHERE ts < ?", (now - KEEP_DAYS * 86400,)).rowcount
            self._db.commit()
        return n

    # ------------------------------------------------------------ reading
    def _item(self, row) -> Item:
        i, ts, t, p, src, d, a = row
        try:
            alts = json.loads(self._dec(a) or "[]")
        except ValueError:
            alts = []
        return Item(i, ts, self._dec(t), self._dec(p), src, d, alts)

    def get(self, item_id: int) -> Item | None:
        with self._lock:
            row = self._db.execute("SELECT id, ts, text, program, source, doubt, alts FROM usage WHERE id = ?",
                                   (item_id,)).fetchone()
        return self._item(row) if row else None

    def to_review(self, limit: int = 10, days: int = 7, now: float | None = None) -> list[Item]:
        """The commands Jarvis was least sure about lately, one per phrasing, most doubtful first."""
        now = now or time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, text, program, source, doubt, alts FROM usage WHERE ts >= ? AND doubt >= ? "
                "AND status IN ('new', 'undone') ORDER BY doubt DESC, ts DESC LIMIT 500",
                (now - days * 86400, REVIEW_MIN_DOUBT)).fetchall()
            settled = {self._dec(t) for (t,) in self._db.execute(
                "SELECT text FROM usage WHERE status IN ('right', 'fixed', 'skipped')").fetchall()}
        settled = {_norm(t) for t in settled if t}
        out, seen = [], set()
        for r in rows:
            it = self._item(r)
            k = _norm(it.text)
            if k in seen or k in settled:
                continue
            seen.add(k)
            out.append(it)
            if len(out) >= limit:
                break
        return out

    def stats(self, start: float, end: float) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT doubt, conf, status FROM usage WHERE ts >= ? AND ts < ?",
                                    (start, end)).fetchall()
        n = len(rows)
        if not n:
            return {"n": 0, "unsure": 0.0, "conf": None, "undone": 0}
        confs = [c for _, c, _ in rows if c is not None]
        return {"n": n, "unsure": sum(d >= REVIEW_MIN_DOUBT for d, _, _ in rows) / n,
                "conf": sum(confs) / len(confs) if confs else None,
                "undone": sum(s == "undone" for _, _, s in rows)}

    def drift(self, now: float | None = None) -> dict | None:
        """This week vs your usual (the 4 weeks before). Returns {week, usual} when it got clearly worse."""
        now = now or time.time()
        week = self.stats(now - WEEK, now)
        usual = self.stats(now - 5 * WEEK, now - WEEK)
        if week["n"] < DRIFT_MIN_COMMANDS or usual["n"] < DRIFT_MIN_COMMANDS:
            return None
        if week["unsure"] >= usual["unsure"] + 0.10 and week["unsure"] >= 1.5 * usual["unsure"]:
            return {"week": week, "usual": usual}
        return None

    def real_texts(self, days: int = 90, limit: int = 3000, now: float | None = None) -> list[dict]:
        """Your real commands (one per phrasing) with what Jarvis ran and any reviewed answer: the report card set."""
        now = now or time.time()
        with self._lock:
            rows = self._db.execute("SELECT text, program, label, status FROM usage WHERE ts >= ? ORDER BY ts DESC",
                                    (now - days * 86400,)).fetchall()
        out, seen = [], set()
        for t, p, lab, st in rows:
            text = self._dec(t)
            k = _norm(text or "")
            if not k or k in seen:
                continue
            seen.add(k)
            out.append({"text": text, "ran": self._dec(p), "label": self._dec(lab), "status": st})
            if len(out) >= limit:
                break
        return out


# ------------------------------------------------------------------ words for the briefing / review
def review_line(n: int) -> str:
    if n <= 0:
        return ""
    return (f"I wasn't sure about {n} of your commands this week. A two-minute review would teach me: "
            f"say \"review my commands\".")


def drift_line(d: dict | None) -> str:
    if not d:
        return ""
    return (f"I've been less sure of your commands this week ({d['week']['unsure']:.0%} unsure, usually "
            f"{d['usual']['unsure']:.0%}). You may be phrasing things in a new way; reviewing them helps me catch up.")


def report_card_line(real: dict | None) -> str:
    """'On 412 of your real commands the new model reads 409 the same way and 3 differently.'"""
    if not real or not real.get("n"):
        return ""
    s = f"On {real['n']} of your real commands it reads {real['same']} the same way"
    if real.get("changed_n"):
        s += f" and {real['changed_n']} differently"
    if real.get("labelled"):
        s += f", and gets {real['labelled_right']} of the {real['labelled']} you reviewed right"
    return s + "."


# ------------------------------------------------------------------ the review, without any window
class ReviewSession:
    """The review logic without any window (unit-tested)."""

    def __init__(self, usage: "UsageLog", items: list[Item], parse_fn: Callable[[str], str | None]):
        self.usage, self.items, self.parse_fn = usage, items, parse_fn
        self.i = 0
        self.done = {"right": 0, "fixed": 0, "skipped": 0}

    @property
    def current(self) -> Item | None:
        return self.items[self.i] if self.i < len(self.items) else None

    def choices(self, item: Item) -> list[tuple[str, str]]:
        """Other things it could have meant: the model's runner-up readings and the backup rules' reading."""
        from .nlu.explain import explain
        from .nlu.rule_parser import rule_parse
        out, seen = [], {item.program}
        cands = [p for _, p in item.alts]
        ruled = rule_parse(item.text)
        if ruled is not None:
            cands.append(str(ruled))
        for p in cands:
            if p and p not in seen and p not in ("unknown()",):
                seen.add(p)
                out.append((p, explain(p)))
        return out[:4]

    def _next(self, kind: str) -> None:
        self.done[kind] += 1
        self.i += 1

    def right(self) -> None:
        it = self.current
        if it is not None and it.program and it.program != "unknown()":
            self.usage.confirm(it.id)
            self._next("right")
        elif it is not None:                           # "it didn't understand, and that's right" = not a command
            self.usage.fix(it.id, "unknown()")
            self._next("fixed")

    def fix(self, program: str) -> None:
        it = self.current
        if it is not None:
            self.usage.fix(it.id, program)
            self._next("fixed")

    def clearer(self, text: str) -> str | None:
        """'Say it more clearly': what Jarvis understands from the clearer wording (shown before you accept)."""
        return self.parse_fn(text.strip()) if text.strip() else None

    def skip(self) -> None:
        it = self.current
        if it is not None:
            self.usage.skip(it.id)
            self._next("skipped")

    def summary(self) -> str:
        d = self.done
        taught = d["right"] + d["fixed"]
        if not taught:
            return "Nothing new to learn this time."
        return (f"Thanks. {d['right']} confirmed, {d['fixed']} corrected. I'll learn them in the next improvement run "
                f"(say \"improve yourself\", or it happens overnight).")
