"""SQLite store: event log (training data for the routine model), undo journal, reminders.

Everything stays local. Event payloads (what you did, when) are encrypted at rest with Windows
DPAPI, so only your Windows account on this PC can read them.
"""
from __future__ import annotations

import json
import sqlite3

from .crypto import protect, unprotect
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,          -- focus | file_added | command | suggestion_feedback ...
    payload TEXT NOT NULL        -- JSON
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,      -- one user command = one batch
    ts REAL NOT NULL,
    op TEXT NOT NULL,            -- e.g. move
    inverse TEXT NOT NULL,       -- JSON describing how to undo
    undone INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_journal_batch ON journal(batch_id);

CREATE TABLE IF NOT EXISTS notes (          -- phase 30: notes and journal entries (text encrypted)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,          -- note | journal
    text TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_ts ON notes(ts);

CREATE TABLE IF NOT EXISTS tasks (          -- phase 33: to-dos with deadlines (text encrypted)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    text TEXT NOT NULL,
    due REAL,                    -- NULL: no deadline
    has_time INTEGER NOT NULL DEFAULT 0,
    every TEXT,                  -- daily | weekdays | weekly | monthly
    priority TEXT NOT NULL DEFAULT 'normal',
    done REAL,                   -- when it was ticked off
    deleted INTEGER NOT NULL DEFAULT 0,
    nudged REAL                  -- when Jarvis last said it was overdue
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    due REAL NOT NULL,
    text TEXT NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path | str, encrypt: bool = True):
        self.encrypt = encrypt
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(journal)")}
        if "status" not in cols:                 # phase 40: write-ahead journal (existing rows are done)
            self._db.execute("ALTER TABLE journal ADD COLUMN status TEXT NOT NULL DEFAULT 'done'")
        if "detail" not in cols:
            self._db.execute("ALTER TABLE journal ADD COLUMN detail TEXT")
        self._db.commit()

    # ------------------------------------------------------------ events
    def log_event(self, kind: str, payload: dict, ts: float | None = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO events(ts, kind, payload) VALUES (?,?,?)",
                             (ts or time.time(), kind, protect(json.dumps(payload), self.encrypt)))
            self._db.commit()

    def events_since(self, ts: float, kind: str | None = None) -> list[tuple[float, str, dict]]:
        q = "SELECT ts, kind, payload FROM events WHERE ts >= ?"
        args: list = [ts]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY ts", args).fetchall()
        return [(t, k, json.loads(unprotect(p))) for t, k, p in rows]

    # ------------------------------------------------------------ journal
    def record(self, batch_id: str, op: str, inverse: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO journal(batch_id, ts, op, inverse) VALUES (?,?,?,?)",
                             (batch_id, time.time(), op, json.dumps(inverse)))
            self._db.commit()

    def begin(self, batch_id: str, op: str, inverse: dict, detail: dict) -> int:
        """Write-ahead: record a step as pending before touching the file."""
        with self._lock:
            cur = self._db.execute("INSERT INTO journal(batch_id, ts, op, inverse, status, detail) "
                                   "VALUES (?,?,?,?,'pending',?)",
                                   (batch_id, time.time(), op, json.dumps(inverse), json.dumps(detail)))
            self._db.commit()
            return cur.lastrowid

    def finish(self, jid: int) -> None:
        with self._lock:
            self._db.execute("UPDATE journal SET status = 'done' WHERE id = ?", (jid,))
            self._db.commit()

    def abort(self, jid: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM journal WHERE id = ?", (jid,))
            self._db.commit()

    def pending(self) -> list[tuple[int, str, dict]]:
        with self._lock:
            rows = self._db.execute("SELECT id, op, detail FROM journal WHERE status = 'pending' ORDER BY id").fetchall()
        return [(i, op, json.loads(d or "{}")) for i, op, d in rows]

    def last_batch(self) -> tuple[str, list[tuple[int, str, dict]]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT batch_id FROM journal WHERE undone = 0 AND status = 'done' ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            rows = self._db.execute(
                "SELECT id, op, inverse FROM journal WHERE batch_id = ? AND undone = 0 AND status = 'done' "
                "ORDER BY id DESC", (row[0],)).fetchall()
        return row[0], [(i, op, json.loads(inv)) for i, op, inv in rows]

    def mark_undone(self, ids: list[int]) -> None:
        with self._lock:
            self._db.executemany("UPDATE journal SET undone = 1 WHERE id = ?", [(i,) for i in ids])
            self._db.commit()

    # ------------------------------------------------------------ reminders
    # ------------------------------------------------------------ notes (phase 30)
    def add_note(self, kind: str, text: str, ts: float | None = None) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO notes(ts, kind, text) VALUES (?,?,?)",
                                   (ts or time.time(), kind, protect(text, self.encrypt)))
            self._db.commit()
            return cur.lastrowid

    def notes(self, kind: str | None = None, since: float | None = None, until: float | None = None,
              deleted: bool = False) -> list[tuple[int, float, str, str]]:
        """[(id, ts, kind, text)] newest first."""
        q, args = "SELECT id, ts, kind, text FROM notes WHERE deleted = ?", [1 if deleted else 0]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        if since is not None:
            q += " AND ts >= ?"
            args.append(since)
        if until is not None:
            q += " AND ts < ?"
            args.append(until)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY ts DESC", args).fetchall()
        return [(i, t, k, unprotect(x)) for i, t, k, x in rows]

    def set_note_deleted(self, note_id: int, deleted: bool) -> None:
        with self._lock:
            self._db.execute("UPDATE notes SET deleted = ? WHERE id = ?", (1 if deleted else 0, note_id))
            self._db.commit()

    # ------------------------------------------------------------ tasks (phase 33)
    TASK_FIELDS = ("text", "due", "has_time", "every", "priority", "done", "deleted", "nudged")

    def add_task(self, text: str, due: float | None = None, has_time: bool = False, every: str | None = None,
                 priority: str = "normal") -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO tasks(created, text, due, has_time, every, priority) VALUES (?,?,?,?,?,?)",
                (time.time(), protect(text, self.encrypt), due, int(has_time), every, priority))
            self._db.commit()
            return cur.lastrowid

    def tasks(self, include_done: bool = False) -> list[dict]:
        q = "SELECT id, created, text, due, has_time, every, priority, done, nudged FROM tasks WHERE deleted = 0"
        if not include_done:
            q += " AND done IS NULL"
        with self._lock:
            rows = self._db.execute(q + " ORDER BY due IS NULL, due, created").fetchall()
        return [{"id": i, "created": c, "text": unprotect(t), "due": d, "has_time": bool(h), "every": e,
                 "priority": p, "done": dn, "nudged": n} for i, c, t, d, h, e, p, dn, n in rows]

    def get_task(self, task_id: int) -> dict | None:
        return next((t for t in self.tasks(include_done=True) if t["id"] == task_id), None)

    def update_task(self, task_id: int, **fields) -> None:
        sets, args = [], []
        for k, v in fields.items():
            if k not in self.TASK_FIELDS:
                raise ValueError(k)
            sets.append(f"{k} = ?")
            args.append(protect(v, self.encrypt) if k == "text" else int(v) if isinstance(v, bool) else v)
        with self._lock:
            self._db.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", (*args, task_id))
            self._db.commit()

    def add_reminder(self, due: float, text: str) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO reminders(due, text) VALUES (?,?)", (due, text))
            self._db.commit()
            return cur.lastrowid

    def upcoming_reminders(self, now: float, limit: int = 3) -> list[tuple[float, str]]:
        with self._lock:
            return self._db.execute("SELECT due, text FROM reminders WHERE fired = 0 AND due > ? ORDER BY due LIMIT ?",
                                    (now, limit)).fetchall()

    def due_reminders(self, now: float) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text FROM reminders WHERE fired = 0 AND due <= ?", (now,)).fetchall()
            self._db.executemany("UPDATE reminders SET fired = 1 WHERE id = ?", [(r[0],) for r in rows])
            self._db.commit()
        return rows
