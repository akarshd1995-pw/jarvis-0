"""SQLite store: event log (training data for the routine model), undo journal, reminders.

Everything stays local. For encryption at rest, swap sqlite3 for SQLCipher later.
"""
from __future__ import annotations

import json
import sqlite3
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

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    due REAL NOT NULL,
    text TEXT NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path | str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ------------------------------------------------------------ events
    def log_event(self, kind: str, payload: dict, ts: float | None = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO events(ts, kind, payload) VALUES (?,?,?)",
                             (ts or time.time(), kind, json.dumps(payload)))
            self._db.commit()

    def events_since(self, ts: float, kind: str | None = None) -> list[tuple[float, str, dict]]:
        q = "SELECT ts, kind, payload FROM events WHERE ts >= ?"
        args: list = [ts]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY ts", args).fetchall()
        return [(t, k, json.loads(p)) for t, k, p in rows]

    # ------------------------------------------------------------ journal
    def record(self, batch_id: str, op: str, inverse: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO journal(batch_id, ts, op, inverse) VALUES (?,?,?,?)",
                             (batch_id, time.time(), op, json.dumps(inverse)))
            self._db.commit()

    def last_batch(self) -> tuple[str, list[tuple[int, str, dict]]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT batch_id FROM journal WHERE undone = 0 ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            rows = self._db.execute(
                "SELECT id, op, inverse FROM journal WHERE batch_id = ? AND undone = 0 ORDER BY id DESC",
                (row[0],)).fetchall()
        return row[0], [(i, op, json.loads(inv)) for i, op, inv in rows]

    def mark_undone(self, ids: list[int]) -> None:
        with self._lock:
            self._db.executemany("UPDATE journal SET undone = 1 WHERE id = ?", [(i,) for i in ids])
            self._db.commit()

    # ------------------------------------------------------------ reminders
    def add_reminder(self, due: float, text: str) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO reminders(due, text) VALUES (?,?)", (due, text))
            self._db.commit()
            return cur.lastrowid

    def due_reminders(self, now: float) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text FROM reminders WHERE fired = 0 AND due <= ?", (now,)).fetchall()
            self._db.executemany("UPDATE reminders SET fired = 1 WHERE id = ?", [(r[0],) for r in rows])
            self._db.commit()
        return rows
