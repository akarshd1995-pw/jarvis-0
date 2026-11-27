"""task(action=add|done|list|delete|move, text, due, every, priority, when)  (phase 33)

    "add task finish the report by friday"        add (due="friday" is read by dates.parse_due)
    "pay rent every month on the 1st"              add, every=monthly
    "done with the report" / "mark rent as done"   done: a recurring task moves to its next date instead
    "what's due today" / "show overdue tasks"      list
    "move the report to monday"                    move
    "delete the gym task"                          delete (asks first)
Everything is undoable. Tasks are encrypted in Jarvis's database like notes.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta

from .. import dates as D
from ..notes import score
from .base import ActionResult, Ctx


class TaskNotFound(Exception):
    def __init__(self, query: str):
        self.query = query
        super().__init__(f"no open task matching {query}")


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip("\"'").strip()


def _end_of_today(now: float) -> float:
    return datetime.fromtimestamp(now).replace(hour=23, minute=59, second=0, microsecond=0).timestamp()


def find(store, query: str, include_done: bool = False) -> dict | None:
    """The open task that best matches what you said (newest deadline first among equals)."""
    rows = [(score(query, t["text"]), t) for t in store.tasks(include_done)]
    rows = [(s, t) for s, t in rows if s >= 0.5]
    rows.sort(key=lambda x: (-x[0], x[1]["due"] or 1e12))
    return rows[0][1] if rows else None


def row(t: dict, now: float) -> dict:
    due = D.nice(t["due"], t["has_time"], now) if t["due"] else ""
    return {"title": t["text"][:120], "text": t["text"], "path": f"task:{t['id']}", "source": "task",
            "mtime": t["due"] or t["created"], "due": due, "overdue": bool(t["due"] and t["due"] < now),
            "every": t["every"], "priority": t["priority"]}


def select(store, when: str | None, now: float) -> list[dict]:
    end_today = _end_of_today(now)
    if when == "done":
        week = now - 7 * 86400
        return [t for t in store.tasks(include_done=True) if t["done"] and t["done"] >= week]
    ts = store.tasks()
    if when == "today":
        return [t for t in ts if t["due"] and t["due"] <= end_today]
    if when == "week":
        return [t for t in ts if t["due"] and t["due"] <= end_today + 6 * 86400]
    if when == "overdue":
        return [t for t in ts if t["due"] and t["due"] < now]
    return ts


def task(args: dict, ctx: Ctx, dry_run: bool, now: float | None = None) -> ActionResult:
    now = time.time() if now is None else now
    act = str(args["action"])
    store = ctx.store
    if act == "list":
        when = str(args["when"]) if args.get("when") else None
        rows = [row(t, now) for t in select(store, when, now)]
        rows.sort(key=lambda r: (not r["overdue"], r["priority"] != "high", r["mtime"]))
        return ActionResult(f"{len(rows)} task(s)", [r["text"] for r in rows], kind="tasks",
                            data={"when": when, "count": len(rows), "results": rows})
    text = _clean(args.get("text", ""))
    if not text:
        raise ValueError("which task?")
    if act == "add":
        due_said = _clean(args["due"]) if args.get("due") else None
        parsed = D.parse_due(due_said, now) if due_said else None
        every = str(args["every"]) if args.get("every") else None
        if parsed is None and every:
            parsed = (_end_of_today(now), False)                 # a recurring task starts today
        prio = str(args.get("priority", "normal"))
        data = {"action": "add", "text": text, "every": every, "priority": prio, "bad_due": bool(due_said and not parsed),
                "due_said": due_said, "due": D.nice(*parsed, now) if parsed else ""}
        if dry_run:
            return ActionResult(f"Add task: {text}", kind="task", data={**data, "dry": True})
        tid = store.add_task(text, parsed[0] if parsed else None, parsed[1] if parsed else False, every, prio)
        store.record(ctx.batch_id, "task", {"op": "task_set", "id": tid, "fields": {"deleted": 1}})
        data["open"] = len(store.tasks())
        return ActionResult(f"Added task: {text}", kind="task", data=data)
    t = find(store, text)
    if t is None:
        raise TaskNotFound(text)
    if act == "done":
        if dry_run:
            return ActionResult(f"Tick off: {t['text']}", kind="task", data={"action": "done", "text": t["text"]})
        if t["every"]:
            nxt = D.next_occurrence(t["due"] or _end_of_today(now), t["every"])
            while nxt < now:
                nxt = D.next_occurrence(nxt, t["every"])
            store.update_task(t["id"], due=nxt, nudged=None)
            store.record(ctx.batch_id, "task", {"op": "task_set", "id": t["id"],
                                                "fields": {"due": t["due"], "nudged": t["nudged"]}})
            return ActionResult(f"Done: {t['text']}", kind="task",
                                data={"action": "done", "text": t["text"], "next": D.nice(nxt, t["has_time"], now)})
        store.update_task(t["id"], done=now)
        store.record(ctx.batch_id, "task", {"op": "task_set", "id": t["id"], "fields": {"done": None}})
        left = select(store, "today", now)
        return ActionResult(f"Done: {t['text']}", kind="task",
                            data={"action": "done", "text": t["text"], "left_today": len(left)})
    if act == "delete":
        if dry_run:
            return ActionResult(f'Delete the task "{t["text"]}"?', [t["text"]], kind="task",
                                data={"action": "delete", "text": t["text"], "dry": True})
        store.update_task(t["id"], deleted=1)
        store.record(ctx.batch_id, "task", {"op": "task_set", "id": t["id"], "fields": {"deleted": 0}})
        return ActionResult(f"Deleted task: {t['text']}", kind="task", data={"action": "delete", "text": t["text"]})
    if act == "move":
        due_said = _clean(args.get("due", ""))
        parsed = D.parse_due(due_said, now) if due_said else None
        if parsed is None:
            raise ValueError(f"I couldn't read the new date \"{due_said}\". Try \"friday\" or \"5th december\"")
        if dry_run:
            return ActionResult(f"Move {t['text']}", kind="task", data={"action": "move", "text": t["text"]})
        store.update_task(t["id"], due=parsed[0], has_time=parsed[1], nudged=None)
        store.record(ctx.batch_id, "task", {"op": "task_set", "id": t["id"],
                                            "fields": {"due": t["due"], "has_time": t["has_time"], "nudged": t["nudged"]}})
        return ActionResult(f"Moved: {t['text']}", kind="task",
                            data={"action": "move", "text": t["text"], "due": D.nice(*parsed, now)})
    raise ValueError(f"unknown task action {act}")


def overdue_to_nudge(store, now: float | None = None) -> list[dict]:
    """Timed tasks that just went past their deadline and haven't been mentioned yet."""
    now = time.time() if now is None else now
    return [t for t in store.tasks() if t["due"] and t["has_time"] and t["due"] < now and not t["nudged"]
            and now - t["due"] < 12 * 3600]


def briefing_lines(store, kind: str, now: float | None = None) -> list[str]:
    """Morning: what's due today and what's overdue. Evening: what's still open from today."""
    now = time.time() if now is None else now
    start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    ts = store.tasks()
    over = [t for t in ts if t["due"] and t["due"] < start]
    today = [t for t in ts if t["due"] and start <= t["due"] <= _end_of_today(now)]

    def names(xs):
        short = [x["text"] if len(x["text"]) <= 50 else x["text"][:49] + "…" for x in xs[:3]]
        return "; ".join(short) + (f", and {len(xs) - 3} more" if len(xs) > 3 else "")
    out = []
    if kind == "morning":
        if today:
            out.append(f"Due today: {names(today)}.")
        if over:
            out.append(f"Overdue: {names(over)}.")
        week = [t for t in ts if t["due"] and _end_of_today(now) < t["due"] <= _end_of_today(now) + 6 * 86400]
        if week and not today:
            out.append(f"Coming up this week: {names(week)}.")
    else:
        left = [t for t in today if t["due"] >= start]
        if left:
            out.append(f"Still open from today: {names(left)}.")
        done = [t for t in store.tasks(include_done=True) if t["done"] and t["done"] >= start]
        if done:
            out.append(f"You finished {len(done)} task{'s' * (len(done) != 1)} today.")
    return out
