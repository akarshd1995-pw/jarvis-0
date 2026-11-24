"""note / find_notes / delete_note (phase 30). Saving is instant and undoable; deleting asks first and is undoable."""
from __future__ import annotations

import re

from .. import notes as N
from .base import ActionResult, Ctx


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip("\"'").strip()


def note(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    kind = str(args.get("kind", "note"))
    text = _clean(args["text"])
    if not text:
        raise ValueError("there's nothing to note")
    data = {"kind": kind, "text": text}
    if dry_run:
        return ActionResult(f"Will save a {kind}: {text}", kind="note", data=data)
    nid = ctx.store.add_note(kind, text)
    ctx.store.record(ctx.batch_id, "note", {"op": "note_forget", "id": nid})      # "undo" removes it again
    since, _ = N.window("today")
    data.update(id=nid, today=len(ctx.store.notes(kind=kind, since=since)))
    return ActionResult(f"Saved {kind}: {text}", kind="note", data=data)


def find_notes(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    query = _clean(args["query"]) if args.get("query") else None
    kind = str(args["kind"]) if args.get("kind") else None
    when = str(args["when"]) if args.get("when") else None
    rows = N.find(ctx.store, query, kind, when)
    results = [{"title": t[:120], "text": t, "path": f"note:{i}", "source": "note", "kind": k, "mtime": ts}
               for i, ts, k, t in rows]
    return ActionResult(f"{len(rows)} note(s)", [r["text"] for r in results], kind="notes",
                        data={"query": query, "kind": kind, "when": when, "count": len(rows), "results": results})


def delete_note(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    query = _clean(args["query"])
    rows = N.find(ctx.store, query, limit=1)
    if not rows:
        raise FileNotFoundError(f"note not found: {query}")
    nid, ts, kind, text = rows[0]
    if dry_run:
        return ActionResult(f'Delete the {kind} "{text[:100]}"?', [text], kind="note_deleted",
                            data={"text": text, "kind": kind, "dry": True})
    ctx.store.set_note_deleted(nid, True)
    ctx.store.record(ctx.batch_id, "note", {"op": "note_restore", "id": nid})
    return ActionResult(f'Deleted {kind}: {text}', kind="note_deleted", data={"text": text, "kind": kind})
