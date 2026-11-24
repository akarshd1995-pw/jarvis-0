"""workspace(action=start|save|close|list|delete, name=...) (phase 31)."""
from __future__ import annotations

from ..workspaces import RealBackend, WorkspaceNotFound, Workspaces, nice_app, norm
from .base import ActionResult, Ctx

BACKEND = None          # tests put a fake here


def _ws(ctx: Ctx) -> Workspaces:
    return Workspaces(ctx.config.data_dir / "workspaces.json", BACKEND or RealBackend(ctx),
                      starter_apps=getattr(ctx.config, "workspace_apps", None))


def workspace(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    act = str(args["action"])
    name = str(args.get("name", "")).strip()
    ws = _ws(ctx)
    if act == "list":
        saved = ws.saved()
        rows = [{"name": n, "saved": n in saved, "apps": sorted({nice_app(e["app"]) for e in saved[n]["windows"]})
                 if n in saved and saved[n].get("windows") else [nice_app(a) for a in ws.get(n)[1].get("apps", [])]}
                for n in ws.names()]
        return ActionResult(f"{len(rows)} workspace(s)", [r["name"] for r in rows], kind="workspace",
                            data={"action": "list", "rows": rows})
    if not name:
        raise ValueError("which workspace? e.g. \"start coding mode\"")
    try:
        if act == "save":
            if dry_run:
                return ActionResult(f"Save the open windows as {norm(name)}", kind="workspace",
                                    data={"action": "save", "name": norm(name), "dry": True})
            key, d, previous = ws.save(name)
            ctx.store.record(ctx.batch_id, "workspace", {"op": "workspace_define", "name": key, "def": previous})
            return ActionResult(f"Saved {len(d['windows'])} windows as {key}", kind="workspace",
                                data={"action": "save", "name": key, "count": len(d["windows"]),
                                      "replaced": previous is not None,
                                      "apps": sorted({nice_app(e["app"]) for e in d["windows"]})})
        if act == "start":
            key, d = ws.get(name)
            if dry_run:
                return ActionResult(f"Start {key}", kind="workspace", data={"action": "start", "name": key, "dry": True})
            rep = ws.start(name)
            ctx.store.record(ctx.batch_id, "workspace", {"op": "workspace_restore", "placements": rep.before})
            return ActionResult(f"{key} ready", kind="workspace",
                                data={"action": "start", "name": key, "arranged": rep.arranged, "opened": rep.opened,
                                      "missing": rep.missing, "music": rep.music, "preset": bool(d.get("preset"))})
        if act == "close":
            key, wins = ws.windows_of(name)
            apps = sorted({nice_app(w.app) for w in wins})
            if dry_run:
                return ActionResult(f"Close {', '.join(apps) or 'nothing'} ({len(wins)} window(s))?", apps,
                                    kind="workspace", data={"action": "close", "name": key, "apps": apps, "dry": True})
            ws.close(name)
            return ActionResult(f"Closed {key}", apps, kind="workspace",
                                data={"action": "close", "name": key, "apps": apps, "count": len(wins)})
        if act == "delete":
            key = norm(name)
            previous = ws.saved().get(key)
            if previous is None:
                raise WorkspaceNotFound(key, ws.names())
            if dry_run:
                return ActionResult(f"Delete your saved {key} layout?", kind="workspace",
                                    data={"action": "delete", "name": key, "dry": True})
            ws.set_definition(key, None)
            ctx.store.record(ctx.batch_id, "workspace", {"op": "workspace_define", "name": key, "def": previous})
            return ActionResult(f"Deleted {key}", kind="workspace", data={"action": "delete", "name": key})
    except WorkspaceNotFound as e:
        return ActionResult(f"No workspace {e.name}", kind="workspace",
                            data={"action": act, "name": e.name, "missing_workspace": True, "known": e.known})
    raise ValueError(f"unknown workspace action {act}")


def undo_op(ctx_config, inverse: dict) -> None:
    """Executor undo: put windows back / bring back the previous saved layout."""
    ws = Workspaces(ctx_config.data_dir / "workspaces.json", BACKEND or RealBackend())
    if inverse["op"] == "workspace_restore":
        ws.restore_placements(inverse["placements"])
    else:
        ws.set_definition(inverse["name"], inverse["def"])
