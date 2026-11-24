"""Turns a validated DSL program into real actions, enforcing the safety tiers."""
from __future__ import annotations

import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from .actions import control, files, system, training, maps, notes, workspace
from .actions.base import ActionResult, Ctx
from .config import Config
from .dsl.grammar import Program
from .safety import Tier, needs_confirmation, tier_of
from .store import Store

Handler = Callable[[dict, Ctx, bool], ActionResult]

REGISTRY: dict[str, Handler] = {
    "open_app": system.open_app,
    "open_url": system.open_url,
    "directions": maps.directions,
    "note": notes.note,
    "find_notes": notes.find_notes,
    "delete_note": notes.delete_note,
    "workspace": workspace.workspace,
    "open_folder": system.open_folder,
    "remind": system.remind,
    "system": system.system,
    "search_memory": system.search_memory,
    "unknown": system.unknown,
    "unsupported": system.unsupported,
    "search_web": system.search_web,
    "play_media": system.play_media,
    "window": control.window,
    "desktop": control.desktop,
    "media": control.media,
    "brightness": control.brightness,
    "set_volume": control.set_volume,
    "radio": control.radio,
    "screenshot": control.screenshot,
    "screen_record": control.screen_record,
    "power": control.power,
    "training": training.training,
    "list_files": files.list_files,
    "move_files": files.move_files,
    "copy_files": files.copy_files,
    "delete_files": files.delete_files,
}

# confirm(tier, summary, preview_items) -> bool
ConfirmFn = Callable[[Tier, str, list[str]], bool]


@dataclass
class RunReport:
    batch_id: str
    results: list[ActionResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    declined: list[ActionResult] = field(default_factory=list)          # user said No at the preview
    failures: list[tuple[str, Exception]] = field(default_factory=list)  # (action, exception)

    def text(self) -> str:
        lines = [r.summary for r in self.results]
        lines += [f"Skipped: {s}" for s in self.skipped]
        lines += [f"Error: {e}" for e in self.errors]
        return "\n".join(lines) or "Nothing to do."


class Executor:
    def __init__(self, config: Config, store: Store, confirm: ConfirmFn, describe=None, screen=None):
        self.config, self.store, self.confirm = config, store, confirm
        self.screen = screen          # phase 38: fills "this"/"here" from File Explorer, results, clipboard
        # describe(preview: ActionResult) -> str: in-character confirmation question (persona layer)
        self.describe = describe

    def run(self, prog: Program) -> RunReport:
        report = RunReport(batch_id=uuid.uuid4().hex[:12])
        ctx = Ctx(self.config, self.store, report.batch_id)
        if self.screen is not None:
            from .context import ContextError, uses_refs
            if uses_refs(prog):
                import copy
                try:
                    prog = self.screen.fill(copy.deepcopy(prog))
                except ContextError as e:
                    report.errors.append(f"context: {e}")
                    report.failures.append((prog.calls[0].action, e))
                    return report
        for call in prog.calls:
            try:
                if call.action == "undo":
                    report.results.append(self.undo())
                    continue
                handler = REGISTRY[call.action]
                tier = tier_of(call.action)
                if needs_confirmation(call.action, getattr(self.config, "confirm_actions", None), call.args):
                    preview = handler(call.args, ctx, True)          # dry run
                    question = self.describe(preview) if self.describe else preview.summary
                    if not self.confirm(tier, question, preview.items):
                        report.skipped.append(preview.summary)
                        report.declined.append(preview)
                        continue
                report.results.append(handler(call.args, ctx, False))
            except Exception as e:  # report and keep going with the next call
                report.errors.append(f"{call.action}: {e}")
                report.failures.append((call.action, e))
        self.store.log_event("command_run", {"program": str(prog), "batch": report.batch_id,
                                             "errors": report.errors, "skipped": len(report.skipped)})
        return report

    def undo(self) -> ActionResult:
        last = self.store.last_batch()
        if not last:
            return ActionResult("Nothing to undo.", kind="undo", data={"count": 0})
        batch_id, entries = last
        if needs_confirmation("undo", getattr(self.config, "confirm_actions", None)) and \
                not self.confirm(Tier.CONFIRM, f"Undo {len(entries)} operation(s) from last command?", []):
            return ActionResult("Undo cancelled.", kind="undo", data={"count": 0, "cancelled": True})
        done, failed = [], []
        for entry_id, _op, inverse in entries:   # newest first; one stuck file doesn't stop the rest
            try:
                if inverse.get("op") in ("note_forget", "note_restore"):          # phase 30: notes
                    self.store.set_note_deleted(inverse["id"], inverse["op"] == "note_forget")
                elif str(inverse.get("op", "")).startswith("workspace_"):        # phase 31: windows / layouts
                    from .actions.workspace import undo_op
                    undo_op(self.config, inverse)
                else:
                    files.apply_inverse(inverse)
                done.append(entry_id)
            except Exception as e:
                name = Path(inverse.get("src") or inverse.get("path") or "?").name
                failed.append((name, files.safeops.reason(e)))
        self.store.mark_undone(done)
        return ActionResult(f"Undid {len(done)} of {len(entries)} operation(s).", kind="undo",
                            data={"count": len(done), "failed": failed, "requested": len(entries)})

    def recover(self) -> dict:
        """Finish or roll back file steps interrupted by a crash or power cut (run at start-up)."""
        return files.safeops.recover(self.store)
