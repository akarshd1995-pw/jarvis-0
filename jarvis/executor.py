"""Turns a validated DSL program into real actions, enforcing the safety tiers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

from .actions import files, system
from .actions.base import ActionResult, Ctx
from .config import Config
from .dsl.grammar import Program
from .safety import Tier, tier_of
from .store import Store

Handler = Callable[[dict, Ctx, bool], ActionResult]

REGISTRY: dict[str, Handler] = {
    "open_app": system.open_app,
    "open_url": system.open_url,
    "open_folder": system.open_folder,
    "remind": system.remind,
    "system": system.system,
    "search_memory": system.search_memory,
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

    def text(self) -> str:
        lines = [r.summary for r in self.results]
        lines += [f"Skipped: {s}" for s in self.skipped]
        lines += [f"Error: {e}" for e in self.errors]
        return "\n".join(lines) or "Nothing to do."


class Executor:
    def __init__(self, config: Config, store: Store, confirm: ConfirmFn):
        self.config, self.store, self.confirm = config, store, confirm

    def run(self, prog: Program) -> RunReport:
        report = RunReport(batch_id=uuid.uuid4().hex[:12])
        ctx = Ctx(self.config, self.store, report.batch_id)
        for call in prog.calls:
            try:
                if call.action == "undo":
                    report.results.append(self.undo())
                    continue
                handler = REGISTRY[call.action]
                tier = tier_of(call.action)
                if tier >= Tier.CONFIRM:
                    preview = handler(call.args, ctx, True)          # dry run
                    if not self.confirm(tier, preview.summary, preview.items):
                        report.skipped.append(preview.summary)
                        continue
                report.results.append(handler(call.args, ctx, False))
            except Exception as e:  # report and keep going with the next call
                report.errors.append(f"{call.action}: {e}")
        self.store.log_event("command_run", {"program": str(prog), "batch": report.batch_id,
                                             "errors": report.errors, "skipped": len(report.skipped)})
        return report

    def undo(self) -> ActionResult:
        last = self.store.last_batch()
        if not last:
            return ActionResult("Nothing to undo.")
        batch_id, entries = last
        if not self.confirm(Tier.CONFIRM, f"Undo {len(entries)} operation(s) from last command?", []):
            return ActionResult("Undo cancelled.")
        done = []
        for entry_id, _op, inverse in entries:   # newest first
            files.apply_inverse(inverse)
            done.append(entry_id)
        self.store.mark_undone(done)
        return ActionResult(f"Undid {len(done)} operation(s).")
