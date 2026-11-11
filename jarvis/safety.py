"""Safety layer: permission tiers, path allowlist, confirmation policy."""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path


class Tier(IntEnum):
    SAFE = 0         # runs immediately (open app, list files, reminders)
    CONFIRM = 1      # needs a yes from the user (move, copy, system actions)
    DESTRUCTIVE = 2  # needs a yes AND shows a dry-run preview first (delete)


ACTION_TIERS: dict[str, Tier] = {
    "open_app": Tier.SAFE,
    "open_url": Tier.SAFE,
    "open_folder": Tier.SAFE,
    "list_files": Tier.SAFE,
    "remind": Tier.SAFE,
    "search_memory": Tier.SAFE,
    "unknown": Tier.SAFE,
    "undo": Tier.CONFIRM,
    "move_files": Tier.CONFIRM,
    "copy_files": Tier.CONFIRM,
    "system": Tier.CONFIRM,
    "delete_files": Tier.DESTRUCTIVE,
}

# Hard caps so a mis-parsed command can never touch thousands of files at once.
MAX_FILES_PER_ACTION = 500


class SafetyError(PermissionError):
    pass


def tier_of(action: str) -> Tier:
    # Unknown actions default to the strictest tier.
    return ACTION_TIERS.get(action, Tier.DESTRUCTIVE)


def check_path(path: Path, allowed_roots: list[str]) -> Path:
    """Resolve and ensure the path is inside one of the allowed roots."""
    resolved = path.expanduser().resolve()
    for root in allowed_roots:
        root_p = Path(root).expanduser().resolve()
        if resolved == root_p or root_p in resolved.parents:
            return resolved
    raise SafetyError(f"{resolved} is outside the allowed folders")
