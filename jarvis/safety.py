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
    "mood": Tier.SAFE,
    "briefing": Tier.SAFE,
    "maintenance": Tier.SAFE,
    "unsupported": Tier.SAFE,
    "search_web": Tier.SAFE,
    "play_media": Tier.SAFE,
    "window": Tier.SAFE,          # close asks (see confirm names below)
    "desktop": Tier.SAFE,
    "media": Tier.SAFE,
    "brightness": Tier.SAFE,
    "set_volume": Tier.SAFE,
    "radio": Tier.SAFE,           # turning a radio off asks
    "screenshot": Tier.SAFE,
    "screen_record": Tier.SAFE,
    "training": Tier.SAFE,
    "power": Tier.DESTRUCTIVE,    # shutdown/restart/sleep/sign out always ask; cancel doesn't
    "undo": Tier.CONFIRM,
    "move_files": Tier.CONFIRM,
    "copy_files": Tier.SAFE,      # non-destructive and undoable
    "system": Tier.SAFE,          # mute, volume, lock, display off: harmless, run instantly
    "delete_files": Tier.DESTRUCTIVE,
}

# Hard caps so a mis-parsed command can never touch thousands of files at once.
MAX_FILES_PER_ACTION = 500


class SafetyError(PermissionError):
    pass


DEFAULT_CONFIRM = ["move_files", "undo", "window_close", "radio_off"]   # editable in settings; delete/power always ask


def confirm_name(action: str, args: dict | None = None) -> str:
    """Name used in the 'ask before' list: actions, or action_variant for risky variants."""
    args = args or {}
    if action == "window" and str(args.get("action")) == "close":
        return "window_close"
    if action == "radio" and str(args.get("state")) == "off":
        return "radio_off"
    return action


def needs_confirmation(action: str, confirm_actions: list[str] | None = None, args: dict | None = None) -> bool:
    if action == "power" and str((args or {}).get("action")) == "cancel":
        return False
    if tier_of(action) >= Tier.DESTRUCTIVE:
        return True
    return confirm_name(action, args) in (DEFAULT_CONFIRM if confirm_actions is None else confirm_actions)


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
