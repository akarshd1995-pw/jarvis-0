"""Which version is trusted (phase 42): update_state.json in Jarvis's data folder.

    {"pending": "<new commit>", "previous": "<commit before the update>", "since": ts}   just updated, unproven
    {"good": "<commit>"}                                                                  ran fine after an update
    {"bad": ["<commit>", ...], "rolled_back": {...}}                                      never install these again

The supervisor rolls back when a pending version crashes before Jarvis confirms it's running (confirm()).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

CONFIRM_AFTER_S = 30          # Jarvis has to run this long after an update before it counts as good


def default_data_dir() -> Path:
    return Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis0"))


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / "update_state.json"


def load(data_dir: Path) -> dict:
    try:
        return json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(data_dir: Path, state: dict) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def started_update(data_dir: Path, previous: str, new: str) -> None:
    s = load(data_dir)
    s.update(pending=new, previous=previous, since=time.time())
    save(data_dir, s)


def mark_bad(data_dir: Path, commit: str, why: str) -> None:
    s = load(data_dir)
    bad = s.get("bad", [])
    if commit not in bad:
        bad.append(commit)
    s["bad"] = bad[-20:]
    s["last_failure"] = {"commit": commit, "why": why[-500:], "at": time.time()}
    save(data_dir, s)


def confirm(data_dir: Path) -> bool:
    """Called by Jarvis once it has run CONFIRM_AFTER_S after an update: the new version is good."""
    s = load(data_dir)
    if not s.get("pending"):
        return False
    s["good"] = s.pop("pending")
    s.pop("since", None)
    save(data_dir, s)
    return True


def pending(data_dir: Path) -> dict | None:
    s = load(data_dir)
    return s if s.get("pending") else None


def rollback(root: Path, data_dir: Path, why: str = "it crashed while starting") -> tuple[bool, str]:
    """Go back to the version before the update (keeping any local changes). Returns (done, message)."""
    s = load(data_dir)
    new, prev = s.get("pending"), s.get("previous")
    if not (new and prev):
        return False, "nothing to roll back"
    r = subprocess.run(["git", "reset", "--keep", prev], cwd=root, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return False, "rollback failed: " + (r.stderr or r.stdout).strip()[-300:]
    mark_bad(data_dir, new, why)
    s = load(data_dir)
    s.pop("pending", None)
    s.pop("since", None)
    s["good"] = prev
    s["rolled_back"] = {"from": new, "to": prev, "why": why, "at": time.time(), "told": False}
    save(data_dir, s)
    return True, f"rolled back from {new[:7]} to {prev[:7]}"


def message_once(data_dir: Path) -> str:
    """After a rollback or a failed update: one message for the next start, then never again."""
    s = load(data_dir)
    rb = s.get("rolled_back")
    if rb and not rb.get("told"):
        rb["told"] = True
        save(data_dir, s)
        return (f"The last update ({rb['from'][:7]}) {rb['why']}, so I went back to the version before it "
                f"({rb['to'][:7]}). Everything else is as you left it; I'll skip that update.")
    return ""
