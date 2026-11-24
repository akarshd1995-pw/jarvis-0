"""One-click update: git pull --ff-only in the project folder, then restart."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def update(root: Path = PROJECT_ROOT) -> tuple[bool, str]:
    """Returns (changed, message). Never discards local changes: fast-forward only."""
    if not (root / ".git").exists():
        return False, "This folder isn't a git clone, so it can't update itself. Clone from GitHub instead."
    try:
        before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=30).stdout.strip()
        r = subprocess.run(["git", "pull", "--ff-only"], cwd=root, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return False, "git isn't installed or isn't on PATH."
    except subprocess.TimeoutExpired:
        return False, "Update timed out (no network?)."
    if r.returncode != 0:
        return False, "Update failed:\n" + (r.stderr or r.stdout).strip()[-600:]
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                           timeout=30).stdout.strip()
    if before == after:
        return False, "Already up to date."
    log = subprocess.run(["git", "log", "--oneline", f"{before}..{after}"], cwd=root, capture_output=True,
                         text=True, timeout=30).stdout.strip()
    return True, f"Updated {before[:7]} -> {after[:7]}:\n{log}"


def restart() -> None:
    """Start a fresh Jarvis: the supervisor restarts us on exit code 3; without one, replace this process."""
    if os.environ.get("JARVIS_SUPERVISED") == "1":
        os._exit(3)
    os.execv(sys.executable, [sys.executable, *sys.argv])
