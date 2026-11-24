"""Safe updates (phase 42): a new version is tested BEFORE Jarvis switches to it, and undone if it can't start.

/update (or the tray's Check for updates):
  1. fetch the new version from GitHub (nothing changes yet)
  2. check it out beside the current one (a git worktree in a temp folder) and test it there:
     the full test suite, then the end-to-end self-test (every action, in a throwaway folder)
  3. only if both pass: fast-forward to it and restart; remember the version that was running before
  4. the supervisor watches the first start: if the new version crashes before it's up and running,
     it goes back to the previous version by itself and tells you (release.py)
A version that failed its tests or had to be rolled back is skipped until a newer one is published.
Your own uncommitted changes are never thrown away (fast-forward only; rollback uses git reset --keep).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from . import release

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TIMEOUT = 15 * 60


def _git(root: Path, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout)


def _head(root: Path, ref: str = "HEAD") -> str:
    return _git(root, "rev-parse", ref, timeout=30).stdout.strip()


def test_version(root: Path, commit: str, progress: Callable[[str], None] = lambda s: None,
                 python: str | None = None) -> tuple[bool, str]:
    """Check out `commit` beside the running version and run its tests there. (passed, what happened)."""
    python = python or sys.executable.replace("pythonw.exe", "python.exe")
    wt = Path(tempfile.mkdtemp(prefix="jarvis-update-"))
    home = Path(tempfile.mkdtemp(prefix="jarvis-update-home-"))
    flags = 0x08000000 if sys.platform == "win32" else 0                  # no console window
    try:
        r = _git(root, "worktree", "add", "--detach", "--force", str(wt), commit)
        if r.returncode != 0:
            return False, "couldn't check out the new version: " + (r.stderr or r.stdout).strip()[-300:]
        env = dict(os.environ, JARVIS_HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
        env.pop("JARVIS_SUPERVISED", None)
        old_req = (root / "requirements.txt").read_text(encoding="utf-8", errors="ignore") \
            if (root / "requirements.txt").exists() else ""
        new_req = (wt / "requirements.txt").read_text(encoding="utf-8", errors="ignore") \
            if (wt / "requirements.txt").exists() else ""
        if new_req != old_req:
            progress("The new version needs new packages: installing them first...")
            r = subprocess.run([python, "-m", "pip", "install", "-q", "-r", str(wt / "requirements.txt")],
                               cwd=wt, capture_output=True, text=True, timeout=TEST_TIMEOUT, creationflags=flags)
            if r.returncode != 0:
                return False, "installing the new packages failed:\n" + (r.stderr or r.stdout).strip()[-500:]
        progress("Testing the new version before switching to it (a few minutes)...")
        r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], cwd=wt, env=env,
                           capture_output=True, text=True, timeout=TEST_TIMEOUT, creationflags=flags)
        tail = (r.stdout or r.stderr).strip().splitlines()
        if r.returncode == 5 or "No module named pytest" in (r.stderr or ""):
            tests = "tests skipped (pytest isn't installed)"
        elif r.returncode != 0:
            failed = [line for line in tail if line.startswith(("FAILED", "ERROR"))][:5]
            return False, "the new version failed its tests:\n" + "\n".join(failed or tail[-8:])
        else:
            tests = tail[-1] if tail else "tests passed"
        progress("Tests passed. Running every action in a throwaway folder...")
        r = subprocess.run([python, "-m", "jarvis.selftest"], cwd=wt, env=env, capture_output=True, text=True,
                           timeout=TEST_TIMEOUT, creationflags=flags)
        out = (r.stdout or r.stderr).strip().splitlines()
        if r.returncode != 0:
            return False, "the new version failed its self-test:\n" + "\n".join(
                [line for line in out if "FAIL" in line or line.startswith("  - ")][:8] or out[-8:])
        return True, f"{tests}; {out[-1] if out else 'self-test passed'}"
    except subprocess.TimeoutExpired:
        return False, "testing the new version took too long"
    finally:
        _git(root, "worktree", "remove", "--force", str(wt), timeout=60)
        shutil.rmtree(wt, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)
        _git(root, "worktree", "prune", timeout=30)


def update(root: Path = PROJECT_ROOT, progress: Callable[[str], None] = lambda s: None,
           data_dir: Path | None = None, test: bool = True, tester=test_version) -> tuple[bool, str]:
    """Returns (changed, message). changed=True means: switched, restart now."""
    if getattr(sys, "frozen", False):
        return False, "This is an installed copy: install a newer JarvisSetup.exe to update it."
    if not (root / ".git").exists():
        return False, "This folder isn't a git clone, so it can't update itself. Clone from GitHub instead."
    data_dir = data_dir or release.default_data_dir()
    try:
        before = _head(root)
        r = _git(root, "fetch", "--quiet")
        if r.returncode != 0:
            return False, "Couldn't reach GitHub:\n" + (r.stderr or r.stdout).strip()[-400:]
        upstream = _head(root, "@{u}")
    except FileNotFoundError:
        return False, "git isn't installed or isn't on PATH."
    except subprocess.TimeoutExpired:
        return False, "Update timed out (no network?)."
    if not upstream:
        return False, "This branch doesn't track a GitHub branch, so there's nothing to update from."
    if upstream == before:
        return False, "Already up to date."
    if _git(root, "merge-base", "--is-ancestor", before, upstream, timeout=30).returncode != 0:
        return False, ("Your copy has commits that aren't on GitHub, so I won't update it automatically. "
                       "Run git pull yourself.")
    state = release.load(data_dir)
    if upstream in state.get("bad", []):
        return False, (f"The newest version ({upstream[:7]}) didn't pass its checks earlier, so I'm staying on "
                       f"{before[:7]}. I'll try again when a newer version is published.")
    summary = ""
    if test:
        ok, summary = tester(root, upstream, progress)
        if not ok:
            release.mark_bad(data_dir, upstream, summary)
            return False, f"I didn't install the update ({upstream[:7]}): {summary}\nYou're still on {before[:7]}."
    r = _git(root, "merge", "--ff-only", upstream)
    if r.returncode != 0:
        return False, "Update failed:\n" + (r.stderr or r.stdout).strip()[-600:]
    release.started_update(data_dir, before, upstream)
    log = _git(root, "log", "--oneline", f"{before}..{upstream}", timeout=30).stdout.strip()
    checked = f"\nChecked first: {summary}" if summary else "\n(Not tested first: forced.)"
    return True, f"Updated {before[:7]} -> {upstream[:7]}:\n{log}{checked}"


def restart() -> None:
    """Start a fresh Jarvis: the supervisor restarts us on exit code 3; without one, replace this process."""
    if os.environ.get("JARVIS_SUPERVISED") == "1":
        os._exit(3)
    os.execv(sys.executable, [sys.executable, *sys.argv])
