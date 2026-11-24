"""Keeps Jarvis running (phase 29).

`python -m jarvis` and the autostart launcher run this supervisor, which runs the real Jarvis as a child:
  exit code 0  -> you quit: stop
  exit code 3  -> restart asked for (after an update): start again straight away
  anything else -> a crash: start again after a short pause; 3 crashes within 10 minutes -> safe mode
                   (right after an update that hasn't proven itself yet: go back to the previous version first)
Safe mode: rules only, no HUD and no background services except reminders, and a message telling you to
run the health check. After 15 minutes without a crash, the next start is normal again.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import release
from .reliability import CrashLog

RESTART = 3
SAFE_AFTER = 3            # crashes ...
SAFE_WINDOW = 600         # ... within this many seconds


def run(child_args: list[str], data_dir: Path) -> int:
    crashes = CrashLog(data_dir / "logs" / "crashes.json")
    env = dict(os.environ, JARVIS_SUPERVISED="1")
    while True:
        safe = crashes.recent(SAFE_WINDOW) >= SAFE_AFTER
        me = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "jarvis"]  # installed exe
        cmd = me + ["--child"] + child_args + (["--safe-mode"] if safe else [])
        started = time.time()
        code = subprocess.call(cmd, env=env)
        if code == 0:
            return 0
        if code == RESTART:
            continue
        if release.pending(data_dir):              # phase 42: the new version crashed before it was up and running
            done, msg = release.rollback(Path(__file__).resolve().parent.parent, data_dir,
                                         why=f"crashed while starting (exit code {code})")
            print(f"[supervisor] {msg}", file=sys.stderr)
            if done:
                continue                             # start the previous version straight away
        crashes.add()
        if safe and time.time() - started > 900:
            crashes.clear()                      # safe mode ran fine for a while: try normal next time
        time.sleep(min(30, 2 * max(1, crashes.recent(SAFE_WINDOW))))
