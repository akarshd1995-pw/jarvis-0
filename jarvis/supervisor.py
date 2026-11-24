"""Keeps Jarvis running (phase 29).

`python -m jarvis` and the autostart launcher run this supervisor, which runs the real Jarvis as a child:
  exit code 0  -> you quit: stop
  exit code 3  -> restart asked for (after an update): start again straight away
  anything else -> a crash: start again after a short pause; 3 crashes within 10 minutes -> safe mode
Safe mode: rules only, no HUD and no background services except reminders, and a message telling you to
run the health check. After 15 minutes without a crash, the next start is normal again.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .reliability import CrashLog

RESTART = 3
SAFE_AFTER = 3            # crashes ...
SAFE_WINDOW = 600         # ... within this many seconds


def run(child_args: list[str], data_dir: Path) -> int:
    crashes = CrashLog(data_dir / "logs" / "crashes.json")
    env = dict(os.environ, JARVIS_SUPERVISED="1")
    while True:
        safe = crashes.recent(SAFE_WINDOW) >= SAFE_AFTER
        cmd = [sys.executable, "-m", "jarvis", "--child"] + child_args + (["--safe-mode"] if safe else [])
        started = time.time()
        code = subprocess.call(cmd, env=env)
        if code == 0:
            return 0
        if code == RESTART:
            continue
        crashes.add()
        if safe and time.time() - started > 900:
            crashes.clear()                      # safe mode ran fine for a while: try normal next time
        time.sleep(min(30, 2 * max(1, crashes.recent(SAFE_WINDOW))))
