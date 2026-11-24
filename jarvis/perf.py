"""Where the time and memory go (phase 41). /perf shows it.

    PERF.mark("ui")                         start-up milestones (seconds since Jarvis started)
    with PERF.stage("model"): ...           per-command stages: understand, act, reply
    PERF.command_done(source)               closes the current command's record
    PERF.model_loaded(name, file, seconds)  which models are in memory, int8 or not
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

T0 = time.perf_counter()


def rss_mb() -> float | None:
    """This process's memory (working set) in MB."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 2**20
            return None
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


class Perf:
    def __init__(self, keep: int = 200):
        self.marks: dict[str, float] = {}
        self.commands: deque = deque(maxlen=keep)
        self.models: dict[str, dict] = {}
        self._cur = threading.local()
        self.timeouts = 0

    # start-up
    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter() - T0)

    # one command
    def begin(self) -> None:
        self._cur.rec = {"t": time.time(), "stages": {}}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            rec = getattr(self._cur, "rec", None)
            if rec is not None:
                rec["stages"][name] = rec["stages"].get(name, 0.0) + time.perf_counter() - t

    def command_done(self, source: str = "") -> dict | None:
        rec = getattr(self._cur, "rec", None)
        if rec is None:
            return None
        rec["source"] = source
        rec["total"] = sum(rec["stages"].values())
        self.commands.append(rec)
        self._cur.rec = None
        return rec

    # models
    def model_loaded(self, name: str, file: Path, seconds: float) -> None:
        try:
            mb = Path(file).stat().st_size / 2**20
        except OSError:
            mb = None
        self.models[name] = {"file": Path(file).name, "int8": ".int8." in Path(file).name, "load_s": seconds,
                             "mb": mb, "at": time.time()}

    def model_released(self, name: str) -> None:
        self.models.pop(name, None)

    # report
    @staticmethod
    def _pct(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0

    def report(self) -> str:
        lines = []
        if self.marks:
            lines.append("Start-up: " + ", ".join(f"{k} {v:.2f}s" for k, v in sorted(self.marks.items(), key=lambda x: x[1])))
        mem = rss_mb()
        if mem is not None:
            lines.append(f"Memory: {mem:.0f} MB")
        if self.models:
            parts = []
            for n, m in self.models.items():
                size = f", {m['mb']:.0f} MB" if m.get("mb") else ""
                parts.append(f"{n} ({'int8' if m['int8'] else 'full precision'}{size}, loaded in {m['load_s']:.2f}s)")
            lines.append("Models in memory: " + ", ".join(parts))
        else:
            lines.append("Models in memory: none yet (they load when first needed)")
        recent = list(self.commands)[-50:]
        if recent:
            stages = sorted({s for r in recent for s in r["stages"]})
            parts = []
            for s in stages:
                xs = [r["stages"][s] for r in recent if s in r["stages"]]
                parts.append(f"{s} {self._pct(xs, .5) * 1000:.0f} ms (slowest 5%: {self._pct(xs, .95) * 1000:.0f} ms)")
            tot = [r["total"] for r in recent]
            lines.append(f"Last {len(recent)} commands: typical {self._pct(tot, .5) * 1000:.0f} ms, "
                         f"slowest 5% {self._pct(tot, .95) * 1000:.0f} ms")
            lines.append("  " + "; ".join(parts))
            slow = max(recent, key=lambda r: r["total"])
            worst = max(slow["stages"].items(), key=lambda x: x[1])
            lines.append(f"  slowest: {slow['total'] * 1000:.0f} ms, mostly {worst[0]} ({slow.get('source', '')})")
        if self.timeouts:
            lines.append(f"Model ran over the time budget {self.timeouts} time(s); rules answered instead.")
        return "\n".join(lines)


PERF = Perf()
