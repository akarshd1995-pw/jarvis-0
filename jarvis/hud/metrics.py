"""Live system metrics for the HUD, gathered on a background thread (stdlib + ctypes, no psutil).

CPU and RAM come from Windows APIs (/proc on Linux, for tests). GPU stats come from `nvidia-smi` if present.
Training progress is read from the training logs ("step 12300/60000 ... ~41 min left").
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


@dataclass
class Snapshot:
    cpu: float | None = None            # percent
    ram: float | None = None            # percent
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    disk: float | None = None           # percent of the system drive
    disk_free_gb: float | None = None
    battery: float | None = None        # percent (None on desktops)
    charging: bool | None = None
    gpu: float | None = None            # percent utilization
    gpu_mem: float | None = None        # percent
    gpu_temp: float | None = None       # °C
    gpu_name: str | None = None
    jobs: list[dict] = field(default_factory=list)   # [{name, pct, eta_min}]
    ts: float = 0.0


# ------------------------------------------------------------------ parsers (pure, unit-tested)
def parse_nvidia_smi(line: str) -> dict | None:
    """'87, 8412, 12227, 64, NVIDIA GeForce RTX 5070' -> util %, mem %, temp, name."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        util, used, total, temp = (float(p) for p in parts[:4])
    except ValueError:
        return None
    return {"gpu": util, "gpu_mem": 100.0 * used / total if total else None, "gpu_temp": temp,
            "gpu_name": parts[4] if len(parts) > 4 else None}


_STEP = re.compile(r"step\s+(\d+)/(\d+).*?~(\d+) min left")


def parse_progress(log_tail: str) -> dict | None:
    """Last 'step N/M ... ~K min left' line -> {pct, eta_min}."""
    matches = _STEP.findall(log_tail)
    if not matches:
        return None
    n, m, eta = (int(x) for x in matches[-1])
    return {"pct": 100.0 * n / m if m else 0.0, "eta_min": eta, "step": n, "steps": m}


def cpu_percent(prev: tuple[int, int] | None, now: tuple[int, int]) -> float | None:
    """From (idle, total) tick counters at two moments."""
    if prev is None:
        return None
    idle, total = now[0] - prev[0], now[1] - prev[1]
    return max(0.0, min(100.0, 100.0 * (1 - idle / total))) if total > 0 else None


# ------------------------------------------------------------------ OS readers
def _cpu_ticks() -> tuple[int, int] | None:
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        val = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        return val(idle), val(kernel) + val(user)        # kernel time already includes idle
    try:
        f = open("/proc/stat").readline().split()[1:]
        nums = [int(x) for x in f]
        return nums[3] + nums[4], sum(nums)
    except OSError:
        return None


def _memory() -> tuple[float, float, float] | None:
    """(percent, used_gb, total_gb)"""
    if IS_WINDOWS:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        total, avail = m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
        return float(m.dwMemoryLoad), total - avail, total
    try:
        info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        total = int(info["MemTotal"].split()[0]) / 2**20
        avail = int(info["MemAvailable"].split()[0]) / 2**20
        return 100 * (1 - avail / total), total - avail, total
    except (OSError, KeyError, ValueError):
        return None


def _battery() -> tuple[float | None, bool | None]:
    if not IS_WINDOWS:
        return None, None
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None, None
    pct = s.BatteryLifePercent & 0xFF
    no_battery = (s.BatteryFlag & 0xFF) == 128 or pct == 255
    return (None, None) if no_battery else (float(pct), s.ACLineStatus == 1)


def _gpu() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        flags = 0x08000000 if IS_WINDOWS else 0          # CREATE_NO_WINDOW: no console flash
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                             creationflags=flags).stdout
        return parse_nvidia_smi(out.splitlines()[0]) if out.strip() else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def _tail(path: Path, n: int = 4000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def job_status(path: Path) -> dict | None:
    """State of a training log: running / finished / crashed, with progress and best score if present."""
    tail = _tail(path, 6000)
    if not tail:
        return None
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None
    prog = parse_progress(tail) or {}
    best = re.findall(r"best (?:score|exact-match|fact-check pass rate) = ([\d.]+)", tail)
    if best:
        return {"state": "finished", "best": float(best[-1]), "age_s": age, **prog}
    lines = tail.strip().splitlines()
    if "Traceback (most recent call last)" in tail or (lines and "Error:" in lines[-1]):
        return {"state": "crashed", "age_s": age, "error": tail.strip().splitlines()[-1][-200:], **prog}
    return {"state": "running", "age_s": age, **prog}


def job_progress(logs: dict[str, Path], active_within_s: float = 180) -> list[dict]:
    jobs = []
    for name, path in logs.items():
        try:
            if time.time() - path.stat().st_mtime > active_within_s:
                continue
        except OSError:
            continue
        prog = parse_progress(_tail(path))
        if prog:
            jobs.append({"name": name, **prog})
    return jobs


class MetricsSampler(threading.Thread):
    """Refreshes a Snapshot every `interval` seconds; the HUD reads .snapshot (atomic swap)."""

    def __init__(self, logs: dict[str, Path], interval: float = 2.0, gpu_every: int = 2):
        super().__init__(daemon=True, name="hud-metrics")
        self.logs, self.interval, self.gpu_every = logs, interval, gpu_every
        self.snapshot = Snapshot()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def sample_once(self, prev_ticks, i: int) -> tuple[Snapshot, tuple | None]:
        s = Snapshot(ts=time.time())
        ticks = _cpu_ticks()
        if ticks:
            s.cpu = cpu_percent(prev_ticks, ticks)
        mem = _memory()
        if mem:
            s.ram, s.ram_used_gb, s.ram_total_gb = mem
        try:
            du = shutil.disk_usage("C:\\" if IS_WINDOWS else "/")
            s.disk = 100.0 * du.used / du.total
            s.disk_free_gb = du.free / 2**30
        except OSError:
            pass
        s.battery, s.charging = _battery()
        old = self.snapshot
        g = _gpu() if i % self.gpu_every == 0 else None
        if g:
            s.gpu, s.gpu_mem, s.gpu_temp, s.gpu_name = g["gpu"], g["gpu_mem"], g["gpu_temp"], g["gpu_name"]
        else:
            s.gpu, s.gpu_mem, s.gpu_temp, s.gpu_name = old.gpu, old.gpu_mem, old.gpu_temp, old.gpu_name
        s.jobs = job_progress(self.logs)
        return s, ticks

    def run(self):
        prev, i = None, 0
        while not self._stop.is_set():
            snap, prev = self.sample_once(prev, i)
            if snap.cpu is None:
                snap.cpu = self.snapshot.cpu
            self.snapshot = snap
            i += 1
            self._stop.wait(self.interval)
