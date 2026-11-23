"""Watches the HUD's metric snapshots and the training logs, and raises alerts.

Rules (thresholds editable in settings):
  GPU temperature, low disk space, low battery (not charging), sustained high RAM
  training run finished / crashed / stalled
  unusual load compared with YOUR normal for this hour (learned baseline, see baseline.py)
Each alert has a key and a cooldown so you're not spammed.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..hud.metrics import MetricsSampler, Snapshot, job_status
from .baseline import Baseline


@dataclass
class Alert:
    key: str                 # e.g. gpu_hot, disk_low, training_done:Training command model
    kind: str                # template key suffix: gpu_hot, disk_low, battery_low, ram_high, training_done, ...
    level: str               # info | warning | critical
    data: dict = field(default_factory=dict)


@dataclass
class Thresholds:
    gpu_temp: float = 85.0
    disk_free_gb: float = 10.0
    battery: float = 20.0
    ram: float = 92.0
    stall_min: float = 10.0
    cooldown_min: float = 30.0


class Monitor:
    def __init__(self, sampler: MetricsSampler, logs: dict[str, Path], on_alert: Callable[[Alert], None],
                 thresholds: Thresholds | None = None, baseline_path: Path | None = None, store=None):
        self.sampler, self.logs, self.on_alert = sampler, logs, on_alert
        self.t = thresholds or Thresholds()
        self.baseline_path = baseline_path
        self.baseline = Baseline.load(baseline_path) if baseline_path else Baseline()
        self.store = store
        self._last_fired: dict[str, float] = {}
        self._ram_streak = 0
        self._job_seen: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ evaluation (pure: easy to test)
    def evaluate(self, s: Snapshot, now: float, hour: int) -> list[Alert]:
        alerts: list[Alert] = []
        t = self.t
        if s.gpu_temp is not None and s.gpu_temp >= t.gpu_temp:
            alerts.append(Alert("gpu_hot", "gpu_hot", "critical", {"temp": round(s.gpu_temp),
                                                                  "training": self._training_running()}))
        if s.disk_free_gb is not None and s.disk_free_gb < t.disk_free_gb:
            alerts.append(Alert("disk_low", "disk_low", "warning", {"free_gb": round(s.disk_free_gb, 1)}))
        if s.battery is not None and s.charging is False and s.battery <= t.battery:
            alerts.append(Alert("battery_low", "battery_low", "warning", {"pct": round(s.battery)}))
        self._ram_streak = self._ram_streak + 1 if (s.ram or 0) >= t.ram else 0
        if self._ram_streak == 15:                       # ~30 s of sustained pressure
            alerts.append(Alert("ram_high", "ram_high", "warning", {"pct": round(s.ram)}))
        for name, path in self.logs.items():
            alerts += self._job_alerts(name, path)
        for metric, value, typical in self.baseline.observe(
                {"cpu": s.cpu, "ram": s.ram, "gpu": s.gpu, "gpu_temp": s.gpu_temp}, hour):
            alerts.append(Alert(f"unusual_{metric}", "unusual", "info",
                                {"metric": metric, "value": round(value), "typical": round(typical)}))
        return [a for a in alerts if self._cooldown_ok(a, now)]

    def _job_alerts(self, name: str, path: Path) -> list[Alert]:
        st = job_status(path) if path.exists() else None
        if not st:
            return []
        prev = self._job_seen.get(name)
        state = st["state"]
        if state == "running" and (path.parent / "PAUSE").exists():
            state = "paused"                               # paused on purpose: not a stall
        elif state == "running" and st["age_s"] > self.t.stall_min * 60 and st.get("step"):
            state = "stalled"
        self._job_seen[name] = state
        if prev is None or prev == state:          # first sight or no change: no alert
            return []
        if state == "finished":
            return [Alert(f"training_done:{name}", "training_done", "info", {"name": name, "best": st.get("best")})]
        if state == "crashed":
            return [Alert(f"training_crashed:{name}", "training_crashed", "critical",
                          {"name": name, "error": st.get("error", "")})]
        if state == "stalled":
            return [Alert(f"training_stalled:{name}", "training_stalled", "warning",
                          {"name": name, "minutes": round(st["age_s"] / 60)})]
        return []

    def _training_running(self) -> bool:
        return any(v == "running" for v in self._job_seen.values())

    def _cooldown_ok(self, a: Alert, now: float) -> bool:
        last = self._last_fired.get(a.key)
        if last is not None and now - last < self.t.cooldown_min * 60:
            return False
        self._last_fired[a.key] = now
        return True

    # ------------------------------------------------------------------ background loop
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(self.sampler.interval):
            s = self.sampler.snapshot
            if not s.ts:
                continue
            now = time.time()
            for a in self.evaluate(s, now, dt.datetime.now().hour):
                try:
                    self.on_alert(a)
                except Exception:
                    pass
            i += 1
            if i % 30 == 0:                                   # about once a minute
                self._persist(s, now)

    def _persist(self, s: Snapshot, now: float) -> None:
        if self.baseline_path:
            try:
                self.baseline.save(self.baseline_path)
            except OSError:
                pass
        if self.store is not None:                            # history for the habit model (phase 10)
            self.store.log_event("metrics", {"cpu": s.cpu, "ram": s.ram, "gpu": s.gpu, "gpu_temp": s.gpu_temp,
                                             "disk_free_gb": s.disk_free_gb, "battery": s.battery}, ts=now)
