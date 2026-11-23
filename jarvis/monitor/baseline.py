"""A small model of YOUR normal system load, learned on this PC from your own history.

For each metric and each hour of the day it keeps an exponentially weighted mean and variance
(trained online, sample by sample). A reading is anomalous when it sits far outside what is normal
for that hour, for several samples in a row, and is also high in absolute terms. Persisted as JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

METRICS = ["cpu", "ram", "gpu", "gpu_temp"]
FLOOR = {"cpu": 70.0, "ram": 80.0, "gpu": 80.0, "gpu_temp": 75.0}   # never flag readings below these


@dataclass
class Stat:
    mean: float = 0.0
    var: float = 25.0
    n: int = 0

    def update(self, x: float, alpha: float) -> None:
        if self.n == 0:
            self.mean, self.n = x, 1
            return
        d = x - self.mean
        self.mean += alpha * d
        self.var = (1 - alpha) * (self.var + alpha * d * d)
        self.n += 1

    def z(self, x: float) -> float:
        return (x - self.mean) / math.sqrt(max(self.var, 4.0))


@dataclass
class Baseline:
    alpha: float = 0.02
    min_samples: int = 120           # per hour slot, before it is trusted (a few days of normal use)
    z_alert: float = 4.0
    persist: int = 3                 # consecutive anomalous samples needed
    stats: dict = field(default_factory=dict)
    _streak: dict = field(default_factory=dict)

    def _slot(self, metric: str, hour: int) -> Stat:
        return self.stats.setdefault(f"{metric}@{hour}", Stat())

    def observe(self, values: dict[str, float | None], hour: int) -> list[tuple[str, float, float]]:
        """Learn from one sample; return [(metric, value, typical)] that are anomalous right now."""
        out = []
        for m in METRICS:
            x = values.get(m)
            if x is None:
                continue
            st = self._slot(m, hour)
            anomalous = st.n >= self.min_samples and x >= FLOOR[m] and st.z(x) >= self.z_alert
            self._streak[m] = self._streak.get(m, 0) + 1 if anomalous else 0
            if self._streak[m] == self.persist:
                out.append((m, x, st.mean))
            if not anomalous:            # don't let an incident become the new normal
                st.update(x, self.alpha)
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: [v.mean, v.var, v.n] for k, v in self.stats.items()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, **kw) -> "Baseline":
        b = cls(**kw)
        try:
            for k, (m, v, n) in json.loads(path.read_text(encoding="utf-8")).items():
                b.stats[k] = Stat(m, v, n)
        except (OSError, ValueError):
            pass
        return b
