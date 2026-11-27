"""Batching, schedule and the stop/resume bookkeeping for Jarvis-LM. NumPy only, so it is tested without PyTorch.

Pretraining data is a set of uint16 token files (*.bin) written by training.lm.corpus. Batches are random windows
picked by a generator seeded with the step number, so a run stopped in the morning resumes exactly where it was
without saving any data position.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

IGNORE = -100


class TokenShards:
    def __init__(self, files: list[Path]):
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in sorted(files)]
        self.maps = [m for m in self.maps if len(m) > 2]
        if not self.maps:
            raise FileNotFoundError("no token files: run `python -m training.lm.corpus encode` first")
        self.sizes = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.total = int(self.sizes.sum())

    def batch(self, step: int, batch: int, ctx: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(inputs, targets) int64 arrays of shape (batch, ctx); targets are inputs shifted by one."""
        rng = np.random.default_rng((seed, step))
        p = np.maximum(self.sizes - ctx - 1, 0).astype(np.float64)
        shard = rng.choice(len(self.maps), size=batch, p=p / p.sum())
        x = np.empty((batch, ctx + 1), dtype=np.int64)
        for i, s in enumerate(shard):
            start = int(rng.integers(0, self.sizes[s] - ctx - 1))
            x[i] = self.maps[s][start:start + ctx + 1]
        return x[:, :-1], x[:, 1:]


def sft_batch(examples: list[tuple[list[int], list[int]]], pad: int = 0, max_len: int | None = None):
    """Chat examples [(ids, mask)] -> (inputs, targets) with the loss only on the reply tokens (others IGNORE)."""
    n = max(len(ids) for ids, _ in examples) - 1
    if max_len:
        n = min(n, max_len)
    x = np.full((len(examples), n), pad, dtype=np.int64)
    y = np.full((len(examples), n), IGNORE, dtype=np.int64)
    for i, (ids, mask) in enumerate(examples):
        ids, mask = ids[: n + 1], mask[: n + 1]
        x[i, : len(ids) - 1] = ids[:-1]
        tgt = np.array(ids[1:], dtype=np.int64)
        tgt[np.array(mask[1:]) == 0] = IGNORE
        y[i, : len(tgt)] = tgt
    return x, y


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    """Linear warmup, then cosine down to floor x peak at the end of the planned run."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def stop_time(until: str | None, now: datetime | None = None) -> float | None:
    """'07:30' -> the next 07:30 as a timestamp (tomorrow if it has passed). None/'' -> no limit."""
    if not until:
        return None
    now = now or datetime.now()
    h, m = (int(x) for x in until.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def eta(tokens_left: int, tokens_per_s: float) -> str:
    if tokens_per_s <= 0:
        return "unknown"
    h = tokens_left / tokens_per_s / 3600
    return f"{h:.1f} h (about {math.ceil(h / 8)} night{'s' if math.ceil(h / 8) != 1 else ''} of 8 h)"


class State:
    """checkpoints/lm_state.json: what the night-by-night runs have done so far (shown by /lm and train_lm.bat)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def update(self, **kw) -> None:
        self.d.update(kw, updated=time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def stage(self) -> str:
        """none | pretraining | pretrained | tuned"""
        return self.d.get("stage", "none")
