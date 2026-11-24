"""One way to open ONNX models (phase 41): the int8 copy when there is one, lean session settings, timings.

    session(path)      -> onnxruntime.InferenceSession  (uses <name>.int8.onnx when present and allowed)
    pick(path)         -> the file that will actually be loaded

int8 copies are made by training/quantize.py (and by every export) and only kept if they agree with the
full-precision model; delete them, or set use_int8 = false in config.json, to go back.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

USE_INT8 = True                 # set from config at start-up


def int8_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".int8" + path.suffix)


def pick(path: Path, int8: bool | None = None) -> Path:
    q = int8_path(path)
    return q if (USE_INT8 if int8 is None else int8) and q.exists() else Path(path)


def session(path: Path, threads: int | None = None, int8: bool | None = None, name: str | None = None):
    import onnxruntime as ort
    from .perf import PERF
    f = pick(path, int8)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = False        # memory goes back to Windows between commands: lower idle RAM
    t = time.perf_counter()
    s = ort.InferenceSession(str(f), opts, providers=["CPUExecutionProvider"])
    PERF.model_loaded(name or Path(path).stem, f, time.perf_counter() - t)
    return s
