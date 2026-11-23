"""Framework-agnostic grammar-constrained beam search.

Used by the ONNX runtime (no PyTorch needed). The model is abstracted as a step function:

    step(seqs: list[list[int]]) -> np.ndarray of shape (len(seqs), vocab) with log-probabilities
                                    for the next token after each sequence

All sequences passed in one call have the same length (beams grow in lockstep).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

StepFn = Callable[[list[list[int]]], np.ndarray]


def constrained_beam(step: StepFn, bos: int, eos: int, k: int = 4, max_len: int = 159,
                     prefix_ok: Callable[[list[int]], bool] | None = None,
                     complete_ok: Callable[[list[int]], bool] | None = None,
                     ) -> list[tuple[float, list[int]]]:
    """Returns up to k (total_logprob, token_ids_without_bos_eos), best first (length-normalized).

    prefix_ok(ids)   -> False rejects a token that makes the output impossible to complete.
    complete_ok(ids) -> False rejects <eos> for an output that isn't a complete, valid program.
    """
    beams: list[tuple[float, list[int]]] = [(0.0, [bos])]
    finished: list[tuple[float, list[int]]] = []
    width = k * 4 if prefix_ok else k
    for _ in range(max_len):
        logp = step([b[1] for b in beams])
        cand: list[tuple[float, list[int]]] = []
        for bi, (score, ids) in enumerate(beams):
            row = logp[bi]
            n = min(width, row.shape[0])
            top = np.argpartition(-row, n - 1)[:n]
            top = top[np.argsort(-row[top])]
            kept = 0
            for tok in top.tolist():
                new = ids + [tok]
                if tok == eos:
                    if complete_ok and not complete_ok(new[1:-1]):
                        continue
                elif prefix_ok and not prefix_ok(new[1:]):
                    continue
                cand.append((score + float(row[tok]), new))
                kept += 1
                if kept == k:
                    break
        cand.sort(key=lambda c: c[0], reverse=True)
        beams = []
        for score, ids in cand:
            (finished if ids[-1] == eos else beams).append((score, ids))
            if len(beams) == k:
                break
        if not beams or len(finished) >= k:
            break
    finished += beams
    finished.sort(key=lambda c: c[0] / max(1, len(c[1])), reverse=True)
    return [(s, [t for t in ids[1:] if t != eos]) for s, ids in finished[:k]]


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
