"""Runs the exported intent model with onnxruntime + numpy. No PyTorch required.

Same behaviour as model_parser.ModelParser: grammar-constrained beam search, confidence gate.
"""
from __future__ import annotations

import math
from pathlib import Path

import json

import numpy as np

from training.tokenizer import BOS, EOS, PAD, BPETokenizer

from ..dsl.grammar import DSLError, Program, is_complete, parse, viable_prefix
from .beam import constrained_beam
from .copy_guard import copy_ok


class OverBudget(Exception):
    """The model took longer than this command's time budget (phase 41)."""


class ONNXParser:
    def __init__(self, encoder: Path, decoder: Path, tokenizer: Path, beam: int = 4,
                 min_confidence: float = 0.55, max_src: int = 160, max_tgt: int = 160,
                 sessions: tuple | None = None, meta: Path | None = None):
        self.tok = BPETokenizer.load(tokenizer)
        if sessions is None:
            from ..onnxutil import session
            name = Path(encoder).stem.replace("_encoder", "")
            sessions = (session(encoder, name=f"{name} model (encoder)"),
                        session(decoder, name=f"{name} model (decoder)"))
        self.enc, self.dec = sessions
        self.knows = None
        try:
            info = json.loads((Path(meta) if meta else Path(encoder).parent / "intent_onnx.json").read_text(encoding="utf-8"))
            self.knows = info.get("knows")
        except (OSError, ValueError, TypeError):
            pass
        self.beam, self.min_confidence = beam, min_confidence
        self.last_alts: list[tuple[float, str]] = []
        self.max_src, self.max_tgt = max_src, max_tgt

    def _encode(self, ids: list[int]):
        src = np.array([ids], dtype=np.int64)
        pad = src == PAD
        mem = self.enc.run(["mem"], {"src": src, "src_pad": pad})[0]
        return mem, pad

    def first_step_logprobs(self, ids: list[int]) -> np.ndarray:
        mem, pad = self._encode(ids)
        return self.dec.run(["logprobs"], {"tgt_in": np.array([[BOS]], dtype=np.int64),
                                           "mem": mem, "src_pad": pad})[0]

    def parse(self, text: str, budget_s: float | None = None) -> tuple[Program | None, float]:
        """Returns (program or None, confidence in 0..1). Past budget_s: (None, 0.0) and self.timed_out."""
        import time
        self.timed_out = False
        deadline = time.perf_counter() + budget_s if budget_s else None
        ids = self.tok.encode(text)[: self.max_src]
        if not ids:
            return None, 0.0
        mem, pad = self._encode(ids)

        def step(seqs: list[list[int]]) -> np.ndarray:
            if deadline is not None and time.perf_counter() > deadline:
                raise OverBudget()
            n = len(seqs)
            return self.dec.run(["logprobs"], {"tgt_in": np.array(seqs, dtype=np.int64),
                                               "mem": np.repeat(mem, n, axis=0),
                                               "src_pad": np.repeat(pad, n, axis=0)})[0]

        decode = self.tok.decode
        results = []
        try:
            for guard in (True, False):  # copy guard first; if it leaves no valid output, decode without it
                results = constrained_beam(step, BOS, EOS, k=self.beam, max_len=self.max_tgt - 1,
                                           prefix_ok=(lambda t: viable_prefix(decode(t)) and copy_ok(decode(t), text))
                                           if guard else (lambda t: viable_prefix(decode(t))),
                                           complete_ok=lambda t: is_complete(decode(t)))
                if results:
                    break
        except OverBudget:
            self.timed_out, self.last_alts = True, []
            return None, 0.0
        self.last_alts = []                       # phase 39: every valid reading, best first
        best = None
        for score, out in results:
            conf = math.exp(score / max(1, len(out) + 1))
            try:
                prog = parse(decode(out))
            except DSLError:
                continue
            self.last_alts.append((conf, str(prog)))
            if best is None:
                best = ((prog if conf >= self.min_confidence else None), conf)
        return best or (None, 0.0)
