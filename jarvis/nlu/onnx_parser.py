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


class ONNXParser:
    def __init__(self, encoder: Path, decoder: Path, tokenizer: Path, beam: int = 4,
                 min_confidence: float = 0.55, max_src: int = 160, max_tgt: int = 160,
                 sessions: tuple | None = None):
        self.tok = BPETokenizer.load(tokenizer)
        if sessions is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            providers = ["CPUExecutionProvider"]
            sessions = (ort.InferenceSession(str(encoder), opts, providers=providers),
                        ort.InferenceSession(str(decoder), opts, providers=providers))
        self.enc, self.dec = sessions
        self.knows = None
        try:
            meta = json.loads((Path(encoder).parent / "intent_onnx.json").read_text(encoding="utf-8"))
            self.knows = meta.get("knows")
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

    def parse(self, text: str) -> tuple[Program | None, float]:
        """Returns (program or None, confidence in 0..1)."""
        ids = self.tok.encode(text)[: self.max_src]
        if not ids:
            return None, 0.0
        mem, pad = self._encode(ids)

        def step(seqs: list[list[int]]) -> np.ndarray:
            n = len(seqs)
            return self.dec.run(["logprobs"], {"tgt_in": np.array(seqs, dtype=np.int64),
                                               "mem": np.repeat(mem, n, axis=0),
                                               "src_pad": np.repeat(pad, n, axis=0)})[0]

        decode = self.tok.decode
        results = []
        for guard in (True, False):      # copy guard first; if it leaves no valid output, decode without it
            results = constrained_beam(step, BOS, EOS, k=self.beam, max_len=self.max_tgt - 1,
                                       prefix_ok=(lambda t: viable_prefix(decode(t)) and copy_ok(decode(t), text))
                                       if guard else (lambda t: viable_prefix(decode(t))),
                                       complete_ok=lambda t: is_complete(decode(t)))
            if results:
                break
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
