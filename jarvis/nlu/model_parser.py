"""Runs the trained intent model inside the daemon.

Decoding = grammar-constrained beam search: at every step, tokens that would make the
output impossible to complete as a valid program are discarded (jarvis.dsl.grammar.viable_prefix),
and <eos> is only accepted for complete programs. Invalid output can never be produced.
"""
from __future__ import annotations

from pathlib import Path

import math

import torch

from training.model import IntentTransformer, ModelConfig
from training.tokenizer import BOS, EOS, PAD, BPETokenizer

from ..dsl.grammar import DSLError, Program, is_complete, parse, viable_prefix


class ModelParser:
    def __init__(self, checkpoint: Path, tokenizer: Path, beam: int = 4, min_confidence: float = 0.55):
        self.tok = BPETokenizer.load(tokenizer)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = IntentTransformer(ModelConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.beam = beam
        # geometric-mean token probability below this -> treat as "not sure" and fall back
        self.min_confidence = min_confidence
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    def parse(self, text: str) -> tuple[Program | None, float]:
        """Returns (program or None, confidence in 0..1)."""
        ids = self.tok.encode(text)[: self.model.cfg.max_src]
        if not ids:
            return None, 0.0
        src = torch.tensor([ids])
        pad = src == PAD
        decode = self.tok.decode
        results = self.model.beam(src, pad, BOS, EOS, k=self.beam,
                                  prefix_ok=lambda ids: viable_prefix(decode(ids)),
                                  complete_ok=lambda ids: is_complete(decode(ids)))
        for score, out in results:
            conf = math.exp(score / max(1, len(out) + 1))
            try:
                prog = parse(self.tok.decode(out))
            except DSLError:
                continue
            return (prog if conf >= self.min_confidence else None), conf
        return None, 0.0
