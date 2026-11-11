"""Runs the trained intent model inside the daemon.

Decoding = beam search + validation: candidates are tried best-first and the first one
that parses and type-checks against the DSL schema wins. Invalid output never executes.
(Next step: true grammar-constrained decoding that masks invalid tokens during search.)
"""
from __future__ import annotations

from pathlib import Path

import torch

from training.model import IntentTransformer, ModelConfig
from training.tokenizer import BOS, EOS, PAD, BPETokenizer

from ..dsl.grammar import DSLError, Program, parse


class ModelParser:
    def __init__(self, checkpoint: Path, tokenizer: Path, beam: int = 4):
        self.tok = BPETokenizer.load(tokenizer)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = IntentTransformer(ModelConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.beam = beam
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    def parse(self, text: str) -> Program | None:
        ids = self.tok.encode(text)[: self.model.cfg.max_src]
        if not ids:
            return None
        src = torch.tensor([ids])
        pad = src == PAD
        for _score, out in self.model.beam(src, pad, BOS, EOS, k=self.beam):
            try:
                return parse(self.tok.decode(out))
            except DSLError:
                continue
        return None
