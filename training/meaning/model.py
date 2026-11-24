"""MeaningEncoder: a small transformer encoder (same blocks as the command model) + mean pooling ->
a unit vector per text. Queries and documents share it; similar meaning = high dot product."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import EncoderBlock, ModelConfig, RMSNorm

MAX_LEN = 128
DIM = 256


class MeaningEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, layers: int = 4, heads: int = 8, dim: int = DIM):
        super().__init__()
        self.cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model, n_heads=heads, n_enc=layers, n_dec=0,
                               d_ff=3 * d_model, max_src=MAX_LEN, max_tgt=1, dropout=0.1)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(self.cfg) for _ in range(layers)])
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, dim, bias=False)

    def forward(self, ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """ids (B,L) token ids, keep (B,L) 1.0 for real tokens / 0.0 for padding -> (B, DIM) unit vectors."""
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.tok(ids) * math.sqrt(self.cfg.d_model) + self.pos(pos)
        mask = (keep > 0.5)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        w = keep.unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)
