"""Jarvis-LM in PyTorch: a decoder-only transformer trained from scratch.

Must stay identical to the NumPy runtime in jarvis/nlg/chat.py (tests/test_lm.py checks the two agree):
RMSNorm (eps 1e-5), rotary positions (rotate-half), SwiGLU, tied input/output embedding, no biases.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim: int, ctx: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ang = torch.outer(torch.arange(ctx, dtype=torch.float32), inv)             # (ctx, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, Dh); cos/sin: (T, Dh/2). Same pairing as the runtime: first half with second half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        d, h = c.d_model, c.hidden
        self.n_head, self.head_dim = c.n_head, d // c.n_head
        self.attn_norm, self.mlp_norm = RMSNorm(d), RMSNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.w1, self.w3 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.attn_norm(x)
        q, k, v = (w(h).view(b, t, self.n_head, self.head_dim).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(o.transpose(1, 2).reshape(b, t, d))
        h = self.mlp_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class JarvisLM(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        cos, sin = rope_tables(c.d_model // c.n_head, c.max_ctx, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for L in self.layers:                    # residual projections start small (GPT-2 style)
            for w in (L.wo, L.w2):
                nn.init.normal_(w.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for L in self.layers:
            x = L(x, cos, sin)
        logits = F.linear(self.norm(x), self.tok_emb.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def param_groups(self, weight_decay: float):
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        rest = [p for n, p in self.named_parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def unwrap(model):
    return getattr(model, "_orig_mod", model)                # torch.compile wraps the module


def to_numpy_weights(model) -> dict:
    """The runtime's layout: matrices as (in, out) so it can do x @ W."""
    sd = {k.replace("_orig_mod.", ""): v.detach().float().cpu().numpy() for k, v in unwrap(model).state_dict().items()}
    out = {"tok_emb": sd["tok_emb.weight"], "norm": sd["norm.weight"]}
    for i in range(unwrap(model).cfg.n_layer):
        p = f"layers.{i}."
        out[f"l{i}.attn_norm"], out[f"l{i}.mlp_norm"] = sd[p + "attn_norm.weight"], sd[p + "mlp_norm.weight"]
        for k in ("wq", "wk", "wv", "wo", "w1", "w3", "w2"):
            out[f"l{i}.{k}"] = sd[p + k + ".weight"].T.copy()
    return out


def export_npz(model, path: Path, dtype=np.float16) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: v.astype(dtype) for k, v in to_numpy_weights(model).items()})
    os.replace(tmp, path)
    return path


def save_checkpoint(path: Path, model, opt=None, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": unwrap(model).state_dict(), "opt": opt.state_dict() if opt is not None else None,
                "cfg": unwrap(model).cfg.to_dict(), **extra}, tmp)
    os.replace(tmp, path)                     # never a half-written checkpoint, even if the PC switches off


def load_checkpoint(path: Path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = JarvisLM(LMConfig(**ck["cfg"]))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return model.to(device), ck
