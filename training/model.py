"""Encoder-decoder transformer for text -> DSL, written from scratch in plain PyTorch.

Pre-norm blocks, RMSNorm, SwiGLU feed-forward, learned positions, tied embeddings.
Default config is ~10M parameters: trains in well under an hour on an RTX 5070.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_enc: int = 4
    n_dec: int = 4
    d_ff: int = 768          # SwiGLU hidden (x3 matrices ~= 1024 classic FFN)
    max_src: int = 160
    max_tgt: int = 160
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, cross: bool = False):
        super().__init__()
        self.h, self.d = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.cross = cross
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d).transpose(1, 2)

    def forward(self, x, ctx=None, mask=None, causal=False):
        ctx = x if ctx is None else ctx
        q = self._split(self.q(x))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        if causal:
            t = x.size(1)
            causal_mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            mask = causal_mask if mask is None else (mask & causal_mask)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop if self.training else 0.0)
        b, _, t, _ = y.shape
        return self.o(y.transpose(1, 2).reshape(b, t, self.h * self.d))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ff = RMSNorm(cfg.d_model), SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, src_mask):
        x = x + self.drop(self.attn(self.n1(x), mask=src_mask))
        return x + self.ff(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.self_attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.cross_attn = RMSNorm(cfg.d_model), Attention(cfg, cross=True)
        self.n3, self.ff = RMSNorm(cfg.d_model), SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, y, mem, mem_mask):
        y = y + self.drop(self.self_attn(self.n1(y), causal=True))
        y = y + self.drop(self.cross_attn(self.n2(y), ctx=mem, mask=mem_mask))
        return y + self.ff(self.n3(y))


class IntentTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_src = nn.Embedding(cfg.max_src, cfg.d_model)
        self.pos_tgt = nn.Embedding(cfg.max_tgt, cfg.d_model)
        self.enc = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_enc))
        self.dec = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_dec))
        self.enc_norm, self.dec_norm = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)
        for name, p in self.named_parameters():     # scaled init for residual projections
            if name.endswith(("o.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * (cfg.n_enc + cfg.n_dec)))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, src, src_pad):
        """src: (B,S) ids; src_pad: (B,S) bool True where padding."""
        pos = torch.arange(src.size(1), device=src.device)
        x = self.drop(self.tok(src) * math.sqrt(self.cfg.d_model) + self.pos_src(pos))
        mask = (~src_pad)[:, None, None, :]          # (B,1,1,S) True = attend
        for blk in self.enc:
            x = blk(x, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt_in, mem, mem_mask):
        pos = torch.arange(tgt_in.size(1), device=tgt_in.device)
        y = self.drop(self.tok(tgt_in) * math.sqrt(self.cfg.d_model) + self.pos_tgt(pos))
        for blk in self.dec:
            y = blk(y, mem, mem_mask)
        return self.head(self.dec_norm(y))

    def forward(self, src, src_pad, tgt_in):
        mem, mem_mask = self.encode(src, src_pad)
        return self.decode(tgt_in, mem, mem_mask)

    @torch.no_grad()
    def greedy(self, src, src_pad, bos: int, eos: int, max_len: int = 128):
        mem, mem_mask = self.encode(src, src_pad)
        out = torch.full((src.size(0), 1), bos, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(min(max_len, self.cfg.max_tgt - 1)):
            nxt = self.decode(out, mem, mem_mask)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == eos
            if done.all():
                break
        return out[:, 1:]

    @torch.no_grad()
    def beam(self, src, src_pad, bos: int, eos: int, k: int = 4, max_len: int = 128):
        """Beam search for a single example; returns list of (logprob, ids) best-first."""
        mem, mem_mask = self.encode(src, src_pad)
        beams = [(0.0, [bos])]
        finished = []
        for _ in range(min(max_len, self.cfg.max_tgt - 1)):
            seqs = torch.tensor([b[1] for b in beams], device=src.device)
            logp = self.decode(seqs, mem.expand(len(beams), -1, -1),
                               mem_mask.expand(len(beams), -1, -1, -1))[:, -1].log_softmax(-1)
            cand = []
            for bi, (score, ids) in enumerate(beams):
                top = logp[bi].topk(k)
                for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((score + lp, ids + [tok]))
            cand.sort(key=lambda c: c[0], reverse=True)
            beams = []
            for score, ids in cand:
                (finished if ids[-1] == eos else beams).append((score, ids))
                if len(beams) == k:
                    break
            if not beams or len(finished) >= k:
                break
        finished += beams
        # length-normalized ranking
        finished.sort(key=lambda c: c[0] / max(1, len(c[1])), reverse=True)
        return [(s, [t for t in ids[1:] if t != eos]) for s, ids in finished[:k]]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
