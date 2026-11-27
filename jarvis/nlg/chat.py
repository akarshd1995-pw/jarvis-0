"""Runs Jarvis-LM (phase 27) with NumPy only: a decoder-only transformer with a key/value cache.

The trainer exports plain weights (checkpoints/lm_chat.npz) plus lm_chat.json (sizes, evaluation, ready flag);
there is no ONNX step to get wrong, and this file is the reference the PyTorch model is tested against.
Architecture (must match training/lm/model.py): RMSNorm, rotary positions, SwiGLU, tied input/output embedding,
no biases.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from training.tokenizer import BOS, EOS, PAD, SEP, BPETokenizer

from .chatfmt import encode_dialog, header, parse_reply, problems

log = logging.getLogger(__name__)


class Decoder:
    def __init__(self, weights: dict, d_model: int, n_layer: int, n_head: int, rope_theta: float = 10000.0,
                 eps: float = 1e-5):
        f = lambda k: np.asarray(weights[k], dtype=np.float32)            # noqa: E731
        self.emb = f("tok_emb")
        self.layers = [{k: f(f"l{i}.{k}") for k in ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w1", "w3", "w2")}
                       for i in range(n_layer)]
        self.norm = f("norm")
        self.d, self.h, self.eps = d_model, n_head, eps
        self.dh = d_model // n_head
        self.inv_freq = 1.0 / (rope_theta ** (np.arange(0, self.dh, 2, dtype=np.float32) / self.dh))

    def _rms(self, x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w

    def _rope(self, x, pos):                                  # x: (H, T, Dh)
        ang = pos[:, None].astype(np.float32) * self.inv_freq[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        half = self.dh // 2
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def forward(self, ids: list[int], cache: list | None = None, all_logits: bool = False):
        """ids: new tokens. cache: [(k, v)] per layer from earlier calls (None = start). Returns (logits, cache).
        The cache passed in is never modified, so one prefill can feed several sampled replies."""
        past = 0 if not cache else cache[0][0].shape[1]
        t = len(ids)
        pos = np.arange(past, past + t)
        x = self.emb[np.asarray(ids)]
        mask = np.triu(np.full((t, past + t), -np.inf, dtype=np.float32), k=past + 1)
        new_cache = []
        for i, L in enumerate(self.layers):
            h = self._rms(x, L["attn_norm"])
            q = (h @ L["wq"]).reshape(t, self.h, self.dh).transpose(1, 0, 2)
            k = (h @ L["wk"]).reshape(t, self.h, self.dh).transpose(1, 0, 2)
            v = (h @ L["wv"]).reshape(t, self.h, self.dh).transpose(1, 0, 2)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache:
                k = np.concatenate([cache[i][0], k], axis=1)
                v = np.concatenate([cache[i][1], v], axis=1)
            new_cache.append((k, v))
            att = q @ k.transpose(0, 2, 1) / np.sqrt(self.dh) + mask
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            x = x + (att @ v).transpose(1, 0, 2).reshape(t, self.d) @ L["wo"]
            h = self._rms(x, L["mlp_norm"])
            a = h @ L["w1"]
            x = x + ((a / (1.0 + np.exp(-a))) * (h @ L["w3"])) @ L["w2"]
        x = self._rms(x, self.norm)
        logits = (x if all_logits else x[-1:]) @ self.emb.T
        return (logits if all_logits else logits[0]), new_cache


def sample(logits: np.ndarray, rng, temperature: float, top_p: float, recent: list[int], penalty: float = 1.15) -> int:
    z = logits.astype(np.float64).copy()
    z[[PAD, BOS]] = -np.inf
    for tok in set(recent):
        z[tok] = z[tok] / penalty if z[tok] > 0 else z[tok] * penalty
    if temperature <= 0:
        return int(np.argmax(z))
    z = (z - z.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[: max(1, int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1)]
    q = p[keep] / p[keep].sum()
    return int(keep[rng.choice(len(keep), p=q)])


class ChatModel:
    def __init__(self, decoder: Decoder, tok: BPETokenizer, max_ctx: int = 512, seed: int | None = None,
                 meta: dict | None = None):
        self.dec, self.tok, self.max_ctx = decoder, tok, max_ctx
        self.rng = np.random.default_rng(seed)
        self.meta = meta or {}
        self.stats = {"replies": 0, "rejected": 0, "web": 0, "cmd": 0, "ms": 0.0}

    @classmethod
    def load(cls, folder: Path, require_ready: bool = True) -> "ChatModel | None":
        folder = Path(folder)
        meta_p, w_p, tok_p = folder / "lm_chat.json", folder / "lm_chat.npz", folder / "lm_tokenizer.json"
        if not (meta_p.exists() and w_p.exists() and tok_p.exists()):
            return None
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if require_ready and not meta.get("ready"):
            log.info("chat model trained but below the quality bar; using the fixed small-talk lines")
            return None
        with np.load(w_p) as w:
            dec = Decoder(dict(w), meta["d_model"], meta["n_layer"], meta["n_head"], meta.get("rope_theta", 10000.0))
        return cls(dec, BPETokenizer.load(tok_p), max_ctx=min(int(meta.get("max_ctx", 512)), 512), meta=meta)

    def generate(self, ids: list[int], cache=None, logits=None, max_new: int = 48, temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        if logits is None:
            logits, cache = self.dec.forward(ids)
        out: list[int] = []
        for _ in range(max_new):
            tok = sample(logits, self.rng, temperature, top_p, out[-24:])
            if tok in (EOS, SEP):
                break
            out.append(tok)
            logits, cache = self.dec.forward([tok], cache)
        return self.tok.decode(out).strip()

    def reply(self, user_text: str, history: list[tuple[str, str]] | None = None, name: str = "Jarvis",
              address: str = "sir", tone: str = "butler", hour: int | None = None,
              tries: int = 3) -> tuple[str, str] | None:
        """('say', text) | ('web', '') | ('cmd', '') | None if no candidate passed the filter."""
        t0 = time.perf_counter()
        hour = time.localtime().tm_hour if hour is None else hour
        max_new = 48
        ids, _ = encode_dialog(self.tok, header(name, address, tone, hour), list(history or []) + [("U", user_text)],
                               open_reply=True, max_len=self.max_ctx - max_new)
        logits, cache = self.dec.forward(ids)                       # read the conversation once
        try:
            for i in range(tries):
                text = self.generate(ids, cache, logits, max_new, temperature=0.3 if i == 0 else 0.8)
                kind, body = parse_reply(text)
                if kind != "say":
                    if i == 0:                                      # the model's most likely reading decides
                        self.stats[kind] += 1
                        return kind, ""
                    continue
                if not problems(body):
                    self.stats["replies"] += 1
                    return "say", body
                self.stats["rejected"] += 1
            return None
        finally:
            self.stats["ms"] = round((time.perf_counter() - t0) * 1000, 1)
