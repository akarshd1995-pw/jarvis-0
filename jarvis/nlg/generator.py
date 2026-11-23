"""Runs the trained reply model (ONNX, no PyTorch) and keeps only fact-checked sentences.

generate() samples a few candidate sentences in one batch (top-k sampling for variety), returns the
first that passes the fact checker, else tries greedy, else returns None so the persona uses its template.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from training.tokenizer import BOS, EOS, PAD, BPETokenizer

from .checker import check, linearize


class ReplyModel:
    def __init__(self, encoder: Path | None, decoder: Path | None, tokenizer: Path, samples: int = 3,
                 temperature: float = 0.8, top_k: int = 8, max_src: int = 192, max_tgt: int = 128,
                 seed: int | None = None, sessions: tuple | None = None):
        self.tok = BPETokenizer.load(tokenizer)
        if sessions is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            sessions = (ort.InferenceSession(str(encoder), opts, providers=["CPUExecutionProvider"]),
                        ort.InferenceSession(str(decoder), opts, providers=["CPUExecutionProvider"]))
        self.enc, self.dec = sessions
        self.samples, self.temperature, self.top_k = samples, temperature, top_k
        self.max_src, self.max_tgt = max_src, max_tgt
        self.rng = np.random.default_rng(seed)
        self.stats = {"generated": 0, "rejected": 0}

    def _decode_batch(self, ids: list[int], n: int, greedy: bool) -> list[str]:
        src = np.array([ids], dtype=np.int64)
        pad = src == PAD
        mem = self.enc.run(["mem"], {"src": src, "src_pad": pad})[0]
        mem, pad = np.repeat(mem, n, axis=0), np.repeat(pad, n, axis=0)
        seqs = [[BOS] for _ in range(n)]
        done = [False] * n
        for _ in range(self.max_tgt - 1):
            lp = self.dec.run(["logprobs"], {"tgt_in": np.array(seqs, dtype=np.int64), "mem": mem, "src_pad": pad})[0]
            for i in range(n):
                if done[i]:
                    seqs[i].append(EOS)
                    continue
                row = lp[i]
                if greedy:
                    tok = int(row.argmax())
                else:
                    top = np.argpartition(-row, self.top_k - 1)[: self.top_k]
                    logits = row[top] / max(1e-3, self.temperature)
                    probs = np.exp(logits - logits.max())
                    tok = int(top[self.rng.choice(len(top), p=probs / probs.sum())])
                seqs[i].append(tok)
                done[i] = tok == EOS
            if all(done):
                break
        out = []
        for s in seqs:
            s = s[1:]
            out.append(self.tok.decode(s[: s.index(EOS)] if EOS in s else s).strip())
        return out

    def generate(self, key: str, fields: dict, tone: str, address: str) -> str | None:
        clean = {k: v for k, v in fields.items() if k[:1].islower()}
        ids = self.tok.encode(linearize(key, clean, tone, address))[: self.max_src]
        for greedy, n in ((False, self.samples), (True, 1)):
            for text in self._decode_batch(ids, n, greedy):
                self.stats["generated"] += 1
                if check(key, clean, tone, address, text):
                    return text
                self.stats["rejected"] += 1
        return None
