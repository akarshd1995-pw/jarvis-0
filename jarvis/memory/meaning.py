"""Meaning vectors for memory search (phase 7 step 3), with our own meaning model on the CPU (ONNX).

Each indexed item gets up to 3 vectors (title + its first passages). A query's vector is compared with all of
them; the best matches are merged with the word-search results (reciprocal rank fusion) in MemoryIndex.search.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_LEN = 128


class Embedder:
    def __init__(self, session, tok):
        self.session, self.tok = session, tok

    @classmethod
    def load(cls, ck: Path, threads: int = 2) -> "Embedder | None":
        m, t = ck / "meaning.onnx", ck / "meaning_tokenizer.json"
        if not (m.exists() and t.exists()):
            return None
        from training.tokenizer import BPETokenizer
        from ..onnxutil import session
        return cls(session(m, threads=threads, name="meaning model"), BPETokenizer.load(t))

    def embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            ids = [self.tok.encode(t.lower())[:MAX_LEN] or [0] for t in texts[i:i + batch]]
            L = max(len(x) for x in ids)
            a = np.zeros((len(ids), L), np.int64)
            k = np.zeros((len(ids), L), np.float32)
            for j, x in enumerate(ids):
                a[j, : len(x)] = x
                k[j, : len(x)] = 1.0
            out.append(self.session.run(["vec"], {"ids": a, "keep": k})[0])
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 256), np.float32)


def chunks(title: str, body: str, n: int = 3, size: int = 400) -> list[str]:
    """What gets a vector: the title on its own words, then title + the first passages."""
    import re
    t = re.sub(r"[_\-.()\[\]]+", " ", re.sub(r"\.[A-Za-z0-9]{1,5}$", "", title or "")).strip()
    out = [t] if t else []
    words = (body or "").split()
    i = 0
    while words[i:] and len(out) < n:
        piece = []
        while words[i:] and sum(len(w) + 1 for w in piece) < size:
            piece.append(words[i])
            i += 1
        out.append((t + " " + " ".join(piece)).strip())
    return out or [title or ""]


_loaded: dict = {}


last_used = 0.0           # phase 41: the indexer frees the model when no search used it for 10 minutes


def get_embedder(ck: Path, searching: bool = False):
    """The trained meaning model (loaded once; reloaded if the files change). None until it's trained."""
    global last_used
    if searching:
        import time as _t
        last_used = _t.time()
    m = ck / "meaning.onnx"
    stamp = m.stat().st_mtime if m.exists() else None
    if _loaded.get("stamp") != stamp:
        try:
            _loaded["emb"] = Embedder.load(ck) if stamp else None
        except Exception:
            _loaded["emb"] = None
        _loaded["stamp"] = stamp
    return _loaded.get("emb")
