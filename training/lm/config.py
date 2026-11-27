"""Model sizes for Jarvis-LM. Plain Python so the runtime, tests and trainer share them without PyTorch."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LMConfig:
    vocab: int = 16384
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    hidden: int = 2048           # SwiGLU width (about 8/3 x d_model, rounded to 128)
    max_ctx: int = 1024
    rope_theta: float = 10000.0

    def params(self) -> int:
        per_layer = 4 * self.d_model ** 2 + 3 * self.d_model * self.hidden + 2 * self.d_model
        return self.vocab * self.d_model + self.n_layer * per_layer + self.d_model

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "tiny": LMConfig(vocab=16384, d_model=64, n_layer=2, n_head=4, hidden=192, max_ctx=256),      # tests, smoke runs
    "small": LMConfig(d_model=512, n_layer=8, n_head=8, hidden=1408, max_ctx=1024),               # ~34M: laptop
    "base": LMConfig(d_model=768, n_layer=12, n_head=12, hidden=2048, max_ctx=1024),              # ~98M: default
    "large": LMConfig(d_model=1024, n_layer=16, n_head=16, hidden=2816, max_ctx=1024),            # ~222M: more nights
}
# Roughly 20-30 training tokens per parameter is where a model this size stops improving quickly.
TOKENS = {"tiny": 5_000_000, "small": 1_000_000_000, "base": 2_800_000_000, "large": 5_000_000_000}
