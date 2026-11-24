"""Export the meaning model to ONNX: checkpoints/meaning.onnx (+ meaning_tokenizer.json from training)."""
from __future__ import annotations

from pathlib import Path

import torch

from training.export_onnx import _export

from .model import MeaningEncoder


def main(ck: Path = Path("checkpoints")):
    state = torch.load(ck / "meaning.pt", map_location="cpu", weights_only=True)
    model = MeaningEncoder(state["vocab"])
    model.load_state_dict(state["model"])
    model.eval()
    ids = torch.randint(0, state["vocab"], (2, 12))
    keep = torch.ones(2, 12)
    _export(model, (ids, keep), ck / "meaning.onnx", ["ids", "keep"], ["vec"],
            {"ids": {0: "n", 1: "len"}, "keep": {0: "n", 1: "len"}, "vec": {0: "n"}})
    print(f"wrote {ck / 'meaning.onnx'} (recall {state.get('recall')})")
    from training.quantize import after_export
    after_export("meaning", ck)


if __name__ == "__main__":
    main()
