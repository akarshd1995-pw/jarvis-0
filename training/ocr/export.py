"""Export the OCR models to ONNX so Jarvis runs them without PyTorch.

    python -m training.ocr.export          # checkpoints/ocr_finder.onnx + checkpoints/ocr_reader.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.export_onnx import _export

from .models import LineFinder, Reader


def export(ck: Path) -> None:
    finder = LineFinder()
    finder.load_state_dict(torch.load(ck / "ocr_finder.pt", map_location="cpu", weights_only=True)["model"])
    finder.eval()
    _export(finder, (torch.rand(1, 1, 384, 640),), ck / "ocr_finder.onnx", ["image"], ["logits"],
            {"image": {2: "h", 3: "w"}, "logits": {2: "h4", 3: "w4"}})
    reader = Reader()
    reader.load_state_dict(torch.load(ck / "ocr_reader.pt", map_location="cpu", weights_only=True)["model"])
    reader.eval()
    _export(reader, (torch.rand(2, 1, 32, 256),), ck / "ocr_reader.onnx", ["line"], ["logprobs"],
            {"line": {0: "n", 3: "w"}, "logprobs": {0: "n", 1: "t"}})
    print(f"wrote {ck / 'ocr_finder.onnx'} and {ck / 'ocr_reader.onnx'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    export(ap.parse_args().checkpoints)


if __name__ == "__main__":
    main()
