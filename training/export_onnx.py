"""Export the trained intent model to ONNX so the agent runs without PyTorch.

    python -m training.export_onnx                 # writes checkpoints/intent_encoder.onnx + intent_decoder.onnx
    python -m training.export_onnx --verify 200    # also checks ONNX == PyTorch on 200 validation examples
    python -m training.export_onnx --checkpoint checkpoints/reply.pt --prefix reply     # the reply model

Two graphs:
  encoder: (src int64 [B,S], src_pad bool [B,S])                  -> mem float32 [B,S,D]
  decoder: (tgt_in int64 [B,T], mem float32 [B,S,D], src_pad bool [B,S]) -> logprobs float32 [B,V] (last position)
The beam search itself runs in plain Python/numpy (jarvis/nlu/beam.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from training.model import IntentTransformer, ModelConfig


class EncoderGraph(nn.Module):
    def __init__(self, model: IntentTransformer):
        super().__init__()
        self.model = model

    def forward(self, src, src_pad):
        mem, _ = self.model.encode(src, src_pad)
        return mem


class DecoderStepGraph(nn.Module):
    def __init__(self, model: IntentTransformer):
        super().__init__()
        self.model = model

    def forward(self, tgt_in, mem, src_pad):
        mem_mask = (~src_pad)[:, None, None, :]
        logits = self.model.decode(tgt_in, mem, mem_mask, last_only=True)[:, -1]
        return torch.log_softmax(logits.float(), dim=-1)


def _export(module, args, path, names_in, names_out, dynamic):
    kw = dict(input_names=names_in, output_names=names_out, dynamic_axes=dynamic, opset_version=17)
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kw)   # TorchScript exporter
    except TypeError:                                                    # older torch: no dynamo flag
        torch.onnx.export(module, args, str(path), **kw)


def export(checkpoint: Path, out_dir: Path, prefix: str = "intent") -> tuple[Path, Path]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = IntentTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path, dec_path = out_dir / f"{prefix}_encoder.onnx", out_dir / f"{prefix}_decoder.onnx"

    src = torch.randint(4, model.cfg.vocab_size, (2, 12))
    pad = torch.zeros(2, 12, dtype=torch.bool)
    pad[1, 9:] = True
    with torch.no_grad():
        _export(EncoderGraph(model), (src, pad), enc_path, ["src", "src_pad"], ["mem"],
                {"src": {0: "b", 1: "s"}, "src_pad": {0: "b", 1: "s"}, "mem": {0: "b", 1: "s"}})
        mem = EncoderGraph(model)(src, pad)
        tgt = torch.randint(4, model.cfg.vocab_size, (2, 5))
        _export(DecoderStepGraph(model), (tgt, mem, pad), dec_path, ["tgt_in", "mem", "src_pad"], ["logprobs"],
                {"tgt_in": {0: "b", 1: "t"}, "mem": {0: "b", 1: "s"}, "src_pad": {0: "b", 1: "s"},
                 "logprobs": {0: "b"}})
    meta = {"config": ckpt["config"], "step": ckpt.get("step"), "metrics": ckpt.get("metrics"),
            "knows": ckpt.get("knows")}
    (out_dir / f"{prefix}_onnx.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return enc_path, dec_path


def verify(checkpoint: Path, out_dir: Path, tokenizer: Path, data: Path, n: int) -> None:
    """Run the same validation examples through PyTorch and ONNX and compare outputs."""
    import numpy as np

    from jarvis.nlu.onnx_parser import ONNXParser
    from training.tokenizer import BOS, EOS, PAD, BPETokenizer

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = IntentTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(tokenizer)
    onnx = ONNXParser(out_dir / "intent_encoder.onnx", out_dir / "intent_decoder.onnx", tokenizer,
                      min_confidence=0.0)
    rows = [json.loads(l) for _, l in zip(range(n), open(data, encoding="utf-8"))]
    same, max_diff = 0, 0.0
    for r in rows:
        ids = tok.encode(r["text"])[: model.cfg.max_src]
        src = torch.tensor([ids])
        pad = src == PAD
        with torch.no_grad():
            mem, mask = model.encode(src, pad)
            t_lp = torch.log_softmax(model.decode(torch.tensor([[BOS]]), mem, mask, last_only=True)[:, -1], -1)
        o_lp = onnx.first_step_logprobs(ids)
        max_diff = max(max_diff, float(np.abs(t_lp.numpy() - o_lp).max()))
        t_out = model.greedy(src, pad, BOS, EOS)[0].tolist()
        t_text = tok.decode(t_out[: t_out.index(EOS)] if EOS in t_out else t_out)
        prog, _ = onnx.parse(r["text"])
        same += prog is not None and str(prog) == t_text
    print(f"checked {len(rows)} examples: max first-step logprob difference {max_diff:.2e}")
    print(f"ONNX constrained beam == PyTorch greedy on {same}/{len(rows)} (small differences are expected:")
    print("beam search with grammar constraints can legitimately pick a better program than greedy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/intent.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--prefix", default="intent", help="intent (command model) or reply (reply model)")
    ap.add_argument("--verify", type=int, default=0, help="compare ONNX vs PyTorch on N validation examples")
    ap.add_argument("--data", type=Path, default=Path("data/intents/val.jsonl"))
    args = ap.parse_args()
    enc, dec = export(args.checkpoint, args.out, args.prefix)
    print(f"wrote {enc} ({enc.stat().st_size / 1e6:.1f} MB) and {dec} ({dec.stat().st_size / 1e6:.1f} MB)")
    if args.verify and args.prefix == "intent":
        verify(args.checkpoint, args.out, args.tokenizer, args.data, args.verify)


if __name__ == "__main__":
    main()
