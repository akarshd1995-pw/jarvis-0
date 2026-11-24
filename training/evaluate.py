"""Detailed evaluation of a trained intent model.

    python -m training.evaluate --split test_ood     (final robustness score; never used during training)
    python -m training.evaluate --split test_iid --errors checkpoints/errors_iid.jsonl

Reports exact match, valid-DSL rate, per-action accuracy, and the two error types that
matter most for an agent:
  * DANGEROUS : predicted a mutating action (move/copy/delete/system) that differs from the gold program
  * MISSED    : gold was a real command but the model said unknown() (safe, just unhelpful)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from jarvis.dsl.grammar import DSLError, canonical
from training.model import IntentTransformer, ModelConfig
from training.tokenizer import BOS, EOS, BPETokenizer
from training.train import collate, eval_batches

MUTATING = ("move_files", "copy_files", "delete_files", "system")


def actions(prog: str) -> list[str]:
    return [c.strip().split("(")[0] for c in prog.split(" ; ")]


def load_model(checkpoint: Path, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    model = IntentTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_rows(path: Path, tok, cfg, limit: int = 10_000) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            src = tok.encode(ex["text"])[: cfg.max_src]
            tgt = tok.encode(ex["program"], bos=True, eos=True)
            if len(tgt) <= cfg.max_tgt:
                rows.append((src, tgt, canonical(ex["program"]), ex["text"]))
            if len(rows) >= limit:
                break
    return rows


def evaluate_rows(model, tok, rows, device) -> dict:
    """{'n', 'exact', 'valid', 'dangerous', 'missed' (rates), 'per_action': {a: (correct, total)}, 'errors': [...]}"""
    stats = Counter()
    per_action = defaultdict(lambda: [0, 0])
    errors = []
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in eval_batches(rows):
        src, src_pad, _, _ = collate(batch, device)
        with torch.no_grad(), torch.autocast(**amp):
            preds = model.greedy(src, src_pad, BOS, EOS)
        for p, (_, _, gold, text) in zip(preds.tolist(), batch):
            raw = tok.decode(p[: p.index(EOS)] if EOS in p else p)
            try:
                pred = canonical(raw)
                stats["valid"] += 1
            except DSLError:
                pred = None
            ok = pred == gold
            stats["exact"] += ok
            stats["n"] += 1
            for a in set(actions(gold)):
                per_action[a][0] += ok
                per_action[a][1] += 1
            if not ok:
                if pred and gold != "unknown()" and pred == "unknown()":
                    stats["missed"] += 1
                elif pred and any(a in MUTATING for a in actions(pred)):
                    stats["dangerous"] += 1
                errors.append({"text": text, "gold": gold, "pred": pred or raw})
    n = max(1, stats["n"])
    return {"n": stats["n"], "exact": stats["exact"] / n, "valid": stats["valid"] / n,
            "dangerous": stats["dangerous"] / n, "missed": stats["missed"] / n,
            "per_action": {a: tuple(v) for a, v in per_action.items()}, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/intents"))
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/intent.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--errors", type=Path, default=None, help="write mistakes here as jsonl")
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(args.tokenizer)
    model, cfg = load_model(args.checkpoint, device)
    rows = load_rows(args.data / f"{args.split}.jsonl", tok, cfg, args.limit)
    r = evaluate_rows(model, tok, rows, device)
    print(f"split={args.split}  n={r['n']}")
    print(f"  exact match : {r['exact']:.4f}")
    print(f"  valid DSL   : {r['valid']:.4f}")
    print(f"  dangerous   : {r['dangerous']:.4f}  (wrong mutating action; confirmation dialog still guards these)")
    print(f"  missed      : {r['missed']:.4f}  (said unknown for a real command)")
    print("  per action:")
    for a, (c, t) in sorted(r["per_action"].items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"    {a:14s} {c / t:.4f}  ({t})")
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with open(args.errors, "w", encoding="utf-8") as f:
            for e in r["errors"]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  wrote {len(r['errors'])} errors to {args.errors}")


if __name__ == "__main__":
    main()
