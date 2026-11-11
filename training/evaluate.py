"""Detailed evaluation of a trained intent model.

    python -m training.evaluate --split test_ood
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
from training.train import batches, collate

MUTATING = ("move_files", "copy_files", "delete_files", "system")


def actions(prog: str) -> list[str]:
    return [c.strip().split("(")[0] for c in prog.split(" ; ")]


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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    model = IntentTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows, texts = [], []
    with open(args.data / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            src = tok.encode(ex["text"])[: cfg.max_src]
            tgt = tok.encode(ex["program"], bos=True, eos=True)
            if len(tgt) <= cfg.max_tgt:
                rows.append((src, tgt, ex["program"]))
                texts.append(ex["text"])
            if len(rows) >= args.limit:
                break
    stats = Counter()
    per_action = defaultdict(lambda: [0, 0])
    errors = []
    i = 0
    for batch in batches(rows, 256, shuffle=False):
        src, src_pad, _, _ = collate(batch, device)
        preds = model.greedy(src, src_pad, BOS, EOS)
        for p, (_, _, gold) in zip(preds.tolist(), batch):
            text = texts[i] if i < len(texts) else ""
            i += 1
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
    print(f"split={args.split}  n={n}")
    print(f"  exact match : {stats['exact'] / n:.4f}")
    print(f"  valid DSL   : {stats['valid'] / n:.4f}")
    print(f"  dangerous   : {stats['dangerous'] / n:.4f}  (wrong mutating action; confirmation dialog still guards these)")
    print(f"  missed      : {stats['missed'] / n:.4f}  (said unknown for a real command)")
    print("  per action:")
    for a, (c, t) in sorted(per_action.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"    {a:14s} {c / t:.4f}  ({t})")
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with open(args.errors, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  wrote {len(errors)} errors to {args.errors}")


if __name__ == "__main__":
    main()
