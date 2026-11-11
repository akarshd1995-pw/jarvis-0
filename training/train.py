"""Train the intent model from scratch.

Night machine (RTX 5070, 12 GB):
    python -m training.train --data data/intents --tokenizer checkpoints/tokenizer.json --out checkpoints/intent.pt

Laptop smoke test (CPU or the small GPU):
    python -m training.train --steps 200 --batch 32 --eval-every 100
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jarvis.dsl.grammar import DSLError, canonical
from training.model import IntentTransformer, ModelConfig
from training.tokenizer import BOS, EOS, PAD, BPETokenizer


def load_split(path: Path, tok: BPETokenizer, cfg: ModelConfig):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            src = tok.encode(ex["text"])[: cfg.max_src]
            tgt = tok.encode(ex["program"], bos=True, eos=True)
            if len(tgt) <= cfg.max_tgt:
                rows.append((src, tgt, ex["program"]))
    return rows


def collate(batch, device):
    s_len = max(len(b[0]) for b in batch)
    t_len = max(len(b[1]) for b in batch)
    src = torch.full((len(batch), s_len), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), t_len), PAD, dtype=torch.long)
    for i, (s, t, _) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
    return src, src == PAD, tgt[:, :-1], tgt[:, 1:]


def batches(rows, bs, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
        # bucket by length so batches have little padding
        chunks = [sorted(idx[i:i + bs * 50], key=lambda j: len(rows[j][0])) for i in range(0, len(idx), bs * 50)]
        groups = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
        random.shuffle(groups)
        for g in groups:
            yield [rows[j] for j in g]
    else:
        for i in range(0, len(idx), bs):
            yield [rows[j] for j in idx[i:i + bs]]


@torch.no_grad()
def evaluate(model, rows, tok, device, limit=2000, bs=256):
    model.eval()
    rows = rows[:limit]
    exact = valid = 0
    loss_sum = n_tok = 0
    for batch in batches(rows, bs, shuffle=False):
        src, src_pad, tin, tout = collate(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(src, src_pad, tin)
        loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                    ignore_index=PAD, reduction="sum").item()
        n_tok += (tout != PAD).sum().item()
        preds = model.greedy(src, src_pad, BOS, EOS)
        for p, (_, _, gold) in zip(preds.tolist(), batch):
            text = tok.decode(p[: p.index(EOS)] if EOS in p else p)
            try:
                pc = canonical(text)
                valid += 1
                exact += pc == gold
            except DSLError:
                pass
    model.train()
    return {"loss": loss_sum / max(1, n_tok), "exact": exact / len(rows), "valid": valid / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/intents"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/intent.pt"))
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="torch.compile (needs Triton; skip on Windows if it errors)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0))

    tok = BPETokenizer.load(args.tokenizer)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model, n_enc=args.layers, n_dec=args.layers,
                      d_ff=3 * args.d_model)
    train_rows = load_split(args.data / "train.jsonl", tok, cfg)
    val_rows = load_split(args.data / "val.jsonl", tok, cfg)
    print(f"train={len(train_rows)} val={len(val_rows)}")

    model = IntentTransformer(cfg).to(device)
    print(f"params: {model.n_params() / 1e6:.2f}M")
    fwd = torch.compile(model) if args.compile else model

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=device.type == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    while step < args.steps:
        for batch in batches(train_rows, args.batch):
            src, src_pad, tin, tout = collate(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = fwd(src, src_pad, tin)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  {time.time() - t0:.0f}s")
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val_rows, tok, device)
                print(f"  eval  loss {m['loss']:.4f}  exact {m['exact']:.4f}  valid {m['valid']:.4f}")
                if m["exact"] > best:
                    best = m["exact"]
                    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step,
                                "metrics": m}, args.out)
                    print(f"  saved {args.out} (exact={best:.4f})")
            if step >= args.steps:
                break
    print(f"done. best exact-match = {best:.4f}")


if __name__ == "__main__":
    main()
