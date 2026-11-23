"""Train the reply model (facts -> in-character sentence) from scratch.

    python -m datagen.generate_replies --n 300000
    python -m training.tokenizer --data data/replies/train.jsonl --fields src,tgt --vocab 2048 --out checkpoints/reply_tokenizer.json
    python -m training.train_replies            # ~6M params, about 10-15 minutes on an RTX 5070
    python -m training.export_onnx --checkpoint checkpoints/reply.pt --prefix reply

The score that matters is the FACT-CHECK PASS RATE: the share of generated replies that state every fact
correctly and invent nothing. At runtime, any reply that fails the check is replaced by the template line.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from array import array
from pathlib import Path

import torch
import torch.nn.functional as F

from jarvis.nlg.checker import check, parse_src
from training.model import IntentTransformer, ModelConfig
from training.tokenizer import BOS, EOS, PAD, BPETokenizer
from training.train import batches, collate, eval_batches


def load(path: Path, tok: BPETokenizer, cfg: ModelConfig):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            src = tok.encode(ex["src"])[: cfg.max_src]
            tgt = tok.encode(ex["tgt"], bos=True, eos=True)
            if len(tgt) <= cfg.max_tgt:
                rows.append((array("H", src), array("H", tgt), ex["src"]))
    return rows


@torch.no_grad()
def evaluate(model, rows, tok, device, limit=1000):
    model.eval()
    rows = rows[:limit]
    passed = 0
    loss_sum = n_tok = 0
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    samples = []
    for batch in eval_batches(rows, 256, 24000):
        src, src_pad, tin, tout = collate(batch, device)
        with torch.autocast(**amp):
            logits = model(src, src_pad, tin)
            preds = model.greedy(src, src_pad, BOS, EOS)
        loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                    ignore_index=PAD, reduction="sum").item()
        n_tok += (tout != PAD).sum().item()
        for p, row in zip(preds.tolist(), batch):
            text = tok.decode(p[: p.index(EOS)] if EOS in p else p)
            key, fields, tone, addr = parse_src(row[2])
            ok = check(key, fields, tone, addr, text)
            passed += ok
            if len(samples) < 6:
                samples.append((row[2][:90], text, ok))
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"loss": loss_sum / max(1, n_tok), "fact_pass": passed / len(rows), "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/replies"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/reply_tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/reply.pt"))
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    tok = BPETokenizer.load(args.tokenizer)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model, n_enc=args.layers, n_dec=args.layers,
                      d_ff=3 * args.d_model, max_src=192, max_tgt=128, dropout=0.1)
    train_rows = load(args.data / "train.jsonl", tok, cfg)
    val_rows = load(args.data / "val.jsonl", tok, cfg)
    model = IntentTransformer(cfg).to(device)
    print(f"train={len(train_rows)} val={len(val_rows)} params: {model.n_params() / 1e6:.2f}M", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
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
        for batch in batches(train_rows, args.batch, max_tokens=args.max_tokens):
            src, src_pad, tin, tout = collate(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(src, src_pad, tin)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                mem = f"  gpu mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB" if device.type == "cuda" else ""
                print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  {el / 60:.0f} min elapsed, "
                      f"~{el / step * (args.steps - step) / 60:.0f} min left{mem}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val_rows, tok, device)
                print(f"  eval  loss {m['loss']:.4f}  fact-check pass {m['fact_pass']:.4f}", flush=True)
                for src_s, text, ok in m["samples"]:
                    print(f"    {'OK ' if ok else 'BAD'} {src_s}  ->  {text}", flush=True)
                if m["fact_pass"] > best:
                    best = m["fact_pass"]
                    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step,
                                "metrics": {"fact_pass": best, "loss": m["loss"]}}, args.out)
                    print(f"  saved {args.out} (fact-check pass {best:.4f})", flush=True)
            if step >= args.steps:
                break
    print(f"done. best fact-check pass rate = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
