"""Train the meaning model (contrastive, in-batch negatives) on concept pairs + pairs from your own files.

    python -m training.meaning.train                     # ~20k steps, about 20-40 min on the RTX 5070
    python -m training.meaning.train --steps 200         # quick check
Reads your memory index (on this PC) for the self-supervised pairs; nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jarvis.actions.training import wait_if_paused
from training.tokenizer import BPETokenizer

from .data import concept_pair, load_user_docs, pair
from .model import MAX_LEN, MeaningEncoder


def default_index() -> Path:
    from jarvis.config import Config
    return Config.load().data_dir / "memory.sqlite"


def batch_tensors(tok, texts: list[str], device):
    ids = [tok.encode(t.lower())[:MAX_LEN] or [0] for t in texts]
    L = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), L, dtype=torch.long)
    keep = torch.zeros(len(ids), L)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
        keep[i, : len(x)] = 1.0
    return pad.to(device), keep.to(device)


def recall_at(model, tok, pairs, device, k: int = 5) -> float:
    """Each query against all documents in the set: is its own document in the top k?"""
    model.eval()
    with torch.no_grad():
        qs, ds = [], []
        for i in range(0, len(pairs), 256):
            chunk = pairs[i:i + 256]
            qs.append(model(*batch_tensors(tok, [q for q, _ in chunk], device)).float())
            ds.append(model(*batch_tensors(tok, [d for _, d in chunk], device)).float())
        q, d = torch.cat(qs), torch.cat(ds)
        top = (q @ d.T).topk(k, dim=1).indices
        hit = (top == torch.arange(len(pairs), device=device)[:, None]).any(1).float().mean().item()
    model.train()
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--index", type=Path, default=None, help="memory index (default: Jarvis's own)")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/meaning.pt"))
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    index = args.index or default_index()
    docs = load_user_docs(index)
    print(f"your files: {len(docs)} documents from {index}", flush=True)
    share = 0.5 if len(docs) >= 50 else 0.0
    if share == 0:
        print("  (too few indexed documents yet: training on generated pairs only)", flush=True)

    # tokenizer: generated text + your files
    texts = []
    for _ in range(120_000):
        q, d = pair(r, docs, share)
        texts += [q.lower(), d.lower()]
    tok_path = args.out.parent / "meaning_tokenizer.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.train(texts, args.vocab, verbose=False)
    tok.save(tok_path)
    print(f"tokenizer: {tok.vocab_size} tokens -> {tok_path}", flush=True)

    vr = random.Random(777)
    val_concept = [concept_pair(vr) for _ in range(2000)]
    val_user = [p for p in (pair(vr, docs, 1.0) for _ in range(2000)) if p] if share else []

    model = MeaningEncoder(tok.vocab_size).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    log_temp = torch.nn.Parameter(torch.tensor(math.log(0.05), device=device))
    opt = torch.optim.AdamW(list(model.parameters()) + [log_temp], lr=args.lr, weight_decay=0.01)
    warm = min(1000, args.steps // 10)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    best, t0 = -1.0, time.time()
    model.train()
    for step in range(1, args.steps + 1):
        wait_if_paused(args.out.parent / "PAUSE")
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        pairs = [pair(r, docs, share) for _ in range(args.batch)]
        with torch.autocast(**amp):
            q = model(*batch_tensors(tok, [a for a, _ in pairs], device))
            d = model(*batch_tensors(tok, [b for _, b in pairs], device))
        logits = (q.float() @ d.float().T) / log_temp.exp().clamp(min=0.01)
        target = torch.arange(len(pairs), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  "
                  f"{el / 60:.0f} min elapsed, ~{el / step * (args.steps - step) / 60:.0f} min left", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            rc = recall_at(model, tok, val_concept, device)
            ru = recall_at(model, tok, val_user, device) if val_user else rc
            score = (rc + ru) / 2
            print(f"  eval  same-meaning recall@5 {rc:.4f}  your-files recall@5 {ru:.4f}", flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "vocab": tok.vocab_size, "step": step,
                            "recall": {"concepts": rc, "files": ru}}, args.out)
                print(f"  saved {args.out} (score {best:.4f})", flush=True)
    print(f"done. best score = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
