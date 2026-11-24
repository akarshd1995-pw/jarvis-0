"""Train the OCR reader (CRNN + CTC) on text lines rendered on the fly.

    python -m training.ocr.train_reader                 # ~60k steps, about 40-60 min on the RTX 5070
    python -m training.ocr.train_reader --steps 300     # quick check that everything runs
"""
from __future__ import annotations

import argparse
import os
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .charset import BLANK, decode_greedy, encode
from jarvis.actions.training import wait_if_paused

from .fonts import find_fonts
from .models import Reader
from .render import MAX_W, FontBank, _need_steps, prepare, render_line

class Lines(IterableDataset):
    def __init__(self, fonts: list[str], seed: int):
        self.fonts, self.seed = fonts, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        r = random.Random(self.seed + (info.id if info else 0) * 7919 + int(time.time()))
        bank = FontBank(self.fonts)
        while True:
            a, s = render_line(r, bank)
            ids = encode(s)
            if not ids:
                continue
            a = prepare(a, ids)
            if a.shape[1] // 4 < _need_steps(ids):
                continue
            yield a, ids


def collate(batch):
    w = max(a.shape[1] for a, _ in batch)
    w = min(MAX_W, ((w + 63) // 64) * 64)              # a few fixed widths: the GPU kernels stay warm
    x = np.stack([np.pad(a, ((0, 0), (0, w - a.shape[1])), mode="edge") for a, _ in batch])[:, None]
    targets = torch.tensor([i for _, ids in batch for i in ids], dtype=torch.long)
    tlen = torch.tensor([len(ids) for _, ids in batch], dtype=torch.long)
    ilen = torch.tensor([a.shape[1] // 4 for a, _ in batch], dtype=torch.long)
    return torch.from_numpy(x), targets, ilen, tlen


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def val_set(fonts: list[str], n: int = 1500):
    r = random.Random(12345)
    bank = FontBank(fonts)
    out = []
    while len(out) < n:
        a, s = render_line(r, bank)
        ids = encode(s)
        if ids:
            out.append((prepare(a, ids), ids, s))
    return out


@torch.no_grad()
def evaluate(model, vals, device, show: int = 0) -> dict:
    model.eval()
    errs = chars = exact = 0
    samples = []
    for i in range(0, len(vals), 64):
        chunk = vals[i:i + 64]
        x, _, ilen, _ = collate([(a, ids) for a, ids, _ in chunk])
        lp = model(x.to(device)).argmax(-1).cpu()
        for k, (_, _, s) in enumerate(chunk):
            pred = decode_greedy(lp[k, : ilen[k]].tolist())
            gold = "".join(ch for ch in s if encode(ch))          # what the reader can output
            errs += edit_distance(pred, gold)
            chars += max(1, len(gold))
            exact += pred == gold
            if len(samples) < show:
                samples.append((gold, pred))
    model.train()
    return {"cer": errs / chars, "exact": exact / len(vals), "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=max(2, min(14, (os.cpu_count() or 8) - 2)),
                    help="CPU processes rendering training images (default: all but 2 threads)")
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/ocr_reader.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False      # line widths vary per batch: re-tuning every shape is slow
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    fonts = find_fonts()
    print(f"fonts: {len(fonts)}", flush=True)
    vals = val_set(fonts)
    model = Reader().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = min(1000, args.steps // 10)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * p)))

    loader = DataLoader(Lines(fonts, args.seed), batch_size=args.batch, collate_fn=collate,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best, step, t0 = 9.9, 0, time.time()
    model.train()
    for x, targets, ilen, tlen in loader:
        wait_if_paused(args.out.parent / "PAUSE")           # "pause training" from Jarvis
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lp = model(x.to(device, non_blocking=True))                 # (N, T, C)
        loss = F.ctc_loss(lp.transpose(0, 1).float(), targets, ilen, tlen, blank=BLANK, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  "
                  f"{el / 60:.0f} min elapsed, ~{el / step * (args.steps - step) / 60:.0f} min left", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, vals, device, show=4)
            print(f"  eval  char error {m['cer']:.4f}  lines exact {m['exact']:.4f}", flush=True)
            for gold, pred in m["samples"]:
                print(f"    {'OK ' if gold == pred else '   '} {gold!r} -> {pred!r}", flush=True)
            if m["cer"] < best:
                best = m["cer"]
                torch.save({"model": model.state_dict(), "step": step, "cer": best}, args.out)
                print(f"  saved {args.out} (char error {best:.4f})", flush=True)
        if step >= args.steps:
            break
    print(f"done. best score = {1 - best:.4f}", flush=True)


if __name__ == "__main__":
    main()
