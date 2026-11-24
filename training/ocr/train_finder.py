"""Train the OCR line finder on fake screenshots rendered on the fly.

    python -m training.ocr.train_finder                 # ~15k steps, about 15-25 min on the RTX 5070
    python -m training.ocr.train_finder --steps 200     # quick check
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from jarvis.actions.training import wait_if_paused

from .fonts import find_fonts
from .models import LineFinder
from .render import FontBank, render_page

PAGE_W, PAGE_H = 640, 384          # multiples of 16


class Pages(IterableDataset):
    def __init__(self, fonts: list[str], seed: int):
        self.fonts, self.seed = fonts, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        r = random.Random(self.seed + (info.id if info else 0) * 7919 + int(time.time()))
        bank = FontBank(self.fonts)
        while True:
            img, mask, _ = render_page(r, bank, PAGE_W, PAGE_H, LineFinder.STRIDE)
            if r.random() < 0.3:
                img = 1.0 - img                               # both polarities
            yield img[None], mask[None]


def collate(batch):
    return (torch.from_numpy(np.stack([b[0] for b in batch])), torch.from_numpy(np.stack([b[1] for b in batch])))


def loss_fn(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=torch.tensor(2.0, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * mask).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, vals, device) -> dict:
    model.eval()
    tp = fp = fn = 0
    for i in range(0, len(vals), 16):
        x, m = collate(vals[i:i + 16])
        p = torch.sigmoid(model(x.to(device))).cpu() > 0.5
        t = m > 0.5
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
    model.train()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"f1": 2 * prec * rec / max(1e-9, prec + rec), "precision": prec, "recall": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/ocr_finder.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    fonts = find_fonts()
    r = random.Random(4242)
    bank = FontBank(fonts)
    vals = []
    for _ in range(160):
        img, mask, _ = render_page(r, bank, PAGE_W, PAGE_H, LineFinder.STRIDE)
        vals.append((img[None], mask[None]))
    model = LineFinder().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = min(500, args.steps // 10)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * p)))

    loader = DataLoader(Pages(fonts, args.seed), batch_size=args.batch, collate_fn=collate,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for x, m in loader:
        wait_if_paused(args.out.parent / "PAUSE")           # "pause training" from Jarvis
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = model(x)
        loss = loss_fn(logits.float(), m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  "
                  f"{el / 60:.0f} min elapsed, ~{el / step * (args.steps - step) / 60:.0f} min left", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(model, vals, device)
            print(f"  eval  f1 {e['f1']:.4f}  precision {e['precision']:.4f}  recall {e['recall']:.4f}", flush=True)
            if e["f1"] > best:
                best = e["f1"]
                torch.save({"model": model.state_dict(), "step": step, "f1": best}, args.out)
                print(f"  saved {args.out} (f1 {best:.4f})", flush=True)
        if step >= args.steps:
            break
    print(f"done. best score = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
