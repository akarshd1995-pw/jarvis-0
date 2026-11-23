"""Train the intent model from scratch.

Night machine (RTX 5070, 12 GB):
    python -m training.train --data data/intents --tokenizer checkpoints/tokenizer.json --out checkpoints/intent.pt

Bigger model if OOD accuracy plateaus (still fits 12 GB easily):
    python -m training.train --d-model 384 --layers 6 --steps 60000

Laptop smoke test (CPU or the small GPU):
    python -m training.train --steps 200 --batch 32 --eval-every 100
"""
from __future__ import annotations

import argparse
import json
from array import array
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
                # uint16 arrays: ~15x less RAM than Python int lists, so 2M+ examples fit easily
                rows.append((array("H", src), array("H", tgt), ex["program"]))
    return rows


def collate(batch, device):
    s_len = max(len(b[0]) for b in batch)
    t_len = max(len(b[1]) for b in batch)
    src = torch.full((len(batch), s_len), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), t_len), PAD, dtype=torch.long)
    for i, row in enumerate(batch):
        s, t = row[0], row[1]
        src[i, : len(s)] = torch.tensor(s.tolist() if isinstance(s, array) else s)
        tgt[i, : len(t)] = torch.tensor(t.tolist() if isinstance(t, array) else t)
    src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
    return src, src == PAD, tgt[:, :-1], tgt[:, 1:]


def batches(rows, bs, shuffle=True, max_tokens=None):
    """Yield batches. With max_tokens, each training batch is capped by padded size
    (rows x (longest source + longest target)) so long multi-command batches can't overflow
    GPU memory. On Windows an overflow doesn't crash: the driver spills into system RAM
    and training silently runs several times slower."""
    idx = list(range(len(rows)))
    if not shuffle:
        for i in range(0, len(idx), bs):
            yield [rows[j] for j in idx[i:i + bs]]
        return
    random.shuffle(idx)
    # bucket by length so batches have little padding
    chunks = [sorted(idx[i:i + bs * 50], key=lambda j: (len(rows[j][0]), len(rows[j][1])))
              for i in range(0, len(idx), bs * 50)]
    groups = []
    for c in chunks:
        g, s_max, t_max = [], 0, 0
        for j in c:
            s2, t2 = max(s_max, len(rows[j][0])), max(t_max, len(rows[j][1]))
            if g and (len(g) >= bs or (max_tokens and (len(g) + 1) * (s2 + t2) > max_tokens)):
                groups.append(g)
                g, s2, t2 = [], len(rows[j][0]), len(rows[j][1])
            g.append(j)
            s_max, t_max = s2, t2
        if g:
            groups.append(g)
    random.shuffle(groups)
    for g in groups:
        yield [rows[j] for j in g]


def eval_batches(rows, bs=256, max_tokens=24000):
    """Deterministic, length-sorted, memory-capped batches for evaluation."""
    order = sorted(range(len(rows)), key=lambda j: (len(rows[j][0]), len(rows[j][1])))
    g, s_max, t_max = [], 0, 0
    for j in order:
        s2, t2 = max(s_max, len(rows[j][0])), max(t_max, len(rows[j][1]))
        if g and (len(g) >= bs or (len(g) + 1) * (s2 + t2) > max_tokens):
            yield [rows[k] for k in g]
            g, s2, t2 = [], len(rows[j][0]), len(rows[j][1])
        g.append(j)
        s_max, t_max = s2, t2
    if g:
        yield [rows[k] for k in g]


@torch.no_grad()
def evaluate(model, rows, tok, device, limit=1500, bs=256, max_tokens=24000):
    model.eval()
    rows = rows[:limit]
    exact = valid = 0
    loss_sum = n_tok = 0
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in eval_batches(rows, bs, max_tokens):
        src, src_pad, tin, tout = collate(batch, device)
        with torch.autocast(**amp):
            logits = model(src, src_pad, tin)
        loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                    ignore_index=PAD, reduction="sum").item()
        n_tok += (tout != PAD).sum().item()
        del logits
        with torch.autocast(**amp):
            preds = model.greedy(src, src_pad, BOS, EOS)
        for p, row in zip(preds.tolist(), batch):
            gold = row[2]
            text = tok.decode(p[: p.index(EOS)] if EOS in p else p)
            try:
                pc = canonical(text)
                valid += 1
                exact += pc == gold
            except DSLError:
                pass
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()   # hand back eval memory so training never inherits a spilled pool
    return {"loss": loss_sum / max(1, n_tok), "exact": exact / len(rows), "valid": valid / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/intents"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/intent.pt"))
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256, help="max sentences per batch")
    ap.add_argument("--max-tokens", type=int, default=24000,
                    help="max padded tokens per batch (source + target); keeps GPU memory under 12 GB")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many evals without improvement (0 = never stop early)")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
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
    # val_ood = held-out sentence frames used ONLY for picking the checkpoint.
    # test_ood is never touched here; it is reported by training.evaluate.
    ood_path = args.data / "val_ood.jsonl"
    ood_rows = load_split(ood_path, tok, cfg) if ood_path.exists() else []
    print(f"train={len(train_rows)} val={len(val_rows)} val_ood={len(ood_rows)}")

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
    best, step, t0, stale = -1.0, 0, time.time(), 0
    model.train()
    while step < args.steps:
        for batch in batches(train_rows, args.batch, max_tokens=args.max_tokens):
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
                el = time.time() - t0
                eta = el / step * (args.steps - step)
                mem = ""
                if device.type == "cuda":
                    peak = torch.cuda.max_memory_reserved() / 2**30
                    total = torch.cuda.get_device_properties(0).total_memory / 2**30
                    mem = f"  gpu mem {peak:.1f}/{total:.0f} GB"
                    if peak > total - 0.8 and step == 100:
                        print("  WARNING: GPU memory nearly full; Windows may spill to system RAM and run "
                              "much slower. Lower --max-tokens.", flush=True)
                print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  "
                      f"{el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left{mem}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val_rows, tok, device)
                print(f"  eval  loss {m['loss']:.4f}  exact {m['exact']:.4f}  valid {m['valid']:.4f}")
                score = m["exact"]
                if ood_rows:
                    o = evaluate(model, ood_rows, tok, device)
                    print(f"  vood  loss {o['loss']:.4f}  exact {o['exact']:.4f}  valid {o['valid']:.4f}")
                    m["val_ood_exact"] = o["exact"]
                    score = (m["exact"] + o["exact"]) / 2   # reward familiar AND unseen phrasing
                if score > best:
                    best, stale = score, 0
                    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step,
                                "metrics": m}, args.out)
                    print(f"  saved {args.out} (score={best:.4f})")
                else:
                    stale += 1
                    if args.patience and stale >= args.patience:
                        print(f"  no improvement for {stale} evals, stopping early")
                        step = args.steps
            if step >= args.steps:
                break
    print(f"done. best score = {best:.4f}")


if __name__ == "__main__":
    main()
