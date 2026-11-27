"""Pretrain Jarvis-LM from scratch on open text, a few hours each night until the token budget is reached.

    python -m training.lm.pretrain --preset base --until 07:30

Run it again the next night: it resumes from checkpoints/lm_pretrain.pt with the same schedule. It pauses while
you game or run on battery (the yield watcher's PAUSE flag), saves every 20 minutes, and at --until it saves and
exits cleanly. A power cut loses at most those 20 minutes.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

from jarvis.actions.training import wait_if_paused
from training.lm.config import PRESETS, TOKENS, LMConfig
from training.lm.data import State, TokenShards, eta, lr_at, stop_time
from training.lm.model import JarvisLM, load_checkpoint, save_checkpoint
from training.tokenizer import BPETokenizer

PEAK_LR = {"tiny": 3e-3, "small": 1e-3, "base": 6e-4, "large": 3e-4}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S "), msg, flush=True)


@torch.no_grad()
def val_loss(model, shards: TokenShards, device, batch: int, ctx: int, n: int = 20) -> float:
    model.eval()
    total = 0.0
    for i in range(n):
        x, y = shards.batch(i, batch, ctx, seed=999)                 # the same windows every time: comparable
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        total += loss.item()
    model.train()
    return total / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--data", type=Path, default=Path("data/lm"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/lm_tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/lm_pretrain.pt"))
    ap.add_argument("--tokens", type=float, default=None, help="total training tokens (default: right for the size)")
    ap.add_argument("--micro", type=int, default=8, help="sequences per GPU step (lower it if you run out of memory)")
    ap.add_argument("--accum", type=int, default=8, help="GPU steps per optimizer step")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--until", default="", help="stop (after saving) at this time, e.g. 07:30")
    ap.add_argument("--save-every-min", type=float, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--compile", action="store_true", help="torch.compile (needs Triton; usually Linux only)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    state = State(args.out.parent / "lm_state.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("no GPU found: this will be very slow (fine for a smoke test with --preset tiny)")
    tok = BPETokenizer.load(args.tokenizer)
    train = TokenShards([p for p in args.data.glob("*.bin") if p.name != "val.bin"])
    val = TokenShards([args.data / "val.bin"]) if (args.data / "val.bin").exists() else train

    if args.out.exists():
        model, ck = load_checkpoint(args.out, device)
        cfg, step, tokens_seen = model.cfg, ck["step"], ck["tokens"]
        plan = ck["plan"]
        log(f"resuming at step {step:,} ({tokens_seen / 1e9:.2f}B of {plan['tokens'] / 1e9:.2f}B tokens)")
    else:
        cfg = LMConfig(**{**PRESETS[args.preset].to_dict(), "max_ctx": args.ctx,
                          "vocab": int(math.ceil(tok.vocab_size / 64) * 64)})
        model = JarvisLM(cfg).to(device)
        step, tokens_seen, ck = 0, 0, None
        plan = {"tokens": int(args.tokens or TOKENS[args.preset]), "micro": args.micro, "accum": args.accum,
                "ctx": args.ctx, "lr": args.lr or PEAK_LR[args.preset], "warmup": args.warmup, "preset": args.preset}
        log(f"new model: {args.preset}, {cfg.params() / 1e6:.0f}M parameters, {plan['tokens'] / 1e9:.2f}B tokens planned")
    per_step = plan["micro"] * plan["accum"] * plan["ctx"]
    total_steps = max(1, plan["tokens"] // per_step)
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=plan["lr"], betas=(0.9, 0.95), eps=1e-8,
                            fused=device.type == "cuda")
    if ck is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    run = torch.compile(model) if args.compile else model
    stop_at = stop_time(args.until)
    state.update(stage="pretraining", preset=plan["preset"], params=cfg.params(), tokens=tokens_seen,
                 tokens_planned=plan["tokens"])

    def save(why: str):
        save_checkpoint(args.out, model, opt, step=step, tokens=tokens_seen, plan=plan)
        state.update(tokens=tokens_seen, step=step)
        log(f"saved ({why}) at step {step:,}")

    last_save, t_log, tok_log = time.time(), time.time(), tokens_seen
    model.train()
    while step < total_steps:
        if stop_at and time.time() >= stop_at:
            save("morning")
            log(f"stopping for the day: {tokens_seen / plan['tokens']:.0%} done. Run train_lm.bat again tonight.")
            return 3
        wait_if_paused()
        lr = lr_at(step, total_steps, plan["lr"], plan["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for m in range(plan["accum"]):
            x, y = train.batch(step * plan["accum"] + m, plan["micro"], plan["ctx"], seed=args.seed)
            x = torch.from_numpy(x).to(device, non_blocking=True)
            y = torch.from_numpy(y).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = run(x, y)
            (loss / plan["accum"]).backward()
            loss_sum += loss.item()
        if not math.isfinite(loss_sum):
            log("loss became NaN/inf: the next run goes back to the last save with half the learning rate")
            if args.out.exists():
                raw = torch.load(args.out, map_location="cpu", weights_only=False)
                raw["plan"]["lr"] = raw["plan"]["lr"] * 0.5
                torch.save(raw, args.out)
            return 4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += per_step
        if step % 20 == 0:
            rate = (tokens_seen - tok_log) / max(1e-6, time.time() - t_log)
            left_min = int((plan["tokens"] - tokens_seen) / max(rate, 1e-6) / 60)
            log(f"step {step}/{total_steps}  loss {loss_sum / plan['accum']:.3f}  lr {lr:.2e}  "
                f"{rate / 1000:.1f}k tok/s  ~{left_min} min left ({eta(plan['tokens'] - tokens_seen, rate)})")
            t_log, tok_log = time.time(), tokens_seen
            state.update(tokens=tokens_seen, step=step, tok_per_s=round(rate))
        if step % args.eval_every == 0:
            v = val_loss(model, val, device, plan["micro"], plan["ctx"])
            log(f"validation loss {v:.3f} (perplexity {math.exp(min(v, 20)):.1f})")
            state.update(val_loss=round(v, 4))
        if time.time() - last_save > args.save_every_min * 60:
            save("regular")
            last_save = time.time()
    save("finished")
    v = val_loss(model, val, device, plan["micro"], plan["ctx"])
    state.update(stage="pretrained", val_loss=round(v, 4))
    log(f"pretraining finished: validation loss {v:.3f} (perplexity {math.exp(min(v, 20)):.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
