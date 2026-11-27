"""Tune the pretrained Jarvis-LM on persona dialogues, export it for the NumPy runtime and grade it.

    python -m datagen.generate_chat --out data/chat
    python -m training.lm.finetune                      # about 20-40 minutes on the RTX 5070

The grade uses the real runtime (jarvis/nlg/chat.py, the same filter Jarvis applies) on conversations whose
phrasing never appeared in training:
- handoff accuracy: factual questions go to [web], missed commands to [cmd], chat gets a reply;
- reply rate: chat prompts that produced a reply the filter accepts;
- name: "who are you" answers use the persona's name.
lm_chat.json gets ready=true only if all three clear the bar; otherwise Jarvis keeps its fixed small-talk lines.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from jarvis.actions.training import wait_if_paused
from jarvis.nlg.chatfmt import encode_dialog
from training.lm.data import State, TokenShards, lr_at, sft_batch
from training.lm.model import export_npz, load_checkpoint, save_checkpoint
from training.tokenizer import BPETokenizer

BAR = {"handoff": 0.85, "reply_rate": 0.85, "name": 0.8}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S "), msg, flush=True)


def load_dialogs(path: Path, tok, max_len: int, limit: int | None = None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            ids, mask = encode_dialog(tok, ex["head"], [tuple(t) for t in ex["turns"]], max_len=max_len)
            if any(mask):
                rows.append((ids, mask))
            if limit and len(rows) >= limit:
                break
    return rows


def grade(folder: Path, test_file: Path, n: int = 300, seed: int = 7) -> dict:
    """Score the exported model through the runtime Jarvis actually uses."""
    from jarvis.nlg.chat import ChatModel
    model = ChatModel.load(folder, require_ready=False)
    model.rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in open(test_file, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    hand_ok = hand_n = said = chat_n = name_ok = name_n = 0
    samples = []
    for ex in rows[:n]:
        name, address, tone, part = ex["head"].split("|")
        hour = {"morning": 9, "afternoon": 14, "evening": 19, "night": 23}[part]
        turns = [tuple(t) for t in ex["turns"]]
        hist, user = turns[:-2], turns[-2][1]                   # the last exchange is the one being graded
        out = model.reply(user, hist, name=name, address=address, tone=tone, hour=hour)
        kind = out[0] if out else "none"
        want = ex["kind"]
        hand_n += 1
        hand_ok += (kind == want) or (want == "chat" and kind in ("say", "none"))
        if want == "chat":
            chat_n += 1
            said += kind == "say"
            if ex.get("tag") == "name":
                name_n += 1
                name_ok += kind == "say" and name.lower() in out[1].lower()
        if len(samples) < 15:
            samples.append(f"[{name}/{tone}] U: {user}\n    J: {out[1] if out and out[0] == 'say' else kind}")
    res = {"handoff": round(hand_ok / max(1, hand_n), 3), "reply_rate": round(said / max(1, chat_n), 3),
           "name": round(name_ok / max(1, name_n), 3) if name_n else 1.0, "graded": hand_n,
           "ms_per_reply": model.stats["ms"]}
    res["ready"] = all(res[k] >= v for k, v in BAR.items())
    return res, samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("checkpoints/lm_pretrain.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/chat"))
    ap.add_argument("--pretrain-data", type=Path, default=Path("data/lm"), help="mixed in a little to keep fluency")
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/lm_tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/lm_chat.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix-every", type=int, default=8, help="every Nth step is a plain-text step (0 = never)")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    folder = args.out.parent
    state = State(folder / "lm_state.json")
    tok = BPETokenizer.load(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.grade_only:
        model, _ = load_checkpoint(args.base, device)
        train = load_dialogs(args.data / "train.jsonl", tok, args.max_len)
        val = load_dialogs(args.data / "val.jsonl", tok, args.max_len, limit=2000)
        log(f"{len(train):,} training dialogues, {len(val):,} for validation, {args.steps} steps")
        plain = None
        if args.mix_every and any(args.pretrain_data.glob("*.bin")):
            plain = TokenShards([p for p in args.pretrain_data.glob("*.bin") if p.name != "val.bin"])
        opt = torch.optim.AdamW(model.param_groups(0.01), lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
        rng = random.Random(args.seed)
        amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

        def run_batch(x, y):
            with torch.autocast(**amp):
                _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            return loss

        @torch.no_grad()
        def val_loss():
            model.eval()
            tot = [run_batch(*sft_batch(val[i:i + args.batch], max_len=args.max_len)).item()
                   for i in range(0, min(len(val), 20 * args.batch), args.batch)]
            model.train()
            return sum(tot) / max(1, len(tot))

        log(f"before tuning: validation loss on replies {val_loss():.3f}")
        model.train()
        for step in range(args.steps):
            wait_if_paused()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, 100)
            if plain is not None and step % args.mix_every == args.mix_every - 1:
                loss = run_batch(*plain.batch(step, args.batch // 4, args.max_len, seed=args.seed))
            else:
                loss = run_batch(*sft_batch(rng.sample(train, args.batch), max_len=args.max_len))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % 50 == 0:
                log(f"step {step + 1}/{args.steps}  loss {loss.item():.3f}")
            if (step + 1) % 500 == 0 or step + 1 == args.steps:
                log(f"validation loss on replies {val_loss():.3f}")
        save_checkpoint(args.out, model, None, step=args.steps)
        export_npz(model, folder / "lm_chat.npz")
        cfg = model.cfg
        meta = {**cfg.to_dict(), "ready": False, "trained": time.strftime("%Y-%m-%d %H:%M"),
                "params": cfg.params(), "max_ctx": min(cfg.max_ctx, 512)}
        (folder / "lm_chat.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log("exported checkpoints/lm_chat.npz; grading it through the runtime Jarvis uses...")

    res, samples = grade(folder, args.data / "test.jsonl")
    meta = json.loads((folder / "lm_chat.json").read_text(encoding="utf-8"))
    meta.update(ready=res["ready"], eval=res)
    (folder / "lm_chat.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    state.update(stage="tuned", eval=res)
    for s in samples:
        print(s)
    log(f"handoff {res['handoff']:.0%} (bar {BAR['handoff']:.0%})  replies {res['reply_rate']:.0%} "
        f"(bar {BAR['reply_rate']:.0%})  name {res['name']:.0%} (bar {BAR['name']:.0%})  "
        f"{res['ms_per_reply']:.0f} ms/reply")
    print(f"best score = {res['handoff']:.3f}", flush=True)             # read by "training status" and the HUD
    log("READY: Jarvis will use it for small talk." if res["ready"] else
        "Below the bar: Jarvis keeps the fixed small-talk lines. More pretraining usually fixes this.")
    return 0 if res["ready"] else 5


if __name__ == "__main__":
    sys.exit(main())
