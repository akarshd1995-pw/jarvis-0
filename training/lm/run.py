"""One command for the whole Jarvis-LM pipeline; run it each night and it carries on from where it stopped.

    python -m training.lm.run                    # base model (~98M), stops at 07:30 and continues next night
    python -m training.lm.run --preset small     # ~34M, about one night, for the laptop
    python -m training.lm.run --until ""         # no stop time
    python -m training.lm.run --retune           # redo only the chat tuning (e.g. after new dialogue data)

Stages: persona dialogues -> download open text -> tokenizer -> token files -> pretraining (several nights)
-> chat tuning + export + grade. Everything is also logged to checkpoints/lm_log.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from training.lm.config import TOKENS
from training.lm.data import State, stop_time

CK = Path("checkpoints")
FILES_FOR = {"tiny": 1, "small": 2, "base": 4, "large": 7}       # ~0.7B tokens per file


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f, self.out = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S "), msg, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=list(FILES_FOR))
    ap.add_argument("--until", default="07:30")
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--chat-examples", type=int, default=300_000)
    args = ap.parse_args(argv)
    sys.stdout = Tee(CK / "lm_log.txt")
    state = State(CK / "lm_state.json")
    log(f"=== Jarvis-LM: stage '{state.stage}', preset {args.preset} ===")
    stop_at = stop_time(args.until)

    if not (Path("data/chat") / "train.jsonl").exists() or args.retune:
        log("[1/6] writing persona dialogues...")
        from datagen import generate_chat
        generate_chat.main(["--n", str(args.chat_examples), "--out", "data/chat"])
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        log("pyarrow is needed to read the open text: pip install pyarrow  (then run this again)")
        return 2
    from training.lm import corpus
    if state.stage in ("none", "pretraining"):
        log(f"[2/6] open text: {FILES_FOR[args.preset]} file(s) of FineWeb-Edu (resumable)...")
        corpus.download(FILES_FOR[args.preset])
        if not (CK / "lm_tokenizer.json").exists():
            log("[3/6] learning the tokenizer (once; 10-30 minutes)...")
            corpus.train_tokenizer(corpus.RAW, Path("data/chat/train.jsonl"), 16384, 120, CK / "lm_tokenizer.json")
        log("[4/6] turning text into token files (once per file)...")
        corpus.encode(corpus.RAW, corpus.OUT, CK / "lm_tokenizer.json")
        log(f"[5/6] pretraining ({TOKENS[args.preset] / 1e9:.1f}B tokens over several nights)...")
        from training.lm import pretrain
        code = pretrain.main(["--preset", args.preset, "--until", args.until])
        if code != 0:
            return code
        state = State(CK / "lm_state.json")
    if state.stage == "pretrained" or args.retune:
        if stop_at and stop_at - time.time() < 75 * 60:
            log("pretraining is done, but tuning needs about an hour: it'll run next time.")
            return 3
        log("[6/6] chat tuning, export and grading...")
        from training.lm import finetune
        return finetune.main([])
    log(f"nothing left to do (stage '{state.stage}'). Grades: {state.get('eval')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
