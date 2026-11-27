"""The pretraining text for Jarvis-LM: download, tokenizer, token files.

    python -m training.lm.corpus download --files 4      # FineWeb-Edu sample, ~2.15 GB and ~0.7B tokens per file
    python -m training.lm.corpus tokenizer               # our own 16k BPE, learned from a slice of it + the chat data
    python -m training.lm.corpus encode                  # text -> uint16 token files in data/lm (all CPU cores)
    python -m training.lm.corpus status

Source: HuggingFaceFW/fineweb-edu, sample-10BT (educational web pages, English), license ODC-By 1.0. Only the
text is used; no model or weights are downloaded. Reading the files needs `pip install pyarrow`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{i:03d}_00000.parquet"
FILES = 14
RAW = Path("data/lm/raw")
OUT = Path("data/lm")
VAL_DOCS = 5000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S "), msg, flush=True)


# ------------------------------------------------------------------ download
def download(n: int, raw: Path = RAW, url: str = URL, opener=urllib.request.urlopen) -> list[Path]:
    """Resumable: a half-downloaded file continues from where it stopped (HTTP Range)."""
    raw.mkdir(parents=True, exist_ok=True)
    done = []
    for i in range(min(n, FILES)):
        dest = raw / f"{i:03d}.parquet"
        if dest.exists():
            done.append(dest)
            continue
        part = dest.with_suffix(".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url.format(i=i), headers={"User-Agent": "jarvis-0", **({"Range": f"bytes={have}-"} if have else {})})
        with opener(req, timeout=60) as r:
            if have and getattr(r, "status", 200) != 206:
                have = 0                                            # the server ignored the range: start over
            total = have + int(r.headers.get("Content-Length") or 0)
            log(f"file {i + 1}/{n}: {'resuming at ' + f'{have / 2**30:.2f} GB' if have else 'starting'} "
                f"({total / 2**30:.2f} GB)")
            t0, last = time.time(), 0.0
            with open(part, "ab" if have else "wb") as f:
                got = have
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    got += len(block)
                    if time.time() - last > 15:
                        rate = (got - have) / max(1e-6, time.time() - t0)
                        log(f"  {got / 2**30:.2f}/{total / 2**30:.2f} GB  {rate / 2**20:.1f} MB/s")
                        last = time.time()
        if total and part.stat().st_size < total:
            raise OSError(f"download of file {i} stopped early; run again to continue")
        part.replace(dest)
        done.append(dest)
    return done


def texts(path: Path, batch: int = 2000):
    """Documents from one parquet file, in batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("Reading the corpus needs pyarrow: pip install pyarrow")
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=["text"]):
        yield [t for t in b.column(0).to_pylist() if t]


# ------------------------------------------------------------------ tokenizer
def chat_texts(chat: Path, limit: int = 200_000) -> list[str]:
    """The persona dialogues, so names, Hinglish and the chat markers get tokens of their own."""
    from jarvis.nlg.chatfmt import BOT, USER
    out = []
    if chat.exists():
        with open(chat, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                ex = json.loads(line)
                out.append(ex["head"])
                out += [(USER if w == "U" else BOT) + t for w, t in ex["turns"]]
    return out


def train_tokenizer(raw: Path, chat: Path, vocab: int, sample_mb: float, out: Path) -> None:
    from training.tokenizer import BPETokenizer
    sample, size = [], 0
    for f in sorted(raw.glob("*.parquet")):
        for docs in texts(f):
            for d in docs:
                sample.append(d)
                size += len(d)
            if size > sample_mb * 2**20:
                break
        if size > sample_mb * 2**20:
            break
    talk = chat_texts(chat)
    log(f"learning {vocab} tokens from {size / 2**20:.0f} MB of text and {len(talk):,} chat lines...")
    tok = BPETokenizer.train(sample + talk * 3, vocab)
    tok.save(out)
    probe = "Good evening, sir. Kya haal hai? [web] who invented the telephone"
    log(f"saved {out}: {tok.vocab_size} tokens; {probe!r} -> {len(tok.encode(probe))} tokens")


# ------------------------------------------------------------------ encode
_TOK = None


def _init(tok_path: str) -> None:
    global _TOK
    from training.tokenizer import BPETokenizer
    _TOK = BPETokenizer.load(Path(tok_path))


def _encode(docs: list[str]) -> tuple[np.ndarray, int]:
    from training.tokenizer import BOS, EOS
    ids: list[int] = []
    for d in docs:
        ids.append(BOS)
        ids += _TOK.encode(d)
        ids.append(EOS)
    return np.asarray(ids, dtype=np.uint16), len(docs)


def encode(raw: Path, out: Path, tok_path: Path, workers: int | None = None) -> None:
    from multiprocessing import Pool
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*.parquet"))
    if not files:
        sys.exit("no corpus files: run `python -m training.lm.corpus download` first")
    with Pool(workers, initializer=_init, initargs=(str(tok_path),)) as pool:
        for k, f in enumerate(files):
            dest = out / f"{f.stem}.bin"
            if dest.exists():
                continue
            t0, n = time.time(), 0
            tmp = dest.with_suffix(".tmp")
            val = open(out / "val.bin", "wb") if k == 0 else None
            seen = 0
            with open(tmp, "wb") as fh:
                for arr, ndocs in pool.imap(_encode, texts(f), chunksize=1):
                    if val is not None and seen < VAL_DOCS:             # the first documents are held out
                        val.write(arr.tobytes())
                    else:
                        fh.write(arr.tobytes())
                        n += len(arr)
                    seen += ndocs
            if val is not None:
                val.close()
            tmp.replace(dest)
            log(f"{f.name}: {n / 1e6:.0f}M tokens in {time.time() - t0:.0f}s ({workers} workers)")


def status(raw: Path = RAW, out: Path = OUT) -> dict:
    files = sorted(raw.glob("*.parquet"))
    bins = [p for p in out.glob("*.bin") if p.name != "val.bin"]
    return {"downloaded": len(files), "gb": round(sum(p.stat().st_size for p in files) / 2**30, 2),
            "encoded": len(bins), "tokens": int(sum(p.stat().st_size for p in bins) // 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--files", type=int, default=4)
    t = sub.add_parser("tokenizer")
    t.add_argument("--chat", type=Path, default=Path("data/chat/train.jsonl"))
    t.add_argument("--vocab", type=int, default=16384)
    t.add_argument("--sample-mb", type=float, default=120)
    t.add_argument("--out", type=Path, default=Path("checkpoints/lm_tokenizer.json"))
    e = sub.add_parser("encode")
    e.add_argument("--tokenizer", type=Path, default=Path("checkpoints/lm_tokenizer.json"))
    e.add_argument("--workers", type=int, default=None)
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if args.cmd == "download":
        download(args.files)
    elif args.cmd == "tokenizer":
        train_tokenizer(RAW, args.chat, args.vocab, args.sample_mb, args.out)
    elif args.cmd == "encode":
        encode(RAW, OUT, args.tokenizer, args.workers)
    else:
        print(json.dumps(status(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
