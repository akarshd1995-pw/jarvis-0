"""Byte-level BPE tokenizer written from scratch (no external libraries).

One shared vocabulary for user text and DSL programs. Digits are kept as single
tokens so times and durations ("18:30", "45m") compose predictably.

    python -m training.tokenizer --data data/intents/train.jsonl --vocab 1024 --out checkpoints/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
PAD, BOS, EOS, SEP = range(4)
OFFSET = len(SPECIALS)          # byte b -> id b + OFFSET
PRETOK = re.compile(r" ?[A-Za-z]+| ?\d| ?[^\sA-Za-z\d]+|\s+")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        # id -> bytes, for decoding
        self.vocab: dict[int, bytes] = {i + OFFSET: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[OFFSET + 256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return OFFSET + 256 + len(self.merges)

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        word_counts = Counter(w for t in texts for w in PRETOK.findall(t))
        words = [[b + OFFSET for b in w.encode("utf-8")] for w in word_counts]
        counts = list(word_counts.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += counts[wi]
                where[pair].add(wi)

        merges = []
        n_merges = vocab_size - OFFSET - 256
        for m in range(n_merges):
            if not pair_counts:
                break
            best, freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if freq < 2:
                break
            new_id = OFFSET + 256 + m
            merges.append(best)
            for wi in list(where[best]):
                ids = words[wi]
                c = counts[wi]
                # remove old pair contributions for this word
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(wi)
                # merge
                out, i = [], 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                words[wi] = out
                for pair in zip(out, out[1:]):
                    pair_counts[pair] += c
                    where[pair].add(wi)
            where.pop(best, None)
            if verbose and (m + 1) % 250 == 0:
                print(f"  merge {m + 1}/{n_merges}  freq={freq}")
        return cls(merges)

    # ------------------------------------------------------------ encode / decode
    def _encode_word(self, w: str) -> list[int]:
        if w in self._cache:
            return self._cache[w]
        ids = [b + OFFSET for b in w.encode("utf-8")]
        while len(ids) > 1:
            pairs = [(self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            ids = ids[:i] + [OFFSET + 256 + rank] + ids[i + 2:]
        self._cache[w] = ids
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        out = [BOS] if bos else []
        for w in PRETOK.findall(text):
            out.extend(self._encode_word(w))
        if eos:
            out.append(EOS)
        return out

    def decode(self, ids) -> str:
        data = b"".join(self.vocab.get(int(i), b"") for i in ids if int(i) >= OFFSET)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(p) for p in raw["merges"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/intents/train.jsonl"))
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/tokenizer.json"))
    args = ap.parse_args()
    texts = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts += [ex["text"], ex["program"]]
    tok = BPETokenizer.train(texts, args.vocab)
    tok.save(args.out)
    sample = texts[0]
    print(f"vocab={tok.vocab_size}  example: {sample!r} -> {len(tok.encode(sample))} tokens")


if __name__ == "__main__":
    main()
