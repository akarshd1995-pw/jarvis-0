"""Realistic input noise applied to the non-slot parts of a rendered command.

Slot spans are wrapped in \\x01...\\x02 by the renderer and never modified, so the
target program always stays correct.
"""
from __future__ import annotations

import random
import re

L, R = "\x01", "\x02"

_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_ADJ: dict[str, str] = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        near = [row[c - 1] if c > 0 else "", row[c + 1] if c + 1 < len(row) else ""]
        if r > 0 and c < len(_ROWS[r - 1]):
            near.append(_ROWS[r - 1][c])
        if r + 1 < len(_ROWS) and c < len(_ROWS[r + 1]):
            near.append(_ROWS[r + 1][c])
        _ADJ[ch] = "".join(near)

CHAT_SPEAK = {"please": ["pls", "plz", "plss"], "you": ["u"], "your": ["ur"], "are": ["r"],
              "to": ["2"], "for": ["4"], "folder": ["fldr", "folde"], "files": ["fils", "filez"],
              "the": ["da", "teh"], "and": ["n", "&"], "reminder": ["remindr"], "minutes": ["mins", "min"],
              "seconds": ["secs"], "hours": ["hrs", "hr"], "tomorrow": ["tmrw", "tmr"], "message": ["msg"],
              "karo": ["kro", "krdo", "kardo"], "kar": ["kr"], "hai": ["h", "he"], "mein": ["me", "mai"],
              "do": ["doo"], "yaar": ["yr", "yrr"], "nahi": ["nhi", "nai"]}
DROPPABLE = {"the", "a", "an", "my", "all", "to", "please", "of"}


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.random()
    if op < 0.3:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]          # transpose
    if op < 0.55:
        return word[:i] + word[i + 1:]                                  # deletion
    if op < 0.75:
        return word[:i] + word[i] + word[i:]                            # duplication
    near = _ADJ.get(word[i].lower())
    if near:
        return word[:i] + rng.choice(near) + word[i + 1:]               # keyboard slip
    return word


def _noise_segment(seg: str, rng: random.Random, level: float) -> str:
    words = seg.split(" ")
    out = []
    for w in words:
        lw = w.lower()
        if lw in DROPPABLE and rng.random() < 0.08 * level:
            continue
        if lw in CHAT_SPEAK and rng.random() < 0.15 * level:
            w = rng.choice(CHAT_SPEAK[lw])
        elif w.isalpha() and rng.random() < 0.05 * level:
            w = typo(w, rng)
        out.append(w)
    seg = " ".join(out)
    if rng.random() < 0.05 * level:                                    # missing space
        seg = re.sub(r"(\w) (\w)", r"\1\2", seg, count=1)
    if rng.random() < 0.05 * level:                                    # doubled space
        seg = seg.replace(" ", "  ", 1)
    return seg


def apply_noise(text: str, rng: random.Random, level: float | None = None) -> str:
    """level 0 = clean, 1 = typical, 2 = very messy. Random per example by default."""
    if level is None:
        level = rng.choices([0.0, 1.0, 2.0], weights=[25, 55, 20])[0]
    casing = rng.choices(["keep", "lower", "upper", "title"], weights=[45, 45, 3, 7])[0]
    out = []
    first_is_slot = None
    for seg in re.split(f"({L}.*?{R})", text):
        if seg.startswith(L):
            out.append(seg[1:-1])
            if first_is_slot is None and seg[1:-1].strip():
                first_is_slot = True
            continue
        seg = _noise_segment(seg, rng, level) if level else seg
        seg = {"lower": seg.lower(), "upper": seg.upper(), "title": seg.title(), "keep": seg}[casing]
        out.append(seg)
        if first_is_slot is None and seg.strip():
            first_is_slot = False
    starts_with_slot = bool(first_is_slot)
    text = "".join(out)
    text = re.sub(r" {3,}", "  ", text).strip()
    if text and not starts_with_slot and rng.random() < 0.3:
        text = text[0].upper() + text[1:]
    r = rng.random()
    if r < 0.12:
        text += rng.choice([".", "!", "?", "...", " pls", " 🙏", " :)"])
    return text
