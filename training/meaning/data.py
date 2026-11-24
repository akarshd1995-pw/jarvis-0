"""Training pairs (query-like text, document-like text) for the meaning model.

1. concept pairs (generated): a document uses one word of a group, the query another
2. your own files (self-supervised, inverse cloze task): a few words cut from a passage -> the passage;
   a file's title -> its text; a passage -> the next one
"""
from __future__ import annotations

import random
import re
import sqlite3
from pathlib import Path

from .concepts import CONCEPTS, FILLER, NAMES


def _num(r: random.Random) -> str:
    return r.choice([f"₹{r.randint(50, 99999):,}", f"#{r.randint(1000, 99999)}", f"{r.randint(1, 28)} "
                     f"{r.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}",
                     str(r.randint(2019, 2027))])


def concept_pair(r: random.Random) -> tuple[str, str]:
    group = r.choice(CONCEPTS)
    a, b = r.sample(group, 2) if len(group) > 1 else (group[0], group[0])
    names = r.sample(NAMES, r.randint(1, 2))
    doc_bits = [a] + names + [r.choice(FILLER) for _ in range(r.randint(2, 6))] + [_num(r) for _ in range(r.randint(0, 2))]
    if r.random() < 0.3:                                  # a second concept as background noise
        doc_bits.append(r.choice(r.choice(CONCEPTS)))
    r.shuffle(doc_bits)
    doc = " ".join(doc_bits)
    if r.random() < 0.3:                                  # file-name style
        doc = "_".join(doc.split()[:6]) + r.choice([".pdf", ".png", ".docx", ".txt", ".jpg", ""])
    q_bits = [b] + ([r.choice(names)] if r.random() < 0.6 else [])
    r.shuffle(q_bits)
    query = " ".join(q_bits)
    if r.random() < 0.3:
        query = r.choice(["my ", "the ", "that ", ""]) + query + r.choice(["", " file", " pdf", " screenshot"])
    return query, doc


def passages(text: str, size: int = 400) -> list[str]:
    words = text.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if sum(len(x) + 1 for x in cur) >= size:
            out.append(" ".join(cur))
            cur = []
    if len(" ".join(cur)) > 40:
        out.append(" ".join(cur))
    return out


def load_user_docs(db: Path, limit: int = 200_000) -> list[tuple[str, list[str]]]:
    """[(title words, [passages])] from the memory index (text files, Office, PDFs, OCR'd images, page titles)."""
    if not db.exists():
        return []
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute("SELECT fts.title, fts.body FROM fts LIMIT ?", (limit,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out = []
    for title, body in rows:
        t = re.sub(r"[_\-.()\[\]]+", " ", re.sub(r"\.[A-Za-z0-9]{1,5}$", "", title or "")).strip()
        ps = passages(body or "")
        if t or ps:
            out.append((t, ps))
    return out


def user_pair(r: random.Random, docs: list) -> tuple[str, str] | None:
    title, ps = r.choice(docs)
    x = r.random()
    if ps and x < 0.5:                                    # inverse cloze: a few words -> their passage
        p = r.choice(ps)
        w = p.split()
        if len(w) < 8:
            return None
        k = r.randint(2, 6)
        i = r.randint(0, len(w) - k)
        span = " ".join(w[i:i + k])
        rest = " ".join(w[:i] + w[i + k:]) if r.random() < 0.6 else p      # usually remove the span itself
        return span, (title + " " + rest).strip()
    if ps and title and x < 0.8:                          # title -> text
        return title, r.choice(ps)
    if len(ps) >= 2:                                      # neighbouring passages
        i = r.randrange(len(ps) - 1)
        return ps[i][:200], ps[i + 1]
    if title:
        w = title.split()
        if len(w) >= 2:
            return " ".join(r.sample(w, max(1, len(w) // 2))), title
    return None


def pair(r: random.Random, docs: list, user_share: float = 0.5) -> tuple[str, str]:
    if docs and r.random() < user_share:
        p = user_pair(r, docs)
        if p:
            return p
    return concept_pair(r)
