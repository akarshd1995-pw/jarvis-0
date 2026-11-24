"""How people describe what they're looking for: kind-of-thing words and time words (phase 7).

Shared by the data engine (training labels) and the repair used for models trained before memory filters.
"""
from __future__ import annotations

import re

from ..dsl.grammar import Duration, Ident, Program

KIND = {   # word -> (ext, where)
    "pdf": ("pdf", None), "pdf file": ("pdf", None), "doc": ("document", None), "document": ("document", None),
    "word doc": ("docx", None), "spreadsheet": ("xlsx", None), "excel sheet": ("xlsx", None), "ppt": ("pptx", None),
    "presentation": ("pptx", None), "photo": ("image", None), "image": ("image", None), "picture": ("image", None),
    "pic": ("image", None), "video": ("video", None), "recording": ("video", None), "song": ("audio", None),
    "zip": ("zip", None), "notebook": ("ipynb", None), "text file": ("txt", None), "file": (None, "files"),
    "screenshot": (None, "screenshots"), "screen grab": (None, "screenshots"), "snip": (None, "screenshots"),
    "website": (None, "web"), "page": (None, "web"), "web page": (None, "web"), "article": (None, "web"),
    "link": (None, "web"), "site": (None, "web"), "tab": (None, "web"), "blog post": (None, "web"),
    "video i watched": (None, "web"),
}
TIME = [("from yesterday", "2d"), ("yesterday", "2d"), ("from today", "1d"), ("today", "1d"),
            ("from last week", "7d"), ("from this week", "7d"), ("last week", "7d"), ("from last month", "30d"),
            ("this month", "30d"), ("from the last 3 days", "3d"), ("in the past 2 weeks", "14d"),
            ("from the last few days", "3d"), ("recently", "7d")]
TIME_HI = [("kal ka", "2d"), ("kal wala", "2d"), ("aaj ka", "1d"), ("pichhle hafte ka", "7d"),
               ("is hafte wala", "7d"), ("pichhle mahine ka", "30d")]


def split_query(text: str, query: str) -> tuple[str, dict]:
    """For older models that put everything in query: 'invoice pdf from last week' -> ('invoice',
    {'ext': 'pdf', 'newer_than': '7d'}). Kind/time words found in the user's text are moved to their own fields."""
    low = " " + text.lower() + " "
    found: dict = {}
    for phrase, d in sorted(TIME + TIME_HI, key=lambda x: -len(x[0])):
        if f" {phrase} " in low:
            found["newer_than"] = d
            query = re.sub(rf"\b{re.escape(phrase)}\b", " ", query, flags=re.I)
            break
    for word in sorted(KIND, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}s?\b", low):
            ext, where = KIND[word]
            if ext:
                found["ext"] = ext
            if where:
                found["where"] = where
            query = re.sub(rf"\b{re.escape(word)}s?\b", " ", query, flags=re.I)
            break
    query = re.sub(r"\b(i (?:read|saw|opened|visited|downloaded|watched|made|wrote|saved)|latest|last|newest|recent"
                   r"|the|that|my|a|an|about|with|of|wala|wali|ka|ki)\b", " ", query, flags=re.I)
    return re.sub(r"\s+", " ", query).strip(), found


def repair(prog: Program, text: str) -> Program:
    for call in prog.calls:
        if call.action != "search_memory" or any(k in call.args for k in ("ext", "where", "newer_than")):
            continue
        q, found = split_query(text, str(call.args.get("query", "")))
        if not found:
            continue
        if q:
            call.args["query"] = q
        else:
            call.args.pop("query", None)
        for k, v in found.items():
            call.args[k] = Duration(int(v[:-1]), v[-1]) if k == "newer_than" else Ident(v)
    return prog
