"""Copy guard for decoding: names, paths, queries and reminder text must be copied from what you typed.

Every quoted value in the command language (query="...", name_contains="...", src="C:/...", text="...")
is, by construction of the training data, an exact substring of the user's sentence. The model still
sometimes garbles a rare name ("datasegekot" -> "dudegekot") or loops ("pe pe pe ..."). During beam
search we reject any token that would make a quoted value stop being a substring of the input, so
those outputs become impossible rather than unlikely.

The one exception is url=: "open zomato" -> url="zomato.com". A url may also be (a prefix of) a known
site address.
"""
from __future__ import annotations

import re

try:
    from datagen.lexicon import KNOWN_SITES as _KS
    KNOWN_URLS = tuple(sorted(set(_KS.values())))
except Exception:                                     # runtime without the data engine: allow any url
    KNOWN_URLS = None

_LIT = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)("?)')


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


def copy_ok(partial: str, source: str) -> bool:
    """False if any quoted value in the (possibly unfinished) program text is not copied from `source`."""
    for m in _LIT.finditer(partial):
        key, body, closed = m.group(1), m.group(2), m.group(3)
        if not closed and body.endswith("\\") and (len(body) - len(body.rstrip("\\"))) % 2:
            body = body[:-1]                          # half-written escape sequence
        value = _unescape(body)
        if not closed:
            value = value.rstrip("\ufffd")             # a multi-byte character (Hindi, Tamil...) half decoded
        if value in source:
            continue
        if key == "url":
            if KNOWN_URLS is None:
                continue
            if any(u == value or (not closed and u.startswith(value)) for u in KNOWN_URLS):
                continue
        return False
    return True
