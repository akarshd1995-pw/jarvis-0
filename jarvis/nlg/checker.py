"""Fact checker for generated replies. A generated sentence is only shown if it:
  * mentions every fact the reply must contain (count, file type, folders, names, query, time...)
  * states no number, number word, folder, site or form of address that isn't in the facts
  * contains no leftover markup
Otherwise Jarvis falls back to the template sentence, so a wrong reply never reaches you.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..persona.templates import SITE_NAMES
from .bank import variants

FIELD_ORDER = ["count", "files", "files_pl", "filt", "src", "dest", "name", "query", "site", "text", "when",
               "request", "items", "action_phrase", "error", "opverb", "target", "level", "device", "delay",
               "pct", "eta", "best", "temp", "free_gb", "minutes", "metric", "value", "typical", "channel", "title", "closest", "place", "age", "more", "version", "time", "km", "via", "what", "thing", "about", "changes", "done", "miss", "tip", "known", "apps", "until", "quiet", "blocking",
               "extra", "left", "focused", "cycle", "msg"]
KNOWN_FOLDERS = ["Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos", "Archive"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
                "dozen", "couple"]
ADDRESSES = ["sir", "ma'am", "madam"]
_PH = re.compile(r"\{(\w+)\}")


def linearize(key: str, fields: dict, tone: str, address: str) -> str:
    """Facts -> model input, e.g. 'tone=butler | addr=sir | key=move | count=fourteen | files=PDFs | ...'"""
    parts = [f"tone={tone}", f"addr={address or '-'}", f"key={key}"]
    for k in FIELD_ORDER:
        v = fields.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={str(v).strip()}")
    return " | ".join(parts)


def parse_src(src: str) -> tuple[str, dict, str, str]:
    """Inverse of linearize -> (key, fields, tone, address)."""
    parts = dict(p.split("=", 1) for p in src.split(" | "))
    tone, addr, key = parts.pop("tone"), parts.pop("addr"), parts.pop("key")
    return key, parts, tone, "" if addr == "-" else addr


@lru_cache(maxsize=None)
def required_fields(tone: str, key: str) -> frozenset[str]:
    """Placeholders present in EVERY phrasing of this key: the facts any good reply must state."""
    sets = [{p.lower() for p in _PH.findall(v)} - {"a"} for v in variants(tone, key)]
    req = set.intersection(*sets) if sets else set()
    if "files_pl" in req:
        req = (req - {"files_pl"}) | {"files"}
    return frozenset(req)


@lru_cache(maxsize=None)
def static_text(tone: str, key: str) -> str:
    return " ".join(_PH.sub(" ", v) for v in variants(tone, key)).lower()


def _contains(out_l: str, value: str) -> bool:
    return value.strip().lower() in out_l


def check(key: str, fields: dict, tone: str, address: str, out: str) -> bool:
    if not out or len(out) > 400 or any(ch in out for ch in "{}|=") or "  " in out:
        return False
    out_l = out.lower()
    facts_l = " ".join(str(v) for k, v in fields.items() if v not in (None, "")).lower()
    for f in required_fields(tone, key):
        if f == "files":
            if not any(_contains(out_l, fields.get(x, "") or "\0") for x in ("files", "files_pl")):
                return False
        elif fields.get(f) and not _contains(out_l, str(fields[f])):
            return False
    static = static_text(tone, key)
    for num in re.findall(r"\d+", out):                                     # no invented numbers
        if num not in facts_l:
            return False
    for w in NUMBER_WORDS:
        if re.search(rf"\b{w}\b", out_l) and not re.search(rf"\b{w}\b", facts_l) \
                and not re.search(rf"\b{w}\b", static):
            return False
    for name in KNOWN_FOLDERS + list(SITE_NAMES.values()):                  # no invented folders or sites
        n = name.lower()
        if re.search(rf"\b{re.escape(n)}\b", out_l) and n not in facts_l and n not in static:
            return False
    for a in ADDRESSES:                                                     # address only as configured
        if re.search(rf"\b{re.escape(a)}\b", out_l) and a != (address or "").lower() and a not in facts_l:
            return False
    return True
