"""One convention for song names, shared by the data engine and the runtime:
a trailing "song" is never part of the name ("play kalaavathi song" -> query="kalaavathi"),
unless it belongs to a phrase like "full song" / "video song" / "title song"."""
from __future__ import annotations

import re

_KEEP_BEFORE = {"full", "video", "title", "theme", "lyrical", "lyric", "new", "old", "best", "love", "sad", "party",
                "dance", "romantic", "item", "hit", "bhakti", "folk"}


def strip_song(query: str) -> str:
    m = re.match(r"^(.*\S)\s+songs?$", query.strip(), re.I)
    if not m:
        return query
    head = m.group(1)
    last = head.split()[-1].lower()
    return query if last in _KEEP_BEFORE or len(head.split()) == 0 else head


def tidy_program_text(dsl: str) -> str:
    """Apply strip_song to every play_media query in a program string."""
    def fix(m):
        return m.group(1) + strip_song(m.group(2)) + '"'
    out = []
    for call in dsl.split(" ; "):
        if call.startswith("play_media("):
            call = re.sub(r'(query=")((?:[^"\\]|\\.)*)"', fix, call)
        out.append(call)
    return " ; ".join(out)
