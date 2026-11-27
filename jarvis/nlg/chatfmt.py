"""Jarvis-LM (phase 27): the one chat format shared by training and runtime, the handoff markers, and the reply
filter. Kept free of numpy/torch so the data generator, the trainer and Jarvis all import the same rules.

A conversation becomes token ids like this:

    <bos> Jarvis|sir|butler|evening <sep> U: hi <sep> J: Good evening, sir. <sep> U: tell me a joke <sep> J: ... <eos>

The header carries the persona (name, how to address you, tone) and the part of the day, so one model speaks
for any name you give it. Only the J: turns are learned during fine-tuning.

The model never acts and never states facts it can't know. Instead of answering it may hand off:
    [web]  a factual question: Jarvis offers a web search instead of guessing
    [cmd]  it reads like a command the command model didn't catch: Jarvis says so rather than pretend
"""
from __future__ import annotations

import re

WEB, CMD = "[web]", "[cmd]"
USER, BOT = "U: ", "J: "
TONES = ("butler", "friendly", "minimal")
MAX_REPLY_WORDS = 60


def part_of_day(hour: int) -> str:
    if hour < 5 or hour >= 22:
        return "night"
    return "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"


def header(name: str, address: str, tone: str, hour: int) -> str:
    clean = lambda s: re.sub(r"[|\n]", " ", str(s or "")).strip()       # noqa: E731
    return f"{clean(name) or 'Jarvis'}|{clean(address)}|{tone if tone in TONES else 'butler'}|{part_of_day(hour)}"


def encode_dialog(tok, head: str, turns: list[tuple[str, str]], open_reply: bool = False,
                  max_len: int | None = None) -> tuple[list[int], list[int]]:
    """turns: [("U"|"J", text)]. Returns (ids, mask) where mask[i] = 1 for tokens the model should learn to say
    (each J reply and the token that ends it). open_reply=True ends with "J: " for the model to continue.
    If max_len is given, the oldest turns are dropped (the header always stays)."""
    from training.tokenizer import BOS, EOS, SEP
    head_ids = [BOS] + tok.encode(head) + [SEP]
    blocks: list[tuple[list[int], list[int]]] = []
    for i, (who, text) in enumerate(turns):
        last = i == len(turns) - 1
        if who == "U":
            ids = tok.encode(USER + text.strip()) + [SEP]
            blocks.append((ids, [0] * len(ids)))
        else:
            pre = tok.encode(BOT)
            body = tok.encode(text.strip()) + [EOS if last and not open_reply else SEP]
            blocks.append((pre + body, [0] * len(pre) + [1] * len(body)))
    tail = tok.encode(BOT) if open_reply else []
    if max_len is not None:
        budget = max_len - len(head_ids) - len(tail)
        while len(blocks) > 1 and sum(len(b[0]) for b in blocks) > budget:
            blocks.pop(0)
    ids, mask = list(head_ids), [0] * len(head_ids)
    for b, m in blocks:
        ids += b
        mask += m
    ids += tail
    mask += [0] * len(tail)
    if max_len is not None and len(ids) > max_len:                       # one huge message: keep its end
        ids, mask = head_ids + ids[-(max_len - len(head_ids)):], [0] * len(head_ids) + mask[-(max_len - len(head_ids)):]
    return ids, mask


# ------------------------------------------------------------------ what a reply may never do
_ACTION_CLAIM = re.compile(
    r"\b(?:i(?:'ve| have| just)? ?(?:opened|deleted|moved|closed|copied|renamed|set|turned (?:on|off)|started|stopped|sent|"
    r"played|created|installed|uninstalled|locked|muted|scheduled|saved|downloaded|cleared|emptied)\b|^(?:all )?done[.!]?$|"
    r"(?:it'?s|that'?s|all) (?:done|sorted|taken care of)\b|i'?ll (?:remind|wake) you\b)", re.I)
_CLOCK = re.compile(r"\b(?:it'?s|it is|the time is)\s+(?:\d{1,2}(?::\d\d)?\s*(?:am|pm)?|noon|midnight)\b|"
                    r"\btoday is (?:mon|tue|wed|thu|fri|sat|sun)", re.I)
_LINK = re.compile(r"https?://|www\.|\b[\w.+-]+@[\w-]+\.\w+|\.com\b|\.in\b", re.I)
_ROLE = re.compile(r"(?:^|\s)(?:U|J):\s")
# Kept short and literal: the model is trained on clean persona data, this only catches leaks from web text.
_BLOCK = re.compile(r"\b(?:fuck\w*|shit\w*|bitch\w*|bastard|cunt|dick(?:head)?|asshole|whore|slut|retard\w*|"
                    r"nigg\w*|fag\w*|chutiya|madarchod|bhenchod|behenchod|gandu|randi|bsdk|mc\b|bc\b)", re.I)


def problems(reply: str) -> list[str]:
    """Why a generated reply can't be shown ([] = fine). Handoffs are checked by parse_reply, not here."""
    out = []
    words = reply.split()
    if not words:
        return ["empty"]
    if len(words) > MAX_REPLY_WORDS:
        out.append("too long")
    if _ACTION_CLAIM.search(reply):
        out.append("claims an action")
    if _CLOCK.search(reply):
        out.append("states the time or date")
    if _LINK.search(reply):
        out.append("link or address")
    if _ROLE.search(reply) or WEB in reply or CMD in reply:
        out.append("format leak")
    if _BLOCK.search(reply):
        out.append("language")
    letters = sum(ch.isalpha() for ch in reply)
    if letters < 0.55 * len(reply.replace(" ", "")):
        out.append("not words")
    low = [w.lower().strip(".,!?") for w in words]
    grams = [tuple(low[i:i + 3]) for i in range(len(low) - 2)]
    if grams and max(grams.count(g) for g in set(grams)) >= 3:
        out.append("repeats itself")
    if reply.count("�"):
        out.append("broken text")
    return out


def parse_reply(text: str) -> tuple[str, str]:
    """('web'|'cmd'|'say', text). A handoff marker anywhere at the start wins over whatever follows it."""
    t = text.strip()
    if t.startswith(WEB):
        return "web", t[len(WEB):].strip()
    if t.startswith(CMD):
        return "cmd", t[len(CMD):].strip()
    return "say", t
