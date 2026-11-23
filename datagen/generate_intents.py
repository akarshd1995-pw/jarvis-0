"""Synthetic (text -> program) data engine, v3.

Program-first generation: sample a valid DSL program, then render it as English or
Hinglish using sentence frames, shared phrase lists, number words, time expressions,
open-vocabulary slots, coreference ("move them", "open it"), out-of-scope requests
(-> unknown()) and noise.

Splits:
  train / val   : in-distribution
  test_iid      : same frames, unseen examples
  val_ood       : HELD-OUT sentence frames (group A) -> used to pick the best checkpoint
  test_ood      : HELD-OUT sentence frames (group B) -> final, untouched robustness score

Only sentence FRAMES (word order / structure) are held out. Every phrase that carries
meaning ("last modified more than X ago", "throw away", "bring up", "khali karo", "A -> B")
lives in shared phrase lists used in training, and tests enforce that held-out frames
contain no word that training never shows.

    python -m datagen.generate_intents --n 2000000 --out data/intents
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from jarvis.dsl.grammar import DSLError, canonical
from jarvis.nlu.media_text import tidy_program_text
from . import lexicon as LX
from .noise import L, R, apply_noise


def slot(s: str) -> str:
    return f"{L}{s}{R}"


def q(s: str) -> str:
    """DSL string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


HOLDOUT_EVERY = 8          # frame index % 8 == 6 -> val_ood, == 7 -> test_ood
SPLIT_SLOT = {"val_ood": HOLDOUT_EVERY - 2, "test_ood": HOLDOUT_EVERY - 1}
ALL_FRAME_LISTS: dict[str, list[str]] = {}   # registry, used by tests for the vocabulary check


def frames(name: str, items: list[str]) -> list[str]:
    ALL_FRAME_LISTS[name] = items
    return items


def frame_split(i: int) -> str:
    r = i % HOLDOUT_EVERY
    for split, k in SPLIT_SLOT.items():
        if r == k:
            return split
    return "train"


class Ctx:
    def __init__(self, rng: random.Random, split: str):
        self.rng, self.split = rng, split   # split: train | val_ood | test_ood
        self.hing = rng.random() < 0.3

    def T(self, templates: list[str]) -> str:
        """Pick a sentence frame belonging to this split (lists shorter than 8 are never held out)."""
        if len(templates) < HOLDOUT_EVERY:
            return self.rng.choice(templates)
        pool = [t for i, t in enumerate(templates) if frame_split(i) == self.split]
        return self.rng.choice(pool or templates)


# ---------------------------------------------------------------- durations and times

def duration(c: Ctx, kind: str) -> tuple[str, str]:
    """kind: 'age' (file filters) or 'delay' (reminders). Returns (dsl, slot-wrapped surface)."""
    rng, h = c.rng, c.hing
    unit = rng.choices(["m", "h", "d", "w", "mo"], weights=[0, 2, 6, 3, 3] if kind == "age" else [7, 4, 1, 0.5, 0])[0]
    if unit == "m":
        # half round numbers, half arbitrary -> the model must copy digits, not snap to 15/30/45
        n = rng.choice([1, 2, 5, 10, 15, 20, 25, 30, 40, 45, 90]) if rng.random() < 0.5 else rng.randint(1, 120)
        special = {30: ["half an hour", "aadha ghanta" if h else "30 mins"], 90: ["an hour and a half",
                   "dedh ghanta" if h else "90 minutes"], 15: ["quarter of an hour", "15 min"]}
        if n in special and rng.random() < 0.35:
            return f"{n}m", slot(rng.choice(special[n]))
        N = LX.num_surface(rng, n, h)
        forms = [f"{N} minutes", f"{N} mins", f"{N} min", f"{n}m", f"{n} minutes"]
        if h:
            forms += [f"{N} minute", f"{N} min"]
        return f"{n}m", slot(rng.choice(forms) if n != 1 else rng.choice(["a minute", "1 min", "one minute"]))
    if unit == "h":
        n = rng.randint(1, 12) if rng.random() < 0.8 else rng.randint(13, 72)
        N = LX.num_surface(rng, n, h)
        if n == 1:
            forms = ["an hour", "1 hour", "one hour", "1 hr"] + (["ek ghanta", "ek ghante"] if h else [])
        else:
            forms = [f"{N} hours", f"{N} hrs", f"{n}h", f"{n} hours"] + ([f"{N} ghante"] if h else [])
        return f"{n}h", slot(rng.choice(forms))
    if unit == "d":
        n = rng.randint(1, 60) if rng.random() < 0.8 else rng.randint(61, 400)
        N = LX.num_surface(rng, n, h)
        if n == 1:
            forms = ["a day", "1 day", "one day"] + (["ek din"] if h else [])
        elif n == 2 and rng.random() < 0.2:
            forms = ["a couple of days"]
        else:
            forms = [f"{N} days", f"{n}d", f"{n} days"] + ([f"{N} din"] if h else [])
        return f"{n}d", slot(rng.choice(forms))
    if unit == "w":
        n = rng.randint(1, 8)
        N = LX.num_surface(rng, n, h)
        if n == 2 and rng.random() < 0.15:
            return "2w", slot("a fortnight")
        forms = (["a week", "1 week", "one week"] + (["ek hafta", "ek hafte"] if h else [])) if n == 1 else \
            [f"{N} weeks", f"{n}w", f"{n} weeks"] + ([f"{N} hafte"] if h else [])
        return f"{n}w", slot(rng.choice(forms))
    n = rng.randint(1, 12)
    if n == 12 and rng.random() < 0.4:
        return "360d", slot(rng.choice(["a year", "one year"] + (["ek saal"] if h else [])))
    N = LX.num_surface(rng, n, h)
    forms = (["a month", "1 month", "one month"] + (["ek mahina", "ek mahine"] if h else [])) if n == 1 else \
        [f"{N} months", f"{n} months"] + ([f"{N} mahine"] if h else [])
    return f"{n * 30}d", slot(rng.choice(forms))


def clock(c: Ctx) -> tuple[str, str]:
    """Returns (dsl HH:MM, phrase including its own preposition, e.g. 'at 6 pm' / 'shaam 6 baje')."""
    rng = c.rng
    r = rng.random()
    if r < 0.06:
        hh, mm, txt = 12, 0, rng.choice(["noon", "12 noon", "midday"])
        return f"{hh:02d}:{mm:02d}", (f"{slot(txt)}" if c.hing else f"at {slot(txt)}")
    if r < 0.1:
        return "00:00", (slot("raat 12 baje") if c.hing else f"at {slot('midnight')}")
    hh = rng.randint(0, 23)
    mm = rng.choice([0, 0, 15, 30, 45]) if rng.random() < 0.35 else rng.randint(0, 59)
    h12, ampm = (hh % 12 or 12), ("am" if hh < 12 else "pm")
    if c.hing and rng.random() < 0.7:
        part = "subah" if 4 <= hh < 12 else "dopahar" if 12 <= hh < 16 else "shaam" if 16 <= hh < 20 else "raat"
        mins = f":{mm:02d}" if mm else ""
        if mm == 30 and rng.random() < 0.4:
            return f"{hh:02d}:{mm:02d}", slot(f"{part} saadhe {h12} baje")
        return f"{hh:02d}:{mm:02d}", slot(f"{part} {h12}{mins} baje")
    forms = [f"{hh:02d}:{mm:02d}"]
    forms += [f"{h12}:{mm:02d} {ampm}", f"{h12}:{mm:02d}{ampm}", f"{h12}:{mm:02d} {ampm.upper()}"] if mm else \
        [f"{h12} {ampm}", f"{h12}{ampm}", f"{h12} {ampm.upper()}", f"{h12}:00 {ampm}"]
    if mm == 30:
        forms.append(f"half past {h12} {ampm}")
    if mm == 15:
        forms.append(f"quarter past {h12} {ampm}")
    if mm == 45:
        forms.append(f"quarter to {(h12 % 12) + 1} {ampm if h12 != 11 else ('pm' if ampm == 'am' else 'am')}")
    if mm == 0:
        part = "in the morning" if 4 <= hh < 12 else "in the afternoon" if 12 <= hh < 17 else \
            "in the evening" if 17 <= hh < 21 else "at night"
        if not (hh == 0 or (hh < 4)):
            forms.append(f"{h12} {part}")
    return f"{hh:02d}:{mm:02d}", f"at {slot(rng.choice(forms))}"

# ---------------------------------------------------------------- folder / file phrases

def folder(c: Ctx, exclude: str | None = None) -> tuple[str, str]:
    """Returns (dsl value, slot surface). 12% of the time a literal custom path."""
    rng = c.rng
    if rng.random() < 0.12:
        parts = [rng.choice(LX.PATH_ROOTS)] + [LX.path_part(rng) for _ in range(rng.randint(1, 3))]
        path = "/".join(parts)
        if rng.random() < 0.3:
            path = path.replace("/", "\\")
        return q(path), slot(path)
    key = rng.choice([k for k in LX.FOLDERS if k != exclude])
    return key, slot(rng.choice(LX.FOLDERS[key]))


GENERIC_WHAT_EN = ["files", "everything", "all files", "the files", "all the stuff", "stuff", "all items",
                   "all the files", "whatever is there"]
GENERIC_WHAT_HI = ["saari files", "sab kuch", "files", "saara saman", "sab files"]


def what_phrase(c: Ctx, args: dict) -> str:
    rng = c.rng
    if rng.random() < 0.72:
        e = rng.choice(LX.EXTS)
        args["ext"] = e
        forms = [f"{e} files", f"{e}s", f".{e} files", f"{e.upper()} files", f"all {e} files", f"all the {e}s",
                 f"every {e} file", f"*.{e} files", f"{e} documents" if e in ("pdf", "docx", "doc") else f"{e} files"]
        forms += LX.EXT_NICKNAMES.get(e, [])
        if c.hing:
            forms += [f"saari {e} files", f"{e} wali files", f"sab {e}", f"saare {e}"]
        return slot(rng.choice(forms))
    return rng.choice(GENERIC_WHAT_HI if c.hing else GENERIC_WHAT_EN)


# Shared filter phrases: never held out, so every way of saying "older"/"newer" is trained.
OLDER_EN = ["older than {d}", "more than {d} old", "from over {d} ago", "that are over {d} old",
            "not touched in {d}", "last modified more than {d} ago", "last modified over {d} ago",
            "aged over {d}", "that haven't changed in {d}", "untouched for {d}", "created more than {d} ago",
            "downloaded more than {d} ago", "at least {d} old", "not opened in {d}", "from before {d} ago",
            "sitting there for over {d}", "that are older than {d}", "over {d} old", "modified more than {d} ago"]
NEWER_EN = ["newer than {d}", "from the last {d}", "less than {d} old", "added in the past {d}",
            "created within {d}", "downloaded in the last {d}", "from the past {d}", "modified in the last {d}",
            "within the last {d}", "added less than {d} ago", "created in the last {d}", "downloaded within {d}",
            "that are under {d} old", "under {d} old", "modified less than {d} ago", "from the recent {d}",
            "last modified less than {d} ago", "that are newer than {d}"]
NAME_EN = ['with "{w}" in the name', "named {w}", "containing {w} in the filename", "that have {w} in their name",
           "whose name has {w}", "matching {w}", "called {w}", "with {w} in the filename", "whose names contain {w}",
           "having {w} in the name"]
OLDER_HI = ["{d} se purani", "jo {d} se purani hain", "{d} se zyada purani", "{d} pehle ki", "{d} se pehle ki",
            "{d} se zyada time se padi", "kam se kam {d} purani", "{d} se purane", "jo {d} se zyada purani hain"]
NEWER_HI = ["pichhle {d} ki", "{d} se nayi", "last {d} wali", "{d} ke andar ki", "pichhle {d} me aayi",
            "{d} se kam purani", "jo pichhle {d} me aayi hain", "{d} ke andar wali"]
NAME_HI = ["jinke naam me {w} hai", "{w} naam wali", "jinme {w} likha hai", "jiske naam me {w} ho", "{w} wali"]


def filters(c: Ctx, args: dict) -> str:
    rng, parts = c.rng, []
    if rng.random() < 0.45:
        d, txt = duration(c, "age")
        args["older_than"] = d
        parts.append(rng.choice(OLDER_HI if c.hing else OLDER_EN).format(d=txt))
    elif rng.random() < 0.2:
        d, txt = duration(c, "age")
        args["newer_than"] = d
        parts.append(rng.choice(NEWER_HI if c.hing else NEWER_EN).format(d=txt))
    if rng.random() < 0.2:
        w = LX.name_fragment(rng)
        args["name_contains"] = q(w)
        parts.append(rng.choice(NAME_HI if c.hing else NAME_EN).format(w=slot(w)))
    rng.shuffle(parts)
    if not parts:
        return ""
    return " " + rng.choice([" ", " ", ", ", " and "]).join(parts)


def dsl_call(action: str, args: dict) -> str:
    return f"{action}({', '.join(f'{k}={v}' for k, v in args.items())})"

# ---------------------------------------------------------------- verbs (shared, never held out)

VERBS = {
    "move": (["move", "shift", "transfer", "put", "relocate", "send", "drag", "move over", "carry", "push"],
             "moved", ["move kar do", "shift karo", "daal do", "move karo", "bhej do", "rakh do", "shift kar do",
                       "le jao", "daal de", "move kardo"]),
    "copy": (["copy", "duplicate", "back up", "clone", "make a copy of", "replicate", "copy over"],
             "copied", ["copy kar do", "copy karo", "ki copy bana do", "copy bana do", "copy kardo", "duplicate karo"]),
    "delete": (["delete", "remove", "trash", "erase", "clean up", "nuke", "throw away", "get rid of", "wipe",
                "clear out", "purge", "bin", "dump", "discard", "wipe out"],
               "deleted", ["delete kar do", "hata do", "uda do", "saaf kar do", "delete karo", "mita do", "hatao",
                           "nikaal do", "delete kardo", "fenk do"]),
    "list": (["list", "show", "show me", "display", "tell me", "give me", "find", "check", "get me", "list out"],
             "listed", ["dikhao", "list karo", "batao", "dikha do", "list kar do", "dikhana"]),
}

# ---------------------------------------------------------------- sentence frames (held out per split)

MOVE_EN = frames("MOVE_EN", [
    "{verb} {what}{filt} from {src} to {dest}",
    "{verb} {what} from {src}{filt} into {dest}",
    "{verb} {what}{filt} in {src} over to {dest}",
    "can you {verb} {what}{filt} from {src} to {dest}",
    "i want to {verb} {what}{filt} from {src} to {dest}",
    "{src} is a mess, {verb} {what}{filt} to {dest}",
    "could you {verb} {what}{filt} in {src} to {dest}",                 # val_ood
    "please {verb} {what}{filt} out of {src} into {dest}",              # test_ood
    "from {src}, {verb} {what}{filt} to {dest}",
    "could you {verb} {what}{filt} sitting in {src} to {dest}",
    "get {what}{filt} out of {src} and {verb} them to {dest}",
    "need {what}{filt} from {src} {pp} to {dest}",
    "please {verb} {what}{filt} from {src} into {dest}",
    "{what}{filt} in {src} should be {pp} to {dest}",
    "{verb} {what}{filt} to {dest} from {src}",                          # val_ood
    "{src}: {verb} {what}{filt} to {dest}",                              # test_ood
    "{verb} {what}{filt}: {src} -> {dest}",
    "help me {verb} {what}{filt} from {src} to {dest}",
    "i'd like {what}{filt} from {src} {pp} into {dest}",
    "quickly {verb} {what}{filt} out of {src} into {dest}",
    "{verb} {what}{filt} from {src} -> {dest}",
    "{src} -> {dest}: {verb} {what}{filt}",
    "can you {verb} {what}{filt} out of {src} and into {dest}",          # val_ood
    "i need {what}{filt} in {src} {pp} over to {dest}",                  # test_ood
])
MOVE_HI = frames("MOVE_HI", [
    "{src} se {what}{filt} {dest} me {hverb}",
    "{what}{filt} {src} se {dest} me {hverb}",
    "{src} wali {what}{filt} {dest} me {hverb}",
    "{dest} me {src} se {what}{filt} {hverb}",
    "{src} se {dest} me {what}{filt} {hverb}",
    "{src} ki {what}{filt} ko {dest} me {hverb}",
    "{src} ki {what}{filt} {dest} me {hverb}",                           # val_ood
    "bhai {what}{filt} ko {src} se {dest} me {hverb}",                   # test_ood
    "bhai {src} se {what}{filt} {dest} me {hverb}",
    "{what}{filt} ko {src} se nikaal ke {dest} me {hverb}",
    "{src} me jo {what}{filt} hai unko {dest} me {hverb}",
    "{src} se {dest} me {what}{filt} ko {hverb}",
    "zara {src} se {what}{filt} {dest} me {hverb}",
    "{what}{filt} {src} se uthake {dest} me {hverb}",
    "zara {src} me jo {what}{filt} hai unko {dest} me {hverb}",         # val_ood
    "{src} se nikaal ke {what}{filt} {dest} me {hverb}",                 # test_ood
])
DELETE_EN = frames("DELETE_EN", [
    "{verb} {what}{filt} from {src}",
    "{verb} {what} in {src}{filt}",
    "{verb} {what}{filt} in {src}",
    "can you {verb} {what}{filt} in {src}",
    "i don't need {what}{filt} in {src} anymore, {verb} them",
    "{src} is full, {verb} {what}{filt}",
    "could you {verb} {what}{filt} from {src}",                          # val_ood
    "{src} is a mess, {verb} {what}{filt}",                              # test_ood
    "{verb} {what}{filt} sitting in {src}",
    "free up space by deleting {what}{filt} from {src}",
    "please {verb} all {what}{filt} from {src}",
    "{src}: {verb} {what}{filt}",
    "{src} is a mess, please {verb} {what}{filt} in it",
    "could you please {verb} {what}{filt} in {src}",
    "i don't need {what}{filt} from {src}, {verb} them",                 # val_ood
    "please {verb} {what}{filt} sitting in {src}",                       # test_ood
    "go ahead and {verb} {what}{filt} in {src}",
    "{verb} {what}{filt} in {src} for good",
])
DELETE_HI = frames("DELETE_HI", [
    "{src} se {what}{filt} {hverb}",
    "{src} me {what}{filt} {hverb}",
    "{what}{filt} {src} se {hverb}",
    "{src} ki {what}{filt} {hverb}",
    "{src} me jo {what}{filt} hai unko {hverb}",
    "{src} khali karo, {what}{filt} {hverb}",
    "{src} ki saari {what}{filt} {hverb}",                               # val_ood
    "{src} me jo {what}{filt} hai {hverb}",                              # test_ood
    "bhai {src} se {what}{filt} {hverb}",
    "{src} ki saari {what}{filt} ko {hverb}",
    "{src} khali kar do, {what}{filt} {hverb}",
    "{what}{filt} ko {src} se {hverb}",
    "zara {src} me {what}{filt} {hverb}",
    "{src} saaf karna hai, {what}{filt} {hverb}",
    "zara {src} khali karo, {what}{filt} {hverb}",                       # val_ood
    "bhai {what}{filt} ko {src} me se {hverb}",                          # test_ood
])
LIST_EN = frames("LIST_EN", [
    "{verb} {what}{filt} in {src}",
    "what {what} are in {src}{filt}",
    "which {what}{filt} do i have in {src}",
    "how many {what}{filt} are in {src}",
    "{verb} {what} from {src}{filt}",
    "any {what}{filt} in {src}",
    "what {what}{filt} do i have in {src}",                              # val_ood
    "how many {what} are there in {src}{filt}",                          # test_ood
    "give me a list of {what}{filt} in {src}",
    "what's in {src}{filt}",
    "check {src} for {what}{filt}",
    "{src}: {verb} {what}{filt}",
    "are there any {what}{filt} in {src}",
    "can you {verb} {what}{filt} in {src}",
    "which {what} are in {src}{filt}",                                   # val_ood
    "what's inside {src}{filt}",                                         # test_ood
    "{verb} {what}{filt} inside {src}",
    "do i have any {what}{filt} in {src}",
])
LIST_HI = frames("LIST_HI", [
    "{src} me {what}{filt} {hverb}",
    "{src} ki {what}{filt} {hverb}",
    "{src} me kaunsi {what}{filt} hai",
    "{src} me kitni {what}{filt} hai",
    "{src} ka {what}{filt} {hverb}",
    "{src} me kya kya hai{filt}",
    "{src} me kitni {what}{filt} hain",                                  # val_ood
    "{src} ki kaunsi {what}{filt} hai",                                  # test_ood
    "zara {src} me {what}{filt} {hverb}",
    "{src} me konsi {what}{filt} hain",
    "{what}{filt} {src} me {hverb}",
    "bhai {src} ki {what}{filt} {hverb}",
])
FILE_FRAMES = {"move": (MOVE_EN, MOVE_HI), "copy": (MOVE_EN, MOVE_HI),
               "delete": (DELETE_EN, DELETE_HI), "list": (LIST_EN, LIST_HI)}


def gen_file_op(c: Ctx, op: str):
    rng = c.rng
    en, hi = FILE_FRAMES[op]
    tpl = c.T(hi if c.hing else en)
    args: dict = {}
    src_v, S = folder(c)
    args["src"] = src_v
    # Only describe a file type if the sentence frame actually mentions one (fixes label leakage).
    what = what_phrase(c, args) if "{what}" in tpl else ""
    filt = filters(c, args) if "{filt}" in tpl else ""
    meta = {"src_args": dict(args)}
    verbs, pp, hverbs = VERBS[op]
    fill = dict(verb=rng.choice(verbs), pp=pp, hverb=rng.choice(hverbs), what=what, filt=filt, src=S)
    if op in ("move", "copy"):
        dest_v, D = folder(c, exclude=src_v)
        if dest_v == src_v:
            dest_v, D = ("Archive", slot("archive")) if src_v != "Archive" else ("Documents", slot("documents"))
        args["dest"] = dest_v
        meta["dest"] = dest_v
        fill["dest"] = D
    return dsl_call(f"{op}_files", args), tpl.format(**fill), meta


REMIND_EN = frames("REMIND_EN", [
    "remind me to {task} {when}",
    "{when} remind me to {task}",
    "set a reminder to {task} {when}",
    "ping me {when} to {task}",
    "don't let me forget to {task} {when}",
    "can you remind me {when} to {task}",
    "nudge me {when} to {task}",                                         # val_ood
    "please set a reminder {when} to {task}",                            # test_ood
    "i need to {task} {when}, nudge me",
    "add a reminder {when}: {task}",
    "remind me {when} to {task}",
    "reminder to {task} {when}",
    "make sure i {task} {when}",
    "please remind me that i have to {task} {when}",
    "i have to {task} {when}, remind me",                                # val_ood
    "don't let me forget {when}: {task}",                                # test_ood
    "set an alarm {when} to {task}",
    "tell me to {task} {when}",
    "schedule a reminder {when} to {task}",
    "{when}, remind me: {task}",
    "i have to {task} {when}, don't let me forget",
    "alert me {when} to {task}",
])
REMIND_HI = ["{when} yaad dilana {task}", "{when} mujhe {task} yaad dila dena", "{task} ke liye {when} remind karna",
             "{when} remind kar dena ki {task}", "{when} bata dena {task}", "yaad dilana {when}: {task}",
             "{when} reminder laga do {task}"]


def gen_remind(c: Ctx):
    rng = c.rng
    task = LX.remind_text(rng)
    if rng.random() < 0.5:
        t, when = clock(c)
        dsl = f"remind(text={q(task)}, at={t})"
        if c.hing and when.startswith("at "):
            when = when[3:]
    else:
        d, txt = duration(c, "delay")
        dsl = f"remind(text={q(task)}, after={d})"
        if c.hing:
            when = rng.choice([f"{txt} me", f"{txt} baad", f"{txt} ke baad"])
        else:
            when = rng.choice([f"in {txt}", f"after {txt}", f"{txt} from now", f"in {txt}"])
    tpl = c.T(REMIND_HI if c.hing else REMIND_EN)
    return dsl, tpl.format(task=slot(task), when=when), {}


APP_V = ["open", "launch", "start", "fire up", "run", "bring up", "boot up", "load", "pull up", "open up"]
APP_EN = frames("APP_EN", [
    "{v} {a}", "can you {v} {a}", "please {v} {a}", "i need {a}", "{v} the {a} app", "{a} please",
    "could you {v} {a}",                                                 # val_ood
    "i want to {v} {a}",                                                 # test_ood
    "i want to use {a}", "could you please {v} {a}", "{v} {a} for me", "quickly {v} {a}",
    "{v} the {a} application", "get {a} running",
    "please {v} the {a} app",                                            # val_ood
    "can you quickly {v} {a}",                                           # test_ood
])
APP_HV = ["khol do", "open karo", "chalu karo", "start kar do", "kholo", "open kar do"]
APP_HI = ["{a} {hv}", "zara {a} {hv}", "{a} ko {hv}", "bhai {a} {hv}", "{a} jaldi {hv}", "mujhe {a} chahiye"]


def gen_open_app(c: Ctx):
    rng = c.rng
    if rng.random() < 0.6:
        key = rng.choice(list(LX.KNOWN_APPS))
        val, surface = key, rng.choice(LX.KNOWN_APPS[key])
    else:
        surface = rng.choice(LX.OTHER_APPS)
        val = q(surface)
    tpl = c.T(APP_HI if c.hing else APP_EN)
    return f"open_app(name={val})", tpl.format(a=slot(surface), v=rng.choice(APP_V), hv=rng.choice(APP_HV)), {}


URL_V = ["open", "go to", "take me to", "browse", "visit", "pull up", "load", "navigate to", "head over to",
         "show me", "bring up"]
URL_EN = frames("URL_EN", [
    "{v} {u}", "can you {v} {u}", "please {v} {u}", "{v} the website {u}", "{v} {u} for me", "i want to {v} {u}",
    "could you {v} {u}",                                                 # val_ood
    "please {v} the website {u}",                                        # test_ood
    "could you please {v} {u}", "{v} the site {u}", "quickly {v} {u}", "i need {u}",
])
URL_HI = ["{u} kholo", "{u} pe le chalo", "{u} open karo", "{u} wali site kholo", "zara {u} kholo"]


def gen_open_url(c: Ctx):
    rng = c.rng
    if rng.random() < 0.45:
        name = rng.choice(list(LX.KNOWN_SITES))
        url, surface = LX.KNOWN_SITES[name], name
        if rng.random() < 0.4:
            surface = url
    else:
        dom = LX.url_name(rng) + rng.choice(LX.TLDS)
        if rng.random() < 0.2:
            dom = rng.choice(["www.", "app.", "docs."]) + dom
        if rng.random() < 0.15:
            dom += rng.choice(["/about", "/login", "/blog", "/docs", "/pricing"])
        if rng.random() < 0.15:
            dom = rng.choice(["https://", "http://"]) + dom
        url = surface = dom
    tpl = c.T(URL_HI if c.hing else URL_EN)
    return f"open_url(url={q(url)})", tpl.format(u=slot(surface), v=rng.choice(URL_V)), {}


FOLDER_V = ["open", "show", "open up", "take me to", "go to", "bring up", "pull up", "navigate to", "pop open"]
FOLDER_EN = frames("FOLDER_EN", [
    "{v} {f}", "can you {v} {f}", "please {v} {f}", "{v} {f} in explorer", "i want to see {f}", "{v} {f} for me",
    "could you {v} {f}",                                                 # val_ood
    "please {v} {f} in explorer",                                        # test_ood
    "quickly {v} {f}", "could you please {v} {f}",
])
FOLDER_HV = ["kholo", "dikhao", "open karo", "khol do"]
FOLDER_HI = ["{f} {hv}", "zara {f} {hv}", "bhai {f} {hv}", "{f} ko {hv}"]


def gen_open_folder(c: Ctx):
    v, F = folder(c)
    tpl = c.T(FOLDER_HI if c.hing else FOLDER_EN)
    return f"open_folder(path={v})", tpl.format(f=F, v=c.rng.choice(FOLDER_V), hv=c.rng.choice(FOLDER_HV)), {}


def gen_system(c: Ctx):
    rng = c.rng
    key = rng.choice(list(LX.SYSTEM))
    phrase = rng.choice(LX.SYSTEM[key])
    pre = rng.choice(["", "", "", "please ", "can you ", "quickly ", "hey "])
    return f"system(action={key})", pre + phrase, {}


SEARCH_V = ["find", "search for", "look up", "locate", "dig up", "find me", "search my files for", "track down"]
SEARCH_EN = frames("SEARCH_EN", [
    "{v} {q}", "can you {v} {q}", "where is {q}", "where did i save {q}", "i'm looking for {q}", "please {v} {q}",
    "could you {v} {q}",                                                 # val_ood
    "please help me {v} {q}",                                            # test_ood
    "do you remember {q}", "{v} {q} in my files", "help me {v} {q}", "search memory for {q}",
    "could you please {v} {q}",
])
SEARCH_HI = ["{q} dhundo", "{q} kahan hai", "{q} search karo", "{q} dhoondh ke do", "mera {q} kahan rakha hai"]


def gen_search(c: Ctx):
    query = LX.search_query(c.rng)
    tpl = c.T(SEARCH_HI if c.hing else SEARCH_EN)
    return f"search_memory(query={q(query)})", tpl.format(q=slot(query), v=c.rng.choice(SEARCH_V)), {}


# ---------------------------------------------------------------- web search ("play X on youtube")
# "play" verbs -> play_media (the video actually starts); "search" verbs -> search_web (results page)
WEB_V = {"play": ["play", "put on", "stream", "start playing", "play me"],
         "music_search": ["search for", "find", "look up", "look for"],
         "other": ["search for", "look up", "find", "search", "check", "look for"]}
WEB_EN = frames("WEB_EN", [
    "{v} {qs} on {site}",
    "open {site} and {v} {qs}",
    "can you {v} {qs} on {site}",
    "{v} {qs} on {site} please",
    "go to {site} and {v} {qs}",
    "search {site} for {qs}",
    "could you {v} {qs} on {site}",                                       # val_ood
    "open {site} and then {v} {qs}",                                      # test_ood
    "i want to {v} {qs} on {site}",
    "{site}: {v} {qs}",
    "use {site} to {v} {qs}",
    "quickly {v} {qs} on {site}",
    "{v} {qs} using {site}",
    "please open {site} and {v} {qs}",
    "go to {site} and then {v} {qs}",                                     # val_ood
    "can you open {site} and {v} {qs}",                                   # test_ood
    "open {site}, then {v} {qs}",
    "on {site}, {v} {qs}",
    "could you please {v} {qs} on {site}",
])
WEB_HI = ["{site} pe {qs} {hv}", "{site} kholo aur {qs} {hv}", "{qs} {site} pe {hv}", "{site} khol ke {qs} {hv}",
          "zara {site} pe {qs} {hv}", "{site} me {qs} {hv}"]
WEB_HV = {"play": ["chalao", "bajao", "lagao", "play karo", "chala do", "sunao"],
          "music_search": ["search karo", "dhundo", "search kar do"],
          "other": ["search karo", "dhundo", "dhoondho", "search kar do", "dekho"]}
NOSITE_MUSIC = ["play {qs}", "put on {qs}", "play the song {qs}", "{qs} chalao", "{qs} bajao", "can you play {qs}",
                "play {qs} for me", "i want to listen to {qs}", "{qs} gaana lagao"]
NOSITE_GOOGLE = ["google {qs}", "search the web for {qs}", "search online for {qs}", "look up {qs} online",
                 "{qs} google karo", "do a google search for {qs}", "web search {qs}"]


def gen_search_web(c: Ctx):
    """Searches (results pages). Playing things is gen_play."""
    rng = c.rng
    site = rng.choice(list(LX.SEARCH_SITES))
    if rng.random() < 0.12:                       # "google X" -> Google
        query = LX.web_query(rng, "google")
        return f"search_web(query={q(query)}, site=google)", rng.choice(NOSITE_GOOGLE).format(qs=slot(query)), {}
    if site in ("youtube", "spotify"):
        query = LX.video_query(rng) if site == "youtube" and rng.random() < 0.5 else LX.song_query(rng)
        kind = "music_search"
    else:
        query, kind = LX.web_query(rng, site), "other"
    tpl = c.T(WEB_HI if c.hing else WEB_EN)
    text = tpl.format(v=rng.choice(WEB_V[kind]), hv=rng.choice(WEB_HV[kind]), qs=slot(query),
                      site=slot(rng.choice(LX.SEARCH_SITES[site])))
    return f"search_web(query={q(query)}, site={site})", text, {}


# ---------------------------------------------------------------- playing things (vast): top / latest / playlist / live / channel
PLAY_TOP_EN = ["play {q}", "put on {q}", "play {q} on youtube", "can you play {q}", "i want to listen to {q}",
               "play me {q}", "play the song {q}", "play the video {q}", "watch {q}", "i want to watch {q}",
               "play {q} for me", "youtube {q} play karo", "open youtube and play {q}", "play {q} please",
               "could you put on {q}", "let's listen to {q}", "throw on {q}", "jarvis play {q}",
               "play {q} song", "play {q} video", "can you play {q} please", "play {q} now", "pls play {q}"]
PLAY_TOP_HI = ["{q} chalao", "{q} bajao", "{q} lagao", "youtube pe {q} chalao", "{q} play karo", "{q} sunao",
               "zara {q} chala do", "{q} ka gaana bajao", "youtube pe {q} lagao", "{q} song chalao",
               "{q} gaana lagao", "{q} play kar do", "{q} baja do", "{q} video chalao"]
PLAY_LATEST_EN = ["play {c} latest video", "play the latest video from {c}", "play {c}'s newest video",
                  "play {c}'s latest upload", "play the new {c} video", "watch {c}'s latest video",
                  "play the most recent {c} video", "put on the newest {c} upload", "play {c} latest video on youtube",
                  "open youtube and play the latest video from {c}", "what did {c} upload last? play it",
                  "play the latest {c} video", "watch the newest video by {c}", "play {c}'s last video",
                  "play the latest upload from {c}", "i want to watch {c}'s newest video"]
PLAY_LATEST_HI = ["{c} ka latest video chalao", "{c} ki nayi video lagao", "{c} ka naya video dikhao",
                  "latest {c} video play karo", "{c} ka last video chalao", "youtube pe {c} ka latest video chalao",
                  "{c} ki latest video dikhao"]
PLAY_PLAYLIST = ["play a {g} playlist", "put on some {g} mix", "play {g} playlist", "play a {g} mix on youtube",
                 "{g} playlist chalao", "{g} mix lagao", "play the best {g} playlist", "start a {g} playlist"]
PLAY_LIVE = ["play {e} live", "watch the {e} live stream", "put on {e} live", "{e} live chalao", "play the {e} livestream",
             "watch {e} live", "open {e} live on youtube"]
OPEN_CHANNEL = ["open {c} channel", "go to {c}'s youtube channel", "show me {c}'s channel", "{c} ka channel kholo",
                "open the {c} youtube channel", "take me to {c}'s channel"]
PLAY_SPOTIFY = ["play {q} on spotify", "spotify pe {q} chalao", "put on {q} on spotify", "open spotify and play {q}"]


def gen_play(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.38:                                                   # the top result
        rq = rng.random()
        query = LX.song_query(rng) if rq < 0.5 else LX.open_query(rng) if rq < 0.65 else LX.video_query(rng)
        tpl = rng.choice(PLAY_TOP_HI if c.hing else PLAY_TOP_EN)
        return f"play_media(query={q(query)}, site=youtube)", tpl.format(q=slot(query)), {}
    if r < 0.65:                                                   # newest upload of a channel
        ch = LX.creator(rng)
        surf = ch if rng.random() < 0.7 else ch.lower()
        tpl = rng.choice(PLAY_LATEST_HI if c.hing else PLAY_LATEST_EN)
        return f"play_media(mode=latest, query={q(surf)}, site=youtube)", tpl.format(c=slot(surf)), {}
    if r < 0.77:
        g = rng.choice(LX.GENRES_MORE + LX.ARTISTS_MORE[:20])
        return f"play_media(mode=playlist, query={q(g)}, site=youtube)", rng.choice(PLAY_PLAYLIST).format(g=slot(g)), {}
    if r < 0.85:
        e = rng.choice(LX.LIVE_THINGS)
        return f"play_media(mode=live, query={q(e)}, site=youtube)", rng.choice(PLAY_LIVE).format(e=slot(e)), {}
    if r < 0.93:
        ch = LX.creator(rng)
        return f"play_media(mode=channel, query={q(ch)}, site=youtube)", rng.choice(OPEN_CHANNEL).format(c=slot(ch)), {}
    query = LX.song_query(rng)
    return f"play_media(query={q(query)}, site=spotify)", rng.choice(PLAY_SPOTIFY).format(q=slot(query)), {}


# ---------------------------------------------------------------- safety data: negations and fragments -> unknown()

NEG_EN = ["don't {verb} {what} in {src}", "do not {verb} {what} from {src}", "never {verb} {what} in {src}",
          "please don't {verb} anything in {src}", "don't {verb} my files", "do not {verb} anything",
          "wait, don't {verb} {what}", "stop, don't {verb} {what} from {src}", "make sure you don't {verb} {what} in {src}",
          "you shouldn't {verb} {what} in {src}", "i said don't {verb} {what}"]
NEG_HI = ["{src} se {what} mat {hverb}", "{src} ki files mat {hverb}", "kuch bhi mat {hverb}", "{what} mat {hverb}",
          "ruko, {src} se kuch mat {hverb}"]
NEG_HVERB = ["hatao", "delete karna", "move karna", "chhedna", "copy karna", "udana"]


def gen_negation(c: Ctx):
    rng = c.rng
    op = rng.choice(["delete", "delete", "move", "copy"])
    src_v, S = folder(c)
    what = what_phrase(c, {})
    if c.hing:
        text = rng.choice(NEG_HI).format(src=S, what=what, hverb=rng.choice(NEG_HVERB))
    else:
        text = rng.choice(NEG_EN).format(verb=rng.choice(VERBS[op][0]), what=what, src=S)
    return "unknown()", text


def gen_fragment(c: Ctx):
    """One or two words that are not a full command (often typo'd) -> unknown()."""
    from .noise import typo
    rng = c.rng
    # ("lock" is excluded: "lock" alone is a valid command -> system(action=lock))
    w = rng.choice(["open", "delete", "move", "copy", "show", "list", "remind", "play", "search", "find",
                    "launch", "start", "remove", "send", "kholo", "hatao", "dikhao", "files", "folder", "the", "it"])
    if rng.random() < 0.5:
        w = typo(w, rng)
    if rng.random() < 0.3:
        w += " " + rng.choice(["it", "that", "this", "please", "now", "pls", "yaar", "the", "kar do"])
    return "unknown()", w


UNDO = ["undo", "undo that", "revert the last command", "undo the last thing", "go back", "cancel what you just did",
        "revert that", "undo last action", "oops undo", "put it back", "reverse that", "wapas kar do",
        "undo kar do", "pehle jaisa kar do", "galti ho gayi, undo karo"]


def gen_undo(c: Ctx):
    return "undo()", c.rng.choice(UNDO), {}


# ---------------------------------------------------------------- phase 5: system control
WIN_PHRASES = {
    "snap_left": ["snap {t} to the left", "put {t} on the left", "move {t} to the left half", "snap {t} left",
                  "{t} left side pe karo", "{t} ko left me lagao"],
    "snap_right": ["snap {t} to the right", "put {t} on the right", "move {t} to the right half", "snap {t} right",
                   "{t} right side pe karo", "{t} ko right me lagao"],
    "maximize": ["maximize {t}", "maximise {t}", "make {t} full screen", "{t} full screen karo", "make {t} bigger"],
    "minimize": ["minimize {t}", "minimise {t}", "hide {t}", "{t} minimize karo", "{t} chhota karo"],
    "restore": ["bring back {t}", "restore {t}", "switch to {t}", "{t} wapas lao", "bring {t} to the front"],
    "close": ["close {t}", "quit {t}", "exit {t}", "{t} band karo", "shut {t}"],
}
WIN_ALL = ["minimize everything", "minimize all windows", "hide all windows", "clear my screen", "sab windows minimize karo",
           "minimise all the windows"]
THIS_WINDOW = ["this window", "this", "the window", "the current window", "it", "this app", "ye window"]


def gen_window(c: Ctx):
    rng = c.rng
    if rng.random() < 0.12:
        return "window(action=minimize_all)", rng.choice(WIN_ALL), {}
    action = rng.choice(list(WIN_PHRASES))
    if rng.random() < 0.5:
        return f"window(action={action})", rng.choice(WIN_PHRASES[action]).format(t=rng.choice(THIS_WINDOW)), {}
    if rng.random() < 0.6:
        key = rng.choice(list(LX.KNOWN_APPS))
        val, surface = key, rng.choice([s for s in LX.KNOWN_APPS[key] if s not in ("the browser", "browser", "my computer")]
                                       or LX.KNOWN_APPS[key])
    else:
        surface = rng.choice(LX.OTHER_APPS)
        val = q(surface)
    return f"window(action={action}, app={val})", rng.choice(WIN_PHRASES[action]).format(t=slot(surface)), {}


DESKTOP_PHRASES = {
    "new": ["new desktop", "create a new virtual desktop", "open a new desktop", "naya desktop banao", "add a desktop"],
    "next": ["next desktop", "switch to the next desktop", "go to the right desktop", "agla desktop", "next virtual desktop"],
    "previous": ["previous desktop", "go back a desktop", "switch to the left desktop", "pichla desktop",
                 "previous virtual desktop"],
    "close": ["close this desktop", "close the virtual desktop", "ye desktop band karo", "remove this desktop"],
}
MEDIA_PHRASES = {
    "play_pause": ["pause", "pause the music", "resume", "resume playback", "pause the video", "play pause",
                   "gaana roko", "music pause karo", "continue playing", "resume the song", "pause it"],
    "next": ["next song", "next track", "skip", "skip this song", "play the next one", "agla gaana", "skip track"],
    "previous": ["previous song", "previous track", "go back a song", "play the last song again", "pichla gaana",
                 "last track"],
    "stop": ["stop the music", "stop playback", "gaana band karo", "stop the song"],
}
BRIGHT_SET = ["set brightness to {n}", "brightness {n}", "brightness to {n} percent", "make the screen {n} percent bright",
              "brightness {n} karo", "screen brightness {n} percent kar do", "set the screen brightness at {n}%"]
BRIGHT_STEP = {"up": ["increase brightness", "brightness up", "make the screen brighter", "brighter please",
                      "brightness badhao", "screen bright karo", "turn the brightness up"],
               "down": ["decrease brightness", "dim the screen", "brightness down", "make the screen darker",
                        "brightness kam karo", "screen dim karo", "turn the brightness down"]}
VOLUME_SET = ["set volume to {n}", "volume {n} percent", "put the volume at {n}", "volume to {n}", "volume {n} kar do",
              "awaaz {n} percent karo", "set the volume at {n}%"]
RADIO_KIND = {"wifi": ["wifi", "wi-fi", "WiFi", "the wifi", "wireless"], "bluetooth": ["bluetooth", "BT", "the bluetooth"]}
RADIO_PHRASES = {"on": ["turn on {k}", "enable {k}", "switch on {k}", "{k} on", "{k} on karo", "{k} chalu karo"],
                 "off": ["turn off {k}", "disable {k}", "switch off {k}", "{k} off", "{k} off karo", "{k} band karo"]}
SHOT_PHRASES = {
    "full": ["take a screenshot", "screenshot", "capture the screen", "grab a screenshot of my screen", "screenshot le lo",
             "screen capture karo", "screenshot the whole screen"],
    "window": ["screenshot this window", "capture the current window", "copy a screenshot of this window",
               "is window ka screenshot lo"],
    "region": ["screenshot an area", "snip part of the screen", "take a partial screenshot",
               "select an area to screenshot", "open the snipping tool", "screen ka hissa capture karo"],
}
RECORD_PHRASES = {"start": ["start recording", "record my screen", "start screen recording", "record this",
                            "recording shuru karo", "screen record karo"],
                  "stop": ["stop recording", "end the recording", "stop screen recording", "recording band karo",
                           "finish recording"]}
POWER_PHRASES = {
    "shutdown": ["shut down", "shut down the pc", "turn off the computer", "power off", "shut down the laptop",
                 "pc band karo", "computer band kar do"],
    "restart": ["restart", "restart the pc", "reboot", "restart my computer", "pc restart karo", "reboot the system"],
    "sleep": ["put the pc to sleep", "go to sleep mode", "sleep mode", "pc ko sula do", "suspend the computer"],
    "sign_out": ["sign out", "log out", "log me out", "logout karo", "sign me out of windows"],
    "cancel": ["cancel shutdown", "abort the shutdown", "don't shut down", "cancel the restart", "shutdown cancel karo",
               "stop the shutdown"],
}
POWER_DELAY = {"shutdown": ["shut down in {d}", "shut down the pc after {d}", "turn off the computer in {d}",
                            "{d} me pc band kar do", "power off in {d}"],
               "restart": ["restart in {d}", "reboot after {d}", "restart the pc in {d}", "{d} baad restart karo"]}


def _level(c: Ctx) -> tuple[int, str]:
    n = c.rng.choice([0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, c.rng.randint(0, 100)])
    return n, slot(str(n))


def gen_desktop(c: Ctx):
    a = c.rng.choice(list(DESKTOP_PHRASES))
    return f"desktop(action={a})", c.rng.choice(DESKTOP_PHRASES[a]), {}


def gen_media(c: Ctx):
    a = c.rng.choice(list(MEDIA_PHRASES))
    return f"media(action={a})", c.rng.choice(MEDIA_PHRASES[a]), {}


def gen_brightness(c: Ctx):
    rng = c.rng
    if rng.random() < 0.5:
        n, surf = _level(c)
        return f"brightness(level={n})", rng.choice(BRIGHT_SET).format(n=surf), {}
    a = rng.choice(["up", "down"])
    return f"brightness(action={a})", rng.choice(BRIGHT_STEP[a]), {}


def gen_set_volume(c: Ctx):
    n, surf = _level(c)
    return f"set_volume(level={n})", c.rng.choice(VOLUME_SET).format(n=surf), {}


def gen_radio(c: Ctx):
    rng = c.rng
    kind, state = rng.choice(["wifi", "bluetooth"]), rng.choice(["on", "off"])
    return (f"radio(kind={kind}, state={state})",
            rng.choice(RADIO_PHRASES[state]).format(k=rng.choice(RADIO_KIND[kind])), {})


def gen_screenshot(c: Ctx):
    mode = c.rng.choice(["full", "full", "window", "region"])
    dsl = f"screenshot(mode={mode})"
    return dsl, c.rng.choice(SHOT_PHRASES[mode]), {}


def gen_record(c: Ctx):
    a = c.rng.choice(["start", "stop"])
    return f"screen_record(action={a})", c.rng.choice(RECORD_PHRASES[a]), {}


def gen_power(c: Ctx):
    rng = c.rng
    a = rng.choice(list(POWER_PHRASES))
    if a in POWER_DELAY and rng.random() < 0.4:
        d, txt = duration(c, "delay")
        return f"power(action={a}, after={d})", rng.choice(POWER_DELAY[a]).format(d=txt), {}
    return f"power(action={a})", rng.choice(POWER_PHRASES[a]), {}


TRAINING_PHRASES = {
    "pause": ["pause training", "pause the training", "hold the training", "pause the training run", "training roko",
              "training pause karo", "put training on hold"],
    "resume": ["resume training", "continue the training", "unpause training", "resume the training run",
               "training wapas shuru karo", "training continue karo"],
    "status": ["how is the training going", "training status", "is the training done", "how far along is training",
               "training kitna hua", "training ka status batao", "how much training is left"],
}


def gen_training(c: Ctx):
    a = c.rng.choice(list(TRAINING_PHRASES))
    return f"training(action={a})", c.rng.choice(TRAINING_PHRASES[a]), {}


# ---------------------------------------------------------------- mood: how the user feels (+ maybe music)
# Self-harm language is deliberately NOT generated: it is caught by fixed rules (jarvis/persona/mood.py), never the model.

FEEL_EN = {
    "low": ["i'm feeling low", "i am feeling sad", "my mood is bad", "i'm in a bad mood", "i feel down", "i had a bad day",
            "today was rough", "i'm not feeling good", "i'm upset", "feeling a bit blue", "i'm heartbroken",
            "i'm really sad today", "my mood is off", "i feel like crying", "i'm not okay", "worst day ever honestly",
            "i feel miserable", "i'm so done with today", "i feel empty today", "kind of feeling down rn",
            "my day was terrible", "i got bad news today", "i failed my exam and feel awful", "my mood is low",
            "i'm feeling gloomy", "nothing is going right today", "i feel unhappy", "i'm having a rough day"],
    "stressed": ["i'm so stressed", "i'm stressed out", "i feel anxious", "i'm really tense", "so much pressure right now",
                 "i'm overwhelmed", "work is killing me", "i'm worried about tomorrow", "deadlines are stressing me out",
                 "i'm nervous about the interview", "too much on my plate", "i'm panicking a bit", "my head is spinning"],
    "tired": ["i'm so tired", "i'm exhausted", "i feel drained", "i'm worn out", "long day, i'm beat", "i'm burnt out",
              "i'm sleepy", "totally drained after work", "i'm dead tired", "i have no energy left"],
    "bored": ["i'm bored", "i'm so bored", "nothing to do", "this is boring", "bored out of my mind", "i'm getting bored",
              "so bored right now"],
    "angry": ["i'm so angry", "i'm frustrated", "i'm irritated", "i'm pissed off", "i'm annoyed", "everything is annoying me",
              "i'm furious right now", "this bug is making me mad"],
    "lonely": ["i feel lonely", "i'm all alone", "i'm missing home", "i feel so alone", "nobody is around", "i'm missing someone"],
    "happy": ["i'm so happy", "i'm in a great mood", "feeling awesome today", "i got the job", "i'm excited", "best day ever",
              "i passed my exam", "i'm feeling great", "good mood today"],
    "focus": ["i need to focus", "i have to study", "time for deep work", "i can't concentrate", "i want to get some work done",
              "exam tomorrow, need to study", "help me focus"],
    "sleep": ["i can't sleep", "i need to sleep", "help me sleep", "trying to fall asleep", "can't fall asleep tonight"],
}
FEEL_HI = {
    "low": ["mood kharab hai", "mera mood off hai", "main bahut udaas hoon", "aaj ka din bura tha", "dil toota hua hai",
            "mood theek nahi hai", "bahut dukhi hoon", "kuch accha nahi lag raha", "man nahi lag raha", "mood low hai yaar"],
    "stressed": ["bahut tension hai", "bahut stress hai", "main pareshaan hoon", "kaam ka bahut pressure hai", "tension ho rahi hai"],
    "tired": ["bahut thak gaya hoon", "thak gayi hoon", "aaj bahut thakaan hai", "energy nahi hai"],
    "bored": ["bore ho raha hoon", "bahut bore ho rahi hoon", "kuch karne ko nahi hai"],
    "angry": ["bahut gussa aa raha hai", "dimaag kharab ho gaya hai", "bahut irritate ho raha hoon"],
    "lonely": ["akela feel ho raha hai", "main akeli hoon", "ghar ki yaad aa rahi hai"],
    "happy": ["aaj bahut khush hoon", "mood mast hai", "aaj ka din accha tha"],
    "focus": ["padhai karni hai", "focus karna hai", "kaam pe dhyan nahi lag raha"],
    "sleep": ["neend nahi aa rahi", "sona hai par neend nahi aa rahi"],
}
MOOD_MUSIC_EN = frames("MOOD_MUSIC_EN", [
    "play some music", "play something to lift my mood", "put on some songs", "play something nice", "play some music for me",
    "can you play something", "play some songs to cheer me up", "put on something to make me feel better",
    "play something relaxing", "play some good music", "throw on some music", "play some tunes", "put some music on",
    "play a few songs", "some music please", "play something to help", "play music that fits", "play me something good",
    "play songs to cheer me up please", "help me feel better with some music, make it up beat",
])
MOOD_MUSIC_HI = frames("MOOD_MUSIC_HI", [
    "kuch gaana lagao", "koi gaana chalao", "kuch music bajao", "kuch accha sunao", "gaane lagao", "koi song chala do",
    "music chalao", "kuch songs bajao", "kuch accha chalao", "koi accha gaana sunao", "songs chalao",
])
MOOD_LIFT = {"low": ["cheer me up", "lift my mood", "play something to cheer me up", "i need a mood booster",
                     "make me feel better", "play some feel good music", "mood fresh kar do", "mood accha kar do",
                     "brighten my day", "i need some good vibes"],
             "stressed": ["help me relax", "calm me down", "play something to calm me down", "i need to relax"],
             "focus": ["play something to help me focus", "help me focus with some music"],
             "sleep": ["play something to help me sleep", "help me fall asleep"]}
MOOD_JOIN = [", ", " ", ". ", " so ", ", can you ", " yaar ", "... ", ", please ", " and ", " - "]


def gen_mood(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.15:                                                   # asks to change the mood, no feeling stated
        kind = rng.choice(["low", "low", "low", "stressed", "focus", "sleep"])
        return f"mood(kind={kind}, music=yes)", rng.choice(MOOD_LIFT[kind])
    if r < 0.27:                                                   # hard negatives: a genre is a song request, not a feeling
        g = rng.choice(["sad", "happy", "romantic", "relaxing", "calm", "motivational", "party", "lonely", "breakup"])
        g += rng.choice([" songs", " music"])
        return f"play_media(query={q(g)}, site=youtube)", rng.choice(["play {g}", "{g} chalao", "put on {g}",
                                                                      "{g} bajao", "play me {g}"]).format(g=slot(g))
    kind = rng.choice(list(FEEL_EN))
    hindi = c.hing and kind in FEEL_HI
    feel = rng.choice(FEEL_HI[kind] if hindi else FEEL_EN[kind])
    if rng.random() < 0.55:                                        # feeling + asks for music
        music = c.T(MOOD_MUSIC_HI if hindi or rng.random() < 0.15 else MOOD_MUSIC_EN)
        text = feel + rng.choice(MOOD_JOIN) + music if rng.random() < 0.8 else music + rng.choice([", ", " because ", " - "]) + feel
        return f"mood(kind={kind}, music=yes)", text
    return f"mood(kind={kind}, music=no)", feel


GENERATORS = [
    (lambda c: gen_file_op(c, "move"), 14), (lambda c: gen_file_op(c, "copy"), 7),
    (lambda c: gen_file_op(c, "delete"), 8), (lambda c: gen_file_op(c, "list"), 6),
    (gen_remind, 15), (gen_open_app, 10), (gen_open_url, 8), (gen_open_folder, 5),
    (gen_system, 7), (gen_search, 6), (gen_search_web, 8), (gen_play, 12),
    (gen_window, 6), (gen_desktop, 3), (gen_media, 5), (gen_brightness, 3), (gen_set_volume, 3), (gen_radio, 3),
    (gen_screenshot, 3), (gen_record, 2.5), (gen_power, 3), (gen_training, 3),
]
_FNS, _WEIGHTS = zip(*GENERATORS)
CONNECTORS = [" and ", " and then ", " then ", ", then ", ". also ", ", ", " after that ", " & ", "; ",
              " aur ", " aur phir ", " phir ", ", uske baad "]
PRE = ["", "", "", "", "hey jarvis ", "jarvis ", "jarvis, ", "please ", "can you ", "could you ", "yo ",
       "ok ", "hey ", "bro ", "quick one: ", "yaar ", "suno, "]
POST = ["", "", "", "", " please", " thanks", " asap", " na", " yaar", " thx", " right now", " for me"]

CO_OPEN = ["open it", "open that folder", "show it to me", "then open it", "open the folder", "usse kholo",
           "woh folder khol do", "take me there", "open it up"]
CO_MOVE = ["move them to {d}", "put them in {d}", "send those to {d}", "unhe {d} me daal do", "shift them to {d}",
           "move those into {d}"]
CO_DELETE = ["delete them", "remove those", "get rid of them", "unhe delete kar do", "trash them", "delete those",
             "unhe hata do"]


def coref(c: Ctx, prev_dsl: str, meta: dict):
    """Follow-up that refers back to the previous call ('open it', 'move them to X')."""
    rng = c.rng
    if "dest" in meta and rng.random() < 0.5:
        return f"open_folder(path={meta['dest']})", rng.choice(CO_OPEN), {}
    if prev_dsl.startswith("list_files") and "src_args" in meta:
        args = dict(meta["src_args"])
        if rng.random() < 0.6:
            dest_v, D = folder(c, exclude=args["src"])
            if dest_v == args["src"]:
                return None
            return dsl_call("move_files", {**args, "dest": dest_v}), rng.choice(CO_MOVE).format(d=D), {}
        return dsl_call("delete_files", args), rng.choice(CO_DELETE), {}
    return None


def sample(rng: random.Random, split: str = "train") -> dict:
    c = Ctx(rng, split)
    r = rng.random()
    if r < 0.06:                                                # chit-chat / unclear
        return {"text": apply_noise(LX.unknown_text(rng), rng), "program": "unknown()"}
    if r < 0.08:                                                # "don't delete ..." -> never act
        d, t = gen_negation(c)
        return {"text": apply_noise(t, rng), "program": d}
    if r < 0.095:                                               # lone / garbled words
        d, t = gen_fragment(c)
        return {"text": apply_noise(t, rng), "program": d}
    if r < 0.116:
        d, t, _ = gen_undo(c)
        return {"text": apply_noise(t, rng), "program": d}
    if r < 0.146:                                               # how the user feels (always its own sentence)
        d, t = gen_mood(c)
        t = rng.choice(["", "", "", "jarvis ", "hey jarvis, ", "yaar ", "bro "]) + t
        return {"text": apply_noise(t, rng), "program": canonical(tidy_program_text(d))}
    n_calls = rng.choices([1, 2, 3, 4], weights=[60, 26, 10, 4])[0]
    dsls, texts, meta = [], [], {}
    for i in range(n_calls):
        out = None
        if i > 0 and rng.random() < 0.35:
            out = coref(c, dsls[-1], meta)
        if out is None and rng.random() < (0.08 if n_calls > 1 else 0.04):
            req = LX.unsupported_text(rng)                      # clear action we can't do: report, don't drop
            out = (f"unsupported(request={q(req)})", slot(req), {})
        if out is None:
            c.hing = rng.random() < (0.7 if c.hing else 0.15)  # code-switching tends to persist
            out = rng.choices(_FNS, weights=_WEIGHTS)[0](c)
        d, t, meta = out
        dsls.append(d)
        texts.append(t)
    text = texts[0]
    for t in texts[1:]:
        text += rng.choice(CONNECTORS) + t
    text = rng.choice(PRE) + text + rng.choice(POST)
    return {"text": apply_noise(text, rng), "program": canonical(tidy_program_text(" ; ".join(dsls)))}

# ---------------------------------------------------------------- vocabulary check for held-out frames

def frame_words(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", re.sub(r"\{\w+\}", " ", t.lower())))


def unseen_frame_words() -> dict[str, set[str]]:
    """Words that appear in held-out frames but in no training frame of the same list. Should be empty."""
    out = {}
    for name, items in ALL_FRAME_LISTS.items():
        if len(items) < HOLDOUT_EVERY:
            continue
        train = set().union(*(frame_words(t) for i, t in enumerate(items) if frame_split(i) == "train"))
        held = set().union(*(frame_words(t) for i, t in enumerate(items) if frame_split(i) != "train"))
        if held - train:
            out[name] = held - train
    return out

# ---------------------------------------------------------------- dataset build

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def build(n_train: int, n_eval: int, seed: int, extra: Path | None = None, extra_weight: int = 20):
    rng = random.Random(seed)
    text_to_prog: dict[str, str] = {}
    conflicts: set[str] = set()

    def take(split: str, n: int, exclude: set[str]) -> list[dict]:
        out, local = [], set()
        attempts = 0
        while len(out) < n and attempts < n * 20:
            attempts += 1
            try:
                ex = sample(rng, split)
            except DSLError:
                continue
            k = _norm(ex["text"])
            prev = text_to_prog.get(k)
            if prev is not None and prev != ex["program"]:
                conflicts.add(k)
                continue
            text_to_prog[k] = ex["program"]
            if k in exclude or k in local:
                continue
            local.add(k)
            out.append(ex)
        return out

    test_ood = take("test_ood", n_eval, set())
    val_ood = take("val_ood", n_eval, set())
    test_iid = take("train", n_eval, set())
    val = take("train", n_eval, set())
    held = {_norm(e["text"]) for e in test_ood + val_ood + test_iid + val}
    train = take("train", n_train, held)
    splits = {"train": train, "val": val, "val_ood": val_ood, "test_iid": test_iid, "test_ood": test_ood}
    for name in splits:  # drop any text that later turned out ambiguous
        splits[name] = [e for e in splits[name] if _norm(e["text"]) not in conflicts]
    if extra and extra.exists():
        rows = [json.loads(l) for l in extra.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [{"text": r["text"], "program": canonical(r["program"])} for r in rows]
        splits["train"] += rows * extra_weight
        rng.shuffle(splits["train"])
        print(f"added {len(rows)} user corrections x{extra_weight}")
    return splits, len(conflicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000, help="training examples")
    ap.add_argument("--n-eval", type=int, default=20_000, help="examples per eval split")
    ap.add_argument("--out", type=Path, default=Path("data/intents"))
    ap.add_argument("--extra", type=Path, default=Path("data/user_corrections.jsonl"),
                    help="your own (text, program) corrections; oversampled")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bad = unseen_frame_words()
    if bad:
        raise SystemExit(f"held-out frames use words training never shows: {bad}")
    splits, n_conf = build(args.n, args.n_eval, args.seed, args.extra)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name:9s} {len(data):>9,d}")
    print(f"dropped {n_conf} ambiguous texts")


if __name__ == "__main__":
    main()
