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
from jarvis import filetypes as FT
from jarvis.memory import query as MQ
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


CAT_HOME = {"video": "Videos", "image": "Pictures", "audio": "Music", "document": "Documents"}
NEW_FOLDER_NAMES = ["videos", "photos", "pics", "music", "songs", "docs", "old stuff", "installers", "zips",
                    "trip pics", "college", "work", "movies", "screenshots", "backup", "misc", "2026", "clips",
                    "wallpapers", "receipts", "setup files", "sorted", "keep", "archive 2025"]


def what_phrase(c: Ctx, args: dict) -> str:
    rng = c.rng
    r = rng.random()
    if 0.5 <= r < 0.72:                         # a whole category: "video files" = mp4, mkv, mov, ...
        cat = rng.choice(list(FT.CATEGORY_WORDS))
        args["ext"] = cat
        w = rng.choice(FT.CATEGORY_WORDS[cat])
        forms = [w, f"all {w}", f"all the {w}", f"my {w}", f"the {w}"]
        if " " not in w and w.endswith("s"):                     # "videos" -> "every video file", "all video files"
            forms += [f"every {w[:-1]} file", f"all {w[:-1]} files"]
        if c.hing:
            forms += [f"saari {w}", f"saare {w}", f"{w} wali files", f"sab {w}"]
        return slot(rng.choice(forms))
    if r < 0.5:
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
        home = CAT_HOME.get(str(args.get("ext")))
        if rng.random() < 0.15:                 # "to a new folder called X": made inside the source
            name = rng.choice(NEW_FOLDER_NAMES) if rng.random() < 0.7 else LX.path_part(rng)
            phrase = rng.choice(["naya folder {n}", "naye folder {n}", "ek naya folder {n}", "new folder {n}"]
                                if c.hing else ["a new folder {n}", "a new folder called {n}", "a new folder named {n}",
                                                "new folder {n}", "a new folder '{n}'", "a fresh folder called {n}"])
            args["new_folder"] = q(name)
            fill["dest"] = phrase.format(n=slot(name))
            return dsl_call(f"{op}_files", args), tpl.format(**fill), meta
        if home and home != src_v and rng.random() < 0.4:      # "move videos from downloads to videos"
            dest_v, D = home, slot(rng.choice(LX.FOLDERS[home]))
        else:
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


# ---------------------------------------------------------------- memory search (phase 7)
# search_memory(query="invoice", ext=pdf, newer_than=7d, where=files|web|screenshots)
# The query holds only WHAT the thing is about; the kind of thing and the time go in their own fields.
MEM_TOPICS = ["invoice", "resume", "login error", "transformer notes", "tax receipt", "offer letter", "budget",
              "flight ticket", "bank statement", "electricity bill", "meeting notes", "rent agreement", "GST",
              "python error", "API key", "wedding photos", "trip itinerary", "salary slip", "insurance policy",
              "lecture notes", "project proposal", "the server crash", "react hooks", "docker setup", "cv",
              "passport scan", "aadhaar", "train booking", "hotel booking", "recipe", "workout plan", "jarvis",
              "everway", "q3 report", "client contract", "design mockup", "pricing table", "cover letter",
              "college admission", "exam timetable", "bug report", "stack trace", "the blue chart", "logo",
              "attention is all you need", "linear regression", "cricket score", "laptop review", "car insurance"]
MEM_KIND, MEM_TIME, MEM_TIME_HI = MQ.KIND, MQ.TIME, MQ.TIME_HI
MEM_EN = frames("MEM_EN", [
    "find the {kind} about {q}{t}", "where's that {kind} with {q}{t}", "find {q}", "find my {q} {kind}{t}",
    "search my files for {q}", "where did i save the {q} {kind}",
    "can you find that {kind} about {q}{t}",                                # val_ood
    "help me find the {kind} with {q}{t}",                                  # test_ood
    "dig up the {kind} about {q}{t}", "i'm looking for the {kind} about {q}{t}", "show me the {kind} with {q}{t}",
    "find that {q} {kind} i saw{t}", "look for {q} in my stuff", "where is my {q}",
    "do you remember the {kind} about {q}{t}",                              # val_ood
    "locate the {q} {kind}{t}",                                             # test_ood
    "find the {q} {kind}{t}", "track down the {kind} about {q}", "search memory for {q}",
    "open the {kind} about {q}{t}", "find my latest {kind}", "show my last {kind}",
    "the {kind} about {q}{t}, where is it",                                 # val_ood
    "find the {kind} i opened{t} about {q}",                                # test_ood
    "i remember a {kind} about {q}, find it", "locate my {q}", "the {q} {kind} i opened{t}",
    "can you help me find {q}", "do you know where my {q} is",
])
MEM_HI = ["{q} wala {kind} dhundo", "{q} ka {kind} kahan hai", "{t}{q} {kind} dhundo", "mera {q} kahan rakha hai",
          "{q} dhundo", "{q} wali {kind} dikhao", "{t}{kind} dhundo jisme {q} tha", "{q} search karo mere files me"]


def gen_search(c: Ctx):
    rng = c.rng
    tpl = c.T(MEM_HI if c.hing else MEM_EN)
    args: dict = {}
    fill: dict = {}
    if "{q}" in tpl:
        topic = rng.choice(MEM_TOPICS) if rng.random() < 0.75 else " ".join(
            LX.pseudo_word(rng).lower().strip("_-") for _ in range(rng.choice([1, 1, 2])))
        args["query"] = q(topic)
        fill["q"] = slot(topic)
    if "{kind}" in tpl:
        word = rng.choice(list(MEM_KIND))
        ext, where = MEM_KIND[word]
        if ext:
            args["ext"] = ext
        if where:
            args["where"] = where
        fill["kind"] = word
    if "{t}" in tpl:
        if rng.random() < 0.45:
            phrase, d = rng.choice(MEM_TIME_HI if c.hing else MEM_TIME)
            args["newer_than"] = d
            fill["t"] = (phrase + " ") if c.hing else (" " + phrase)
        else:
            fill["t"] = ""
    return dsl_call("search_memory", args), tpl.format(**fill), {}


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
    "improve": ["improve yourself", "learn from my corrections", "fine tune the model now", "retrain your model",
                "get better from my fixes", "update the command model", "self improve tonight", "improve the model",
                "train yourself on my corrections", "khud ko improve karo", "apne aap ko behtar karo",
                "model ko fine tune karo", "learn from your mistakes"],
    "review": ["review my commands", "let's do the weekly review", "show me what you weren't sure about",
               "go through my commands", "check the commands you weren't sure about", "weekly review",
               "open the review", "review this week's commands", "let's review your mistakes", "label my commands",
               "commands review karo", "review karte hain", "what did you get wrong this week",
               "teach you from my commands", "go through your doubts"],
    "rollback": ["roll back the model", "go back to the previous model", "undo the model update", "revert the model",
                 "switch back to the old model", "the new model is worse, roll it back", "purana model wapas lao",
                 "model rollback karo", "restore the previous model"],
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


# What a model trained on this data has learned beyond plain actions. Jarvis reads this from the model
# files and only uses its hand-written rules/repairs for what the model has NOT learned.
TEACHES = ["play_modes", "copy_names", "mood", "briefing", "file_categories", "new_folder", "system_control",
           "training_control", "song_suffix", "self_improve", "memory_filters", "maintenance", "directions", "usage_review", "context_refs", "notes", "workspaces", "focus_gaming", "tasks", "your_words", "wellbeing", "system_care"]


# ---------------------------------------------------------------- directions (routes, travel time)
PLACES = ["koramangala", "indiranagar", "whitefield", "electronic city", "hsr layout", "jayanagar", "mg road",
          "the airport", "kempegowda airport", "majestic", "bangalore railway station", "mysore", "goa", "pune",
          "mumbai", "andheri", "bandra", "connaught place", "gurgaon", "noida sector 62", "hitech city",
          "gachibowli", "anna nagar", "t nagar", "marine drive", "phoenix mall", "forum mall", "orion mall",
          "the office", "work", "home", "my college", "the gym", "apollo hospital", "the stadium", "lalbagh",
          "cubbon park", "nandi hills", "chennai central", "hyderabad", "delhi airport", "manipal", "ooty",
          "my parents' place", "rahul's house", "the metro station", "infosys campus", "manyata tech park"]
MODE_PHRASES = {"driving": ["by car", "driving", "in my car"], "two_wheeler": ["on a bike", "on my scooter", "by bike"],
                "walking": ["on foot", "walking"], "transit": ["by metro", "by bus", "by train", "by public transport"],
                "bicycling": ["by cycle", "cycling"]}
MODE_HI = {"driving": ["car se"], "two_wheeler": ["bike se", "scooty se"], "walking": ["paidal"],
           "transit": ["metro se", "bus se"], "bicycling": ["cycle se"]}
DIR_EN = frames("DIR_EN", [
    "fastest route from {o} to {d}{m}", "what's the fastest way to {d}{m}", "how long to get to {d}{m}",
    "directions to {d}{m}", "how long will it take to reach {d} from {o}{m}", "how far is {d} from {o}",
    "quickest way to {d}{m}",                                               # 6 val_ood
    "how much time to reach {d}{m}",                                        # 7 test_ood
    "route to {d}{m}", "best route from {o} to {d}", "navigate to {d}", "how long is the drive to {d}",
    "get me directions to {d} from {o}", "how far is {d}",
    "what's the eta to {d}{m}",                                             # val_ood
    "travel time from {o} to {d}{m}",                                       # test_ood
    "show me the way to {d}", "take me to {d}", "how long does it take to reach {d} from {o}{m}",
    "shortest route to {d}{m}", "how long to reach {d}", "directions from {o} to {d}{m}",
    "fastest way to get to {d}",                                            # val_ood
    "how long will it take to get to {d}",                                  # test_ood
    "what's the time to {d} from {o}", "the best way to reach {d}{m}", "eta from {o} to {d}",
    "quickest route from {o} to {d}", "how much time from {o} to {d}{m}", "travel time to {d}",
    "how far is it to {d}", "how far is it to {d} from {o}", "how long is it to {d}{m}",
    "how far is it from {o} to {d}", "how long will it take to get to {d} from {o}{m}", "how much distance is it to {d}",
])
DIR_HI = frames("DIR_HI", [
    "{d} tak kitna time lagega", "{o} se {d} kaise jaun", "{d} kitna door hai", "{d} kaise pahunchu",
    "{o} se {d} tak kitna time lagega", "{m}{d} kaise jaun", "{d} pahunchne me kitna time lagega",
    "{o} se {d} ka rasta dikhao", "{d} ka route batao", "{m}{d} tak kitna time lagega",
    "{d} pahunchne me kitna samay lagega", "{d} ka rasta dikhao",
])


def gen_directions(c: Ctx):
    rng = c.rng
    tpl = c.T(DIR_HI if c.hing else DIR_EN)
    pool = PLACES + [" ".join(LX.pseudo_word(rng).lower().strip("_-") for _ in range(rng.randint(1, 2)))
                     for _ in range(3)]
    d = rng.choice(pool)
    args = {"destination": q(d)}
    fill = {"d": slot(d)}
    if "{o}" in tpl:
        o = rng.choice([p for p in pool if p != d])
        args["origin"] = q(o)
        fill["o"] = slot(o)
    if "{m}" in tpl:
        if rng.random() < 0.5:
            mode = rng.choice(list(MODE_PHRASES))
            args["mode"] = mode
            fill["m"] = (rng.choice(MODE_HI[mode]) + " ") if c.hing else (" " + rng.choice(MODE_PHRASES[mode]))
        else:
            fill["m"] = ""
    return dsl_call("directions", args), tpl.format(**fill), {}


# ---------------------------------------------------------------- context references (phase 38)
# "this/these/it/the selected files" -> src=this, "here/this folder" -> src=here or dest=here,
# "google this" -> query=this, "open the second one" -> open_result(n=2). Jarvis fills them in from the screen.
THIS_EN = ["this", "these", "that", "those", "it", "them", "this file", "these files", "that file", "those files",
           "the selected files", "the selected ones", "selected files", "what i selected", "the files i selected",
           "the highlighted files", "these ones", "this one", "the stuff i selected", "everything i selected"]
THIS_KIND_EN = ["these {k}", "those {k}", "the selected {k}", "selected {k}", "the {k} i selected", "these {k} here"]
THIS_HI = ["ise", "inhe", "ye", "ye files", "ye wali files", "selected files ko", "inko", "isko", "is file ko"]
CTX_FILES_EN = frames("CTX_FILES_EN", [
    "{v} {t} to {d}", "{v} {t} into {d}", "please {v} {t} to {d}", "can you {v} {t} to {d}", "{v} {t} over to {d}",
    "i want to {v} {t} to {d}",
    "could you {v} {t} into {d}",                                   # 6 val_ood
    "{v} {t} straight to {d}",                                      # 7 test_ood
    "go ahead and {v} {t} to {d}", "{v} {t} to {d} for me", "just {v} {t} to {d}", "{v} {t} to {d} now",
    "quickly {v} {t} to {d}", "{v} {t} in {d}",
    "{v} {t} to {d} please",                                        # val_ood
    "jarvis {v} {t} to {d}",                                        # test_ood
    "could you {v} {t} to {d}", "jarvis, {v} {t} into {d}", "{v} {t} straight into {d}",
])
CTX_DEL_EN = frames("CTX_DEL_EN", [
    "{v} {t}", "please {v} {t}", "{v} {t} now", "can you {v} {t}", "just {v} {t}", "{v} {t} for me",
    "i don't need {t}, {v} them",                                  # 6 val_ood
    "{v} {t} right away",                                          # 7 test_ood
    "{v} {t} please", "go ahead and {v} {t}", "{v} {t}, i don't need them", "{v} all of {t}",
    "{v} {t} right now", "{v} {t} and throw them away",
])
CTX_FILES_HI = frames("CTX_FILES_HI", [
    "{t} {d} me {hv}", "{t} {d} mein {hv}", "{t} {d} folder me {hv}", "jarvis {t} {d} me {hv}", "{t} ko {d} me {hv}",
    "{t} jaldi se {d} me {hv}",
    "zara {t} {d} me {hv}",                                          # val_ood
    "{t} abhi {d} me {hv}",                                          # test_ood
    "{t} seedha {d} me {hv}", "please {t} {d} me {hv}",
    "abhi {t} {d} me {hv}", "zara {t} ko {d} me {hv}",
])
CTX_DEL_HI = ["{t} {hv}", "{t} abhi {hv}", "{t} please {hv}", "jarvis {t} {hv}"]
HERE_EN = ["here", "in this folder", "in here", "in the current folder", "in the open folder", "from here",
           "in this directory"]
CTX_HERE_EN = frames("CTX_HERE_EN", [
    "{v} {w} {h} to {d}", "{v} all {w} {h} to {d}", "{v} the {w} {h} into {d}", "can you {v} {w} {h} to {d}",
    "{v} everything {h} to {d}", "please {v} {w} {h} to {d}",
    "{v} every {w} {h} over to {d}",                                 # val_ood
    "take the {w} {h} and {v} them to {d}",                          # test_ood
    "{v} {w} {h} to {d} for me", "i want to {v} {w} {h} to {d}",
    "take {w} {h} and {v} them over to {d}", "{v} every {w} {h} to {d}",
])
CTX_LIST_HERE = frames("CTX_LIST_HERE", [
    "{lv} {w} {h}", "what {w} are {h}", "{lv} all {w} {h}", "what's {h}", "{lv} the {w} {h}", "which {w} are {h}",
    "are there any {w} {h}",                                         # val_ood
    "how many {w} are {h}",                                          # test_ood
    "{lv} everything {h}", "{lv} me the {w} {h}",
    "how many {w} are there {h}", "is there any {w} {h}",
])
CTX_TO_HERE = frames("CTX_TO_HERE", [
    "{v} {w} from {s} here", "{v} the {w} from {s} to here", "bring {w} from {s} here", "{v} all {w} in {s} here",
    "get {w} from {s} in here", "{v} my {w} from {s} into this folder",
    "fetch the {w} from {s} here",                                   # val_ood
    "pull {w} out of {s} into here",                                 # test_ood
    "{v} {w} from {s} into the current folder", "bring the {w} in {s} over here",
    "fetch {w} from {s} here", "pull the {w} out of {s} here",
])
CTX_HERE_HI = ["yahan ki {w} {d} me {hv}", "is folder ki {w} {d} me {hv}", "yahan se {w} {d} me {hv}",
               "is folder ki saari files {d} me {hv}"]
CTX_TO_HERE_HI = ["{s} se {w} yahan le aao", "{s} ki {w} yahan {hv}", "{s} se {w} is folder me {hv}"]
CTX_TEXT = {
    "search_web": frames("CTX_SEARCH", [
        "google this", "search this on {site}", "look this up on {site}", "search what i copied on {site}",
        "find this on {site}", "search for this on {site}", "look up what i copied on {site}",
        "check this on {site}",                                     # test_ood
        "can you google this", "search {site} for this", "look this up", "google what i copied",
        "search this", "search that on {site}",
        "what does {site} say about this",                          # val_ood
        "search the copied text on {site}",                         # test_ood
        "ise google karo", "ise {site} pe dhundo", "ye {site} pe search karo", "isko google karo"
    "check what i copied on {site}", "what does google say about the copied text",
    "check this on google", "please check this on {site}", "quickly check this on {site}",
    "please search this on {site}", "quickly google this",
]),
    "open_url": frames("CTX_URL", [
        "open this link", "open the link i copied", "open the copied link", "go to this link", "open that link",
        "visit the link i copied", "open this url",
        "take me to the link i copied",                             # test_ood
        "ye link kholo", "is link ko kholo", "open the url i copied", "open this website"
    "take me to this link",
    "take me to the copied link", "please take me to this url", "take me to that link",
]),
    "play_media": frames("CTX_PLAY", [
        "play this", "play what i copied", "play this song", "play this video", "play that", "play this link",
        "play the song i copied",                                   # val_ood
        "play the video i copied",                                  # test_ood
        "play this on {psite}", "ye chalao", "ye gaana bajao", "ise chalao", "play that song"
    "play the link i copied",
]),
    "directions": frames("CTX_DIR", [
        "directions to this address", "route to this place", "navigate to this", "how far is this address",
        "how long to get to this address", "take me to this address",
        "fastest way to the address i copied",                      # val_ood
        "how long to reach this place",                             # test_ood
        "directions to the address i copied", "is jagah tak kitna time lagega", "is address ka rasta dikhao"
    "fastest way to this place", "how long to reach the address i copied",
    "the fastest route to this address", "fastest route to this place", "what is the fastest way to this address",
    "what is the route to this place", "how long to get there, to this address",
    "what is the way to this address", "what is the distance to this place",
]),
    "remind": ["remind me about this in {n} minutes", "remind me of this in {n} min", "remind me about this in {n} mins",
               "yaad dilana ise {n} min baad"],
    "search_memory": ["find files about this", "search my files for this", "do i have files about this",
                      "is bare me files dhundo", "find my documents about this"],
}
ORD_EN = {1: ["first", "1st"], 2: ["second", "2nd"], 3: ["third", "3rd"], 4: ["fourth", "4th"], 5: ["fifth", "5th"],
          6: ["sixth", "6th"]}
ORD_HI = {1: ["pehla", "pehli"], 2: ["dusra", "dusri"], 3: ["teesra", "teesri"], 4: ["chautha", "chauthi"]}
CTX_RESULT = frames("CTX_RESULT", [
    "open the {o} one", "open the {o} result", "open result {n}", "open number {n}", "open the {o} file",
    "open the {o} link", "show me the {o} one",
    "open up the {o} result",                                       # test_ood
    "open the {o}", "{o} one please", "open the {o} item", "can you open the {o} one",
    "the {o} one, open it",                                         # val_ood (index 14 % 8 == 6)
    "open the {o} of those",                                        # test_ood
    "show me the {o} result", "open up the {o} one",
    "show me the {o} file", "open me the {o} one", "pull up the {o} result", "show the {o} one",
])


def _this_kind(c: Ctx, args: dict) -> str:
    """'these pdfs' (ext=pdf) or plain 'these'."""
    rng = c.rng
    if c.hing:
        return rng.choice(THIS_HI)
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            cat = rng.choice(list(FT.CATEGORY_WORDS))
            args["ext"] = cat
            k = rng.choice([w for w in FT.CATEGORY_WORDS[cat] if " " not in w] or [cat + "s"])
        else:
            e = rng.choice(LX.EXTS)
            args["ext"] = e
            k = rng.choice([f"{e}s", f"{e} files"] + LX.EXT_NICKNAMES.get(e, []))
        return rng.choice(THIS_KIND_EN).format(k=slot(k))
    return rng.choice(THIS_EN)


def gen_context(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.38:                                                  # files: this / these
        op = rng.choice(["move"] * 5 + ["copy"] * 2 + ["delete"] * 3)
        verbs, _, hverbs = VERBS[op]
        args: dict = {"src": "this"}
        t = _this_kind(c, args)
        if op == "delete":
            tpl = c.T(CTX_DEL_HI if c.hing else CTX_DEL_EN)
            return dsl_call("delete_files", args), tpl.format(v=rng.choice(verbs), t=t, hv=rng.choice(hverbs)), {}
        if rng.random() < 0.15:
            name = rng.choice(NEW_FOLDER_NAMES)
            args["new_folder"] = q(name)
            d = rng.choice(["naya folder {n}", "ek naya folder {n}"] if c.hing else
                           ["a new folder called {n}", "a new folder {n}", "a new folder named {n}"]).format(n=slot(name))
        else:
            dv, d = folder(c)
            args["dest"] = dv
        tpl = c.T(CTX_FILES_HI if c.hing else CTX_FILES_EN)
        if c.hing and "new_folder" in args:
            tpl = "{t} {d} me {hv}"
        return dsl_call(f"{op}_files", args), tpl.format(v=rng.choice(verbs), t=t, d=d, hv=rng.choice(hverbs)), {}
    if r < 0.6:                                                   # here: this folder
        args = {}
        w = what_phrase(c, args) if rng.random() < 0.75 else rng.choice(["files", "all files", "everything", "stuff"])
        kind = rng.random()
        if kind < 0.4:
            op = rng.choice(["move", "move", "copy", "delete"])
            verbs, _, hverbs = VERBS[op]
            args = {"src": "here", **args}
            if op == "delete":
                return dsl_call("delete_files", args), \
                    rng.choice(["{v} {w} {h}", "{v} all {w} {h}", "{v} the {w} {h}"]).format(
                        v=rng.choice(verbs), w=w, h=rng.choice(HERE_EN)), {}
            dv, d = folder(c)
            args["dest"] = dv
            tpl = rng.choice(CTX_HERE_HI) if c.hing else c.T(CTX_HERE_EN)
            return dsl_call(f"{op}_files", args), tpl.format(v=rng.choice(verbs), w=w, h=rng.choice(HERE_EN), d=d,
                                                             hv=rng.choice(hverbs)), {}
        if kind < 0.7:
            args = {"src": "here", **args}
            tpl = rng.choice(["{w} yahan dikhao", "is folder me kya hai", "yahan ki {w} dikhao"]) if c.hing \
                else c.T(CTX_LIST_HERE)
            return dsl_call("list_files", args), tpl.format(lv=rng.choice(VERBS["list"][0]), w=w,
                                                            h=rng.choice(HERE_EN)), {}
        op = rng.choice(["move", "copy"])
        verbs, _, hverbs = VERBS[op]
        sv, sp = folder(c)
        args = {"src": sv, **args, "dest": "here"}
        tpl = rng.choice(CTX_TO_HERE_HI) if c.hing else c.T(CTX_TO_HERE)
        return dsl_call(f"{op}_files", args), tpl.format(v=rng.choice(verbs), w=w, s=sp, hv=rng.choice(hverbs)), {}
    if r < 0.85:                                                  # text: what you copied
        action = rng.choices(list(CTX_TEXT), [5, 2, 3, 2, 1.5, 1.5])[0]
        tpl = c.T(CTX_TEXT[action])
        if action == "search_web":
            site = rng.choice(["google", "youtube", "amazon", "flipkart", "wikipedia", "reddit", "github"])
            if "{site}" not in tpl:
                site = "google"
            return f"search_web(query=this, site={site})", tpl.format(site=site), {}
        if action == "play_media":
            site = rng.choice(["youtube", "spotify"]) if "{psite}" in tpl else "youtube"
            return f"play_media(query=this, site={site})", tpl.format(psite=site), {}
        if action == "remind":
            n = rng.choice([5, 10, 15, 20, 30, 45, 60, 90])
            return f"remind(text=this, after={n}m)", tpl.format(n=n), {}
        key = {"open_url": "url", "directions": "destination", "search_memory": "query"}[action]
        return f"{action}({key}=this)", tpl, {}
    n = rng.choice([1, 1, 2, 2, 2, 3, 3, 4, 5, 6])                # "open the second one"
    if c.hing and n in ORD_HI:
        return f"open_result(n={n})", rng.choice(["{o} wala kholo", "{o} wali kholo", "{o} wala open karo"]).format(
            o=rng.choice(ORD_HI[n])), {}
    if n == 1 and rng.random() < 0.4:
        return "open_result(n=1)", rng.choice(["open this", "open it", "open that", "open the selected file",
                                               "open the top one", "open the top result"]), {}
    return f"open_result(n={n})", c.T(CTX_RESULT).format(o=rng.choice(ORD_EN[n]), n=n), {}


# ---------------------------------------------------------------- notes & journal (phase 30)
NOTE_EN = frames("NOTE_EN", [
    "note: {t}", "note that {t}", "make a note: {t}", "jot down {t}", "take a note: {t}", "add a note: {t}",
    "note to self: {t}",                                            # 6 val_ood
    "can you write down that {t}",                                  # 7 test_ood
    "write down {t}", "note down {t}", "quick note: {t}", "save a note saying {t}", "make a note that {t}",
    "please note {t}",
    "put down a note: {t}",                                         # val_ood
    "jot this down: {t}",                                           # test_ood
    "can you note that {t}", "note {t}", "add note {t}", "make a note of {t}", "jarvis note that {t}",
    "write this down: {t}", "please write down {t}", "jot down that {t}",
    "note to self {t}", "put a note: {t}", "note this for me: {t}", "put in a note that {t}",
])
NOTE_HI = ["note karo: {t}", "{t} note kar lo", "{t} note kar do", "likh lo: {t}", "{t} likh lo", "ek note banao: {t}",
           "{t} note karna", "yeh note karo: {t}"]
JOURNAL_EN = frames("JOURNAL_EN", [
    "add to my journal: {t}", "journal: {t}", "dear diary {t}", "journal entry: {t}", "write in my journal that {t}",
    "log in my journal: {t}",
    "put this in my journal: {t}",                                  # val_ood
    "for my journal: {t}",                                          # test_ood
    "add a journal entry: {t}", "today's journal: {t}", "journal that {t}", "in my journal write {t}",
    "add this to my journal: {t}", "new journal entry: {t}",
    "put in my journal: {t}", "for today's journal: {t}", "journal for today: {t}",
    "put this down in my journal: {t}", "please put in my journal: {t}",
])
JOURNAL_HI = ["diary me likho: {t}", "journal me likho: {t}", "{t} journal me daal do", "aaj ki diary: {t}"]
NOTE_REF = ["note this", "note that down", "save this as a note", "note down what i copied", "make a note of this",
            "add this to my notes", "ise note kar lo", "isko note karo", "jot this down"]
JOURNAL_REF = ["add this to my journal", "put this in my journal", "ise journal me daal do"]
FIND_EN = frames("FIND_NOTES_EN", [
    "what did i note about {q}", "show my notes about {q}", "find my note on {q}", "what did i write down about {q}",
    "read my notes about {q}", "do i have a note about {q}",
    "what was that note about {q}",                                 # val_ood
    "pull up my note on {q}",                                       # test_ood
    "show me my notes on {q}", "get my note about {q}", "search my notes for {q}", "any notes about {q}",
    "what did i jot down about {q}", "open my note about {q}",
    "i made a note about {q}, show it",                             # val_ood
    "look up my notes on {q}",                                      # test_ood
    "find the note about {q}", "what's in my notes about {q}",
    "what was my note about {q}", "pull up my notes about {q}", "look up my note about {q}", "i made a note on {q}, find it", "what was in that note about {q}", "look up {q} in my notes",
    "that note about {q}, show it", "show that note on {q}",
])
FIND_HI = ["{q} ke baare me kya note kiya tha", "{q} wala note dikhao", "{q} ka note dikhao", "{q} ke notes dikhao"]
WHEN_EN = {"today": ["today's", "today"], "yesterday": ["yesterday's", "yesterday"], "week": ["this week's", "this week"],
           "month": ["this month's", "this month"]}
FIND_WHEN = ["show {w1} notes", "read {w1} notes", "what did i note {w2}", "my notes from {w2}", "list {w1} notes",
             "show {w1} journal", "read {w1} journal entries", "what did i write in my journal {w2}",
             "my journal from {w2}"]
FIND_ALL = ["show all my notes", "read my notes", "list my notes", "show my notes", "read my journal",
            "show my journal", "open my journal", "notes dikhao", "mera journal dikhao"]
DEL_NOTE = ["delete the note about {q}", "remove my note on {q}", "forget the note about {q}", "erase my note about {q}",
            "delete my {q} note", "{q} wala note hata do", "{q} ka note delete karo"]
NOTE_TOPICS = ["the bank", "wifi password", "rahul", "parking", "the meeting", "rent", "the card", "passport",
               "gym timings", "the landlord", "insurance", "flight", "ideas", "the project", "doctor", "dentist",
               "car service", "electricity bill", "priya's birthday", "laptop warranty", "the exam", "jarvis"]


def note_text(rng) -> str:
    """Open-ended things people note: numbers, passwords, places, ideas, to-dos, made-up words."""
    name = rng.choice(LX.NAMES)
    word = LX.pseudo_word(rng).lower()
    digits = "".join(rng.choice("0123456789") for _ in range(rng.choice([4, 5, 10])))
    forms = [
        LX.remind_text(rng), f"{name}'s number is {digits}", f"wifi password is {word}{digits[:2]}",
        f"parked at level {rng.randint(1, 5)} spot {rng.choice('ABCD')}{rng.randint(1, 90)}",
        f"{name} owes me {rng.choice([200, 500, 1500, 2000, 4300])} rupees", f"idea: an app that {rng.choice(['tracks', 'sorts', 'finds', 'plays'])} {word}s",
        f"meeting with {name} moved to {rng.choice(['monday', 'friday', 'tomorrow', '5 pm', 'next week'])}",
        f"buy {rng.choice(['milk', 'batteries', 'a phone charger', 'coffee', 'printer ink', word])}",
        f"the {word} project deadline is {rng.choice(['the 15th', 'friday', 'march 3', 'end of month'])}",
        f"{rng.choice(['good', 'rough', 'long', 'productive', 'quiet'])} day, {rng.choice(['finished', 'started', 'fixed', 'shipped'])} the {word} {rng.choice(['bug', 'feature', 'report', 'installer'])}",
        f"call {name} about the {rng.choice(['rent', 'car', 'trip', 'invoice', word])}",
        f"{word} {LX.pseudo_word(rng).lower()}",
    ]
    return rng.choice(forms)


def gen_notes(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.45:                                          # a note
        if rng.random() < 0.1:
            return "note(text=this)", rng.choice(NOTE_REF), {}
        t = note_text(rng)
        tpl = rng.choice(NOTE_HI) if c.hing else c.T(NOTE_EN)
        return dsl_call("note", {"text": q(t)}), tpl.format(t=slot(t)), {}
    if r < 0.6:                                           # a journal entry
        if rng.random() < 0.1:
            return "note(kind=journal, text=this)", rng.choice(JOURNAL_REF), {}
        t = note_text(rng)
        tpl = rng.choice(JOURNAL_HI) if c.hing else c.T(JOURNAL_EN)
        return dsl_call("note", {"text": q(t), "kind": "journal"}), tpl.format(t=slot(t)), {}
    if r < 0.85:                                          # finding them
        k = rng.random()
        if k < 0.6:
            topic = rng.choice(NOTE_TOPICS) if rng.random() < 0.7 else LX.pseudo_word(rng).lower()
            query = topic[4:] if topic.startswith("the ") else topic
            tpl = rng.choice(FIND_HI) if c.hing else c.T(FIND_EN)
            return dsl_call("find_notes", {"query": q(query)}), tpl.format(q=slot(topic)), {}
        if k < 0.85:
            when = rng.choice(list(WHEN_EN))
            tpl = rng.choice(FIND_WHEN)
            args = {"kind": "journal"} if "journal" in tpl else {}
            args["when"] = when
            w1, w2 = WHEN_EN[when]
            return dsl_call("find_notes", args), tpl.format(w1=w1, w2=w2), {}
        tpl = rng.choice(FIND_ALL)
        return ("find_notes(kind=journal)" if "journal" in tpl else "find_notes()"), tpl, {}
    topic = rng.choice(NOTE_TOPICS) if rng.random() < 0.7 else LX.pseudo_word(rng).lower()
    query = topic[4:] if topic.startswith("the ") else topic
    return dsl_call("delete_note", {"query": q(query)}), rng.choice(DEL_NOTE).format(q=slot(topic)), {}


# ---------------------------------------------------------------- workspaces (phase 31)
WS_NAMES = ["coding", "study", "work", "writing", "design", "editing", "research", "meeting", "streaming", "office",
            "college", "video editing", "trading", "client work", "music production", "accounts", "exam prep",
            "side project", "freelance", "content"]
WS_START = frames("WS_START", [
    "start {n} mode", "{n} mode", "open my {n} workspace", "switch to {n} mode", "load my {n} layout",
    "set up for {n}",
    "bring up my {n} setup",                                       # 6 val_ood
    "get my {n} workspace ready",                                  # 7 test_ood
    "launch {n} mode", "let's start {n}", "{n} mode please", "enter {n} mode", "start my {n} setup",
    "open the {n} layout",
    "fire up {n} mode",                                            # val_ood
    "time for {n} mode",                                           # test_ood
    "go to {n} mode", "activate {n} mode", "begin {n} mode", "start the {n} workspace", "get my {n} setup ready",
    "bring up {n} mode", "fire up my {n} workspace", "it's time for {n}",
    "it is time for {n} mode", "fire up {n}", "time to start {n}", "ok time for {n}",
    "ok it's {n} time", "it's {n} time, set it up",
])
WS_START_HI = ["{n} mode chalu karo", "{n} mode shuru karo", "{n} wala setup lagao", "{n} mode on karo", "{n} setup kholo"]
WS_SAVE = frames("WS_SAVE", [
    "save this layout as {n}", "save this as my {n} workspace", "remember this layout as {n}",
    "save these windows as {n}", "save my windows as {n} mode", "store this setup as {n}",
    "keep this arrangement as {n}",                                # val_ood
    "save the current layout for {n}",                             # test_ood
    "remember this setup as {n}", "save this workspace as {n}", "save the layout as {n}", "save this as {n} mode",
    "keep this layout as {n}", "remember these windows as my {n} setup", "save current windows as {n}",
    "save the current setup as {n}",
    "save this arrangement as {n}", "save my current layout as {n}", "save this setup for {n}", "keep the current arrangement for {n}",
])
WS_SAVE_HI = ["ye layout {n} naam se save karo", "is setup ko {n} ke naam se save karo", "ye windows {n} mode me save karo"]
WS_CLOSE = ["close {n} mode", "end {n} mode", "close my {n} workspace", "stop {n} mode", "i'm done with {n}",
            "exit {n} mode", "shut down my {n} setup", "{n} mode band karo", "{n} wala setup band karo"]
WS_DELETE = ["delete the {n} workspace", "forget my {n} layout", "remove {n} mode", "delete my {n} layout",
             "{n} wala layout delete karo"]
WS_LIST = ["list my workspaces", "what workspaces do i have", "show my layouts", "show my workspaces",
           "which modes have i saved", "my saved layouts", "workspaces dikhao"]


def gen_workspace(c: Ctx):
    rng = c.rng
    name = rng.choice(WS_NAMES) if rng.random() < 0.75 else LX.pseudo_word(rng).lower().strip("_-")
    r = rng.random()
    if r < 0.45:
        tpl = rng.choice(WS_START_HI) if c.hing else c.T(WS_START)
        return dsl_call("workspace", {"action": "start", "name": q(name)}), tpl.format(n=slot(name)), {}
    if r < 0.7:
        tpl = rng.choice(WS_SAVE_HI) if c.hing else c.T(WS_SAVE)
        return dsl_call("workspace", {"action": "save", "name": q(name)}), tpl.format(n=slot(name)), {}
    if r < 0.82:
        return dsl_call("workspace", {"action": "close", "name": q(name)}), rng.choice(WS_CLOSE).format(n=slot(name)), {}
    if r < 0.9:
        return dsl_call("workspace", {"action": "delete", "name": q(name)}), rng.choice(WS_DELETE).format(n=slot(name)), {}
    return "workspace(action=list)", rng.choice(WS_LIST), {}


# ---------------------------------------------------------------- focus & gaming (phase 32)
FOCUS_DUR = [(15, ["15 minutes", "15 mins", "fifteen minutes"]), (20, ["20 minutes", "20 min"]),
             (25, ["25 minutes", "25 mins"]), (30, ["30 minutes", "half an hour", "30 mins", "aadha ghanta"]),
             (40, ["40 minutes"]), (45, ["45 minutes", "45 mins", "forty five minutes"]), (50, ["50 minutes"]),
             (60, ["an hour", "1 hour", "one hour", "60 minutes", "ek ghanta"]), (90, ["90 minutes", "1.5 hours", "an hour and a half"]),
             (120, ["2 hours", "two hours", "do ghante"]), (180, ["3 hours"])]
FOCUS_START = frames("FOCUS_START", [
    "focus for {d}", "start a focus session for {d}", "i need to focus for {d}", "deep work for {d}",
    "focus mode for {d}", "let me focus for {d}",
    "lock in for {d}",                                             # 6 val_ood
    "i want {d} of deep work",                                     # 7 test_ood
    "help me focus for {d}", "start focusing for {d}", "focus session {d}", "{d} focus session", "{d} of focus",
    "do not disturb for {d}",
    "i'm locking in for {d}",                                      # val_ood
    "give me {d} without distractions",                            # test_ood
    "begin a {d} focus block", "time to focus for {d}", "i'll work for {d}, keep me focused", "start {d} of focus",
    "lock in, {d}", "{d} of deep work", "no distractions for {d}", "give me {d} to focus",
    "i want to focus for {d}", "i'm going to focus for {d}", "{d} with no distractions", "give me {d} of focus",
    "{d} without any distractions", "start locking in for {d}", "no distractions please, {d}", "i'm working for {d}",
    "please let me focus for {d}", "working for {d}, focus please",
])
FOCUS_PLAIN = ["focus mode", "focus mode on", "start focusing", "i need to focus", "start a focus session",
               "deep work mode", "help me focus", "let me focus", "turn on focus mode", "focus time",
               "focus karna hai", "focus mode chalu karo", "lock in"]
POMO = ["start a pomodoro", "pomodoro", "start pomodoro", "pomodoro mode", "begin a pomodoro session",
        "pomodoro timer please", "pomodoro chalu karo"]
POMO_DUR = ["pomodoro with {d} rounds", "pomodoro for {d}", "{d} pomodoro", "start a {d} pomodoro"]
BLOCK_YES = [" and block distractions", " and block youtube", ", strict mode", " and block social media",
             ", no distractions", " and block my distracting apps"]
BLOCK_NO = [" without blocking anything", " but don't block anything", ", no blocking"]
TOTAL = [", no interruptions at all", ", not even reminders", ", completely quiet", ", koi disturb nahi"]
FOCUS_HI = ["{d} focus karna hai", "{d} ke liye focus mode on karo", "{d} tak disturb mat karna", "{d} padhai karni hai, focus mode"]
FOCUS_STOP = ["stop focus", "end focus mode", "focus mode off", "i'm done focusing", "stop the pomodoro", "end my focus session",
              "cancel focus", "finish the focus session", "focus band karo", "pomodoro band karo", "stop deep work"]
FOCUS_STATUS = ["how long is left", "how much time is left in my focus", "focus status", "how long have i been focusing",
                "how much focus time is left", "time left in this session", "kitna time bacha hai", "focus kitna bacha hai"]
GAME_ON = ["gaming mode on", "gaming mode", "start gaming mode", "i'm going to play", "game mode on", "i'm gonna game now",
           "turn on gaming mode", "gaming mode chalu karo", "time to play some games"]
GAME_OFF = ["gaming mode off", "done gaming", "stop gaming mode", "i'm done playing", "turn off gaming mode", "exit game mode",
            "gaming mode band karo"]


def gen_focus(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.55:
        args: dict = {"action": "start"}
        k = rng.random()
        if k < 0.6:
            mins, says = rng.choice(FOCUS_DUR)
            d = rng.choice(says)
            args["minutes"] = mins
            text = (rng.choice(FOCUS_HI) if c.hing else c.T(FOCUS_START)).format(d=d)
        elif k < 0.75:
            text = rng.choice(FOCUS_PLAIN)
        else:
            args["mode"] = "pomodoro"
            if rng.random() < 0.4:
                mins, says = rng.choice(FOCUS_DUR[:7])
                args["minutes"] = mins
                text = rng.choice(POMO_DUR).format(d=rng.choice(says))
            else:
                text = rng.choice(POMO)
        x = rng.random()
        if not c.hing and x < 0.18:
            args["block"] = "yes"
            text += rng.choice(BLOCK_YES)
        elif not c.hing and x < 0.25:
            args["block"] = "no"
            text += rng.choice(BLOCK_NO)
        if not c.hing and rng.random() < 0.08:
            args["quiet"] = "total"
            text += rng.choice(TOTAL)
        return dsl_call("focus", args), text, {}
    if r < 0.7:
        return "focus(action=stop)", rng.choice(FOCUS_STOP), {}
    if r < 0.82:
        return "focus(action=status)", rng.choice(FOCUS_STATUS), {}
    if r < 0.93:
        return "gaming(action=on)", rng.choice(GAME_ON), {}
    return "gaming(action=off)", rng.choice(GAME_OFF), {}


# ---------------------------------------------------------------- tasks & deadlines (phase 33)
TASK_VERBS = ["finish", "submit", "send", "review", "call", "pay", "renew", "book", "fix", "write", "prepare", "update",
              "clean", "return", "buy", "file", "reply to", "schedule", "print", "back up"]
TASK_OBJS = ["the report", "the assignment", "the invoice", "the presentation", "rent", "the electricity bill",
             "my passport", "flight tickets", "the PRs", "the {w} bug", "the {w} deck", "the resume", "my taxes",
             "the {w} proposal", "the car insurance", "the library books", "groceries", "{n}'s email", "the {w} doc"]
DUE_SAID = [("friday", ["by friday", "on friday", "before friday", "friday"]), ("monday", ["by monday", "on monday"]),
            ("tomorrow", ["tomorrow", "by tomorrow"]), ("tomorrow 5 pm", ["tomorrow 5 pm", "by tomorrow 5 pm"]),
            ("today evening", ["today evening", "by today evening"]), ("tonight", ["tonight"]),
            ("5th december", ["by 5th december", "on 5th december"]), ("december 15", ["by december 15", "on december 15"]),
            ("next week", ["next week", "by next week"]), ("in 3 days", ["in 3 days"]), ("end of month", ["by end of month"]),
            ("31/12", ["by 31/12"]), ("this weekend", ["this weekend"]), ("next friday", ["by next friday", "next friday"]),
            ("wednesday 11 am", ["by wednesday 11 am", "on wednesday at 11 am"]), ("the 1st", ["on the 1st"])]
DUE_HI = [("friday", "friday"), ("kal", "kal"), ("parso", "parso"), ("kal shaam", "kal shaam"), ("monday", "monday"),
          ("5 baje", "5 baje")]
EVERY_SAID = {"daily": [" every day", " daily"], "weekdays": [" every weekday", " on weekdays"],
              "weekly": [" every week", " weekly"], "monthly": [" every month", " monthly"]}
TASK_ADD = frames("TASK_ADD", [
    "add task {t}{dp}", "add a task: {t}{dp}", "new task {t}{dp}", "to-do: {t}{dp}", "add {t} to my to-do list",
    "i need to {t}{dp}",
    "put {t} on my task list{dp}",                                 # 6 val_ood
    "can you track {t}{dp}",                                       # 7 test_ood
    "task: {t}{dp}", "make a task to {t}{dp}", "add to my list: {t}{dp}", "remember i have to {t}{dp}",
    "add a to-do to {t}{dp}", "new to-do: {t}{dp}",
    "please track {t}{dp}",                                        # val_ood
    "i have to {t}{dp}, add it",                                   # test_ood
    "add a task to {t}{dp}", "create a task: {t}{dp}", "put {t} on my list{dp}", "can you add {t} to my tasks",
    "track this task: {t}{dp}", "i have to {t}{dp}", "please add {t} to my to-do{dp}", "add it: {t}{dp}",
    "please add a task: {t}{dp}", "add it to my tasks: {t}{dp}", "please put {t} on my list{dp}",
])
TASK_ADD_HI = ["{t} karna hai {d} tak", "{t} {d} tak karna hai", "task add karo: {t}", "{t} ko list me daal do",
               "{d} tak {t} karna hai"]
TASK_DONE = ["done with {q}", "mark {q} as done", "{q} is done", "i finished {q}", "tick off {q}", "completed {q}",
             "i'm done with {q}", "check off {q}", "{q} ho gaya", "finished {q}"]
TASK_MOVE = ["move {q} to {d}", "postpone {q} to {d}", "push {q} to {d}", "reschedule {q} for {d}", "shift {q} to {d}",
             "{q} ko {d} pe shift karo"]
TASK_DEL = ["delete the task {q}", "remove {q} from my list", "{q} wala task hata do", "cancel the {q} task",
            "delete my {q} task"]
TASK_LIST = {"today": ["what's due today", "what do i have to do today", "today's tasks", "show my tasks for today",
                       "aaj kya karna hai", "what's on my list today"],
             "week": ["what's due this week", "show my tasks for this week", "this week's tasks", "is hafte kya karna hai"],
             "overdue": ["what's overdue", "show overdue tasks", "anything overdue", "which tasks are late"],
             "done": ["what did i finish this week", "show completed tasks", "what have i done this week"],
             None: ["show my tasks", "my to-do list", "show my to-do list", "list my tasks", "what's on my list",
                    "mere tasks dikhao"]}


def task_text(rng) -> str:
    obj = rng.choice(TASK_OBJS).format(w=LX.pseudo_word(rng).lower(), n=rng.choice(LX.NAMES))
    return f"{rng.choice(TASK_VERBS)} {obj}"           # never a date inside the task text: that's the due slot


def gen_tasks(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.45:
        t = task_text(rng)
        args: dict = {"action": "add", "text": q(t)}
        if c.hing:
            due, said = rng.choice(DUE_HI)
            args["due"] = q(due)
            tpl = rng.choice(TASK_ADD_HI)
            if "{d}" not in tpl:
                args.pop("due")
            return dsl_call("task", args), tpl.format(t=slot(t), d=slot(said)), {}
        dp = ""
        if rng.random() < 0.7:
            due, says = rng.choice(DUE_SAID)
            args["due"] = q(due)
            phrase = rng.choice(says)
            dp = " " + phrase.replace(due, slot(due))
        if rng.random() < 0.12:
            every = rng.choice(list(EVERY_SAID))
            args["every"] = every
            dp = rng.choice(EVERY_SAID[every]) + dp
        if rng.random() < 0.1:
            args["priority"] = "high"
            dp += rng.choice([", it's urgent", ", it's important", ", high priority", " asap"])
        tpl = c.T(TASK_ADD)
        if "{dp}" not in tpl:
            args.pop("due", None), args.pop("every", None), args.pop("priority", None)
        return dsl_call("task", args), tpl.format(t=slot(t), dp=dp), {}
    t = task_text(rng)
    words = t.split()
    query = " ".join(words[1:]) if len(words) > 1 and rng.random() < 0.6 else t
    query = query[4:] if query.startswith("the ") else query
    if r < 0.65:
        return dsl_call("task", {"action": "done", "text": q(query)}), rng.choice(TASK_DONE).format(q=slot(query)), {}
    if r < 0.75:
        due, says = rng.choice(DUE_SAID)
        return dsl_call("task", {"action": "move", "text": q(query), "due": q(due)}), \
            rng.choice(TASK_MOVE).format(q=slot(query), d=slot(due)), {}
    if r < 0.82:
        return dsl_call("task", {"action": "delete", "text": q(query)}), rng.choice(TASK_DEL).format(q=slot(query)), {}
    when = rng.choice(list(TASK_LIST))
    prog = "task(action=list" + (f", when={when})" if when else ")")
    return prog, rng.choice(TASK_LIST[when]), {}


# ---------------------------------------------------------------- your words (phase 43)
NICKS = [("dl", "downloads"), ("docs", "documents"), ("vs", "cursor"), ("yt", "youtube"), ("ammu", "amrutha"),
         ("proj", "prototypes"), ("pics", "pictures"), ("bro", "rahul"), ("insta", "instagram"), ("wa", "whatsapp")]
VOCAB_TEACH = ["when i say {w} i mean {m}", "{w} means {m}", "learn that {w} means {m}", "remember that {w} means {m}",
               "when i say '{w}', i mean '{m}'", "from now on {w} means {m}", "jab main {w} bolun to {m} samajhna",
               "{w} matlab {m}"]
VOCAB_FORGET = ["forget {w}", "forget the nickname {w}", "unlearn {w}", "stop using {w} as a nickname",
                "{w} isn't a nickname", "{w} bhool jao"]
VOCAB_LIST = ["show my nicknames", "what nicknames do you know", "list my words", "show my shortcuts",
              "my vocabulary", "mere nicknames dikhao"]


def gen_vocab(c: Ctx):
    rng = c.rng
    w, m = rng.choice(NICKS) if rng.random() < 0.5 else (LX.pseudo_word(rng).lower()[:6], rng.choice(
        ["downloads", "documents", "cursor", "chrome", "spotify", rng.choice(LX.NAMES).lower(), "prototypes"]))
    r = rng.random()
    if r < 0.6:
        return dsl_call("vocab", {"action": "teach", "word": q(w), "means": q(m)}), \
            rng.choice(VOCAB_TEACH).format(w=slot(w), m=slot(m)), {}
    if r < 0.85:
        return dsl_call("vocab", {"action": "forget", "word": q(w)}), rng.choice(VOCAB_FORGET).format(w=slot(w)), {}
    return "vocab(action=list)", rng.choice(VOCAB_LIST), {}


# ---------------------------------------------------------------- system care (phase 37)
STARTUP_APPS = ["spotify", "discord", "steam", "onedrive", "teams", "skype", "zoom", "epic games", "slack", "dropbox",
                "google drive", "whatsapp", "telegram", "nvidia", "logitech", "razer synapse", "adobe", "utorrent"]
CARE_DISK = frames("CARE_DISK", [
    "how much disk space do i have", "check my disk space", "what's taking up space", "why is my c drive full",
    "disk usage", "show storage",
    "my drive is almost full, what's using it",                    # 6 val_ood
    "which folders are the biggest",                               # 7 test_ood
    "how much space is left", "storage report", "what is eating my disk space", "is my disk full",
    "show me the biggest folders", "how full is my drive",
    "where did all my space go",                                   # val_ood
    "what's filling up my drive",                                  # test_ood
    "where did my space go", "which folders are filling my disk", "where is it all going", "what are the big folders",
    "did something fill my drive", "which folders take the most",
    "disk space check",                                            # val_ood
    "how much free space",                                         # test_ood
    "space left on my pc", "check storage", "what's using all my space", "my drive is full", "it says my disk is full",
    "what is filling my pc",
    "biggest folders on my pc",                                    # val_ood
    "show what's using my space",                                  # test_ood
    "all my space is gone", "almost no free space left",
])
CARE_DISK_HI = ["disk mein kitni jagah hai", "kitni jagah bachi hai", "storage kitna bacha hai"]
CARE_CLEAN = {
    "temp": ["clear temp files", "delete temporary files", "clean the temp folder", "remove old temp files",
             "wipe temp files", "temp files saaf karo", "get rid of temp files"],
    "installers": ["delete old installers", "remove setup files from downloads", "clear old installer files",
                   "get rid of installers in downloads", "purane installers hata do"],
    "duplicates": ["remove duplicate downloads", "find duplicates in downloads", "delete duplicate files",
                   "clean up duplicates", "duplicate files hata do"],
    "all": ["free up space", "clean up my pc", "free up some disk space", "clean up the computer", "pc saaf karo",
            "jagah khali karo", "clear junk files", "clean my laptop"],
}
CARE_STARTUP_LIST = ["startup apps", "what starts with windows", "show startup programs", "what runs at startup",
                     "which apps open when my pc starts", "list startup items"]
CARE_STARTUP_OFF = ["turn off {a} at startup", "stop {a} from starting with windows", "disable {a} on startup",
                    "don't open {a} at startup", "remove {a} from startup", "startup se {a} hata do",
                    "stop {a} launching when my pc starts"]
CARE_STARTUP_ON = ["turn on {a} at startup", "start {a} with windows", "enable {a} on startup",
                   "launch {a} when my pc starts", "{a} ko startup pe chalu karo"]
CARE_TEMPS = ["gpu temperature history", "how hot did my gpu get this week", "gpu temps this week",
              "what was my highest gpu temperature", "show gpu temperatures", "gpu kitna garam hua"]
CARE_TRASH = ["empty your trash", "empty jarvis trash", "clear your trash"]


def gen_care(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.2:
        return "care(action=disk)", (rng.choice(CARE_DISK_HI) if c.hing else c.T(CARE_DISK)), {}
    if r < 0.45:
        what = rng.choice(list(CARE_CLEAN))
        return f"care(action=cleanup, what={what})", rng.choice(CARE_CLEAN[what]), {}
    if r < 0.55:
        return "care(action=startup_list)", rng.choice(CARE_STARTUP_LIST), {}
    if r < 0.85:
        app = rng.choice(STARTUP_APPS) if rng.random() < 0.8 else LX.pseudo_word(rng).lower()
        on = rng.random() < 0.3
        tpl = rng.choice(CARE_STARTUP_ON if on else CARE_STARTUP_OFF)
        return dsl_call("care", {"action": "startup_on" if on else "startup_off", "name": q(app)}), \
            tpl.format(a=slot(app)), {}
    if r < 0.95:
        return "care(action=temps)", rng.choice(CARE_TEMPS), {}
    return "care(action=empty_trash)", rng.choice(CARE_TRASH), {}


# ---------------------------------------------------------------- wellbeing (phase 34)
LIMIT_APPS = ["youtube", "instagram", "netflix", "reddit", "twitter", "facebook", "steam", "valorant", "discord",
              "hotstar", "prime video", "twitch", "whatsapp", "spotify", "chess.com", "linkedin"]
LIMIT_DUR = [(15, ["15 minutes"]), (20, ["20 minutes"]), (30, ["30 minutes", "half an hour"]), (45, ["45 minutes"]),
             (60, ["an hour", "1 hour", "one hour", "60 minutes"]), (90, ["90 minutes", "1.5 hours"]),
             (120, ["2 hours", "two hours"]), (180, ["3 hours"])]
LIMIT_SET = frames("LIMIT_SET", [
    "limit {a} to {d} a day", "limit {a} to {d} per day", "only {d} of {a} a day", "set a {d} limit on {a}",
    "{a} limit {d}", "cap {a} at {d} a day",
    "don't let me use {a} more than {d} a day",                   # 6 val_ood
    "i only want {d} of {a} each day",                            # 7 test_ood
    "keep {a} under {d} a day", "max {d} of {a} daily", "give me {d} of {a} per day", "set my {a} limit to {d}",
    "restrict {a} to {d} a day", "{a} only {d} a day",
    "no more than {d} of {a} every day",                          # val_ood
    "allow just {d} of {a} each day",                             # test_ood
    "i want to use {a} for only {d} a day", "limit my {a} time to {d}", "cap my {a} usage at {d}",
    "only allow {d} of {a} a day", "each day {d} of {a} max", "just {d} of {a} every day",
    "don't let me spend more than {d} on {a}", "no more than {d} of {a}", "let me use {a} for {d} a day",
    "{d} of {a} is enough, limit it", "it's too much {a}, limit it to {d}", "don't let me go over {d} of {a}",
    "no more {a} than {d} a day", "i spend too long on {a}, cap it at {d}", "never more than {d} of {a} a day",
    "spend at most {d} on {a}",
    "never let {a} go past {d} a day", "at most {d} of {a}",
])
LIMIT_HI = ["{a} sirf {d} roz", "{a} ka limit {d} kar do", "roz {d} se zyada {a} nahi"]
UNLIMIT = ["remove the {a} limit", "no limit on {a}", "unlimit {a}", "delete my {a} limit", "{a} ka limit hata do",
           "stop limiting {a}"]
APP_TIME = ["how much {a} today", "how long have i spent on {a} today", "how much time on {a} today",
            "{a} screen time today", "aaj kitna {a} dekha"]
SCREEN_REPORT = ["screen time today", "my screen time", "how much screen time today", "show my screen time",
                 "what did i spend my time on today", "how long have i been on the computer today",
                 "aaj kitna screen time hua", "screen time report"]
BREAKS_OFF = [("all", ["stop break reminders", "turn off break reminders", "no more breaks", "disable break reminders",
                       "break reminders band karo"]),
              ("eyes", ["turn off eye break reminders", "stop the eye breaks", "no more 20-20-20 reminders"]),
              ("stretch", ["stop stretch reminders", "no more stretch reminders", "turn off stretch reminders"])]
BREAKS_ON = [("all", ["turn on break reminders", "start break reminders again", "enable break reminders",
                      "break reminders chalu karo"]),
             ("eyes", ["turn on eye break reminders", "remind me to rest my eyes"]),
             ("stretch", ["turn on stretch reminders", "remind me to stretch"])]
SNOOZE = ["snooze breaks for {d}", "pause break reminders for {d}", "no breaks for {d}", "hold the break reminders for {d}",
          "{d} tak break reminders mat do"]
BREAK_STATUS = ["when was my last break", "break reminder status", "how long have i been at the screen",
                "how long since my last break"]


def gen_wellbeing(c: Ctx):
    rng = c.rng
    r = rng.random()
    app = rng.choice(LIMIT_APPS) if rng.random() < 0.8 else LX.pseudo_word(rng).lower()
    if r < 0.35:
        mins, says = rng.choice(LIMIT_DUR)
        tpl = rng.choice(LIMIT_HI) if c.hing else c.T(LIMIT_SET)
        return dsl_call("screen_time", {"action": "limit", "app": q(app), "minutes": mins}), \
            tpl.format(a=slot(app), d=rng.choice(says)), {}
    if r < 0.45:
        return dsl_call("screen_time", {"action": "unlimit", "app": q(app)}), rng.choice(UNLIMIT).format(a=slot(app)), {}
    if r < 0.58:
        return dsl_call("screen_time", {"action": "report", "app": q(app)}), rng.choice(APP_TIME).format(a=slot(app)), {}
    if r < 0.7:
        return "screen_time(action=report)", rng.choice(SCREEN_REPORT), {}
    if r < 0.8:
        kind, says = rng.choice(BREAKS_OFF)
        return "breaks(action=off" + (f", kind={kind})" if kind != "all" else ")"), rng.choice(says), {}
    if r < 0.88:
        kind, says = rng.choice(BREAKS_ON)
        return "breaks(action=on" + (f", kind={kind})" if kind != "all" else ")"), rng.choice(says), {}
    if r < 0.95:
        mins, says = rng.choice(LIMIT_DUR[:6])
        return f"breaks(action=snooze, minutes={mins})", rng.choice(SNOOZE).format(d=rng.choice(says)), {}
    return "breaks(action=status)", rng.choice(BREAK_STATUS), {}


# ---------------------------------------------------------------- maintenance (phase 29)
HEALTH = frames("HEALTH", [
    "run a health check", "health check", "check yourself", "are you working properly", "run diagnostics",
    "is everything ok with you", "do a self check",                       # 6 val_ood, 7 test_ood
    "run a system check", "check if everything is fine", "jarvis are you ok", "run the doctor",
    "check your models and files", "diagnose yourself", "is everything working",
    "are you healthy",                                                     # val_ood
    "something feels off, check yourself",                                 # test_ood
    "apna health check karo", "sab theek hai kya check karo", "run a health check please",
    "do a quick self check", "check everything is ok", "self check now",
    "check yourself please", "run a health check now",                  # (22, 23: held out)
    "system check please", "something is off, are you healthy", "feels slow, check yourself",
])
BACKUP = frames("BACKUP", [
    "back up yourself", "make a backup", "back up jarvis", "take a backup of your data", "save a backup now",
    "create a backup", "backup your settings and models",                 # 6 val_ood, 7 test_ood
    "back up your brain", "make a backup of jarvis", "take a backup now", "jarvis ka backup lo",
    "save your data somewhere safe", "backup everything",
    "can you back yourself up",                                            # val_ood
    "please make a fresh backup",                                          # test_ood
    "backup bana do", "back up your models", "take a backup of jarvis please", "make a backup now",
    "back up my jarvis", "create a fresh backup",
    "back up your settings and your brain",                              # 21
    "make a backup please", "take a backup",                              # (22, 23: held out)
    "backup do", "backup bana lo",
])


def gen_maintenance(c: Ctx):
    if c.rng.random() < 0.6:
        return "maintenance(action=health_check)", c.T(HEALTH), {}
    return "maintenance(action=backup)", c.T(BACKUP), {}


# ---------------------------------------------------------------- briefings (phase 24)
BRIEF_MORNING_EN = frames("BRIEF_MORNING_EN", [
    "good morning", "good morning jarvis", "morning briefing", "give me my morning briefing", "what's my day look like",
    "what do i have today", "what's on today", "what's on my agenda today", "read me the morning briefing",
    "start my morning briefing", "morning update please", "what's the plan for today", "how's my day looking",
    "anything on today", "morning summary", "give me the morning report", "what's planned for today",
    "what's my schedule today", "brief me on today", "tell me about my day", "what does today look like",
    "give me today's briefing", "today's update please", "morning rundown",
    "morning agenda summary please", "the morning report and rundown",
])
BRIEF_MORNING_HI = frames("BRIEF_MORNING_HI", [
    "aaj ka plan kya hai", "aaj kya hai", "aaj ke liye kya hai", "aaj ka din kaisa hai", "subah ki briefing do",
    "aaj ka schedule batao", "aaj kya kya hai batao", "good morning yaar, aaj ka plan batao", "aaj ka update do",
    "aaj ka briefing do", "good morning yaar",
])
BRIEF_EVENING_EN = frames("BRIEF_EVENING_EN", [
    "how was my day", "evening summary", "give me my evening briefing", "summarize my day", "what did i do today",
    "what did i work on today", "wrap up the day", "end of day summary", "daily report", "give me the day recap",
    "evening update please", "how did my day go", "recap my day", "evening report", "day summary please",
    "what did i get done today", "give me today's summary", "read me the evening briefing", "end of day report",
    "how productive was i today", "show me my day", "evening rundown", "tell me how today went", "day wrap up",
    "tell me what i get done and wrap it up", "how it went today, sum it up",
])
BRIEF_EVENING_HI = frames("BRIEF_EVENING_HI", [
    "aaj maine kya kiya", "aaj ka summary do", "aaj ka hisaab batao", "aaj din kaisa gaya", "aaj kitna kaam kiya",
    "shaam ki briefing do", "aaj ka recap do", "aaj kya kya kiya batao", "din ka summary batao", "aaj ka report do",
    "din ka recap",
])
BRIEF_ANY = frames("BRIEF_ANY", [
    "brief me", "briefing", "give me a briefing", "catch me up", "status update", "sitrep", "update me",
    "give me the rundown", "what's the status", "give me an update", "quick briefing please", "run the briefing",
    "read me my briefing", "what did i miss", "daily briefing", "jarvis briefing", "update de do", "briefing do",
    "daily rundown jarvis",
])


def gen_briefing(c: Ctx):
    r = c.rng.random()
    if r < 0.4:
        return "briefing(kind=morning)", c.T(BRIEF_MORNING_HI if c.hing else BRIEF_MORNING_EN), {}
    if r < 0.8:
        return "briefing(kind=evening)", c.T(BRIEF_EVENING_HI if c.hing else BRIEF_EVENING_EN), {}
    return "briefing()", c.T(BRIEF_ANY), {}


GENERATORS = [
    (lambda c: gen_file_op(c, "move"), 14), (lambda c: gen_file_op(c, "copy"), 7),
    (lambda c: gen_file_op(c, "delete"), 8), (lambda c: gen_file_op(c, "list"), 6),
    (gen_remind, 15), (gen_open_app, 10), (gen_open_url, 8), (gen_open_folder, 5),
    (gen_system, 7), (gen_search, 6), (gen_search_web, 8), (gen_play, 12),
    (gen_window, 6), (gen_desktop, 3), (gen_media, 5), (gen_brightness, 3), (gen_set_volume, 3), (gen_radio, 3),
    (gen_screenshot, 3), (gen_record, 2.5), (gen_power, 3), (gen_training, 3), (gen_briefing, 3), (gen_maintenance, 2), (gen_directions, 4), (gen_context, 6), (gen_notes, 6), (gen_workspace, 5), (gen_focus, 5), (gen_tasks, 7), (gen_vocab, 2), (gen_wellbeing, 4), (gen_care, 4),
]
_FNS, _WEIGHTS = zip(*GENERATORS)


def use_personal(vocab: dict) -> int:
    """Phase 43: mix in examples made from your own apps, projects, folders and typos. Returns how many kinds."""
    global _FNS, _WEIGHTS
    from .personal import personal_generators
    extra = personal_generators(vocab, list(GENERATORS))
    if extra:
        _FNS, _WEIGHTS = zip(*(list(GENERATORS) + extra))
    return len(extra)


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
        latest = {}                                   # the same text corrected twice: the latest answer wins
        for r in rows:
            latest[_norm(r["text"])] = {"text": r["text"], "program": canonical(r["program"])}
        rows = list(latest.values())
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
    ap.add_argument("--vocab", type=Path, default=None, help="your vocab.json (default: Jarvis's data folder)")
    ap.add_argument("--no-personal", action="store_true", help="leave out examples made from your own words")
    args = ap.parse_args()
    if not args.no_personal:
        from .personal import load_vocab
        kinds = use_personal(load_vocab(args.vocab))
        if kinds:
            print(f"Adding personal examples from your words ({kinds} kinds: apps / projects & folders / typos)")
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
    actions = sorted({a for ex in splits["train"] for a in re.findall(r"(\w+)\(", ex["program"])})
    (args.out / "meta.json").write_text(json.dumps({"actions": actions, "features": TEACHES}, indent=2))
    print(f"dropped {n_conf} ambiguous texts; teaches {len(actions)} actions + {', '.join(TEACHES)}")


if __name__ == "__main__":
    main()
