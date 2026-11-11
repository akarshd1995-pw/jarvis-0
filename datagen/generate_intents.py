"""Synthetic (text -> program) data engine, v2.

Program-first generation: sample a valid DSL program, then render it as English or
Hinglish using many templates, number words, time expressions, open-vocabulary slots,
coreference ("move them", "open it"), out-of-scope requests (-> unknown()) and noise.

Splits:
  train / val   : in-distribution
  test_iid      : same templates, unseen examples
  test_ood      : HELD-OUT templates never seen in training -> measures robustness to new phrasing

    python -m datagen.generate_intents --n 1000000 --out data/intents
    python -m datagen.generate_intents --n 1000000 --extra data/user_corrections.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from jarvis.dsl.grammar import DSLError, canonical
from . import lexicon as LX
from .noise import L, R, apply_noise


def slot(s: str) -> str:
    return f"{L}{s}{R}"


def q(s: str) -> str:
    """DSL string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


HOLDOUT_EVERY = 6   # every 6th template is reserved for test_ood


class Ctx:
    def __init__(self, rng: random.Random, split: str):
        self.rng, self.split = rng, split
        self.hing = rng.random() < 0.3

    def T(self, templates: list[str]) -> str:
        """Pick a template; held-out ones only appear in the OOD split."""
        ood = self.split == "ood"
        pool = [t for i, t in enumerate(templates) if (i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1) == ood]
        return self.rng.choice(pool or templates)

# ---------------------------------------------------------------- durations and times

def duration(c: Ctx, kind: str) -> tuple[str, str]:
    """kind: 'age' (file filters) or 'delay' (reminders). Returns (dsl, slot-wrapped surface)."""
    rng, h = c.rng, c.hing
    unit = rng.choices(["m", "h", "d", "w", "mo"], weights=[0, 2, 6, 3, 3] if kind == "age" else [7, 4, 1, 0.5, 0])[0]
    if unit == "m":
        n = rng.choice([1, 2, 5, 10, 15, 20, 25, 30, 40, 45, 90])
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
        n = rng.randint(1, 12)
        N = LX.num_surface(rng, n, h)
        if n == 1:
            forms = ["an hour", "1 hour", "one hour", "1 hr"] + (["ek ghanta", "ek ghante"] if h else [])
        else:
            forms = [f"{N} hours", f"{N} hrs", f"{n}h", f"{n} hours"] + ([f"{N} ghante"] if h else [])
        return f"{n}h", slot(rng.choice(forms))
    if unit == "d":
        n = rng.randint(1, 60)
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
    mm = rng.choice([0, 0, 0, 15, 30, 45, 10, 20, 40, 50, rng.randint(0, 59)])
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
        parts = [rng.choice(LX.PATH_ROOTS)] + rng.sample(LX.PATH_PARTS, rng.randint(1, 2))
        path = "/".join(parts)
        if rng.random() < 0.3:
            path = path.replace("/", "\\")
        return q(path), slot(path)
    key = rng.choice([k for k in LX.FOLDERS if k != exclude])
    return key, slot(rng.choice(LX.FOLDERS[key]))


def what_phrase(c: Ctx, args: dict) -> str:
    rng = c.rng
    if rng.random() < 0.72:
        e = rng.choice(LX.EXTS)
        args["ext"] = e
        forms = [f"{e} files", f"{e}s", f".{e} files", f"{e.upper()} files", f"all {e} files", f"all the {e}s",
                 f"every {e} file", f"*.{e} files"] + LX.EXT_NICKNAMES.get(e, [])
        if c.hing:
            forms += [f"saari {e} files", f"{e} wali files", f"sab {e}", f"saare {e}"]
        return slot(rng.choice(forms))
    generic = ["files", "everything", "all files", "the files", "all the stuff", "stuff", "all items"]
    if c.hing:
        generic += ["saari files", "sab kuch", "files"]
    return rng.choice(generic)


OLDER_EN = ["older than {d}", "more than {d} old", "from over {d} ago", "that are over {d} old",
            "not touched in {d}", "last modified more than {d} ago", "aged over {d}", "that haven't changed in {d}"]
NEWER_EN = ["newer than {d}", "from the last {d}", "less than {d} old", "added in the past {d}",
            "created within {d}", "downloaded in the last {d}"]
NAME_EN = ['with "{w}" in the name', "named {w}", "containing {w} in the filename", "that have {w} in their name",
           "whose name has {w}", "matching {w}", "called {w}"]
OLDER_HI = ["{d} se purani", "jo {d} se purani hain", "{d} se zyada purani", "{d} pehle ki"]
NEWER_HI = ["pichhle {d} ki", "{d} se nayi", "last {d} wali", "{d} ke andar ki"]
NAME_HI = ["jinke naam me {w} hai", "{w} naam wali", "jinme {w} likha hai"]


def filters(c: Ctx, args: dict) -> str:
    rng, parts = c.rng, []
    if rng.random() < 0.45:
        d, txt = duration(c, "age")
        args["older_than"] = d
        parts.append(c.T(OLDER_HI if c.hing else OLDER_EN).format(d=txt))
    elif rng.random() < 0.15:
        d, txt = duration(c, "age")
        args["newer_than"] = d
        parts.append(c.T(NEWER_HI if c.hing else NEWER_EN).format(d=txt))
    if rng.random() < 0.2:
        w = LX.name_fragment(rng)
        args["name_contains"] = q(w)
        parts.append(c.T(NAME_HI if c.hing else NAME_EN).format(w=slot(w)))
    rng.shuffle(parts)
    if not parts:
        return ""
    return " " + rng.choice([" ", " ", ", ", " and "]).join(parts)


def dsl_call(action: str, args: dict) -> str:
    return f"{action}({', '.join(f'{k}={v}' for k, v in args.items())})"

# ---------------------------------------------------------------- actions

MOVE_EN = ["{verb} {what}{filt} from {src} to {dest}", "{verb} {what} from {src}{filt} into {dest}",
           "{verb} {what}{filt} in {src} over to {dest}", "can you {verb} {what}{filt} from {src} to {dest}",
           "i want to {verb} {what}{filt} from {src} to {dest}", "{src} is a mess, {verb} {what}{filt} to {dest}",
           "from {src}, {verb} {what}{filt} to {dest}", "{verb} {what}{filt} to {dest} from {src}",
           "need {what}{filt} from {src} {pp} to {dest}", "get {what}{filt} out of {src} and {verb} them to {dest}",
           "{verb} over {what}{filt} from {src} into {dest}", "could you {verb} {what}{filt} sitting in {src} to {dest}",
           "{what}{filt} in {src} should be {pp} to {dest}", "quickly {verb} {what}{filt} from {src} to {dest}",
           "{verb} {what}{filt} out of {src} into {dest}", "help me {verb} {what}{filt} from {src} to {dest}",
           "i'd like {what}{filt} from {src} {pp} into {dest}", "{verb} {what}{filt}: {src} -> {dest}"]
MOVE_HI = ["{src} se {what}{filt} {dest} me {hverb}", "{what}{filt} {src} se {dest} me {hverb}",
           "{src} wali {what}{filt} {dest} me {hverb}", "{dest} me {src} se {what}{filt} {hverb}",
           "{src} se {dest} me {what}{filt} {hverb}", "{src} ki {what}{filt} ko {dest} me {hverb}",
           "bhai {src} se {what}{filt} {dest} me {hverb}"]
DELETE_EN = ["{verb} {what}{filt} from {src}", "{verb} {what} in {src}{filt}", "clear out {what}{filt} from {src}",
             "get rid of {what}{filt} in {src}", "i don't need {what}{filt} in {src} anymore, delete them",
             "{verb} {what}{filt} sitting in {src}", "can you {verb} {what}{filt} in {src}",
             "{src} is full, {verb} {what}{filt}", "free up space by deleting {what}{filt} from {src}",
             "{verb} all {what}{filt} from {src}", "wipe {what}{filt} from {src}", "throw away {what}{filt} in {src}"]
DELETE_HI = ["{src} se {what}{filt} delete kar do", "{src} me {what}{filt} hata do", "{what}{filt} {src} se uda do",
             "{src} ki {what}{filt} saaf kar do", "{src} me jo {what}{filt} hai unko delete karo",
             "{src} khali karo, {what}{filt} hata do"]
LIST_EN = ["{verb} {what}{filt} in {src}", "what {what} are in {src}{filt}", "which {what}{filt} do i have in {src}",
           "how many {what}{filt} are in {src}", "show me {what} from {src}{filt}", "{verb} {what}{filt} from {src}",
           "any {what}{filt} in {src}", "give me a list of {what}{filt} in {src}", "what's in {src}{filt}",
           "check {src} for {what}{filt}", "find {what}{filt} in {src}", "{src}: {verb} {what}{filt}"]
LIST_HI = ["{src} me {what}{filt} dikhao", "{src} ki {what}{filt} list karo", "{src} me kaunsi {what}{filt} hai",
           "{src} me kitni {what}{filt} hai", "{src} ka {what}{filt} batao"]

VERBS = {"move": (["move", "shift", "transfer", "put", "relocate", "send", "drag"], "moved",
                  ["move kar do", "shift karo", "daal do", "move karo", "bhej do", "rakh do"]),
         "copy": (["copy", "duplicate", "back up", "clone", "make a copy of"], "copied",
                  ["copy kar do", "copy karo", "ki copy bana do"])}


def gen_file_op(c: Ctx, op: str):
    rng = c.rng
    args: dict = {}
    src_v, S = folder(c)
    args["src"] = src_v
    what = what_phrase(c, args)
    filt = filters(c, args)
    meta = {"src_args": dict(args)}
    if op in ("move", "copy"):
        dest_v, D = folder(c, exclude=src_v)
        if dest_v == src_v:
            dest_v, D = ("Archive", slot("archive")) if src_v != "Archive" else ("Documents", slot("documents"))
        args["dest"] = dest_v
        meta["dest"] = dest_v
        verbs, pp, hverbs = VERBS[op]
        tpl = c.T(MOVE_HI if c.hing else MOVE_EN)
        text = tpl.format(verb=rng.choice(verbs), pp=pp, hverb=rng.choice(hverbs), what=what, filt=filt, src=S, dest=D)
    elif op == "delete":
        tpl = c.T(DELETE_HI if c.hing else DELETE_EN)
        text = tpl.format(verb=rng.choice(["delete", "remove", "trash", "erase", "clean up", "nuke"]),
                          what=what, filt=filt, src=S)
    else:
        tpl = c.T(LIST_HI if c.hing else LIST_EN)
        text = tpl.format(verb=rng.choice(["list", "show", "show me", "display", "tell me"]), what=what, filt=filt, src=S)
    return dsl_call(f"{op}_files", args), text, meta


REMIND_EN = ["remind me to {task} {when}", "{when} remind me to {task}", "set a reminder to {task} {when}",
             "ping me {when} to {task}", "don't let me forget to {task} {when}", "can you remind me {when} to {task}",
             "i need to {task} {when}, remind me", "nudge me {when} to {task}", "add a reminder {when}: {task}",
             "remind me {when} to {task}", "reminder to {task} {when}", "make sure i {task} {when}",
             "set an alarm {when} to {task}", "tell me to {task} {when}", "{when}, remind me: {task}",
             "please remind me that i have to {task} {when}", "schedule a reminder {when} to {task}",
             "i have to {task} {when}, don't let me forget"]
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


APP_EN = ["open {a}", "launch {a}", "start {a}", "fire up {a}", "run {a}", "bring up {a}", "can you open {a}",
          "i need {a}", "get {a} running", "open up {a}", "boot up {a}", "load {a}", "please start {a}",
          "pull up {a}", "{a} please", "open the {a} app", "start the {a} application", "i want to use {a}"]
APP_HI = ["{a} khol do", "{a} open karo", "{a} chalu karo", "{a} start kar do", "{a} kholo", "zara {a} khol do"]


def gen_open_app(c: Ctx):
    rng = c.rng
    if rng.random() < 0.6:
        key = rng.choice(list(LX.KNOWN_APPS))
        val, surface = key, rng.choice(LX.KNOWN_APPS[key])
    else:
        surface = rng.choice(LX.OTHER_APPS)
        val = q(surface)
    return f"open_app(name={val})", c.T(APP_HI if c.hing else APP_EN).format(a=slot(surface)), {}


URL_EN = ["open {u}", "go to {u}", "take me to {u}", "browse {u}", "visit {u}", "pull up {u}", "load {u}",
          "open the website {u}", "navigate to {u}", "show me {u}", "head over to {u}", "can you open {u}"]
URL_HI = ["{u} kholo", "{u} pe le chalo", "{u} open karo", "{u} wali site kholo"]


def gen_open_url(c: Ctx):
    rng = c.rng
    r = rng.random()
    if r < 0.45:
        name = rng.choice(list(LX.KNOWN_SITES))
        url, surface = LX.KNOWN_SITES[name], name
        if rng.random() < 0.4:
            surface = url
    else:
        dom = "".join(rng.sample(LX.URL_WORDS, rng.choice([1, 2]))) + rng.choice(LX.TLDS)
        if rng.random() < 0.2:
            dom = rng.choice(["www.", "app.", "docs."]) + dom
        if rng.random() < 0.15:
            dom += rng.choice(["/about", "/login", "/blog", "/docs", "/pricing"])
        if rng.random() < 0.15:
            dom = rng.choice(["https://", "http://"]) + dom
        url = surface = dom
    return f"open_url(url={q(url)})", c.T(URL_HI if c.hing else URL_EN).format(u=slot(surface)), {}


FOLDER_EN = ["open {f}", "show {f}", "open up {f}", "take me to {f}", "go to {f}", "bring up {f}",
             "open {f} in explorer", "i want to see {f}", "pop open {f}", "navigate to {f}"]
FOLDER_HI = ["{f} kholo", "{f} dikhao", "{f} open karo", "{f} khol do"]


def gen_open_folder(c: Ctx):
    v, F = folder(c)
    return f"open_folder(path={v})", c.T(FOLDER_HI if c.hing else FOLDER_EN).format(f=F), {}


def gen_system(c: Ctx):
    rng = c.rng
    key = rng.choice(list(LX.SYSTEM))
    phrase = rng.choice(LX.SYSTEM[key])
    pre = rng.choice(["", "", "", "please ", "can you ", "quickly ", "hey "])
    return f"system(action={key})", pre + phrase, {}


SEARCH_EN = ["find {q}", "search for {q}", "where is {q}", "look up {q}", "where did i save {q}", "find me {q}",
             "i'm looking for {q}", "search my files for {q}", "locate {q}", "dig up {q}", "can you find {q}",
             "do you remember {q}", "pull up {q} from my files", "search memory for {q}"]
SEARCH_HI = ["{q} dhundo", "{q} kahan hai", "{q} search karo", "{q} dhoondh ke do", "mera {q} kahan rakha hai"]


def gen_search(c: Ctx):
    query = LX.search_query(c.rng)
    return f"search_memory(query={q(query)})", c.T(SEARCH_HI if c.hing else SEARCH_EN).format(q=slot(query)), {}


UNDO = ["undo", "undo that", "revert the last command", "undo the last thing", "go back", "cancel what you just did",
        "revert that", "undo last action", "oops undo", "put it back", "reverse that", "wapas kar do",
        "undo kar do", "pehle jaisa kar do", "galti ho gayi, undo karo"]


def gen_undo(c: Ctx):
    return "undo()", c.rng.choice(UNDO), {}


GENERATORS = [
    (lambda c: gen_file_op(c, "move"), 14), (lambda c: gen_file_op(c, "copy"), 7),
    (lambda c: gen_file_op(c, "delete"), 7), (lambda c: gen_file_op(c, "list"), 6),
    (gen_remind, 16), (gen_open_app, 10), (gen_open_url, 8), (gen_open_folder, 5),
    (gen_system, 7), (gen_search, 6),
]
_FNS, _WEIGHTS = zip(*GENERATORS)
CONNECTORS = [" and ", " and then ", " then ", ", then ", ". also ", ", ", " after that ", " & ", "; ",
              " aur ", " aur phir ", " phir ", ", uske baad "]
PRE = ["", "", "", "", "hey jarvis ", "jarvis ", "jarvis, ", "please ", "can you ", "could you ", "yo ",
       "ok ", "hey ", "bro ", "quick one: ", "yaar ", "suno, "]
POST = ["", "", "", "", " please", " thanks", " asap", " na", " yaar", " thx", " right now", " for me"]

CO_OPEN = ["open it", "open that folder", "show it to me", "then open it", "open the folder", "usse kholo",
           "woh folder khol do"]
CO_MOVE = ["move them to {d}", "put them in {d}", "send those to {d}", "unhe {d} me daal do", "shift them to {d}"]
CO_DELETE = ["delete them", "remove those", "get rid of them", "unhe delete kar do", "trash them"]


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
    if rng.random() < 0.08:                                     # out of scope
        return {"text": apply_noise(LX.unknown_text(rng), rng), "program": "unknown()"}
    if rng.random() < 0.015:
        d, t, _ = gen_undo(c)
        return {"text": apply_noise(t, rng), "program": d}
    n_calls = rng.choices([1, 2, 3, 4], weights=[65, 25, 8, 2])[0]
    dsls, texts, meta = [], [], {}
    for i in range(n_calls):
        out = None
        if i > 0 and rng.random() < 0.35:
            out = coref(c, dsls[-1], meta)
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
    return {"text": apply_noise(text, rng), "program": canonical(" ; ".join(dsls))}


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

    test_ood = take("ood", n_eval, set())
    test_iid = take("train", n_eval, set())
    val = take("train", n_eval, set())
    held = {_norm(e["text"]) for e in test_ood + test_iid + val}
    train = take("train", n_train, held)
    splits = {"train": train, "val": val, "test_iid": test_iid, "test_ood": test_ood}
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
    ap.add_argument("--n", type=int, default=1_000_000, help="training examples")
    ap.add_argument("--n-eval", type=int, default=10_000, help="examples per eval split")
    ap.add_argument("--out", type=Path, default=Path("data/intents"))
    ap.add_argument("--extra", type=Path, default=Path("data/user_corrections.jsonl"),
                    help="your own (text, program) corrections; oversampled")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
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
