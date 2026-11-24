"""Training data for the dialog model (phase 6): what you say NEXT, given what just happened.

Input  = context + " || " + what you said, where context is one of
           PREV: <the program Jarvis just ran>                  -> follow-ups: "no, to Desktop", "same for Documents"
           ASK: <program 1> | <program 2> [| <program 3>]       -> answering "did you mean ...?": "the second one"
           CONFIRM: <program>                                    -> "yes" / "no" / "no, to Desktop"
Output = the full program to run now, or unknown() when it isn't a reply to the context (a fresh command, which
         the command model handles) or you said no / neither.

    python -m datagen.generate_dialog --n 400000 --out data/dialog
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from jarvis.dsl.grammar import DSLError, canonical, parse

from . import lexicon as LX
from .generate_intents import GENERATORS, Ctx, PLACES, q

SEP = " || "
TEACHES = ["followups", "clarify"]
HOLD = 8            # frame index % 8: 6 -> val_ood, 7 -> test_ood (same rule as the command data)

FOLDER_IDS = list(LX.FOLDERS)


def T(rng: random.Random, split: str, items: list[str]) -> str:
    """A frame belonging to this split; short lists are never held out."""
    if len(items) < HOLD:
        return rng.choice(items)
    want = {"val_ood": 6, "test_ood": 7}.get(split)
    pool = [x for i, x in enumerate(items) if (i % HOLD == want if want is not None else i % HOLD < 6)]
    return rng.choice(pool)


def lit(v) -> str:
    return q(v) if isinstance(v, str) else str(v)


def build(action: str, args: dict) -> str:
    return canonical(f"{action}({', '.join(f'{k}={lit(v) if not isinstance(v, Raw) else v}' for k, v in args.items())})")


class Raw(str):
    """An already-formatted DSL literal (ident, duration, number)."""


def folder(rng) -> tuple[str, str]:
    """(DSL value, how you say it): known folders or a path."""
    if rng.random() < 0.8:
        f = rng.choice(FOLDER_IDS)
        return Raw(f), rng.choice(LX.FOLDERS[f])
    p = f"{rng.choice(LX.PATH_ROOTS)}/{rng.choice(LX.PATH_PARTS)}"
    return p, p


def fresh(rng, split: str) -> tuple[str, str]:
    fns, weights = zip(*GENERATORS)
    fn = rng.choices(fns, weights)[0]
    prog, text, _ = fn(Ctx(rng, split))
    return canonical(prog), text.replace("\x01", "").replace("\x02", "")


# ------------------------------------------------------------------ follow-ups to a program that just ran
AGAIN = ["do it again", "again", "once more", "repeat that", "same thing again", "do that again", "one more time",
         "phir se karo", "dobara karo", "ek baar aur", "again please", "redo that", "run it again", "same again"]
DEST = ["no, to {F}", "actually put them in {F}", "not there, {F}", "{F} instead", "i meant {F}", "nahi, {F} me",
        "make that {F}",                                                          # val_ood
        "wrong folder, {F}",                                                      # test_ood
        "no no, {F}", "put them in {F} instead", "move them to {F} instead", "actually {F}", "change it to {F}",
        "send them to {F}", "{F} me daalo", "into {F} please"]
SRC = ["do the same for {F}", "same for {F}", "and {F} too", "now do {F}", "{F} ka bhi", "what about {F}",
       "now the same in {F}",                                                    # val_ood
       "repeat that for {F}",                                                    # test_ood
       "same with {F}", "do {F} as well", "{F} me bhi karo", "also do it for {F}", "and in {F}?", "{F} too"]
EXT = ["only the {E} files", "just {E} files", "{E} instead", "sirf {E}", "only {E}", "just the {E} ones",
       "make it {E} files",                                                      # val_ood
       "{E} files only",                                                         # test_ood
       "no, {E} files", "only {E} wali", "do it for {E} files", "i meant {E} files"]
ALL_FILES = ["all of them, not just those", "all files", "saari files", "every file, not only {E}", "all types"]
OLDER = ["only the ones older than a month", "just old ones", "only files older than {N} days", "purani wali sirf",
         "only the old stuff", "older than {N} days only"]
NEWER = ["only from this week", "just the recent ones", "only new ones", "sirf nayi wali", "from the last {N} days only"]
SITE = ["on {S} instead", "now on {S}", "search that on {S}", "{S} pe dekho", "try {S}", "what about on {S}",
        "check {S}",                                                              # val_ood
        "look on {S} instead",                                                    # test_ood
        "{S} me dhundo", "and on {S}?", "same on {S}", "on {S}"]
PLAY_SITE = ["on spotify instead", "play it on spotify", "spotify pe chalao", "on youtube", "youtube pe", "play it on youtube"]
PLAY_MODE = {"latest": ["the latest one", "their newest video", "latest wala", "the new one instead"],
             "playlist": ["a playlist instead", "make it a playlist", "playlist chalao", "the whole playlist"],
             "live": ["the live version", "live wala", "play the live one"]}
QUERY = ["no, {Q}", "i meant {Q}", "not that, {Q}", "nahi, {Q}", "wrong one, {Q}", "actually {Q}"]
LEVEL_UP = ["a bit more", "louder", "more", "higher", "thoda aur", "increase it a bit", "a little more"]
LEVEL_DOWN = ["a bit less", "lower", "less", "too much, lower it", "thoda kam", "decrease it a bit", "a little lower"]
LEVEL_SET = ["make it {N}", "{N} instead", "no, {N}", "{N} please", "set it to {N}", "{N} kar do"]
REMIND_AFTER = ["make it {N} minutes", "in {N} min instead", "change it to {N} minutes", "{N} minute baad karo",
                "no, {N} minutes"]
DIR_MODE = {"walking": ["what about walking", "and on foot?", "paidal kitna"], "two_wheeler": ["what about by bike", "on a bike?", "bike se?"],
            "transit": ["and by metro?", "what about by bus", "metro se kitna"], "driving": ["and by car?", "what if i drive"]}
DIR_ORIGIN = ["and from {P}?", "what about from {P}", "{P} se?", "from {P} instead"]
DIR_DEST = ["how about to {P}", "and to {P}?", "what about {P}", "{P} tak?", "and {P}?"]
CLOSE = {"close": ["close it", "close that", "band karo", "shut it"], "maximize": ["maximize it", "make it full screen"],
         "minimize": ["minimize it", "hide it"], "snap_left": ["put it on the left"], "snap_right": ["put it on the right"]}
UNDO = ["undo that", "undo", "take it back", "wapas karo", "revert that"]
MEM_FILTER = {("ext", "pdf"): ["only pdfs", "just the pdf", "sirf pdf"],
              ("newer_than", "7d"): ["from last week", "only recent ones", "this week only"],
              ("where", "web"): ["in my browser history", "check the websites i visited", "browser me dekho"],
              ("where", "screenshots"): ["in my screenshots", "check screenshots"]}


def followup(p: str, rng, split: str) -> tuple[str, str] | None:
    """(utterance, new program) for a program that just ran, or None if no follow-up fits."""
    call = parse(p).calls[-1]
    a, act = dict(call.args), call.action
    r = rng.random()
    if r < 0.08:
        return T(rng, split, AGAIN), p
    if r < 0.12 and act not in ("undo", "unknown", "unsupported"):
        return rng.choice(UNDO), "undo()"
    if act in ("move_files", "copy_files", "delete_files", "list_files"):
        k = rng.random()
        if act in ("move_files", "copy_files") and k < 0.35:
            v, said = folder(rng)
            a.pop("new_folder", None)
            a["dest"] = v
            return T(rng, split, DEST).format(F=said), build(act, a)
        if k < 0.65:
            v, said = folder(rng)
            a["src"] = v
            return T(rng, split, SRC).format(F=said), build(act, a)
        if k < 0.85:
            e = rng.choice(LX.EXTS)
            a["ext"] = Raw(e)
            return T(rng, split, EXT).format(E=rng.choice(LX.EXT_NICKNAMES.get(e, [e]))), build(act, a)
        if k < 0.9 and "ext" in a:
            e = str(a.pop("ext"))
            return rng.choice(ALL_FILES).format(E=e), build(act, a)
        n = rng.choice([7, 14, 30, 60, 90, 180, 365])
        if k < 0.95:
            a.pop("newer_than", None)
            a["older_than"] = Raw(f"{n}d")
            t = rng.choice(OLDER)
            if "{N}" not in t:
                a["older_than"] = Raw("30d")
            return t.format(N=n), build(act, a)
        a.pop("older_than", None)
        t = rng.choice(NEWER)
        a["newer_than"] = Raw(f"{n}d" if "{N}" in t else "7d")
        return t.format(N=n), build(act, a)
    if act == "play_media":
        k = rng.random()
        if k < 0.35:
            t = rng.choice(PLAY_SITE)
            a["site"] = Raw("spotify" if "spotify" in t else "youtube")
            if a["site"] == "spotify":
                a.pop("mode", None)
            return t, build(act, a)
        if k < 0.65:
            m = rng.choice(list(PLAY_MODE))
            a["mode"], a["site"] = Raw(m), Raw("youtube")
            return rng.choice(PLAY_MODE[m]), build(act, a)
        new, _ = fresh_query(rng, split)
        a["query"] = new
        return rng.choice(QUERY).format(Q=new), build(act, a)
    if act == "search_web":
        if rng.random() < 0.7:
            s = rng.choice([x for x in ("google", "youtube", "amazon", "flipkart", "github", "wikipedia", "reddit",
                                         "stackoverflow", "linkedin") if x != str(a.get("site"))])
            a["site"] = Raw(s)
            return T(rng, split, SITE).format(S=s), build(act, a)
        new, _ = fresh_query(rng, split)
        a["query"] = new
        return rng.choice(QUERY).format(Q=new), build(act, a)
    if act == "search_memory":
        (k, v), says = rng.choice(list(MEM_FILTER.items()))
        a[k] = Raw(v)
        return rng.choice(says), build(act, a)
    if act in ("set_volume", "brightness") and "level" in a:
        lv = int(str(a["level"]))
        k = rng.random()
        if k < 0.35:
            a["level"] = Raw(str(min(100, lv + 10)))
            return rng.choice(LEVEL_UP), build(act, a)
        if k < 0.7:
            a["level"] = Raw(str(max(0, lv - 10)))
            return rng.choice(LEVEL_DOWN), build(act, a)
        n = rng.choice(range(0, 101, 5))
        a["level"] = Raw(str(n))
        return rng.choice(LEVEL_SET).format(N=n), build(act, a)
    if act == "remind":
        n = rng.choice([2, 5, 10, 15, 20, 30, 45, 60, 90])
        a.pop("at", None)
        a["after"] = Raw(f"{n}m")
        return rng.choice(REMIND_AFTER).format(N=n), build(act, a)
    if act == "directions":
        k = rng.random()
        if k < 0.4:
            m = rng.choice([x for x in DIR_MODE if x != str(a.get("mode"))])
            a["mode"] = Raw(m)
            return rng.choice(DIR_MODE[m]), build(act, a)
        place = rng.choice(PLACES)
        if k < 0.7:
            a["origin"] = place
            return rng.choice(DIR_ORIGIN).format(P=place), build(act, a)
        a["destination"] = place
        return rng.choice(DIR_DEST).format(P=place), build(act, a)
    if act == "open_app":
        w = rng.choice(list(CLOSE))
        return rng.choice(CLOSE[w]), build("window", {"action": Raw(w), "app": str(a["name"])})
    return None


def fresh_query(rng, split) -> tuple[str, str]:
    for _ in range(20):
        p, text = fresh(rng, split)
        c = parse(p).calls[0]
        if c.action in ("play_media", "search_web") and isinstance(c.args.get("query"), str):
            return c.args["query"], text
    return "lofi beats", ""


# ------------------------------------------------------------------ answering "did you mean ...?"
ORD = {0: ["the first one", "first", "1", "option 1", "pehla", "pehla wala", "the 1st", "number one", "first one please"],
       1: ["the second one", "second", "2", "option 2", "dusra", "dusra wala", "the 2nd", "number two", "second one"],
       2: ["the third one", "third", "3", "option 3", "teesra", "the 3rd", "number three"]}
LAST = ["the last one", "last one", "the last", "aakhri wala"]
NEITHER = ["neither", "none of those", "neither of them", "koi nahi", "cancel", "never mind", "no, none",
           "nothing", "leave it", "rehne do", "forget it"]
YES = ["yes", "yeah", "yep", "do it", "go ahead", "sure", "haan", "haan karo", "ok", "okay do it", "correct",
       "yes please", "that's right", "right", "ji haan", "theek hai", "confirm"]
NO = ["no", "nope", "don't", "cancel", "nahi", "mat karo", "no thanks", "stop", "wrong", "not that"]


def _key_word(p: str) -> str | None:
    c = parse(p).calls[0]
    for k in ("query", "name", "destination", "text", "dest"):
        v = c.args.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def ask(rng, split) -> tuple[str, str, str]:
    n = 2 if rng.random() < 0.75 else 3
    base, _ = fresh(rng, split)
    opts = [base]
    tries = 0
    while len(opts) < n and tries < 30:                  # options usually share an action: a real "torn" case
        tries += 1
        cand, _ = fresh(rng, split)
        if cand in opts:
            continue
        if parse(cand).calls[0].action == parse(base).calls[0].action or rng.random() < 0.3:
            opts.append(cand)
    ctx = "ASK: " + " | ".join(opts)
    r = rng.random()
    if r < 0.12:
        return ctx, rng.choice(NEITHER), "unknown()"
    if r < 0.2:
        return ctx, fresh(rng, split)[1], "unknown()"     # ignored the question, said something new
    i = rng.randrange(len(opts))
    kw = _key_word(opts[i])
    others = [_key_word(o) for j, o in enumerate(opts) if j != i]
    if r < 0.5 and kw and kw not in others and all(kw.lower() not in (o or "").lower() for o in others):
        say = rng.choice(["{K}", "the {K} one", "{K} wala", "i meant {K}", "{K} please"]).format(K=kw)
    elif i == len(opts) - 1 and rng.random() < 0.2:
        say = rng.choice(LAST)
    else:
        say = rng.choice(ORD[i])
    return ctx, say, opts[i]


def confirm(rng, split) -> tuple[str, str, str]:
    p, _ = fresh(rng, split)
    ctx = "CONFIRM: " + p
    r = rng.random()
    if r < 0.45:
        return ctx, rng.choice(YES), p
    if r < 0.75:
        return ctx, rng.choice(NO), "unknown()"
    fu = followup(p, rng, split)                          # "no, to Desktop": a correction is also an answer
    if fu and fu[1] not in (p, "undo()"):
        return ctx, fu[0], fu[1]
    return ctx, rng.choice(YES), p


def example(rng, split) -> dict:
    r = rng.random()
    if r < 0.42:
        for _ in range(10):
            p, _ = fresh(rng, split)
            fu = followup(p, rng, split)
            if fu:
                say, out = fu
                return {"text": f"PREV: {p}{SEP}{say}", "program": out}
    if r < 0.72:                                          # a fresh command is not a follow-up
        p, _ = fresh(rng, split)
        _, say = fresh(rng, split)
        return {"text": f"PREV: {p}{SEP}{say}", "program": "unknown()"}
    if r < 0.88:
        ctx, say, out = ask(rng, split)
    else:
        ctx, say, out = confirm(rng, split)
    return {"text": f"{ctx}{SEP}{say}", "program": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--n-eval", type=int, default=6_000)
    ap.add_argument("--out", type=Path, default=Path("data/dialog"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sizes = {"train": args.n, "val": args.n_eval, "test_iid": args.n_eval, "val_ood": args.n_eval,
             "test_ood": args.n_eval}
    seen_train: set[str] = set()
    for name, n in sizes.items():
        split = name if name in ("val_ood", "test_ood") else "train"
        rows, bad = [], 0
        while len(rows) < n:
            try:
                ex = example(rng, split)
                canonical(ex["program"])
            except (DSLError, ValueError, KeyError):
                bad += 1
                continue
            if name == "train":
                seen_train.add(ex["text"])
            elif ex["text"] in seen_train:
                continue
            rows.append(ex)
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows):,} examples ({bad} skipped)")
    (args.out / "meta.json").write_text(json.dumps({"actions": [], "features": TEACHES}), encoding="utf-8")


if __name__ == "__main__":
    main()
