"""Synthetic (text -> program) generator for training the intent model from scratch.

Program-first generation: sample a valid DSL program, then render it into natural
English or Hinglish with randomized templates, word order, fillers and typo noise.
Slot values are protected from noise so the target program stays correct.

    python -m datagen.generate_intents --n 300000 --out data/intents
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from jarvis.dsl.grammar import canonical

L, R = "\x01", "\x02"   # markers around protected slot text


def slot(s: str) -> str:
    return f"{L}{s}{R}"

# ---------------------------------------------------------------- vocab

FOLDERS = {
    "Downloads": ["downloads", "Downloads", "my downloads", "the downloads folder", "downloads folder"],
    "Documents": ["documents", "Documents", "my documents", "the documents folder", "docs"],
    "Desktop": ["desktop", "Desktop", "my desktop", "the desktop"],
    "Pictures": ["pictures", "Pictures", "my pictures", "the pictures folder", "photos folder"],
    "Music": ["music", "Music", "my music folder"],
    "Videos": ["videos", "Videos", "my videos folder"],
    "Archive": ["archive", "Archive", "the archive", "the archive folder", "my archive"],
}
EXTS = ["pdf", "png", "jpg", "zip", "exe", "docx", "mp4", "csv", "xlsx", "pptx", "mp3", "txt", "msi", "iso"]
NAME_WORDS = ["invoice", "resume", "screenshot", "report", "bill", "ticket", "draft", "notes", "assignment", "receipt"]
REMIND_TEXTS = ["call mom", "drink water", "submit the assignment", "check the archive", "join the standup",
                "push my code", "pay the electricity bill", "stretch", "take a break", "reply to the email",
                "book the train ticket", "review the PR", "water the plants", "start the training run"]
APPS = {
    "vscode": ["vscode", "VS Code", "vs code", "visual studio code", "code editor"],
    "chrome": ["chrome", "Chrome", "google chrome", "the browser"],
    "edge": ["edge", "microsoft edge"],
    "notepad": ["notepad", "Notepad"],
    "explorer": ["file explorer", "explorer"],
    "terminal": ["terminal", "the terminal", "windows terminal", "cmd"],
    "spotify": ["spotify", "Spotify", "music app"],
    "calculator": ["calculator", "calc"],
    "settings": ["settings", "windows settings"],
}
URLS = ["github.com", "youtube.com", "mail.google.com", "leetcode.com", "kaggle.com", "linkedin.com",
        "stackoverflow.com", "huggingface.co", "vercel.com", "news.ycombinator.com"]
SYSTEM = {
    "lock": ["lock the screen", "lock my pc", "lock the computer", "lock", "screen lock kar do", "pc lock karo"],
    "mute": ["mute", "mute the sound", "mute audio", "sound band karo", "awaaz band kar do"],
    "volume_up": ["volume up", "increase the volume", "make it louder", "awaaz badhao", "volume badha do"],
    "volume_down": ["volume down", "decrease the volume", "make it quieter", "awaaz kam karo", "volume kam kar do"],
    "sleep_display": ["turn off the display", "sleep the monitor", "screen off kar do", "display band karo"],
}
SEARCH_QUERIES = ["the error I copied yesterday", "notes about GST", "the blue chart screenshot",
                  "my resume", "the train ticket pdf", "that python snippet", "the meeting notes",
                  "the invoice from last month", "the API key docs"]

# ---------------------------------------------------------------- surface helpers

def s_folder(rng, key):
    return slot(rng.choice(FOLDERS[key]))


def s_ext(rng, e, hinglish=False):
    forms = [f"{e} files", f"{e}s", f".{e} files", f"{e.upper()} files", f"all {e}s", f"all the {e} files"]
    if hinglish:
        forms += [f"saari {e} files", f"{e} wali files", f"sab {e}"]
    return slot(rng.choice(forms))


def sample_duration(rng):
    unit = rng.choice(["d", "d", "w", "h", "month"])
    if unit == "month":
        n = rng.randint(1, 6)
        return f"{n * 30}d", rng.choice([f"{n} months" if n > 1 else "a month", f"{n} month" + ("s" if n > 1 else "")])
    n = rng.randint(1, 30 if unit == "d" else 8 if unit == "w" else 48)
    word = {"d": "day", "w": "week", "h": "hour"}[unit]
    one = {"d": "a day", "w": "a week", "h": "an hour"}[unit]
    text = one if n == 1 and rng.random() < 0.5 else f"{n} {word}{'s' if n > 1 else ''}"
    return f"{n}{unit}", text


def s_older(rng, txt, hinglish):
    forms = [f"older than {slot(txt)}", f"more than {slot(txt)} old", f"from over {slot(txt)} ago"]
    if hinglish:
        forms += [f"{slot(txt)} se purani", f"jo {slot(txt)} se purani hain"]
    return rng.choice(forms)


def s_newer(rng, txt, hinglish):
    forms = [f"newer than {slot(txt)}", f"from the last {slot(txt)}", f"less than {slot(txt)} old"]
    if hinglish:
        forms += [f"pichhle {slot(txt)} ki", f"{slot(txt)} se nayi"]
    return rng.choice(forms)


def s_name(rng, w, hinglish):
    forms = [f'with "{slot(w)}" in the name', f"named {slot(w)}", f"containing {slot(w)} in the filename",
             f"that have {slot(w)} in their name"]
    if hinglish:
        forms += [f"jinke naam me {slot(w)} hai", f"{slot(w)} naam wali"]
    return rng.choice(forms)

# ---------------------------------------------------------------- per-action samplers
# each returns (dsl_string, surface_text)

def gen_file_op(rng, op):
    hing = rng.random() < 0.25
    src = rng.choice(list(FOLDERS))
    args = {"src": src}
    filt_text = []
    if rng.random() < 0.75:
        e = rng.choice(EXTS)
        args["ext"] = e
        ext_txt = s_ext(rng, e, hing)
    else:
        ext_txt = slot(rng.choice(["files", "everything", "all files", "the files"]))
    if rng.random() < 0.45:
        dsl_d, txt = sample_duration(rng)
        args["older_than"] = dsl_d
        filt_text.append(s_older(rng, txt, hing))
    elif rng.random() < 0.15:
        dsl_d, txt = sample_duration(rng)
        args["newer_than"] = dsl_d
        filt_text.append(s_newer(rng, txt, hing))
    if rng.random() < 0.2:
        w = rng.choice(NAME_WORDS)
        args["name_contains"] = w
        filt_text.append(s_name(rng, w, hing))
    rng.shuffle(filt_text)
    filt = (" " + " ".join(filt_text)) if filt_text else ""

    if op in ("move", "copy"):
        dest = rng.choice([f for f in FOLDERS if f != src])
        args["dest"] = dest
        S, D = s_folder(rng, src), s_folder(rng, dest)
        if hing:
            verb = {"move": rng.choice(["move kar do", "shift karo", "daal do"]),
                    "copy": rng.choice(["copy kar do", "copy karo"])}[op]
            text = rng.choice([f"{S} se {ext_txt}{filt} {D} me {verb}",
                               f"{ext_txt}{filt} {S} se {D} me {verb}"])
        else:
            verb = {"move": rng.choice(["move", "shift", "transfer", "put"]),
                    "copy": rng.choice(["copy", "duplicate", "back up"])}[op]
            text = rng.choice([f"{verb} {ext_txt}{filt} from {S} to {D}",
                               f"{verb} {ext_txt} from {S}{filt} into {D}",
                               f"please {verb} {ext_txt}{filt} in {S} over to {D}",
                               f"can you {verb} {ext_txt}{filt} from {S} to {D}"])
    elif op == "delete":
        S = s_folder(rng, src)
        if hing:
            text = rng.choice([f"{S} se {ext_txt}{filt} delete kar do", f"{S} me {ext_txt}{filt} hata do"])
        else:
            verb = rng.choice(["delete", "remove", "get rid of", "clean up", "trash"])
            text = rng.choice([f"{verb} {ext_txt}{filt} from {S}", f"{verb} {ext_txt} in {S}{filt}",
                               f"{verb} {ext_txt}{filt} in {S}"])
    else:  # list
        S = s_folder(rng, src)
        if hing:
            text = rng.choice([f"{S} me {ext_txt}{filt} dikhao", f"{S} ki {ext_txt}{filt} list karo"])
        else:
            verb = rng.choice(["list", "show", "show me", "what are the"])
            text = rng.choice([f"{verb} {ext_txt}{filt} in {S}", f"{verb} {ext_txt} from {S}{filt}"])

    parts = [f"{k}={v}" for k, v in args.items()]
    if "name_contains" in args:
        parts = [p if not p.startswith("name_contains=") else f'name_contains="{args["name_contains"]}"' for p in parts]
    return f"{op}_files({', '.join(parts)})", text


def gen_remind(rng):
    hing = rng.random() < 0.25
    task = rng.choice(REMIND_TEXTS)
    if rng.random() < 0.5:
        h, m = rng.randint(0, 23), rng.choice([0, 0, 15, 30, 45, rng.randint(0, 59)])
        if rng.random() < 0.6:
            h12, ampm = (h % 12 or 12), ("am" if h < 12 else "pm")
            when_txt = f"{h12}:{m:02d} {ampm}" if m else rng.choice([f"{h12} {ampm}", f"{h12}{ampm}"])
        else:
            when_txt = f"{h:02d}:{m:02d}"
        dsl = f'remind(text="{task}", at={h:02d}:{m:02d})'
        when = f"at {slot(when_txt)}"
        when_h = f"{slot(when_txt)} baje"
    else:
        unit = rng.choice(["m", "h"])
        n = rng.choice([5, 10, 15, 20, 30, 45, 90]) if unit == "m" else rng.randint(1, 8)
        word = "minute" if unit == "m" else "hour"
        txt = f"{n} {word}{'s' if n > 1 else ''}"
        if unit == "m" and rng.random() < 0.3:
            txt = f"{n} mins"
        dsl = f'remind(text="{task}", after={n}{unit})'
        when = f"in {slot(txt)}"
        when_h = f"{slot(txt)} me"
    T = slot(task)
    if hing:
        text = rng.choice([f"{when_h} yaad dilana {T}", f"{when_h} mujhe remind karna {T}"])
    else:
        text = rng.choice([f"remind me to {T} {when}", f"{when} remind me to {T}",
                           f"set a reminder to {T} {when}", f"ping me {when} to {T}"])
    return dsl, text


def gen_open_app(rng):
    key = rng.choice(list(APPS))
    A = slot(rng.choice(APPS[key]))
    text = rng.choice([f"open {A}", f"launch {A}", f"start {A}", f"fire up {A}", f"{A} khol do", f"{A} open karo"])
    return f"open_app(name={key})", text


def gen_open_url(rng):
    u = rng.choice(URLS)
    U = slot(u)
    text = rng.choice([f"open {U}", f"go to {U}", f"take me to {U}", f"{U} kholo", f"browse {U}"])
    return f'open_url(url="{u}")', text


def gen_open_folder(rng):
    key = rng.choice(list(FOLDERS))
    F = s_folder(rng, key)
    text = rng.choice([f"open {F}", f"show {F}", f"open up {F}", f"{F} kholo"])
    return f"open_folder(path={key})", text


def gen_system(rng):
    key = rng.choice(list(SYSTEM))
    return f"system(action={key})", slot(rng.choice(SYSTEM[key]))


def gen_search(rng):
    q = rng.choice(SEARCH_QUERIES)
    Q = slot(q)
    text = rng.choice([f"find {Q}", f"search for {Q}", f"where is {Q}", f"look up {Q}", f"{Q} dhundo"])
    return f'search_memory(query="{q}")', text


def gen_undo(rng):
    return "undo()", rng.choice(["undo", "undo that", "revert the last command", "undo the last thing",
                                 "wapas kar do", "go back", "cancel what you just did"])


GENERATORS = [
    (lambda r: gen_file_op(r, "move"), 18), (lambda r: gen_file_op(r, "copy"), 8),
    (lambda r: gen_file_op(r, "delete"), 8), (lambda r: gen_file_op(r, "list"), 6),
    (gen_remind, 16), (gen_open_app, 12), (gen_open_url, 8), (gen_open_folder, 6),
    (gen_system, 8), (gen_search, 6), (gen_undo, 2),
]
_FNS, _WEIGHTS = zip(*GENERATORS)
CONNECTORS = [" and ", " and then ", " then ", ", then ", ". also ", " aur ", " aur phir ", " & "]
FILLERS_PRE = ["", "", "", "hey jarvis ", "jarvis ", "please ", "can you ", "could you ", "yo ", "ok "]
FILLERS_POST = ["", "", "", " please", " thanks", " asap", " na", " yaar"]

# ---------------------------------------------------------------- noise

def _typo(word: str, rng) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.random()
    if op < 0.4:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # swap
    if op < 0.7:
        return word[:i] + word[i + 1:]                          # drop
    return word[:i] + word[i] + word[i:]                        # double


def apply_noise(text: str, rng, typo_p=0.04) -> str:
    lower = rng.random() < 0.5
    starts_with_slot = text.lstrip().startswith(L)
    out = []
    for seg in re.split(f"({L}.*?{R})", text):
        if seg.startswith(L):
            out.append(seg[1:-1])  # protected slot: untouched
            continue
        words = [(_typo(w, rng) if w.isalpha() and rng.random() < typo_p else w) for w in seg.split(" ")]
        seg = " ".join(words)
        out.append(seg.lower() if lower else seg)
    text = "".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    if rng.random() < 0.3 and text and not starts_with_slot:
        text = text[0].upper() + text[1:]
    if rng.random() < 0.15:
        text += rng.choice([".", "!", "?"])
    return text


def sample(rng: random.Random) -> dict:
    n_calls = rng.choices([1, 2, 3], weights=[70, 24, 6])[0]
    dsls, texts = [], []
    for _ in range(n_calls):
        d, t = rng.choices(_FNS, weights=_WEIGHTS)[0](rng)
        if d == "undo()" and n_calls > 1:
            d, t = gen_open_app(rng)
        dsls.append(d)
        texts.append(t)
    text = texts[0]
    for t in texts[1:]:
        text += rng.choice(CONNECTORS) + t
    text = rng.choice(FILLERS_PRE) + text + rng.choice(FILLERS_POST)
    return {"text": apply_noise(text, rng), "program": canonical(" ; ".join(dsls))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--out", type=Path, default=Path("data/intents"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    seen, rows = set(), []
    while len(rows) < args.n:
        ex = sample(rng)
        if ex["text"] not in seen:
            seen.add(ex["text"])
            rows.append(ex)
    rng.shuffle(rows)
    n_val = n_test = max(1000, args.n // 50)
    splits = {"test": rows[:n_test], "val": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}
    for name, data in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} examples")


if __name__ == "__main__":
    main()
