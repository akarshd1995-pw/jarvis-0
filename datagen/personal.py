"""Personal training examples from your own words (phase 43), mixed in when vocab.json exists.

    your Start-menu apps   "open obsidian", "whatsapp beta kholo"          -> open_app(name="...")
    your projects          "open my everway project"                        -> open_folder(path="everway")
    your folders           "move the pdfs in downloads to tax 2026"         -> move_files(..., dest="tax 2026")
    your typos             a normal example with one of your typos in it, the program unchanged:
                           "opne my donwloads folder" -> open_folder(path=Downloads)
The vocabulary is read from JARVIS_VOCAB, or vocab.json in Jarvis's data folder (JARVIS_HOME or ~/.jarvis0).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import lexicon as LX


def vocab_path() -> Path:
    if os.environ.get("JARVIS_VOCAB"):
        return Path(os.environ["JARVIS_VOCAB"])
    return Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis0")) / "vocab.json"


def load_vocab(path: Path | None = None) -> dict:
    try:
        return json.loads((path or vocab_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _clean(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.+-]", " ", name)).strip().lower()


APP_FRAMES = ["open {a}", "launch {a}", "start {a}", "{a} kholo", "open up {a}", "can you open {a}", "run {a}",
              "fire up {a}", "{a} open karo", "bring up {a}"]
PROJECT_FRAMES = ["open my {p} project", "open the {p} project", "open {p} folder", "{p} project kholo",
                  "open the {p} repo", "go to my {p} project", "show me the {p} folder", "open {p}"]
MOVE_FRAMES = ["move the {w} in downloads to {p}", "move {w} from downloads into {p}", "copy the {w} in downloads to {p}",
               "downloads ki {w} {p} me daal do", "put the {w} from downloads in {p}"]
WHAT = [("pdf", "pdfs"), ("image", "images"), ("document", "documents"), ("video", "videos"), ("archive", "zip files")]


def personal_generators(vocab: dict, base_generators: list) -> list:
    """(generator, weight) pairs for this person's words; empty when there's nothing personal yet."""
    from .generate_intents import Ctx, dsl_call, q, slot   # noqa: F401
    known = {s.lower() for forms in LX.KNOWN_APPS.values() for s in forms} | {k.lower() for k in LX.KNOWN_APPS}
    # apps the model already knows by name keep their usual label (open_app(name=spotify)), never a second one
    apps = [a for a in (_clean(x) for x in vocab.get("apps") or []) if 2 <= len(a) <= 30 and a not in known]
    places = [p for p in (_clean(x) for x in list((vocab.get("projects") or {})) + list((vocab.get("folders") or {})))
              if 2 <= len(p) <= 40]
    typos = vocab.get("typos") or {}
    by_word: dict[str, list[str]] = {}
    for t, w in typos.items():
        by_word.setdefault(w, []).append(t)
    out = []

    if apps:
        def gen_my_apps(c):
            a = c.rng.choice(apps)
            return dsl_call("open_app", {"name": q(a)}), c.rng.choice(APP_FRAMES).format(a=slot(a)), {}
        out.append((gen_my_apps, 3))
    if places:
        def gen_my_places(c):
            p = c.rng.choice(places)
            if c.rng.random() < 0.6:
                return dsl_call("open_folder", {"path": q(p)}), c.rng.choice(PROJECT_FRAMES).format(p=slot(p)), {}
            ext, said = c.rng.choice(WHAT)
            tpl = c.rng.choice(MOVE_FRAMES)
            op = "copy_files" if tpl.startswith("copy") else "move_files"
            return dsl_call(op, {"src": "Downloads", "ext": ext, "dest": q(p)}), tpl.format(w=said, p=slot(p)), {}
        out.append((gen_my_places, 3))
    if by_word:
        fns, weights = zip(*base_generators)

        def gen_my_typos(c):
            for _ in range(20):
                fn = c.rng.choices(fns, weights)[0]
                prog, text, extra = fn(c)
                # never inside text that's copied into the program (a song or file name is not "corrected");
                # a slot whose words aren't copied (a folder said as "downloads" -> Downloads) is fine
                parts = re.split(r"(\x01.*?\x02)", text)
                copied = {m.lower() for m in re.findall(r'"([^"]*)"', prog)}
                spots = [(i, w) for i, part in enumerate(parts)
                         if not (part.startswith("\x01") and part.strip("\x01\x02").lower() in copied)
                         for w in re.findall(r"[a-z]+", part) if w in by_word]
                if spots:
                    i, w = c.rng.choice(spots)
                    parts[i] = re.sub(rf"\b{w}\b", c.rng.choice(by_word[w]), parts[i], count=1)
                    return prog, "".join(parts), extra
            return prog, text, extra
        out.append((gen_my_typos, 2))
    return out
