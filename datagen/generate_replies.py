"""Synthetic (facts -> reply) data for the reply model.

Random action results are turned into facts by the SAME code the agent uses at runtime
(Persona.plan / plan_confirm / plan_error), then rendered with a random phrasing from the
paraphrase bank. Every target is verified by the fact checker before it is written.

    python -m datagen.generate_replies --n 300000 --out data/replies
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from jarvis.actions.base import ActionResult as AR
from jarvis.nlg.bank import variants
from jarvis.nlg.checker import check, linearize
from jarvis.persona.persona import Persona, render_template
from jarvis.persona.templates import ADDRESS_RATE, SITE_NAMES, SYSTEM_PHRASES
from jarvis.safety import SafetyError

from . import lexicon as LX

TONES = (["butler"] * 5) + (["friendly"] * 3) + (["minimal"] * 2)
ADDRESSES = ["sir"] * 6 + ["ma'am"] * 2 + [""] * 3 + ["Akash", "Priya", "Rahul", "Neha", "Sam", "boss"]
EXT_POOL = LX.EXTS + [None] * 10


def _folder(rng) -> str:
    if rng.random() < 0.75:
        return rng.choice(list(LX.FOLDERS))
    parts = [rng.choice(LX.PATH_ROOTS)] + [LX.path_part(rng) for _ in range(rng.randint(1, 3))]
    return "/".join(parts)


def _count(rng) -> int:
    r = rng.random()
    return 0 if r < 0.12 else rng.randint(1, 12) if r < 0.6 else rng.randint(13, 120) if r < 0.92 else rng.randint(121, 499)


def _file_data(rng, op: str, dry: bool = False) -> dict:
    src = _folder(rng)
    dest = _folder(rng) if op in ("move", "copy") else None
    older = newer = None
    if rng.random() < 0.45:
        older = f"{rng.randint(1, 60)}{rng.choice('dwh')}" if rng.random() < 0.8 else f"{rng.randint(1, 12) * 30}d"
    elif rng.random() < 0.2:
        newer = f"{rng.randint(1, 30)}{rng.choice('dwh')}"
    return {"op": op, "count": _count(rng), "dry": dry, "src": src, "dest": dest, "ext": rng.choice(EXT_POOL),
            "older_than": older, "newer_than": newer,
            "name_contains": LX.name_fragment(rng) if rng.random() < 0.15 else None}


def _filenames(rng, n: int) -> list[str]:
    return [f"{LX.name_fragment(rng).strip()}{rng.randint(1, 99) if rng.random() < 0.3 else ''}.{rng.choice(LX.EXTS)}"
            for _ in range(n)]


def random_case(rng: random.Random, p: Persona) -> tuple[str, dict]:
    """Returns (key, fields) planned by the persona from a random situation."""
    if rng.random() < 0.12:
        return _control_case(rng, p)
    r = rng.random()
    now = time.time()
    if r < 0.28:
        return p.plan(AR("", kind="files", data=_file_data(rng, rng.choice(["move", "move", "copy", "delete"]))))
    if r < 0.36:
        d = _file_data(rng, "list")
        items = _filenames(rng, d["count"] if d["count"] < 8 else rng.randint(6, 9))
        return p.plan(AR("", items if d["count"] else [], kind="list", data=d))
    if r < 0.44:
        d = _file_data(rng, rng.choice(["move", "copy", "delete"]), dry=True)
        d["count"] = max(1, d["count"])
        return p.plan_confirm(AR("", kind="files", data=d))
    if r < 0.46:
        return p.plan_confirm(AR("", kind="system", data={"action": rng.choice(list(SYSTEM_PHRASES))}))
    if r < 0.54:
        what = rng.choice(["app", "app", "url", "folder"])
        name = {"app": rng.choice(list(LX.KNOWN_APPS) + LX.OTHER_APPS), "url": LX.url_name(rng) + rng.choice(LX.TLDS),
                "folder": _folder(rng)}[what]
        return p.plan(AR("", kind="open", data={"what": what, "name": name}))
    if r < 0.64:
        rel = rng.random() < 0.5
        due = now + (rng.randint(1, 600) * 60 if rel else rng.randint(1, 72) * 3600 + rng.randint(0, 59) * 60)
        return p.plan(AR("", kind="remind", data={"text": LX.remind_text(rng), "due": due, "relative": rel}))
    if r < 0.69:
        a = rng.choice(list(SYSTEM_PHRASES))
        return p.plan(AR("", kind="system", data={"action": a, "unsupported_os": rng.random() < 0.03}))
    if r < 0.75:
        site = rng.choice(list(SITE_NAMES))
        return p.plan(AR("", kind="search", data={"query": LX.web_query(rng, site), "site": site}))
    if r < 0.82:
        site = "spotify" if rng.random() < 0.2 else "youtube"
        return p.plan(AR("", kind="play", data={"query": LX.web_query(rng, site), "site": site,
                                                 "fallback": site == "youtube" and rng.random() < 0.1}))
    if r < 0.84:
        return p.plan(AR("", kind="memory", data={"query": LX.search_query(rng)}))
    if r < 0.89:
        return p.plan(AR("", kind="unsupported", data={"request": LX.unsupported_text(rng)}))
    if r < 0.91:
        return p.plan(AR("", kind="unknown"))
    if r < 0.93:
        c = rng.choice([0, 0, rng.randint(1, 40)])
        return p.plan(AR("", kind="undo", data={"count": c, "cancelled": c == 0 and rng.random() < 0.3}))
    if r < 0.96:
        action = rng.choice(["move_files", "copy_files", "delete_files", "list_files", "open_folder"])
        e = rng.choice([FileNotFoundError(f"folder not found: {_folder(rng)}"),
                        SafetyError(f"{_folder(rng)} is outside the allowed folders"),
                        SafetyError(f"{rng.randint(501, 5000)} files matched; cap is 500. Narrow the filter."),
                        PermissionError("access denied")])
        return p.plan_error(action, e)
    if r < 0.98:
        return "reminder_due", {"text": LX.remind_text(rng)}
    return rng.choice(["declined", "unparsed"]), {}


def _control_case(rng: random.Random, p: Persona) -> tuple[str, dict]:
    """Phase 5 system-control replies and confirmations."""
    kind = rng.choice(["window", "window", "desktop", "media", "brightness", "volume_set", "radio", "screenshot",
                       "record", "power", "power"])
    app = rng.choice(["", "", "chrome", "vscode", "spotify", "explorer"] + LX.OTHER_APPS[:20])
    if kind == "window":
        act = rng.choice(["snap_left", "snap_right", "maximize", "minimize", "restore", "close", "minimize_all"])
        if act == "close" and rng.random() < 0.4:
            return p.plan_confirm(AR("", kind="window", data={"action": "close", "app": app, "dry": True}))
        return p.plan(AR("", kind="window", data={"action": act, "app": app}))
    if kind == "desktop":
        return p.plan(AR("", kind="desktop", data={"action": rng.choice(["new", "next", "previous", "close"])}))
    if kind == "media":
        return p.plan(AR("", kind="media", data={"action": rng.choice(["play_pause", "next", "previous", "stop"])}))
    if kind == "brightness":
        return p.plan(AR("", kind="brightness", data={"level": rng.randint(0, 100),
                                                      "action": rng.choice(["set", "up", "down"])}))
    if kind == "volume_set":
        return p.plan(AR("", kind="volume_set", data={"level": rng.randint(0, 100)}))
    if kind == "radio":
        d = {"kind": rng.choice(["wifi", "bluetooth"]), "state": rng.choice(["on", "off"])}
        if d["state"] == "off" and rng.random() < 0.4:
            return p.plan_confirm(AR("", kind="radio", data={**d, "dry": True}))
        return p.plan(AR("", kind="radio", data=d))
    if kind == "screenshot":
        return p.plan(AR("", kind="screenshot", data={"mode": rng.choice(["full", "window", "region"])}))
    if kind == "record":
        return p.plan(AR("", kind="record", data={"action": rng.choice(["start", "stop"])}))
    d = {"action": rng.choice(["shutdown", "restart", "sleep", "sign_out", "cancel"]),
         "seconds": rng.choice([30, 30, 60, 300, 600, 1800, 3600])}
    if d["action"] != "cancel" and rng.random() < 0.5:
        return p.plan_confirm(AR("", kind="power", data={**d, "dry": True}))
    return p.plan(AR("", kind="power", data=d))


def make_example(rng: random.Random) -> dict | None:
    tone, address = rng.choice(TONES), rng.choice(ADDRESSES)
    p = Persona(tone=tone, address=address, seed=rng.randint(0, 1 << 30))
    key, fields = random_case(rng, p)
    if not key:
        return None
    use_addr = bool(address) and rng.random() < max(ADDRESS_RATE.get(tone, 0), 0.4 if tone == "butler" else 0)
    tgt = render_template(rng.choice(variants(tone, key)), fields, address if use_addr else "")
    clean = {k: v for k, v in fields.items() if k[:1].islower()}
    if not check(key, clean, tone, address, tgt):
        raise AssertionError(f"bank phrasing fails its own fact check: {key} {clean} -> {tgt!r}")
    return {"src": linearize(key, clean, tone, address), "tgt": tgt, "key": key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("data/replies"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    while len(rows) < args.n + 2 * args.n_eval:
        ex = make_example(rng)
        if ex:
            rows.append(ex)
    rng.shuffle(rows)
    splits = {"val": rows[:args.n_eval], "test": rows[args.n_eval:2 * args.n_eval], "train": rows[2 * args.n_eval:]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name:6s} {len(data):>8,d}")


if __name__ == "__main__":
    main()
