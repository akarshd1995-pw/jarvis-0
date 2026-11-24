"""Conversation & clarification (phase 6).

Jarvis remembers the last thing it did for a few minutes, so you can follow up:
    "move the pdfs in downloads to documents"  ->  "no, to Desktop"  /  "do the same for Pictures"  /  "again"
and when the command model is torn between two readings it asks instead of guessing:
    "Did you mean 1. play "super machi" on YouTube  2. play "sup" on YouTube?"  ->  "the first one" / "2" / "neither"
Risky low-confidence moves get a yes/no first ("Move the PDFs from Downloads to Documents?" -> "no, to Desktop").

A small dialog model (trained from scratch, datagen/generate_dialog.py) reads   <context> || <what you said>
and writes the program to run, or unknown() when it isn't a reply. Until that model is trained, a few backup
rules cover "again", "same for <folder>", "<folder> instead", ordinals and yes/no.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .dsl.grammar import DSLError, canonical, parse

log = logging.getLogger(__name__)

SEP = " || "
PREV_TTL = 180          # follow-ups work for 3 minutes after a command
ASK_TTL = 120
MIN_CONF = 0.6
TORN_RATIO = 0.85       # runner-up at least 85% as likely as the best reading -> ask
TORN_MAX_CONF = 0.9
RISKY_LOW_CONF = 0.7    # a move/copy the model is less than 70% sure of -> confirm first
RISKY = ("move_files", "copy_files")


@dataclass
class Context:
    kind: str                         # prev | ask | confirm
    programs: list[str]
    ts: float
    text: str = ""                    # what you originally said (ask/confirm)
    usage_id: int | None = None
    extra: dict = field(default_factory=dict)

    def as_text(self) -> str:
        if self.kind == "prev":
            return "PREV: " + self.programs[0]
        if self.kind == "ask":
            return "ASK: " + " | ".join(self.programs)
        return "CONFIRM: " + self.programs[0]


class DialogModel:
    """The trained dialog model (checkpoints/dialog_*.onnx), or None when it hasn't been trained."""

    def __init__(self, ckpt_dir: Path):
        enc, dec = ckpt_dir / "dialog_encoder.onnx", ckpt_dir / "dialog_decoder.onnx"
        tok = ckpt_dir / "dialog_tokenizer.json"
        from .nlu.onnx_parser import ONNXParser
        self.parser = ONNXParser(enc, dec, tok, min_confidence=0.0, meta=ckpt_dir / "dialog_onnx.json")

    @classmethod
    def load(cls, ckpt_dir: Path) -> "DialogModel | None":
        if not (ckpt_dir / "dialog_encoder.onnx").exists():
            return None
        try:
            return cls(ckpt_dir)
        except Exception as e:
            log.warning("dialog model unavailable (%s); using backup follow-up rules", e)
            return None

    def read(self, ctx: Context, text: str) -> tuple[str | None, float]:
        prog, conf = self.parser.parse(ctx.as_text() + SEP + text)
        return (str(prog) if prog is not None else None), conf


# ------------------------------------------------------------------ backup rules (no dialog model yet)
_F = {"downloads": "Downloads", "download": "Downloads", "documents": "Documents", "docs": "Documents",
      "desktop": "Desktop", "pictures": "Pictures", "photos": "Pictures", "pics": "Pictures", "music": "Music",
      "songs": "Music", "videos": "Videos", "movies": "Videos", "archive": "Archive"}
_FOLDER = r"(?:the |my )?(" + "|".join(_F) + r")(?: folder)?"
_AGAIN = re.compile(r"^(?:do (?:it|that) again|again|once more|repeat(?: that)?|same (?:thing )?again|one more time"
                    r"|phir se(?: karo)?|dobara(?: karo)?|ek baar aur)[.! ]*$", re.I)
_SAME_FOR = re.compile(rf"^(?:(?:now )?do the same (?:for|in|with)|same (?:for|in|with)|and|what about|now do)\s+{_FOLDER}"
                       rf"(?: too| as well)?\??$|^{_FOLDER} (?:ka|ke|me) bhi(?: karo)?$", re.I)
_INSTEAD = re.compile(rf"^(?:no,? |nahi,? |actually,? |i meant )?(?:to |into |in )?{_FOLDER}(?: instead| me| please)?$"
                      rf"|^(?:no,? )?(?:put|move|send) them (?:in|to|into) {_FOLDER}(?: instead)?$", re.I)
_ORD = [re.compile(r"^(?:the )?(?:first|1st|1|one|option 1|number one|pehla)(?: one| wala)?$", re.I),
        re.compile(r"^(?:the )?(?:second|2nd|2|two|option 2|number two|dusra)(?: one| wala)?$", re.I),
        re.compile(r"^(?:the )?(?:third|3rd|3|three|option 3|number three|teesra)(?: one| wala)?$", re.I)]
_YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go ahead|do it|haan|ha|ji haan|theek hai|correct|right|confirm)"
                  r"(?: please| do it| karo)?[.! ]*$", re.I)
_NO = re.compile(r"^(?:no|nope|nah|nahi|cancel|don'?t|mat karo|neither|none(?: of (?:those|them))?|never ?mind"
                 r"|rehne do|forget it|koi nahi)[.! ]*$", re.I)


def rules_read(ctx: Context, text: str) -> str | None:
    t = text.strip().rstrip(".!")
    if ctx.kind in ("ask", "confirm"):
        if _NO.match(t):
            return "unknown()"
        if ctx.kind == "confirm" and _YES.match(t):
            return ctx.programs[0]
        if ctx.kind == "ask":
            for i, pat in enumerate(_ORD):
                if pat.match(t) and i < len(ctx.programs):
                    return ctx.programs[i]
            if re.match(r"^(?:the )?last(?: one)?$", t, re.I):
                return ctx.programs[-1]
            hits = [p for p in ctx.programs if len(t) > 2 and t.lower() in p.lower()]
            if len(hits) == 1:
                return hits[0]
    base = ctx.programs[0]
    if ctx.kind == "ask":
        return None
    try:
        call = parse(base).calls[-1]
    except DSLError:
        return None
    if _AGAIN.match(t):
        return base
    m = _SAME_FOR.match(t)
    if m and "src" in call.args:
        return edit(base, src=Ident_(_F[(m.group(1) or m.group(2)).lower()]))
    m = _INSTEAD.match(t)
    if m and call.action in RISKY:
        return edit(base, dest=Ident_(_F[(m.group(1) or m.group(2)).lower()]), new_folder=None)
    return None


class Ident_(str):
    """A folder name to write without quotes."""


def edit(program: str, **changes) -> str | None:
    """program with some arguments changed (None removes one). Folders as Ident_ are written as identifiers."""
    try:
        call = parse(program).calls[-1]
    except DSLError:
        return None
    args = dict(call.args)
    for k, v in changes.items():
        if v is None:
            args.pop(k, None)
        else:
            args[k] = v

    def lit(v):
        if isinstance(v, Ident_) or not isinstance(v, str):
            return str(v)
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    try:
        return canonical(f"{call.action}({', '.join(f'{k}={lit(v)}' for k, v in args.items())})")
    except DSLError:
        return None


# ------------------------------------------------------------------ the conversation
class Conversation:
    def __init__(self, model: DialogModel | None = None, clarify: bool = True):
        self.model, self.clarify = model, clarify
        self.ctx: Context | None = None

    def active(self, now: float | None = None) -> Context | None:
        now = time.time() if now is None else now
        c = self.ctx
        if c is None:
            return None
        if now - c.ts > (PREV_TTL if c.kind == "prev" else ASK_TTL):
            self.ctx = None
            return None
        return c

    def ran(self, program: str | None, now: float | None = None) -> None:
        """Jarvis just ran this: follow-ups refer to it."""
        try:
            program = canonical(program) if program else None
        except DSLError:
            program = None
        self.ctx = (Context("prev", [program], time.time() if now is None else now)
                    if program and program != "unknown()" else None)

    def asking(self, kind: str, programs: list[str], text: str, usage_id: int | None, now: float | None = None):
        self.ctx = Context(kind, programs, time.time() if now is None else now, text=text, usage_id=usage_id)

    def clear(self):
        self.ctx = None

    def read(self, text: str, now: float | None = None) -> tuple[str | None, str]:
        """What you said, as a reply to the context: (program, source), program 'unknown()' = not a reply / no."""
        c = self.active(now)
        if c is None:
            return None, ""
        if self.model is not None:
            try:
                prog, conf = self.model.read(c, text)
            except Exception as e:
                log.warning("dialog model failed: %s", e)
                prog, conf = None, 0.0
            if prog is not None and (conf >= MIN_CONF or prog == "unknown()"):
                return prog, f"follow-up {conf:.0%}"
            ruled = rules_read(c, text)
            return (ruled, "follow-up rules (dialog model unsure)") if ruled else (None, "")
        ruled = rules_read(c, text)
        return (ruled, "follow-up rules") if ruled else (None, "")

    # -------------------------------------------------------------- should Jarvis ask instead of acting?
    def should_ask(self, program: str, source: str, detail: dict | None) -> tuple[str, list[str]] | None:
        """('ask', [options]) when two readings are nearly tied, ('confirm', [program]) for an unsure move/copy."""
        if not self.clarify or not source.startswith("model") or "check" in source:
            return None
        detail = detail or {}
        conf, alts = detail.get("conf"), detail.get("alts") or []
        opts: list[str] = []
        for c, p in alts:
            if p not in opts and p != "unknown()" and not p.startswith("unsupported("):
                opts.append(p)
        if program in opts:
            opts.remove(program)
        opts.insert(0, program)
        if len(alts) > 1 and conf is not None and conf < TORN_MAX_CONF and alts[0][0] > 0:
            close = [p for c, p in alts[1:] if c / alts[0][0] >= TORN_RATIO]
            close = [p for p in close if p in opts and p != program]
            if close:
                return "ask", [program] + close[:2]
        if conf is not None and conf < RISKY_LOW_CONF:
            try:
                acts = [c.action for c in parse(program).calls]
            except DSLError:
                return None
            if any(a in RISKY for a in acts):
                return "confirm", [program]
        return None
