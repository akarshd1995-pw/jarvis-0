"""Text -> Program resolution chain: raw DSL, then trained model, then rules.

Model backends, in order of preference:
  1. ONNX  (checkpoints/intent_encoder.onnx + intent_decoder.onnx, needs only onnxruntime + numpy)
  2. PyTorch checkpoint (checkpoints/intent.pt, needs torch)
  3. none  -> regex rules only
"""
from __future__ import annotations

import logging
from pathlib import Path

import re

from .media_text import strip_song

from ..dsl.grammar import DSLError, Ident, Program, parse
from .rule_parser import rule_parse

log = logging.getLogger(__name__)

RULES_FIRST = {"window", "desktop", "media", "brightness", "set_volume", "radio", "screenshot", "screen_record",
               "power", "training", "briefing", "maintenance", "directions",
               "note", "find_notes", "delete_note"}
# New values of an action the model already knows: rules handle them until the model learned that feature
NEW_VALUES = {("training", "review"): "usage_review"}
PLAY_WORDS = re.compile(r"\b(play|put on|stream|start playing|chalao|chala do|bajao|lagao|sunao|play karo)\b", re.I)


_LATEST_WORDS = r"(?:latest|newest|new|recent|most recent|last|nayi|naya|naye)"
_VID = r"(?:videos?|uploads?|vids?|episodes?)"
_LATEST_PATTERNS = [
    re.compile(rf"^(?:the )?{_LATEST_WORDS} {_VID} (?:of|from|by|on) (?:the )?(.+?)(?: channel)?$", re.I),
    re.compile(rf"^(.+?) (?:ka|ki|ke) {_LATEST_WORDS} {_VID}$", re.I),
    re.compile(rf"^(.+?)(?:'s|s'|'| channel'?s?)? {_LATEST_WORDS} {_VID}$", re.I),
    re.compile(rf"^{_LATEST_WORDS} (.+?) {_VID}$", re.I),
]


def normalize_media(prog: Program) -> Program:
    """Older models put 'yes theory latest video' straight into the query. Turn that into
    play_media(query="yes theory", mode=latest), and '... playlist' / '... live' into their modes."""
    for call in prog.calls:
        if call.action == "play_media" and isinstance(call.args.get("query"), str):
            call.args["query"] = strip_song(call.args["query"])
        if call.action != "play_media" or "mode" in call.args or str(call.args.get("site", "youtube")) != "youtube":
            continue
        q = call.args["query"].strip()
        for pat in _LATEST_PATTERNS:
            m = pat.match(q)
            if m and m.group(1).strip():
                call.args["query"], call.args["mode"] = m.group(1).strip(), Ident("latest")
                break
        else:
            m = re.match(r"^(?:a |an |some )?(.+?) (?:playlist|mix)$", q, re.I)
            if m:
                call.args["query"], call.args["mode"] = m.group(1), Ident("playlist")
                continue
            m = re.match(r"^(.+?) (?:live(?: ?stream)?)$", q, re.I)
            if m:
                call.args["query"], call.args["mode"] = m.group(1), Ident("live")
    return prog


def upgrade_play(prog: Program, text: str) -> Program:
    """Models trained before play_media existed emit search_web(site=youtube) for "play X".
    If the user actually asked to play, turn it into play_media so the video starts."""
    if not PLAY_WORDS.search(text):
        return normalize_media(prog)
    for call in prog.calls:
        if call.action == "search_web" and str(call.args.get("site", "")) in ("youtube", "spotify"):
            call.action = "play_media"
    return normalize_media(prog)


# Words the model may legitimately drop from a free-text slot ("play dear god song" -> query="dear god")
_FILLER = {"song", "songs", "video", "videos", "music", "the", "a", "an", "some", "please", "pls", "jarvis",
           "on", "in", "youtube", "spotify", "yt", "for", "me", "now", "track", "gaana", "gana", "wala", "vala",
           "chalao", "bajao", "lagao", "sunao", "karo", "do", "chala", "play", "search", "up", "look"}
_TEXT_SLOTS = {"play_media": "query", "search_web": "query"}


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"(?<=\w)'s?\b", " ", s.lower()))


def slot_faithful(prog: Program, text: str) -> bool:
    """A free-text slot must be copied from what the user typed: whole words, nothing cut off.
    Catches a model that starts copying an unfamiliar name and stops early ("super machi" -> "sup")."""
    said = _words(text)
    joined = " " + " ".join(said) + " "
    for call in prog.calls:
        key = _TEXT_SLOTS.get(call.action)
        if not key or key not in call.args:
            continue
        got = _words(str(call.args[key]))
        if not got or " " + " ".join(got) + " " not in joined:
            return False          # not a whole-word span of the input (cut mid-word or invented)
    return True


def repair_slots(prog: Program, text: str) -> Program | None:
    """If the rules read the same request and kept content words the model dropped, take the rules' slot.
    Returns the repaired program, or None if nothing needed repair."""
    ruled = rule_parse(text)
    if ruled is None or len(ruled.calls) != len(prog.calls):
        return None
    changed = False
    for mc, rc in zip(prog.calls, ruled.calls):
        key = _TEXT_SLOTS.get(mc.action)
        if not key or rc.action not in _TEXT_SLOTS or key not in mc.args or key not in rc.args:
            continue
        lost = set(_words(str(rc.args[key]))) - set(_words(str(mc.args[key]))) - _FILLER
        if lost:
            mc.args[key] = rc.args[key]
            changed = True
    return prog if changed else None


_FILE_OPS = {"move_files", "copy_files", "delete_files", "list_files"}
_NEW_FOLDER = re.compile(r"\b(?:in)?to (?:a |an )?new folder(?: called| named)? ['\"]?([\w .-]+?)['\"]?\s*$", re.I)


def repair_file_kind(prog: Program, text: str) -> Program:
    """Never act on every file when the user named a type. Older models don't know categories:
    'move all video files in downloads' -> add ext=video (mp4, mkv, mov, ...). Also 'to a new folder X'."""
    from .. import filetypes
    from .rule_parser import _ext
    ops = [c for c in prog.calls if c.action in _FILE_OPS]
    if len(ops) != 1:                               # chained commands: can't tell which clause a type belongs to
        return prog
    call = ops[0]
    single = _ext(filetypes.object_words(text))
    cat = filetypes.category_in(text)
    have = str(call.args["ext"]) if "ext" in call.args else None
    if have is None and (single or cat):
        call.args["ext"] = Ident(single or cat)
    elif have and cat and not single and have in filetypes.CATEGORIES.get(cat, set()):
        call.args["ext"] = Ident(cat)                # said "videos", model picked just "mp4"
    m = _NEW_FOLDER.search(text)
    if m and call.action in ("move_files", "copy_files") and "new_folder" not in call.args:
        call.args.pop("dest", None)
        call.args["new_folder"] = m.group(1).strip()
    return prog


class Brain:
    def __init__(self, checkpoint: Path | None = None, tokenizer: Path | None = None,
                 min_confidence: float = 0.55, prefer: str = "auto", budget_s: float | None = None,
                 background: bool = False):
        """background=True: load the model in a thread so Jarvis shows up at once (phase 41); commands in the
        first moment wait up to a few seconds for it, then use the rules."""
        import threading
        self._args = (checkpoint, tokenizer, min_confidence, prefer)
        self.budget_s = budget_s
        self.last_detail: dict = {"conf": None, "alts": []}
        self._stamp = self._files_stamp()
        self._ready = threading.Event()
        self.model, self.backend = None, "loading"
        if background:
            def load():
                try:
                    self._load(*self._args)
                except Exception as e:
                    log.warning("intent model failed to load: %s", e)
                    self.model, self.backend = None, "rules"
                finally:
                    self._ready.set()
            threading.Thread(target=load, daemon=True, name="load-intent-model").start()
        else:
            self._load(*self._args)
            self._ready.set()

    def _files_stamp(self) -> tuple:
        checkpoint, tokenizer = self._args[0], self._args[1]
        if not tokenizer:
            return ()
        d = (checkpoint or tokenizer).parent
        out = []
        for name in ("intent_onnx.json", "intent.pt", "intent_encoder.int8.onnx"):
            try:
                out.append((d / name).stat().st_mtime)
            except OSError:
                out.append(None)
        return tuple(out)

    def maybe_reload(self) -> bool:
        """The self-improving loop (or a rollback) swapped the model files: load the new model, no restart."""
        stamp = self._files_stamp()
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        old_model, old_backend = self.model, self.backend
        try:
            self._load(*self._args)
            log.info("switched to the updated intent model (%s)", self.backend)
            return True
        except Exception as e:
            self.model, self.backend = old_model, old_backend
            log.warning("couldn't load the updated model (%s); keeping the previous one", e)
            return False

    def _load(self, checkpoint, tokenizer, min_confidence, prefer):
        self.model = None
        self.backend = "rules"
        if not (tokenizer and tokenizer.exists()):
            return
        ckpt_dir = (checkpoint or tokenizer).parent
        enc, dec = ckpt_dir / "intent_encoder.onnx", ckpt_dir / "intent_decoder.onnx"
        if prefer in ("auto", "onnx") and enc.exists() and dec.exists():
            try:
                from .onnx_parser import ONNXParser
                self.model = ONNXParser(enc, dec, tokenizer, min_confidence=min_confidence)
                self.backend = "onnx"
                log.info("intent model loaded (ONNX) from %s", ckpt_dir)
                return
            except Exception as e:
                log.warning("ONNX model unavailable (%s); trying PyTorch", e)
        if prefer in ("auto", "torch") and checkpoint and checkpoint.exists():
            try:
                from .model_parser import ModelParser
                self.model = ModelParser(checkpoint, tokenizer, min_confidence=min_confidence)
                self.backend = "torch"
                log.info("intent model loaded (PyTorch) from %s", checkpoint)
            except Exception as e:  # torch missing, bad checkpoint, ...
                log.warning("intent model unavailable (%s); using rules only", e)

    # ------------------------------------------------------------ what the loaded model has learned
    def knows(self, what: str) -> bool:
        """True if the model was trained on this action ('briefing') or feature ('file_categories').
        Models trained before this was recorded know none of the newer things, so rules cover those."""
        k = getattr(self.model, "knows", None) if self.model is not None else None
        return bool(k) and (what in k.get("actions", []) or what in k.get("features", []))

    def _post(self, prog: Program, text: str) -> Program:
        """Hand-written repairs, only for what the model hasn't learned."""
        if not self.knows("file_categories"):
            prog = repair_file_kind(prog, text)
        if not self.knows("play_modes"):
            prog = upgrade_play(prog, text)
        if not self.knows("memory_filters"):
            from ..memory.query import repair as memory_repair
            prog = memory_repair(prog, text)
        return prog

    def resolve(self, text: str) -> tuple[Program | None, str]:
        """(program, how it was understood). self.last_detail keeps the model's confidence and runner-up
        readings for real-usage learning (phase 39)."""
        self.last_detail = {"conf": None, "alts": []}
        if not self._ready.wait(timeout=4.0):           # still loading right after start-up: rules for now
            ruled = rule_parse(text.strip())
            return ruled, "rules (model still loading)" if ruled is not None else "unparsed"
        self.maybe_reload()
        text = text.strip()
        if not text:
            return None, "empty"
        try:
            return parse(text), "dsl"
        except DSLError:
            pass
        # System-control actions (phase 5) are matched by precise rules first, so they work even with a
        # model trained before they existed. After retraining, the model and the rules agree.
        ruled_first = rule_parse(text)
        from ..context import uses_refs
        if ruled_first is not None and uses_refs(ruled_first) and not self.knows("context_refs"):
            return ruled_first, "rules (model not trained on this yet)"     # phase 38: this / these / here
        if ruled_first is not None and any(c.action in RULES_FIRST and not self.knows(c.action)
                                           or (c.action, str(c.args.get("action", ""))) in NEW_VALUES
                                           and not self.knows(NEW_VALUES[(c.action, str(c.args.get("action", "")))])
                                           for c in ruled_first.calls):
            return ruled_first, "rules (model not trained on this yet)"
        model_unknown = None
        if self.model is not None:
            try:
                prog, conf = self.model.parse(text, budget_s=self.budget_s) if self.budget_s else self.model.parse(text)
            except TypeError:                            # a parser without a time budget (PyTorch / tests)
                prog, conf = self.model.parse(text)
            if getattr(self.model, "timed_out", False):  # phase 41: too slow this time -> rules answer
                from ..perf import PERF
                PERF.timeouts += 1
                ruled = rule_parse(text)
                if ruled is not None:
                    return ruled, "rules (model over its time budget)"
            self.last_detail = {"conf": conf, "alts": list(getattr(self.model, "last_alts", []) or [])[:4]}
            if prog is not None:
                gave_up = any(c.action in ("unknown", "unsupported") for c in prog.calls)
                if not gave_up:
                    if not slot_faithful(prog, text):
                        ruled = rule_parse(text)
                        if ruled is not None and slot_faithful(ruled, text):
                            return self._post(ruled, text), "rules (model cut the name short)"
                    fixed = repair_slots(prog, text)
                    if fixed is not None:
                        return self._post(fixed, text), f"model {conf:.0%} + name check"
                    return self._post(prog, text), f"model {conf:.0%}"
                # The model says it can't: a newer rule may know better (e.g. actions added after it was trained)
                ruled = rule_parse(text)
                if ruled is not None and not any(c.action in ("unknown", "unsupported") for c in ruled.calls):
                    return ruled, "rules"
                if str(prog) != "unknown()":
                    return self._post(prog, text), f"model {conf:.0%}"
                model_unknown = (prog, f"model {conf:.0%}")
        prog = rule_parse(text)
        if prog is not None:
            return prog, "rules (model unsure)" if self.model is not None else "rules (no model)"
        if model_unknown:
            return model_unknown       # model is confident this is out of scope -> polite decline
        return None, "unparsed"
