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

from ..dsl.grammar import DSLError, Program, parse
from .rule_parser import rule_parse

log = logging.getLogger(__name__)

RULES_FIRST = {"window", "desktop", "media", "brightness", "set_volume", "radio", "screenshot", "screen_record",
               "power"}
PLAY_WORDS = re.compile(r"\b(play|put on|stream|start playing|chalao|chala do|bajao|lagao|sunao|play karo)\b", re.I)


def upgrade_play(prog: Program, text: str) -> Program:
    """Models trained before play_media existed emit search_web(site=youtube) for "play X".
    If the user actually asked to play, turn it into play_media so the video starts."""
    if not PLAY_WORDS.search(text):
        return prog
    for call in prog.calls:
        if call.action == "search_web" and str(call.args.get("site", "")) in ("youtube", "spotify"):
            call.action = "play_media"
    return prog


class Brain:
    def __init__(self, checkpoint: Path | None = None, tokenizer: Path | None = None,
                 min_confidence: float = 0.55, prefer: str = "auto"):
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

    def resolve(self, text: str) -> tuple[Program | None, str]:
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
        if ruled_first is not None and any(c.action in RULES_FIRST for c in ruled_first.calls):
            return ruled_first, "rules"
        model_unknown = None
        if self.model is not None:
            prog, conf = self.model.parse(text)
            if prog is not None:
                gave_up = any(c.action in ("unknown", "unsupported") for c in prog.calls)
                if not gave_up:
                    return upgrade_play(prog, text), f"model {conf:.0%}"
                # The model says it can't: a newer rule may know better (e.g. actions added after it was trained)
                ruled = rule_parse(text)
                if ruled is not None and not any(c.action in ("unknown", "unsupported") for c in ruled.calls):
                    return ruled, "rules"
                if str(prog) != "unknown()":
                    return upgrade_play(prog, text), f"model {conf:.0%}"
                model_unknown = (prog, f"model {conf:.0%}")
        prog = rule_parse(text)
        if prog is not None:
            return prog, "rules"
        if model_unknown:
            return model_unknown       # model is confident this is out of scope -> polite decline
        return None, "unparsed"
