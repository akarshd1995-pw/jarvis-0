"""Text -> Program resolution chain: raw DSL, then trained model, then rules.

Model backends, in order of preference:
  1. ONNX  (checkpoints/intent_encoder.onnx + intent_decoder.onnx, needs only onnxruntime + numpy)
  2. PyTorch checkpoint (checkpoints/intent.pt, needs torch)
  3. none  -> regex rules only
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..dsl.grammar import DSLError, Program, parse
from .rule_parser import rule_parse

log = logging.getLogger(__name__)


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
        model_unknown = None
        if self.model is not None:
            prog, conf = self.model.parse(text)
            if prog is not None:
                if str(prog) != "unknown()":
                    return prog, f"model {conf:.0%}"
                model_unknown = (prog, f"model {conf:.0%}")
        prog = rule_parse(text)
        if prog is not None:
            return prog, "rules"
        if model_unknown:
            return model_unknown       # model is confident this is out of scope -> polite decline
        return None, "unparsed"
