"""Text -> Program resolution chain: raw DSL, then trained model, then rules."""
from __future__ import annotations

import logging
from pathlib import Path

from ..dsl.grammar import DSLError, Program, parse
from .rule_parser import rule_parse

log = logging.getLogger(__name__)


class Brain:
    def __init__(self, checkpoint: Path | None = None, tokenizer: Path | None = None):
        self.model = None
        if checkpoint and tokenizer and checkpoint.exists() and tokenizer.exists():
            try:
                from .model_parser import ModelParser
                self.model = ModelParser(checkpoint, tokenizer)
                log.info("intent model loaded from %s", checkpoint)
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
