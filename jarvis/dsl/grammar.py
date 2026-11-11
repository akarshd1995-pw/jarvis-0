"""Jarvis action DSL.

A program is a sequence of calls separated by ';':

    move_files(src=Downloads, ext=pdf, older_than=30d, dest=Archive) ; remind(text="check archive", at=18:00)

This is the target language the intent model learns to emit. Keeping it tiny,
typed and canonical (args sorted) makes it easy to learn and safe to execute.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- value types

@dataclass(frozen=True)
class Duration:
    amount: int
    unit: str  # m h d w

    def seconds(self) -> int:
        return self.amount * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[self.unit]

    def __str__(self) -> str:
        return f"{self.amount}{self.unit}"


@dataclass(frozen=True)
class ClockTime:
    hour: int
    minute: int

    def __str__(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"


@dataclass(frozen=True)
class Ident:
    name: str

    def __str__(self) -> str:
        return self.name


Value = str | int | Duration | ClockTime | Ident

# ---------------------------------------------------------------- schema
# arg -> type; types: str, int, dur, time, ident, ident|str
SCHEMA: dict[str, dict] = {
    "open_app":      {"args": {"name": "ident|str"}, "required": ["name"]},
    "open_url":      {"args": {"url": "str"}, "required": ["url"]},
    "open_folder":   {"args": {"path": "ident|str"}, "required": ["path"]},
    "list_files":    {"args": {"src": "ident|str", "ext": "ident", "older_than": "dur",
                               "newer_than": "dur", "name_contains": "str"}, "required": ["src"]},
    "move_files":    {"args": {"src": "ident|str", "dest": "ident|str", "ext": "ident",
                               "older_than": "dur", "newer_than": "dur", "name_contains": "str"},
                      "required": ["src", "dest"]},
    "copy_files":    {"args": {"src": "ident|str", "dest": "ident|str", "ext": "ident",
                               "older_than": "dur", "newer_than": "dur", "name_contains": "str"},
                      "required": ["src", "dest"]},
    "delete_files":  {"args": {"src": "ident|str", "ext": "ident", "older_than": "dur",
                               "newer_than": "dur", "name_contains": "str"}, "required": ["src"]},
    "remind":        {"args": {"text": "str", "at": "time", "after": "dur"}, "required": ["text"],
                      "one_of": [["at", "after"]]},
    "system":        {"args": {"action": "ident"}, "required": ["action"],
                      "enum": {"action": ["lock", "mute", "volume_up", "volume_down", "sleep_display"]}},
    "search_memory": {"args": {"query": "str"}, "required": ["query"]},
    "undo":          {"args": {}, "required": []},
}


class DSLError(ValueError):
    pass


@dataclass
class Call:
    action: str
    args: dict[str, Value] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{k}={format_value(v)}" for k, v in sorted(self.args.items())]
        return f"{self.action}({', '.join(parts)})"


@dataclass
class Program:
    calls: list[Call]

    def __str__(self) -> str:
        return " ; ".join(str(c) for c in self.calls)


def format_value(v: Value) -> str:
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(v)

# ---------------------------------------------------------------- lexer

_TOKEN_RE = re.compile(r"""
    (?P<WS>\s+)
  | (?P<STRING>"(?:[^"\\]|\\.)*")
  | (?P<TIME>\d{1,2}:\d{2})
  | (?P<DUR>\d+[mhdw])(?![A-Za-z0-9_])
  | (?P<NUM>\d+)(?![A-Za-z_])
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_.\-]*)
  | (?P<PUNCT>[(),=;])
""", re.VERBOSE)


def _lex(src: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise DSLError(f"unexpected character {src[pos]!r} at {pos}")
        kind = m.lastgroup
        if kind != "WS":
            out.append((kind, m.group()))
        pos = m.end()
    return out

# ---------------------------------------------------------------- parser

class _Parser:
    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("EOF", "")

    def take(self, kind=None, text=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (text and tok[1] != text):
            raise DSLError(f"expected {text or kind}, got {tok[1] or 'end of input'!r}")
        self.i += 1
        return tok

    def program(self) -> Program:
        calls = [self.call()]
        while self.peek() == ("PUNCT", ";"):
            self.take()
            calls.append(self.call())
        if self.peek()[0] != "EOF":
            raise DSLError(f"trailing input at {self.peek()[1]!r}")
        return Program(calls)

    def call(self) -> Call:
        name = self.take("IDENT")[1]
        self.take("PUNCT", "(")
        args: dict[str, Value] = {}
        if self.peek() != ("PUNCT", ")"):
            while True:
                key = self.take("IDENT")[1]
                self.take("PUNCT", "=")
                if key in args:
                    raise DSLError(f"duplicate argument {key!r}")
                args[key] = self.value()
                if self.peek() == ("PUNCT", ","):
                    self.take()
                    continue
                break
        self.take("PUNCT", ")")
        return Call(name, args)

    def value(self) -> Value:
        kind, text = self.take()
        if kind == "STRING":
            return re.sub(r"\\(.)", r"\1", text[1:-1])
        if kind == "TIME":
            h, m = map(int, text.split(":"))
            if not (0 <= h < 24 and 0 <= m < 60):
                raise DSLError(f"invalid time {text}")
            return ClockTime(h, m)
        if kind == "DUR":
            return Duration(int(text[:-1]), text[-1])
        if kind == "NUM":
            return int(text)
        if kind == "IDENT":
            return Ident(text)
        raise DSLError(f"unexpected token {text!r}")


_TYPE_CHECK = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int),
    "dur": lambda v: isinstance(v, Duration),
    "time": lambda v: isinstance(v, ClockTime),
    "ident": lambda v: isinstance(v, Ident),
    "ident|str": lambda v: isinstance(v, (Ident, str)),
}


def validate(prog: Program) -> Program:
    for call in prog.calls:
        spec = SCHEMA.get(call.action)
        if spec is None:
            raise DSLError(f"unknown action {call.action!r}")
        for k, v in call.args.items():
            t = spec["args"].get(k)
            if t is None:
                raise DSLError(f"{call.action}: unknown argument {k!r}")
            if not _TYPE_CHECK[t](v):
                raise DSLError(f"{call.action}: argument {k!r} must be {t}")
        for k in spec["required"]:
            if k not in call.args:
                raise DSLError(f"{call.action}: missing argument {k!r}")
        for group in spec.get("one_of", []):
            if sum(k in call.args for k in group) != 1:
                raise DSLError(f"{call.action}: exactly one of {group} required")
        for k, allowed in spec.get("enum", {}).items():
            if k in call.args and str(call.args[k]) not in allowed:
                raise DSLError(f"{call.action}: {k} must be one of {allowed}")
    return prog


def parse(src: str) -> Program:
    """Parse and validate a DSL program. Raises DSLError on any problem."""
    return validate(_Parser(_lex(src)).program())


def canonical(src: str) -> str:
    """Round-trip to the canonical form (sorted args, normalized spacing)."""
    return str(parse(src))
