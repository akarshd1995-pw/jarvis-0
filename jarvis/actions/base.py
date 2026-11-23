from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config import Config
from ..store import Store


@dataclass
class Ctx:
    config: Config
    store: Store
    batch_id: str
    notify: Callable[[str, str], None] = lambda title, msg: print(f"[{title}] {msg}")


@dataclass
class ActionResult:
    """summary: plain technical text (logs, CLI). kind + data: structured facts the persona layer
    turns into an in-character reply (counts, folders, file types, what happened)."""
    summary: str
    items: list[str] = field(default_factory=list)
    kind: str = ""
    data: dict = field(default_factory=dict)
