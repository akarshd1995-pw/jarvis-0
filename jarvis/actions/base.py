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
    summary: str
    items: list[str] = field(default_factory=list)
