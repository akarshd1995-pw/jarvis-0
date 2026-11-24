"""Models that load the first time they're needed (phase 41), so start-up is quick and idle memory stays low."""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)


class Lazy:
    """Stands in for a model: the first attribute access loads it (once, thread-safe). If loading fails or the
    loader returns None, the proxy is falsy and attribute access raises AttributeError."""

    def __init__(self, loader: Callable[[], object | None], name: str):
        self._loader, self._name = loader, name
        self._obj, self._done = None, False
        self._lock = threading.Lock()

    def _get(self):
        if not self._done:
            with self._lock:
                if not self._done:
                    try:
                        self._obj = self._loader()
                    except Exception as e:
                        log.warning("%s unavailable: %s", self._name, e)
                        self._obj = None
                    self._done = True
        return self._obj

    def warm(self) -> None:
        """Load now (in the background after start-up, so the first use isn't slower)."""
        threading.Thread(target=self._get, daemon=True, name=f"warm-{self._name}").start()

    @property
    def loaded(self) -> bool:
        return self._done and self._obj is not None

    def __bool__(self) -> bool:
        return self._get() is not None

    def __getattr__(self, item):
        obj = self._get()
        if obj is None:
            raise AttributeError(item)
        return getattr(obj, item)
