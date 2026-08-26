"""Append-only JSONL event log for live progress monitoring.

The annotation pipeline emits one JSON line per event to ``<out>/tmp/events.jsonl``
(legacy: ``build/events.jsonl``);
``vmem_bench.web.server`` tails this file over the shared filesystem and streams it
to the browser via SSE. Writes are line-buffered + fsync'd so a reader on another host
sees events promptly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, kind: str, **data) -> None:
        record = {"ts": round(time.time(), 3), "kind": kind, **data}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


class NullEventLog:
    """No-op stand-in (tests / callers that do not care)."""

    def emit(self, kind: str, **data) -> None:  # noqa: ARG002
        return
