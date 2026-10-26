"""Contract between the closed-loop runner and a realization backend.

The runner decides which references a segment gets (that is the memory under test); the backend only
renders. A backend must not read gold, other segments, or the memory store, and must honour ``seed``
for every random source it controls, so that a condition rerun with the same seed follows the same
path wherever the underlying model allows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RealizationRequest:
    segment_id: int
    prompt: str
    references: tuple[Path, ...]
    seed: int
    out_dir: Path


@dataclass
class RealizationResult:
    clip: Path
    keyframe: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RealizationBackend(Protocol):
    name: str

    def start(self) -> None:
        """Load models or attach to resident services; idempotent."""

    def realize(self, request: RealizationRequest) -> RealizationResult:
        """Render one segment; raise on failure instead of returning a partial result."""

    def stop(self) -> None:
        """Release services this backend started; leave services it only attached to."""
