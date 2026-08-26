from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from vmem_bench.common.media import MediaInfo


class BoundaryReason(str, Enum):
    """Semantic reason why a chunk boundary was placed."""

    HARD_CUT = "hard_cut"
    GRADUAL_TRANSITION = "gradual_transition"
    MAX_DURATION_WITHIN_SHOT = "max_duration_within_shot"
    FIXED_DURATION_LIMIT = "fixed_duration_limit"
    VIDEO_START = "video_start"
    VIDEO_END = "video_end"


class BoundarySource(str, Enum):
    """Provenance of the boundary."""

    OBSERVED = "observed"
    POLICY = "policy"
    METADATA = "metadata"


@dataclass(frozen=True, slots=True)
class ChunkBoundary:
    """A precise point in time separating two video chunks."""

    timestamp_sec: float
    reason: BoundaryReason
    source: BoundarySource
    confidence: float = 1.0
    camera_relation: str | None = None


@dataclass(frozen=True, slots=True)
class SbdRequest:
    """Request payload for shot boundary detection."""

    video_path: Path
    method: str = "auto"
    scenedetect_threshold: float = 27.0
    ffmpeg_threshold: float = 0.35
    min_scene_len_sec: float = 1.0
    use_dino_refinement: bool = True
    dino_window_size: int = 5


@dataclass(frozen=True, slots=True)
class SbdResult:
    """Result of shot boundary detection with method provenance."""

    boundaries: list[ChunkBoundary]
    media_info: MediaInfo
    requested_method: str
    used_method: str
    notes: list[str] = field(default_factory=list)
