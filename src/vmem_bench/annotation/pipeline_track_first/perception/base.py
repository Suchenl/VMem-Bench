"""Perception backend contract (annotation_tracking_internals.md §3.1b).

Self-contained: imports only stdlib + this benchmark's own modules. Backends may load external
model weights by path (Principle 7 allows external *weights*, forbids external *code*).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from vmem_bench.annotation.pipeline_track_first.tracking import Tracklet


@dataclass(slots=True)
class Frame:
    """One sampled shot frame on disk (both backends reuse the Path-based detectors/embedders)."""

    frame_index: int  # absolute frame index in the source video (for frame_span math)
    path: Path  # decoded frame image on disk (chunks/frames/...); crops are cut from this


@dataclass(slots=True)
class RosterEntry:
    """One cast-roster entry: what the detector/segmenter should look for (§3.2)."""

    name: str
    kind: str  # character | prop | location
    grounding_phrase: str  # open-vocab text prompt (GroundingDINO phrase / SAM3 concept)
    static_attributes: dict[str, str] = field(default_factory=dict)
    # Route B (exemplar-grounded): path to this entity's visual exemplar crop. Identity is then
    # decided by DINOv3 similarity to this crop, never by language matching. Empty -> no exemplar
    # (route A, or an entity the exemplar collection step could not anchor).
    exemplar_crop: str = ""
    # Human-seeded production mode: all detector phrases/aliases for one canonical identity point
    # to this stable id.  Empty preserves proposal/legacy behavior.
    canonical_entity_id: str = ""
    identity_scope: str = "individual"  # individual | category | scene
    aliases: tuple[str, ...] = ()
    exemplar_crops: tuple[str, ...] = ()
    allowed_state_events: tuple[str, ...] = ()


@runtime_checkable
class PerceptionBackend(Protocol):
    """frames + roster -> tracklets (persistent local track ids within one shot).

    ``next_track_id`` lets a movie-level caller keep track ids globally unique across shots.
    Implementations must be deterministic given the same frames/roster/weights.
    """

    name: str  # provenance fingerprint, e.g. "gdino_track" / "sam3_track"

    def track_shot(
        self,
        frames: list[Frame],
        roster: list[RosterEntry],
        *,
        next_track_id: int = 0,
    ) -> list[Tracklet]:
        ...


def get_backend(name: str, **kwargs) -> PerceptionBackend:
    """Factory: resolve a backend by ``AnnotationConfig.perception_backend``.

    Lazy per-backend import so importing this module never requires GroundingDINO or SAM3 to be
    installed (only the selected backend's heavy deps load).
    """
    if name == "gdino_track":
        from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import GdinoTrackBackend
        return GdinoTrackBackend(**kwargs)
    if name == "sam3_track":
        from vmem_bench.annotation.pipeline_track_first.perception.sam3_track import Sam3TrackBackend
        return Sam3TrackBackend(**kwargs)
    if name == "fusion_track":
        from vmem_bench.annotation.pipeline_track_first.perception.fusion_track import FusionTrackBackend
        return FusionTrackBackend(**kwargs)
    raise ValueError(f"unknown perception_backend: {name!r} "
                     "(expected gdino_track | sam3_track | fusion_track)")
