"""Shared, pipeline-local artifact contracts.

These are operational contracts for S1--S7.  Public benchmark wire contracts
remain in :mod:`vmem_bench.common.schemas`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PIPELINE_STAGES = (
    "s1_vlm_annotation",
    "s2_annotation_postprocess",
    "s3_segment_auto_review_revise",
    "s4_segment_sampling_human_review",
    "s5_entities_visual_crop_acquisition",
    "s6_entities_visual_crop_human_review",
    "s7_freeze_publish",
)
S4_MODES = ("auto", "blocking", "nonblocking")


def resolve_s4_mode(requested: str, *, skip_human: bool) -> str:
    mode = str(requested or "auto")
    if mode not in S4_MODES:
        raise ValueError(f"invalid s4_mode {mode!r}; expected one of {S4_MODES}")
    if mode == "auto":
        return "nonblocking" if skip_human else "blocking"
    if mode == "blocking" and skip_human:
        raise ValueError("s4_mode=blocking is incompatible with skip_human")
    return mode


@dataclass(slots=True)
class MovieManifest:
    """One source-video item known to the annotation batch."""

    dataset: str
    movie_id: str
    movie_dir: str
    source_video: str
    vlm_output: str
    source_duration_seconds: float | None = None
    source_fps: float | None = None
    source_sha256: str = ""
    status: str = "source_ready"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MovieManifest":
        return cls(
            dataset=str(value["dataset"]),
            movie_id=str(value["movie_id"]),
            movie_dir=str(value["movie_dir"]),
            source_video=str(value["source_video"]),
            vlm_output=str(value["vlm_output"]),
            source_duration_seconds=(
                float(value["source_duration_seconds"])
                if value.get("source_duration_seconds") is not None
                else None
            ),
            source_fps=float(value["source_fps"]) if value.get("source_fps") is not None else None,
            source_sha256=str(value.get("source_sha256") or ""),
            status=str(value.get("status") or "source_ready"),
            notes=[str(note) for note in value.get("notes", [])],
        )

    @property
    def root(self) -> Path:
        return Path(self.movie_dir)


@dataclass(slots=True)
class StageState:
    """Restart-safe per-movie pipeline status."""

    movie_id: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"movie_id": self.movie_id, "stages": self.stages}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageState":
        return cls(movie_id=str(value["movie_id"]), stages=dict(value.get("stages") or {}))
