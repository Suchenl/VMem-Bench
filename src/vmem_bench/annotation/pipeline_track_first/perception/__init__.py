"""Pluggable perception backends (annotation_tracking_internals.md §3.1b).

A backend turns one shot's sampled frames + the cast roster into ``Tracklet``s (persistent local
track ids). Everything downstream (re-ID, per-crop QA, presence, naming, prompt) is backend-
agnostic, so the two research routes can be A/B compared as a paper ablation:

  - ``gdino_track`` (route A, default): GroundingDINO open-vocab detection + self-contained
    class-aware tracking (tracking.py). Modality-agnostic, mature, box-only.
  - ``sam3_track``  (route B, ablation): SAM3 / SAM3.1 concept segmentation + native video
    propagation. One step yields detection+tracking+masks; cleaner crops, robust to occlusion.

Both emit the SAME ``Tracklet`` contract (route B may additionally attach a mask). Select via
``AnnotationConfig.perception_backend``.
"""

from vmem_bench.annotation.pipeline_track_first.perception.base import (
    Frame,
    PerceptionBackend,
    RosterEntry,
    get_backend,
)

__all__ = ["Frame", "PerceptionBackend", "RosterEntry", "get_backend"]
