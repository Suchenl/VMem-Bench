"""Legacy track-first annotation stack (compat / copy source for ``annotation.pipeline``).

Public orchestrator helpers are re-exported from ``pipeline_track_first.py`` so existing
``from vmem_bench.annotation.pipeline_track_first import …`` call sites keep working.
"""

from __future__ import annotations

from vmem_bench.annotation.pipeline_track_first.pipeline_track_first import (
    _present_payloads,
    _write_identity_resolution_artifact,
    _write_seed_assignment_artifact,
    annotate_movie_track_first,
    cluster_scene_locations,
    entity_time_metadata,
    frame_to_chunk_fn,
    is_closeup_shot,
    merge_spans,
    presence_for_chunks,
    prune_scratch,
    reslug_entities,
    shots_from_boundaries_csv,
    should_reuse_location_without_tracklets,
)

__all__ = [
    "annotate_movie_track_first",
    "cluster_scene_locations",
    "entity_time_metadata",
    "frame_to_chunk_fn",
    "is_closeup_shot",
    "merge_spans",
    "presence_for_chunks",
    "prune_scratch",
    "reslug_entities",
    "shots_from_boundaries_csv",
    "should_reuse_location_without_tracklets",
    "_present_payloads",
    "_write_identity_resolution_artifact",
    "_write_seed_assignment_artifact",
]
