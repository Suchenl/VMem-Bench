"""Non-semantic deterministic normalization for parseable v5 JSON."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied annotation with deterministic order and synced counts.

    Ordering is canonicalized so downstream stages get a reproducible artifact.
    ``counts`` is recomputed from collection lengths (derived field); other
    natural-language values, references, and segment boundaries are left alone.
    """
    normalized = deepcopy(annotation)
    for collection, id_key in (
        ("characters", "char_id"),
        ("props", "prop_id"),
        ("locations", "loc_id"),
    ):
        values = normalized.get(collection)
        if isinstance(values, list):
            values.sort(key=lambda value: _text_key(value, id_key))

    scenes = normalized.get("screenplay", {}).get("scenes")
    n_segments = 0
    if isinstance(scenes, list):
        scenes.sort(key=lambda value: _time_key(value, "start_seconds"))
        for scene in scenes:
            if isinstance(scene, dict) and isinstance(scene.get("visual_segments"), list):
                scene["visual_segments"].sort(key=lambda value: _time_key(value, "start_seconds"))
                n_segments += len(scene["visual_segments"])

    normalized["counts"] = {
        "characters": len(normalized["characters"]) if isinstance(normalized.get("characters"), list) else 0,
        "props": len(normalized["props"]) if isinstance(normalized.get("props"), list) else 0,
        "locations": len(normalized["locations"]) if isinstance(normalized.get("locations"), list) else 0,
        "scenes": len(scenes) if isinstance(scenes, list) else 0,
        "visual_segments": n_segments,
    }
    return normalized


def _text_key(value: Any, key: str) -> tuple[int, str]:
    return (0, value[key]) if isinstance(value, dict) and isinstance(value.get(key), str) else (1, "")


def _time_key(value: Any, key: str) -> tuple[int, float]:
    time = value.get(key) if isinstance(value, dict) else None
    return (0, float(time)) if isinstance(time, (int, float)) and not isinstance(time, bool) else (1, 0.0)
