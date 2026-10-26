"""Deterministic structural lint for v5 annotation JSON."""

from __future__ import annotations

from typing import Any


def finding(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def lint_structure(annotation: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    actual_counts = {
        "characters": 0,
        "props": 0,
        "locations": 0,
        "scenes": 0,
        "visual_segments": 0,
    }
    if not isinstance(annotation, dict):
        errors.append(finding("INVALID_ROOT", "$", "annotation root must be an object"))
        return {"errors": errors, "warnings": warnings, "actual_counts": actual_counts}

    for key in ("characters", "props", "locations"):
        value = annotation.get(key)
        if not isinstance(value, list):
            errors.append(finding("INVALID_COLLECTION", f"$.{key}", "must be a list"))
        else:
            actual_counts[key] = len(value)

    screenplay = annotation.get("screenplay")
    scenes: list[Any] = []
    if not isinstance(screenplay, dict):
        errors.append(finding("INVALID_SCREENPLAY", "$.screenplay", "must be an object"))
    else:
        scenes = screenplay.get("scenes") or []
        if not isinstance(scenes, list):
            errors.append(finding("INVALID_SCENES", "$.screenplay.scenes", "must be a list"))
            scenes = []
        else:
            actual_counts["scenes"] = len(scenes)
            for scene in scenes:
                if not isinstance(scene, dict):
                    continue
                segs = scene.get("visual_segments") or []
                if isinstance(segs, list):
                    actual_counts["visual_segments"] += len(segs)

    # ``counts`` is a derived summary field. Mismatches are auto-corrected in
    # ``normalize_annotation``; surface them as warnings so S2 does not block.
    counts = annotation.get("counts")
    if isinstance(counts, dict):
        for key, actual in actual_counts.items():
            declared = counts.get(key)
            if declared is not None and declared != actual:
                warnings.append(
                    finding(
                        "COUNT_MISMATCH",
                        f"$.counts.{key}",
                        f"declared {declared} != actual {actual}; will sync in normalize",
                    )
                )
    else:
        warnings.append(
            finding("MISSING_COUNTS", "$.counts", "counts object missing; will sync in normalize")
        )

    return {"errors": errors, "warnings": warnings, "actual_counts": actual_counts}
