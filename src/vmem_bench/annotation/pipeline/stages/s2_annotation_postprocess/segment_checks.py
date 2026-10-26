"""Segment-level deterministic checks for v5 annotation JSON."""

from __future__ import annotations

from typing import Any


def finding(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def check_segments(annotation: dict[str, Any], entity_ids: set[str]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    segment_index: list[dict[str, Any]] = []
    known_segment_ids: set[str] = set()

    scenes = ((annotation.get("screenplay") or {}).get("scenes") or []) if isinstance(annotation, dict) else []
    if not isinstance(scenes, list):
        errors.append(finding("INVALID_SCENES", "$.screenplay.scenes", "must be a list"))
        return {"errors": errors, "warnings": warnings, "segment_index": segment_index}

    for scene_i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(finding("INVALID_SCENE", f"$.screenplay.scenes[{scene_i}]", "must be an object"))
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{scene_i}")
        segments = scene.get("visual_segments") or []
        if not isinstance(segments, list):
            errors.append(
                finding("INVALID_SEGMENTS", f"$.screenplay.scenes[{scene_i}].visual_segments", "must be a list")
            )
            continue
        for seg_i, segment in enumerate(segments):
            path = f"$.screenplay.scenes[{scene_i}].visual_segments[{seg_i}]"
            if not isinstance(segment, dict):
                errors.append(finding("INVALID_SEGMENT", path, "must be an object"))
                continue
            segment_id = str(segment.get("segment_id") or "")
            start = segment.get("start_seconds")
            end = segment.get("end_seconds")
            duration = segment.get("duration_seconds")
            if segment_id:
                if segment_id in known_segment_ids:
                    errors.append(finding("DUPLICATE_SEGMENT_ID", f"{path}.segment_id", segment_id))
                known_segment_ids.add(segment_id)
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(finding("INVALID_SEGMENT_TIME", path, "start/end must be numbers"))
            else:
                if start > end:
                    errors.append(finding("INVALID_SEGMENT_TIME", path, "start_seconds > end_seconds"))
                if isinstance(duration, (int, float)) and abs(float(duration) - (float(end) - float(start))) > 1e-6:
                    errors.append(
                        finding(
                            "SEGMENT_DURATION_MISMATCH",
                            f"{path}.duration_seconds",
                            "duration_seconds must equal end-start",
                        )
                    )
                segment_index.append(
                    {
                        "segment_id": segment_id,
                        "scene_id": scene_id,
                        "start_seconds": float(start),
                        "end_seconds": float(end),
                    }
                )

            present = segment.get("present_entity_ids") or []
            if not isinstance(present, list):
                errors.append(finding("INVALID_PRESENT", f"{path}.present_entity_ids", "must be a list"))
            else:
                seen_present: set[str] = set()
                for eid in present:
                    if not isinstance(eid, str) or not eid:
                        errors.append(finding("INVALID_PRESENT", f"{path}.present_entity_ids", "empty id"))
                        continue
                    if eid in seen_present:
                        errors.append(
                            finding("DUPLICATE_PRESENT_ENTITY_ID", f"{path}.present_entity_ids", eid)
                        )
                    seen_present.add(eid)
                    if eid not in entity_ids:
                        errors.append(
                            finding("UNKNOWN_PRESENT_ENTITY_ID", f"{path}.present_entity_ids", eid)
                        )

    self_check = annotation.get("self_check") if isinstance(annotation, dict) else None
    if isinstance(self_check, dict):
        low = self_check.get("low_confidence_segment_ids") or []
        if isinstance(low, list):
            for sid in low:
                if sid not in known_segment_ids:
                    errors.append(
                        finding("UNKNOWN_SELF_CHECK_SEGMENT", "$.self_check.low_confidence_segment_ids", str(sid))
                    )

    return {"errors": errors, "warnings": warnings, "segment_index": segment_index}
