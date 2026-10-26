"""Entity-level deterministic checks for v5 annotation JSON."""

from __future__ import annotations

from typing import Any


def finding(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def check_entities(annotation: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    entity_ids: list[str] = []
    seen: set[str] = set()
    duration = annotation.get("video_duration_seconds")
    has_duration = isinstance(duration, (int, float)) and not isinstance(duration, bool)

    for collection, id_key in (
        ("characters", "char_id"),
        ("props", "prop_id"),
        ("locations", "loc_id"),
    ):
        values = annotation.get(collection) or []
        if not isinstance(values, list):
            continue
        for index, entity in enumerate(values):
            path = f"$.{collection}[{index}]"
            if not isinstance(entity, dict):
                errors.append(finding("INVALID_ENTITY", path, "must be an object"))
                continue
            entity_id = entity.get(id_key)
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(finding("MISSING_ENTITY_ID", f"{path}.{id_key}", "missing id"))
                continue
            if entity_id in seen:
                errors.append(finding("DUPLICATE_ENTITY_ID", f"{path}.{id_key}", f"duplicate {entity_id}"))
            seen.add(entity_id)
            entity_ids.append(entity_id)
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(finding("EMPTY_ENTITY_NAME", f"{path}.name", "name required"))
            _check_presence_range(entity, path, duration, has_duration, warnings)
            _check_state_changes(entity, path, duration, has_duration, seen, warnings)

    return {"errors": errors, "warnings": warnings, "entity_ids": entity_ids}


def _check_presence_range(
    entity: dict[str, Any],
    path: str,
    duration: Any,
    has_duration: bool,
    warnings: list[dict[str, str]],
) -> None:
    first = entity.get("first_presence_seconds")
    last = entity.get("last_presence_seconds")
    if first is not None and not isinstance(first, (int, float)):
        warnings.append(finding("INVALID_PRESENCE_TIME", f"{path}.first_presence_seconds", "must be a number"))
        return
    if last is not None and not isinstance(last, (int, float)):
        warnings.append(finding("INVALID_PRESENCE_TIME", f"{path}.last_presence_seconds", "must be a number"))
        return
    if isinstance(first, (int, float)) and isinstance(last, (int, float)) and first > last:
        warnings.append(
            finding("PRESENCE_TIME_OUT_OF_RANGE", path, "first_presence_seconds > last_presence_seconds")
        )
    if has_duration:
        if isinstance(first, (int, float)) and (first < 0 or first > float(duration)):
            warnings.append(
                finding("PRESENCE_TIME_OUT_OF_RANGE", f"{path}.first_presence_seconds", "outside video duration")
            )
        if isinstance(last, (int, float)) and (last < 0 or last > float(duration)):
            warnings.append(
                finding("PRESENCE_TIME_OUT_OF_RANGE", f"{path}.last_presence_seconds", "outside video duration")
            )


def _check_state_changes(
    entity: dict[str, Any],
    path: str,
    duration: Any,
    has_duration: bool,
    declared_ids: set[str],
    warnings: list[dict[str, str]],
) -> None:
    changes = entity.get("state_changes")
    if changes is None:
        return
    if not isinstance(changes, list):
        warnings.append(finding("INVALID_STATE_CHANGES", f"{path}.state_changes", "must be a list"))
        return
    for index, change in enumerate(changes):
        change_path = f"{path}.state_changes[{index}]"
        if not isinstance(change, dict):
            warnings.append(finding("INVALID_STATE_EVENT", change_path, "must be an object"))
            continue
        seconds = change.get("seconds")
        if seconds is not None and not isinstance(seconds, (int, float)):
            warnings.append(finding("INVALID_STATE_EVENT_TIME", f"{change_path}.seconds", "must be a number"))
        elif has_duration and isinstance(seconds, (int, float)) and (seconds < 0 or seconds > float(duration)):
            warnings.append(
                finding("INVALID_STATE_EVENT_TIME", f"{change_path}.seconds", "outside video duration")
            )
