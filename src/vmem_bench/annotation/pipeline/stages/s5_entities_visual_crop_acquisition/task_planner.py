"""S5 crop task planning: coverage library (default) or per-slot ablation.

Default mode builds a capped per-entity visual library (first appearance,
re-appearance after absence, then temporally spaced fills). Scoring / gold
still need one resolvable crop per ``(chunk, entity)`` slot — that is filled
later by ``expand_library_to_slots`` using ≤t same-state binding
(``scoring.md`` ``current_appearance``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

TaskMode = Literal["coverage", "per_slot"]


@dataclass(slots=True)
class CoverageCaps:
    """Per-kind acquire caps for the coverage library."""

    character: int = 8
    prop: int = 5
    location: int = 12  # soft: higher than char/prop; no hard film-wide location tax
    max_total_acquire: int | None = None

    def for_kind(self, kind: str) -> int:
        if kind == "character":
            return max(1, int(self.character))
        if kind == "prop":
            return max(1, int(self.prop))
        if kind == "location":
            return max(1, int(self.location))
        return max(1, int(self.prop))


@dataclass(slots=True)
class CropTask:
    chunk_id: int
    segment_id: str
    entity_id: str
    kind: str
    name: str
    description: str
    action: str
    start_seconds: float
    end_seconds: float
    task_kind: str = "acquire"  # acquire | slot_bind
    priority: int = 0
    reason: str = ""
    bind_source_chunk_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _roster(annotation: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for raw in annotation.get(group) or []:
            entity_id = str(raw.get(id_key) or "")
            if entity_id:
                output[entity_id] = {
                    "kind": kind,
                    "name": str(raw.get("name") or ""),
                    "description": str(raw.get("description") or ""),
                }
    return output


def _segments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        segments.extend(scene.get("visual_segments") or [])
    segments.sort(key=lambda item: (float(item["start_seconds"]), str(item["segment_id"])))
    return segments


def _task_from_segment(
    *,
    chunk_id: int,
    segment: dict[str, Any],
    entity_id: str,
    entity: dict[str, str],
    task_kind: str = "acquire",
    priority: int = 0,
    reason: str = "",
    bind_source_chunk_id: int | None = None,
) -> CropTask:
    return CropTask(
        chunk_id=chunk_id,
        segment_id=str(segment["segment_id"]),
        entity_id=entity_id,
        kind=entity["kind"],
        name=entity["name"],
        description=entity["description"],
        action=str(segment.get("action") or ""),
        start_seconds=float(segment["start_seconds"]),
        end_seconds=float(segment["end_seconds"]),
        task_kind=task_kind,
        priority=priority,
        reason=reason,
        bind_source_chunk_id=bind_source_chunk_id,
    )


def derive_tasks(annotation: dict[str, Any]) -> list[CropTask]:
    """Ablation: one current-GT crop task per segment × present known entity."""
    roster = _roster(annotation)
    segments = _segments(annotation)
    tasks: list[CropTask] = []
    for chunk_id, segment in enumerate(segments):
        for entity_id in dict.fromkeys(str(x) for x in segment.get("present_entity_ids") or []):
            entity = roster.get(entity_id)
            if entity is None:
                continue
            tasks.append(
                _task_from_segment(
                    chunk_id=chunk_id,
                    segment=segment,
                    entity_id=entity_id,
                    entity=entity,
                    task_kind="acquire",
                    reason="per_slot",
                )
            )
    return tasks


def _select_coverage_chunks(presence_chunks: list[int], *, cap: int) -> list[tuple[int, str]]:
    """Return ``(chunk_id, reason)`` pairs for one entity, capped."""
    if not presence_chunks:
        return []
    ordered = sorted(presence_chunks)
    chosen: dict[int, str] = {}

    first = ordered[0]
    chosen[first] = "first_appearance"

    for index in range(1, len(ordered)):
        prev_c, cur_c = ordered[index - 1], ordered[index]
        if cur_c > prev_c + 1:
            chosen.setdefault(cur_c, "reappearance")

    # Temporally spaced fills until cap (prefer mid-span diversity).
    remaining = [c for c in ordered if c not in chosen]
    need = max(0, cap - len(chosen))
    if need > 0 and remaining:
        if len(remaining) <= need:
            for chunk_id in remaining:
                chosen[chunk_id] = "coverage_fill"
        else:
            # Evenly sample across remaining presence.
            for i in range(need):
                pos = int(round(i * (len(remaining) - 1) / max(need - 1, 1))) if need > 1 else 0
                chunk_id = remaining[min(pos, len(remaining) - 1)]
                chosen.setdefault(chunk_id, "coverage_fill")

    ranked = sorted(chosen.items(), key=lambda item: item[0])
    return ranked[:cap]


def derive_tasks_coverage(
    annotation: dict[str, Any],
    *,
    caps: CoverageCaps | None = None,
) -> tuple[list[CropTask], dict[str, Any]]:
    """Plan capped acquire tasks that fill a per-entity visual coverage library."""
    caps = caps or CoverageCaps()
    roster = _roster(annotation)
    segments = _segments(annotation)

    presence: dict[str, list[int]] = {entity_id: [] for entity_id in roster}
    for chunk_id, segment in enumerate(segments):
        for entity_id in dict.fromkeys(str(x) for x in segment.get("present_entity_ids") or []):
            if entity_id in presence:
                presence[entity_id].append(chunk_id)

    acquire: list[CropTask] = []
    per_entity: dict[str, Any] = {}
    for entity_id, chunks in presence.items():
        entity = roster[entity_id]
        cap = caps.for_kind(entity["kind"])
        selected = _select_coverage_chunks(chunks, cap=cap)
        per_entity[entity_id] = {
            "kind": entity["kind"],
            "n_presence": len(chunks),
            "cap": cap,
            "n_acquire": len(selected),
            "chunks": [{"chunk_id": c, "reason": r} for c, r in selected],
        }
        for priority, (chunk_id, reason) in enumerate(selected):
            acquire.append(
                _task_from_segment(
                    chunk_id=chunk_id,
                    segment=segments[chunk_id],
                    entity_id=entity_id,
                    entity=entity,
                    task_kind="acquire",
                    priority=priority,
                    reason=reason,
                )
            )

    # Deterministic order: time, then entity id.
    acquire.sort(key=lambda task: (task.chunk_id, task.entity_id))

    if caps.max_total_acquire is not None and len(acquire) > caps.max_total_acquire:
        # Keep first-appearance tasks, then fill remaining by earlier chunks.
        firsts = [t for t in acquire if t.reason == "first_appearance"]
        rest = [t for t in acquire if t.reason != "first_appearance"]
        budget = max(0, caps.max_total_acquire - len(firsts))
        acquire = firsts + rest[:budget]
        acquire.sort(key=lambda task: (task.chunk_id, task.entity_id))

    stats = {
        "mode": "coverage",
        "n_segments": len(segments),
        "n_entities": len(roster),
        "n_acquire": len(acquire),
        "n_per_slot_would_be": sum(len(v) for v in presence.values()),
        "caps": {
            "character": caps.character,
            "prop": caps.prop,
            "location": caps.location,
            "max_total_acquire": caps.max_total_acquire,
        },
        "per_entity": per_entity,
    }
    return acquire, stats


def plan_tasks(
    annotation: dict[str, Any],
    *,
    mode: TaskMode = "coverage",
    caps: CoverageCaps | None = None,
) -> tuple[list[CropTask], dict[str, Any]]:
    """Return acquire tasks + plan stats for the selected mode."""
    if mode == "per_slot":
        tasks = derive_tasks(annotation)
        return tasks, {
            "mode": "per_slot",
            "n_acquire": len(tasks),
            "n_per_slot_would_be": len(tasks),
        }
    return derive_tasks_coverage(annotation, caps=caps)


def write_tasks(tasks: list[CropTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([task.to_dict() for task in tasks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "CoverageCaps",
    "CropTask",
    "TaskMode",
    "derive_tasks",
    "derive_tasks_coverage",
    "plan_tasks",
    "write_tasks",
]
