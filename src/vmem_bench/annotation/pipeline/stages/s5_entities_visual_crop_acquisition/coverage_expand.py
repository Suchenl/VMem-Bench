"""Coverage library helpers: attribute prune + ≤t same-state slot binding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from vmem_bench.common.attribute_dedup import select_attribute_diverse
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CoverageCaps,
    CropTask,
    derive_tasks,
)


def proposal_state(proposal: dict[str, Any]) -> str:
    attrs = proposal.get("crop_attributes")
    if isinstance(attrs, dict) and attrs.get("state_angle"):
        return str(attrs["state_angle"])
    return str(proposal.get("state") or "default")


def proposal_bucket(proposal: dict[str, Any]) -> tuple[str, str, str, str]:
    attrs = proposal.get("crop_attributes") if isinstance(proposal.get("crop_attributes"), dict) else {}
    return (
        str(attrs.get("spatial_angle") or "unknown"),
        str(attrs.get("state_angle") or "unknown"),
        str(attrs.get("shot_size") or "unknown"),
        str(attrs.get("lighting") or "unknown"),
    )


def resolve_current_appearance(
    library: list[dict[str, Any]],
    *,
    chunk_id: int,
    state: str = "default",
    allow_any_state_fallback: bool = True,
) -> dict[str, Any] | None:
    """Nearest library crop with ``chunk_id ≤ t`` and matching state (v3 rule).

    If no same-state crop exists and ``allow_any_state_fallback``, fall back to
    nearest ≤t of any state (marked by caller via ``state_fallback``).
    """
    same = [
        item
        for item in library
        if int(item.get("chunk_id", -1)) <= chunk_id and proposal_state(item) == state
    ]
    if same:
        return max(same, key=lambda item: int(item["chunk_id"]))
    if not allow_any_state_fallback:
        return None
    any_state = [item for item in library if int(item.get("chunk_id", -1)) <= chunk_id]
    if not any_state:
        return None
    return max(any_state, key=lambda item: int(item["chunk_id"]))


def prune_library_by_attributes(
    proposals: list[dict[str, Any]],
    *,
    caps: CoverageCaps | None = None,
) -> list[dict[str, Any]]:
    """Keep attribute-diverse accepted crops per entity (post-acquire)."""
    caps = caps or CoverageCaps()
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for proposal in proposals:
        if proposal.get("accepted") and proposal.get("crop_path"):
            by_entity[str(proposal["entity_id"])].append(proposal)
        else:
            rejected.append(proposal)

    kept: list[dict[str, Any]] = []
    for entity_id, items in by_entity.items():
        kind = str(items[0].get("kind") or "prop")
        max_keep = caps.for_kind(kind)
        if len(items) <= max_keep:
            kept.extend(items)
            continue
        buckets = [proposal_bucket(item) for item in items]
        qualities = [
            float((item.get("qa") or {}).get("sharpness") or item.get("detector_score") or 1.0)
            for item in items
        ]
        indices = select_attribute_diverse(
            bucket_keys=buckets,
            quality=qualities,
            max_keep=max_keep,
        )
        kept.extend(items[i] for i in indices)
    return kept + rejected


def expand_library_to_slots(
    *,
    annotation: dict[str, Any],
    library_proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand sparse library into one proposal per present ``(chunk, entity)`` slot.

    Exact acquire wins; otherwise bind ≤t current_appearance (same-state preferred).
    """
    library_accepted = [
        item
        for item in library_proposals
        if item.get("accepted") and item.get("crop_path") and item.get("task_kind", "acquire") != "slot_bind"
    ]
    # Also treat missing task_kind as acquire.
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact: dict[tuple[int, str], dict[str, Any]] = {}
    for proposal in library_accepted:
        entity_id = str(proposal["entity_id"])
        chunk_id = int(proposal["chunk_id"])
        by_entity[entity_id].append(proposal)
        exact[(chunk_id, entity_id)] = proposal
    for items in by_entity.values():
        items.sort(key=lambda item: int(item["chunk_id"]))

    slot_tasks: list[CropTask] = derive_tasks(annotation)
    expanded: list[dict[str, Any]] = []
    for task in slot_tasks:
        key = (task.chunk_id, task.entity_id)
        if key in exact:
            proposal = dict(exact[key])
            proposal["task_kind"] = "acquire"
            proposal.setdefault("reason", task.reason or "coverage_acquire")
            expanded.append(proposal)
            continue

        source = resolve_current_appearance(
            by_entity.get(task.entity_id, []),
            chunk_id=task.chunk_id,
            state="default",
            allow_any_state_fallback=True,
        )
        if source is None:
            expanded.append(
                {
                    **task.to_dict(),
                    "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
                    "accepted": False,
                    "reason": "no_library_crop_for_slot",
                    "task_kind": "slot_bind",
                    "crop_path": None,
                }
            )
            continue

        bound = dict(source)
        bound.update(
            {
                "chunk_id": task.chunk_id,
                "segment_id": task.segment_id,
                "entity_id": task.entity_id,
                "kind": task.kind,
                "name": task.name,
                "description": task.description,
                "action": task.action,
                "start_seconds": task.start_seconds,
                "end_seconds": task.end_seconds,
                "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
                "task_kind": "slot_bind",
                "reason": "current_appearance_le_t",
                "bind_source_chunk_id": int(source["chunk_id"]),
                "bind_source_representation_id": source.get("representation_id"),
                "accepted": True,
            }
        )
        expanded.append(bound)
    return expanded


__all__ = [
    "expand_library_to_slots",
    "proposal_bucket",
    "proposal_state",
    "prune_library_by_attributes",
    "resolve_current_appearance",
]
