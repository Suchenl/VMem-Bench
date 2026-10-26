"""Deterministic identity safety gates shared by staged crop pipelines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

_CONFIDENCE_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _identity_score(item: dict[str, Any]) -> float:
    pick = item.get("pick") if isinstance(item.get("pick"), dict) else {}
    confidence = _CONFIDENCE_SCORE.get(str(pick.get("confidence") or "").lower(), 0.0)
    grounding = item.get("grounding") if isinstance(item.get("grounding"), dict) else {}
    detector = float(grounding.get("detector_score") or 0.0)
    return confidence + 0.05 * detector


def _reject(item: dict[str, Any], peers: list[str], reason: str) -> None:
    item["accepted"] = False
    item["reason"] = reason
    qa = dict(item.get("qa") or {})
    qa["accepted"] = False
    qa["reasons"] = list(dict.fromkeys([*(qa.get("reasons") or []), reason]))
    item["qa"] = qa
    item["identity_gate"] = {
        "passed": False,
        "reason": reason,
        "conflicting_entity_ids": peers,
    }


def apply_cross_entity_conflict_gate(
    proposals: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.95,
    winner_margin: float = 0.15,
) -> list[dict[str, Any]]:
    """Reject ambiguous same-frame crops assigned to different same-kind IDs.

    A uniquely higher-confidence closed-set pick may survive. If confidence
    cannot distinguish the assignments, all conflicting proposals fail closed
    so slot binding cannot propagate a mixed identity library.
    """
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in proposals:
        if not item.get("accepted") or item.get("task_kind") != "acquire":
            continue
        if str(item.get("kind") or "") == "location":
            continue
        bbox = item.get("bbox_norm") or []
        if len(bbox) != 4:
            continue
        key = (
            int(item.get("chunk_id", -1)),
            int(item.get("frame_index", -1)),
            str(item.get("kind") or ""),
        )
        groups[key].append(item)

    for items in groups.values():
        adjacency: dict[int, set[int]] = defaultdict(set)
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                same_crop = str(items[left].get("crop_path") or "") == str(
                    items[right].get("crop_path") or ""
                )
                overlaps = _bbox_iou(
                    list(items[left]["bbox_norm"]),
                    list(items[right]["bbox_norm"]),
                ) >= iou_threshold
                if same_crop or overlaps:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        seen: set[int] = set()
        for start in adjacency:
            if start in seen:
                continue
            component: set[int] = set()
            stack = [start]
            while stack:
                index = stack.pop()
                if index in component:
                    continue
                component.add(index)
                stack.extend(adjacency[index])
            seen.update(component)
            ranked = sorted(
                component,
                key=lambda index: _identity_score(items[index]),
                reverse=True,
            )
            keep: int | None = None
            if len(ranked) == 1:
                keep = ranked[0]
            elif (
                _identity_score(items[ranked[0]])
                - _identity_score(items[ranked[1]])
                >= winner_margin
            ):
                keep = ranked[0]
            entity_ids = [str(items[index].get("entity_id") or "") for index in component]
            for index in component:
                if index == keep:
                    items[index]["identity_gate"] = {
                        "passed": True,
                        "reason": "unique_confident_assignment",
                        "conflicting_entity_ids": [
                            entity_id
                            for entity_id in entity_ids
                            if entity_id != str(items[index].get("entity_id") or "")
                        ],
                    }
                    continue
                peers = [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id != str(items[index].get("entity_id") or "")
                ]
                _reject(items[index], peers, "cross_entity_bbox_conflict")
    return proposals

