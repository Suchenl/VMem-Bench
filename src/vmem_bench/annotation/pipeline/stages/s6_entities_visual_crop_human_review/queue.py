"""Decision-oriented crop review queue for S6.

Human review only sees machine-accepted **library** crops (``task_kind=acquire``),
deduped by ``crop_path``. Near-duplicate bboxes (high IoU) keep the sharper crop.

``slot_bind`` copies stay in ``crop_proposals.json`` and follow library decisions
on apply — they must not clutter the review UI.

Do **not** hard-cap per entity: aggressive diversity pruning hid good acquires
and left worse high-sharpness dark crops on the board.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.geometry import (
    bbox_iou,
)


# Machine outcomes that are not worth a human click — hide from S6 UI.
_AUTO_HIDE_REASONS = frozenset(
    {
        "not_accepted",
        "picker_rejected",
        "picker_request_failed",
        "identity_mismatch",
        "cross_entity_bbox_conflict",
        "no_library_crop_for_slot",
        "missing_crop",
    }
)

# Only collapse near-identical frames; do not temporal-cap the review board.
_IOU_DEDUP = 0.85


def _crop_key(proposal: dict[str, Any]) -> str:
    """Stable crop identity across absolute / MemStrata-relative path spellings.

    Candidate filenames are unique within an entity directory
    (``c00003_00001313.png``), so the leaf name collapses abs vs relative
    duplicates that previously leaked onto the S6 board.
    """
    raw = str(proposal.get("crop_path") or "").strip().replace("\\", "/")
    if not raw:
        return ""
    leaf = Path(raw).name
    return leaf or raw


def _proposal_quality(proposal: dict[str, Any]) -> tuple:
    qa = proposal.get("qa") if isinstance(proposal.get("qa"), dict) else {}
    task_kind = str(proposal.get("task_kind") or "acquire")
    rid = str(proposal.get("representation_id") or "")
    return (
        0 if "@human_add_" in rid else 1,  # prefer canonical over accidental human_add dupes
        1 if task_kind != "slot_bind" else 0,
        1 if proposal.get("accepted") is True else 0,
        float(qa.get("sharpness") or proposal.get("detector_score") or 0.0),
        -int(proposal.get("chunk_id") or 0),
    )


def _should_hide_from_human(proposal: dict[str, Any], reasons: list[str]) -> bool:
    task_kind = str(proposal.get("task_kind") or "acquire")
    if task_kind == "slot_bind":
        return True
    if proposal.get("accepted") is False:
        return True
    if not _crop_key(proposal):
        return True
    if any(reason in _AUTO_HIDE_REASONS for reason in reasons):
        return True
    return False


def _collect_reasons(proposal: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    qa = proposal.get("qa") if isinstance(proposal.get("qa"), dict) else {}
    for reason in list(qa.get("reasons") or []):
        text = str(reason or "").strip()
        if text and text not in reasons:
            reasons.append(text)
    if proposal.get("accepted") is False:
        text = str(proposal.get("reason") or "not_accepted")
        if text and text not in reasons:
            reasons.append(text)
    elif isinstance(proposal.get("reason"), str) and proposal["reason"].strip():
        text = proposal["reason"].strip()
        if text not in {"current_appearance_le_t", "coverage_acquire"} and text not in reasons:
            reasons.append(text)
    return reasons


def _dedupe_near_identical(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate bboxes; keep the higher-quality crop of each cluster."""
    ranked = sorted(proposals, key=_proposal_quality, reverse=True)
    kept: list[dict[str, Any]] = []
    for proposal in ranked:
        bbox = proposal.get("bbox_norm") or []
        duplicate = False
        if len(bbox) == 4:
            for other in kept:
                other_bbox = other.get("bbox_norm") or []
                if len(other_bbox) == 4 and bbox_iou(bbox, other_bbox) >= _IOU_DEDUP:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(proposal)
    return sorted(kept, key=lambda item: int(item.get("chunk_id") or 0))


def build_review_queue(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build S6 queue: acquire-only, unique path, near-identical bbox collapse."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        reasons = _collect_reasons(proposal)
        if _should_hide_from_human(proposal, reasons):
            continue
        entity_id = str(proposal.get("entity_id") or "")
        crop_path = _crop_key(proposal)
        if not entity_id or not crop_path:
            continue
        key = (entity_id, crop_path)
        prev = best.get(key)
        if prev is None or _proposal_quality(proposal) > _proposal_quality(prev["proposal"]):
            best[key] = {"proposal": proposal, "reasons": reasons}

    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reason_by_id: dict[str, list[str]] = {}
    for item in best.values():
        proposal = item["proposal"]
        entity_id = str(proposal.get("entity_id") or "")
        by_entity[entity_id].append(proposal)
        reason_by_id[str(proposal.get("representation_id") or "")] = item["reasons"]

    queue: list[dict[str, Any]] = []
    for entity_id, items in by_entity.items():
        diverse = _dedupe_near_identical(items)
        for proposal in diverse:
            reasons = reason_by_id.get(str(proposal.get("representation_id") or ""), [])
            identity_block = any(
                reason
                in {
                    "cross_entity_bbox_conflict",
                    "picker_rejected",
                    "picker_request_failed",
                    "identity_mismatch",
                }
                for reason in reasons
            )
            queue.append(
                {
                    "kind": "crop",
                    "card_id": f"crop:{proposal.get('representation_id', '')}",
                    "chunk_id": proposal.get("chunk_id"),
                    "segment_id": proposal.get("segment_id"),
                    "entity_id": proposal.get("entity_id"),
                    "proposal": proposal,
                    "recommended_action": "keep",
                    "review_tier": "must" if identity_block else "spot_check",
                    "reasons": reasons,
                }
            )
    queue.sort(
        key=lambda item: (
            0 if item["review_tier"] == "must" else 1,
            str(item.get("entity_id") or ""),
            int(item.get("chunk_id") or 0),
        )
    )
    return queue


def write_review_queue(proposals_path: Path, out_path: Path) -> list[dict[str, Any]]:
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    queue = build_review_queue(proposals if isinstance(proposals, list) else [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed S6 crop review queue")
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = write_review_queue(args.proposals, args.output)
    print(
        json.dumps(
            {
                "n_queue": len(queue),
                "n_must": sum(1 for item in queue if item.get("review_tier") == "must"),
            }
        )
    )


if __name__ == "__main__":
    main()
