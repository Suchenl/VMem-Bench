"""S6 human review persistence helpers on top of crop decision patching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.patch import (
    apply_crop_decisions,
)
from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.queue import (
    build_review_queue,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_s6_queue(movie_dir: Path) -> list[dict[str, Any]]:
    """Rebuild the S6 queue from current S5 proposals (filters may change)."""
    movie_dir = Path(movie_dir)
    s5 = movie_dir / "tmp" / "pipeline" / "s5_entities_visual_crop_acquisition" / "crop_proposals.json"
    s6_dir = movie_dir / "tmp" / "pipeline" / "s6_entities_visual_crop_human_review"
    queue_path = s6_dir / "review_queue.json"
    if not s5.is_file():
        return []
    proposals = _read_json(s5)
    queue = build_review_queue(proposals if isinstance(proposals, list) else [])
    _write_json(queue_path, queue)
    return queue


def _normalize_crop_path(path: Any) -> str:
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        return ""
    return Path(raw).name or raw


def _reassign_payload(decision: dict[str, Any], *, reason: str) -> dict[str, Any]:
    payload = {
        "action": "reassign",
        "entity_id": str(decision.get("entity_id") or ""),
        "reason": reason,
    }
    if decision.get("from_entity_id") is not None:
        payload["from_entity_id"] = str(decision.get("from_entity_id") or "")
    if decision.get("name") is not None:
        payload["name"] = decision.get("name")
    if decision.get("kind") is not None:
        payload["kind"] = decision.get("kind")
    if "description" in decision:
        payload["description"] = decision.get("description")
    replacement = decision.get("replacement")
    if isinstance(replacement, dict) and replacement.get("crop_path"):
        payload["replacement"] = dict(replacement)
    return payload


def expand_decisions_to_siblings(
    *,
    proposals: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Propagate keep/reject/replace/reassign from reviewed library crops to slot_bind copies.

    Matching is by shared ``crop_path`` leaf (same image) or ``bind_source_representation_id``.
    ``reassign`` only propagates to siblings that still belong to the source entity
    (never to another entity that happens to share a crop leaf).
    Machine-rejected proposals without a human decision stay untouched.
    """
    by_id = {
        str(item.get("representation_id") or ""): item
        for item in proposals
        if isinstance(item, dict) and item.get("representation_id")
    }
    expanded = {str(k): dict(v) for k, v in decisions.items() if isinstance(v, dict)}

    # Map crop_path -> winning decision from an explicit human choice.
    # reassign is excluded: attribution moves are entity-scoped, not global by image leaf.
    path_decision: dict[str, dict[str, Any]] = {}
    source_decision: dict[str, dict[str, Any]] = {}
    reassign_by_source: list[tuple[str, dict[str, Any], str, str]] = []
    for rep_id, decision in list(expanded.items()):
        proposal = by_id.get(rep_id) or {}
        action = str(decision.get("action") or "")
        crop_path = _normalize_crop_path(proposal.get("crop_path"))
        if not crop_path and action == "add":
            payload = decision.get("proposal") if isinstance(decision.get("proposal"), dict) else {}
            crop_path = _normalize_crop_path(payload.get("crop_path"))
        if not crop_path and action in {"replace", "reassign"}:
            replacement = decision.get("replacement") if isinstance(decision.get("replacement"), dict) else {}
            crop_path = _normalize_crop_path(replacement.get("crop_path")) or crop_path
        if action == "reassign":
            from_entity = str(
                decision.get("from_entity_id") or proposal.get("entity_id") or ""
            ).strip()
            reassign_by_source.append((rep_id, decision, from_entity, crop_path))
        elif crop_path:
            path_decision[crop_path] = decision
        source_decision[rep_id] = decision
        bind_src = str(proposal.get("bind_source_representation_id") or "")
        if bind_src:
            source_decision.setdefault(bind_src, decision)

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        rep_id = str(proposal.get("representation_id") or "")
        if not rep_id or rep_id in expanded:
            continue
        crop_path = _normalize_crop_path(proposal.get("crop_path"))
        bind_src = str(proposal.get("bind_source_representation_id") or "")
        decision = None
        if crop_path and crop_path in path_decision:
            decision = path_decision[crop_path]
        elif bind_src and bind_src in source_decision:
            decision = source_decision[bind_src]
        elif rep_id in source_decision:
            decision = source_decision[rep_id]
        if decision is None:
            continue
        # Only auto-propagate keep/reject; replace/add need an explicit crop.
        action = str(decision.get("action") or "")
        if action == "keep":
            expanded[rep_id] = {"action": "keep", "reason": decision.get("reason") or "propagated_from_library"}
        elif action == "reject":
            expanded[rep_id] = {
                "action": "reject",
                "reason": decision.get("reason") or "propagated_from_library",
            }
        elif action == "add":
            # human_add of an already-known library crop → keep the canonical id
            expanded[rep_id] = {"action": "keep", "reason": "deduped_human_add"}
        # reassign handled below (source-entity scoped)

    for source_id, decision, from_entity, crop_path in reassign_by_source:
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            rep_id = str(proposal.get("representation_id") or "")
            if not rep_id or rep_id in expanded:
                continue
            entity_id = str(proposal.get("entity_id") or "").strip()
            if from_entity and entity_id and entity_id != from_entity:
                continue
            bind_src = str(proposal.get("bind_source_representation_id") or "")
            same_path = bool(crop_path) and _normalize_crop_path(proposal.get("crop_path")) == crop_path
            if bind_src == source_id or same_path:
                expanded[rep_id] = _reassign_payload(
                    decision,
                    reason=str(decision.get("reason") or "propagated_from_library"),
                )
    return expanded


def _materialize_add_decisions(
    *,
    proposals: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append human-promoted alternate crops into the proposal list before patching.

    Skip adds whose crop leaf already exists for the same entity — those are
    duplicates created when alternate→existing_id mapping failed.
    """
    by_id = {
        str(item.get("representation_id") or "")
        for item in proposals
        if isinstance(item, dict) and item.get("representation_id")
    }
    existing_keys = {
        (
            str(item.get("entity_id") or ""),
            _normalize_crop_path(item.get("crop_path")),
        )
        for item in proposals
        if isinstance(item, dict) and item.get("crop_path")
    }
    out = list(proposals)
    for rep_id, decision in decisions.items():
        if str(decision.get("action") or "") != "add":
            continue
        if rep_id in by_id:
            continue
        payload = decision.get("proposal")
        if not isinstance(payload, dict) or not payload.get("crop_path"):
            continue
        entity_id = str(payload.get("entity_id") or "")
        crop_key = _normalize_crop_path(payload.get("crop_path"))
        if (entity_id, crop_key) in existing_keys:
            continue
        item = dict(payload)
        item["representation_id"] = rep_id
        item["accepted"] = True
        item["task_kind"] = str(item.get("task_kind") or "acquire")
        out.append(item)
        by_id.add(rep_id)
        existing_keys.add((entity_id, crop_key))
    return out


def apply_s6_decisions(*, movie_dir: Path, decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    movie_dir = Path(movie_dir)
    pipeline = movie_dir / "tmp" / "pipeline"
    proposals_path = pipeline / "s5_entities_visual_crop_acquisition" / "crop_proposals.json"
    s6_dir = pipeline / "s6_entities_visual_crop_human_review"
    if not proposals_path.is_file():
        raise FileNotFoundError(f"missing crop proposals: {proposals_path}")
    proposals = _read_json(proposals_path)
    if not isinstance(proposals, list):
        raise ValueError("crop_proposals.json must be a list")
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("decisions must be a non-empty object keyed by representation_id")

    proposals = _materialize_add_decisions(proposals=proposals, decisions=decisions)
    expanded = expand_decisions_to_siblings(proposals=proposals, decisions=decisions)
    accepted = apply_crop_decisions(proposals=proposals, decisions=expanded)
    kept = [item for item in accepted if item.get("accepted") and item.get("crop_path")]
    patch = {"version": 1, "decisions": decisions, "expanded_decisions": expanded}
    _write_json(proposals_path, accepted)
    _write_json(s6_dir / "review_patch.draft.json", patch)
    _write_json(s6_dir / "review_patch.applied.json", patch)
    _write_json(s6_dir / "accepted_crops.json", kept)
    ensure_s6_queue(movie_dir)
    audit = {
        "mode": "human_reviewed",
        "human_reviewed": True,
        "human_review_skipped": False,
        "accepted_count": len(kept),
        "proposal_count": len(proposals),
        "decision_count": len(decisions),
        "expanded_decision_count": len(expanded),
    }
    _write_json(s6_dir / "review_audit.json", audit)

    state_path = pipeline / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {"stages": {}}
    stages = dict(state.get("stages") or {})
    stages["s6_entities_visual_crop_human_review"] = {
        "status": "human_reviewed",
        "n_accepted": len(kept),
        "n_decisions": len(decisions),
        "n_expanded_decisions": len(expanded),
    }
    state["stages"] = stages
    _write_json(state_path, state)
    return {
        "ok": True,
        "accepted_count": len(kept),
        "proposal_count": len(proposals),
        "accepted_crops": str(s6_dir / "accepted_crops.json"),
    }


def load_s6_draft(movie_dir: Path) -> dict[str, Any]:
    path = Path(movie_dir) / "tmp" / "pipeline" / "s6_entities_visual_crop_human_review" / "review_patch.draft.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}
