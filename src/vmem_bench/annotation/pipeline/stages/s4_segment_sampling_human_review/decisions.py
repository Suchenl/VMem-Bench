"""Human decisions for S4 segment review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (
    action_has_entity_list_coda,
    missing_canonical_names,
)

VALID_ACTIONS = {
    "accept",
    "edit_action",
    "edit_present",
    "edit_both",
    "request_retry",
    "reject_film",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_present_overrides(annotation: dict[str, Any], overrides: dict[str, list[str]]) -> dict[str, Any]:
    """Return a deep-copied annotation with segment present_entity_ids overridden."""
    payload = json.loads(json.dumps(annotation))
    for scene in (payload.get("screenplay") or {}).get("scenes") or []:
        for segment in scene.get("visual_segments") or []:
            segment_id = str(segment.get("segment_id") or "")
            if segment_id in overrides:
                segment["present_entity_ids"] = list(overrides[segment_id])
    return payload


def apply_annotation_overrides(
    annotation: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply action and/or present edits to a deep-copied annotation."""
    payload = json.loads(json.dumps(annotation))
    for scene in (payload.get("screenplay") or {}).get("scenes") or []:
        for segment in scene.get("visual_segments") or []:
            segment_id = str(segment.get("segment_id") or "")
            decision = decisions.get(segment_id) or {}
            action = str(decision.get("action") or "")
            if action in {"edit_present", "edit_both"}:
                segment["present_entity_ids"] = list(decision["present_entity_ids"])
            if action in {"edit_action", "edit_both"}:
                segment["action"] = str(decision["revised_action"])
    return payload


def _normalize_decision(segment_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    action = str(decision.get("action") or "accept")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid S4 action {action!r} for {segment_id}")
    item: dict[str, Any] = {
        "action": action,
        "reason": str(decision.get("reason") or ""),
    }
    if action in {"edit_present", "edit_both"}:
        present = decision.get("present_entity_ids")
        if not isinstance(present, list) or not all(isinstance(x, str) and x for x in present):
            raise ValueError(f"{action} requires present_entity_ids for {segment_id}")
        item["present_entity_ids"] = list(dict.fromkeys(present))
    if action in {"edit_action", "edit_both"}:
        revised_action = str(decision.get("revised_action") or "").strip()
        if not revised_action:
            raise ValueError(f"{action} requires revised_action for {segment_id}")
        item["revised_action"] = revised_action
    return item


def _validate_edited_segments(
    annotation: dict[str, Any],
    edited_segment_ids: set[str],
) -> None:
    roster: dict[str, dict[str, str]] = {}
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for entity in annotation.get(group) or []:
            entity_id = str(entity.get(id_key) or "")
            if entity_id:
                roster[entity_id] = {
                    "entity_id": entity_id,
                    "name": str(entity.get("name") or ""),
                    "kind": kind,
                }

    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        for segment in scene.get("visual_segments") or []:
            segment_id = str(segment.get("segment_id") or "")
            if segment_id not in edited_segment_ids:
                continue
            present = [str(item) for item in segment.get("present_entity_ids") or []]
            unknown = [entity_id for entity_id in present if entity_id not in roster]
            if unknown:
                raise ValueError(f"{segment_id} contains unknown entity ids: {unknown}")
            action = str(segment.get("action") or "")
            if action_has_entity_list_coda(action):
                raise ValueError(f"{segment_id} action contains an entity-list coda")
            missing = missing_canonical_names(
                action=action,
                present_entity_ids=present,
                roster_by_id=roster,
            )
            if missing:
                names = [item.get("name") or item["entity_id"] for item in missing]
                raise ValueError(f"{segment_id} action missing canonical names: {names}")


def save_s4_draft(
    *,
    movie_dir: Path,
    decisions: dict[str, dict[str, Any]],
    film_verdict: str = "accept",
    reason: str = "",
) -> dict[str, Any]:
    """Persist a partial draft; does not require every queue item."""
    movie_dir = Path(movie_dir)
    s4_dir = movie_dir / "tmp" / "pipeline" / "s4_segment_sampling_human_review"
    queue_path = s4_dir / "review_queue.json"
    if not queue_path.is_file():
        raise FileNotFoundError(f"missing S4 review queue: {queue_path}")
    queue = _read_json(queue_path)
    if not isinstance(queue, list):
        raise ValueError("S4 review_queue.json must be a list")
    queue_ids = {str(item.get("segment_id") or "") for item in queue}
    normalized: dict[str, dict[str, Any]] = {}
    for segment_id, decision in decisions.items():
        if segment_id not in queue_ids:
            continue
        normalized[segment_id] = _normalize_decision(segment_id, decision)
    if film_verdict not in {"accept", "reject_for_reannotation"}:
        raise ValueError(f"invalid film_verdict {film_verdict!r}")
    patch = {
        "version": 1,
        "film_verdict": film_verdict,
        "reason": reason,
        "decisions": normalized,
        "n_decided": len(normalized),
        "n_queue": len(queue_ids),
        "complete": len(normalized) >= len(queue_ids) and len(queue_ids) > 0,
    }
    _write_json(s4_dir / "review_patch.draft.json", patch)
    return {"ok": True, "draft": True, **{k: patch[k] for k in ("n_decided", "n_queue", "complete")}}


def apply_s4_decisions(
    *,
    movie_dir: Path,
    decisions: dict[str, dict[str, Any]],
    film_verdict: str = "accept",
    reason: str = "",
) -> dict[str, Any]:
    """Finalize S4 review for the full queue without mutating S3 auto_revised_annotation.json."""
    movie_dir = Path(movie_dir)
    pipeline = movie_dir / "tmp" / "pipeline"
    s3_ann = pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json"
    s4_dir = pipeline / "s4_segment_sampling_human_review"
    queue_path = s4_dir / "review_queue.json"
    if not queue_path.is_file():
        raise FileNotFoundError(f"missing S4 review queue: {queue_path}")
    if not s3_ann.is_file():
        raise FileNotFoundError(f"missing S3 annotation: {s3_ann}")

    queue = _read_json(queue_path)
    if not isinstance(queue, list):
        raise ValueError("S4 review_queue.json must be a list")
    queue_ids = {str(item.get("segment_id") or "") for item in queue}
    normalized: dict[str, dict[str, Any]] = {}
    for segment_id, decision in decisions.items():
        if segment_id not in queue_ids:
            raise ValueError(f"segment {segment_id!r} is not in the S4 review queue")
        normalized[segment_id] = _normalize_decision(segment_id, decision)

    missing = sorted(queue_ids - set(normalized))
    if missing:
        raise ValueError(
            f"S4 decisions incomplete ({len(normalized)}/{len(queue_ids)}); "
            f"missing e.g. {', '.join(missing[:8])}"
        )

    if film_verdict not in {"accept", "reject_for_reannotation"}:
        raise ValueError(f"invalid film_verdict {film_verdict!r}")

    annotation = _read_json(s3_ann)
    revised = apply_annotation_overrides(annotation, normalized)
    edited_ids = {
        segment_id
        for segment_id, decision in normalized.items()
        if decision["action"] in {"edit_action", "edit_present", "edit_both"}
    }
    _validate_edited_segments(revised, edited_ids)
    retry_ids = sorted(
        segment_id
        for segment_id, decision in normalized.items()
        if decision["action"] == "request_retry"
    )
    n_present_overrides = sum(
        1 for decision in normalized.values()
        if decision["action"] in {"edit_present", "edit_both"}
    )
    n_action_overrides = sum(
        1 for decision in normalized.values()
        if decision["action"] in {"edit_action", "edit_both"}
    )
    resolved_verdicts = {
        segment_id: (
            "RETRYABLE_ERROR" if decision["action"] == "request_retry" else "PASS"
        )
        for segment_id, decision in normalized.items()
    }
    patch = {
        "version": 1,
        "film_verdict": film_verdict,
        "reason": reason,
        "decisions": normalized,
        "n_overrides": n_present_overrides + n_action_overrides,
        "n_present_overrides": n_present_overrides,
        "n_action_overrides": n_action_overrides,
        "retry_segment_ids": retry_ids,
        "resolved_verdicts": resolved_verdicts,
        "n_queue": len(queue_ids),
        "complete": True,
    }
    _write_json(s4_dir / "review_patch.draft.json", patch)
    _write_json(s4_dir / "review_patch.applied.json", patch)
    _write_json(s4_dir / "human_revised_annotation.json", revised)
    previous_audit_path = s4_dir / "review_audit.json"
    previous_audit = (
        _read_json(previous_audit_path)
        if previous_audit_path.is_file()
        else {}
    )
    audit = {
        "mode": "human_reviewed",
        "human_reviewed": not retry_ids,
        "blocks_pipeline": bool(retry_ids),
        "film_verdict": film_verdict,
        "reason": reason,
        "sampled_segments": len(queue),
        "queue_policy": "stratified_sample",
        "n_overrides": n_present_overrides + n_action_overrides,
        "n_present_overrides": n_present_overrides,
        "n_action_overrides": n_action_overrides,
        "n_retry_requested": len(retry_ids),
        "decided_segments": len(normalized),
        "s4_mode_requested": previous_audit.get("s4_mode_requested", "auto"),
        "s4_mode_effective": previous_audit.get("s4_mode_effective", "blocking"),
    }
    _write_json(s4_dir / "review_audit.json", audit)

    state_path = pipeline / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {"stages": {}}
    stages = dict(state.get("stages") or {})
    stages["s4_segment_sampling_human_review"] = {
        "status": (
            "retry_requested"
            if retry_ids
            else (
                "rejected_for_reannotation"
                if film_verdict == "reject_for_reannotation"
                else "human_reviewed"
            )
        ),
        "n_sampled": len(queue),
        "n_overrides": n_present_overrides + n_action_overrides,
        "n_retry_requested": len(retry_ids),
    }
    state["stages"] = stages
    _write_json(state_path, state)
    return {
        "ok": True,
        "film_verdict": film_verdict,
        "n_decisions": len(normalized),
        "n_overrides": n_present_overrides + n_action_overrides,
        "n_retry_requested": len(retry_ids),
        "human_revised_annotation": str(s4_dir / "human_revised_annotation.json"),
        "applied_patch": str(s4_dir / "review_patch.applied.json"),
    }


def load_s4_draft(movie_dir: Path) -> dict[str, Any]:
    path = Path(movie_dir) / "tmp" / "pipeline" / "s4_segment_sampling_human_review" / "review_patch.draft.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}
