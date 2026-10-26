"""Reversible human crop decisions for S6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ACTIONS = {"keep", "reject", "replace", "add", "reassign"}


def apply_crop_decisions(
    *,
    proposals: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply narrow, auditable decisions without editing S1/S3 annotations.

    ``add`` inserts a new library crop promoted from an alternate candidate.
    ``reassign`` moves an existing crop's attribution to another entity (keeps
    ``representation_id``; ``entity_id`` / name / kind are authoritative).
    """
    by_id = {str(item.get("representation_id")): dict(item) for item in proposals}
    order = [str(item.get("representation_id")) for item in proposals]

    for representation_id, decision in decisions.items():
        action = str(decision.get("action") or "")
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid crop decision {action!r} for {representation_id}")

        if action == "add":
            payload = decision.get("proposal")
            if not isinstance(payload, dict) or not payload.get("crop_path"):
                raise ValueError(f"add requires a proposal with crop_path for {representation_id}")
            new_item = dict(payload)
            new_item["representation_id"] = representation_id
            new_item["accepted"] = True
            new_item["task_kind"] = str(new_item.get("task_kind") or "acquire")
            new_item["review_reason"] = str(decision.get("reason") or "human_promoted_alt")
            if representation_id in by_id:
                by_id[representation_id].update(new_item)
            else:
                by_id[representation_id] = new_item
                order.append(representation_id)
            continue

        if representation_id not in by_id:
            raise ValueError(f"unknown representation {representation_id!r}")
        proposal = by_id[representation_id]
        if action == "keep":
            proposal["accepted"] = True
        elif action == "reject":
            proposal["accepted"] = False
            proposal["review_reason"] = str(decision.get("reason") or "human_rejected")
        elif action == "reassign":
            target_entity = str(decision.get("entity_id") or "").strip()
            if not target_entity:
                raise ValueError(f"reassign requires entity_id for {representation_id}")
            from_entity = str(
                decision.get("from_entity_id") or proposal.get("entity_id") or ""
            ).strip()
            if from_entity and from_entity != target_entity:
                proposal.setdefault("original_entity_id", from_entity)
                proposal["reassigned_from_entity_id"] = from_entity
            proposal["entity_id"] = target_entity
            if decision.get("name") is not None:
                proposal["name"] = str(decision.get("name") or target_entity)
            if decision.get("kind") is not None:
                proposal["kind"] = str(decision.get("kind") or proposal.get("kind") or "")
            if "description" in decision:
                proposal["description"] = str(decision.get("description") or "")
            replacement = decision.get("replacement")
            if isinstance(replacement, dict) and replacement.get("crop_path"):
                proposal.update(replacement)
            proposal["accepted"] = True
            proposal["review_reason"] = str(decision.get("reason") or "human_reassigned")
        else:
            replacement = decision.get("replacement")
            if not isinstance(replacement, dict) or not replacement.get("crop_path"):
                raise ValueError(f"replace requires a crop replacement for {representation_id}")
            proposal.update(replacement)
            proposal["accepted"] = True
            proposal["review_reason"] = str(decision.get("reason") or "human_replaced")
    return [by_id[rid] for rid in order if rid in by_id]


def apply_patch_files(*, proposals_path: Path, decisions_path: Path, out_path: Path) -> list[dict[str, Any]]:
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(decisions, dict):
        raise ValueError("crop decisions must be an object keyed by representation_id")
    accepted = apply_crop_decisions(proposals=proposals, decisions=decisions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    return accepted
