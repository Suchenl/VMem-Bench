"""Lightweight S7 freeze gates over S3/S4 review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _final_s3_verdicts(pipeline_root: Path) -> list[dict[str, str]]:
    """Return per-segment final verdicts after S4 resolved_verdicts overrides."""
    reviews = _read_jsonl(
        pipeline_root / "s3_segment_auto_review_revise" / "segment_audit.jsonl"
    )
    patch_path = (
        pipeline_root
        / "s4_segment_sampling_human_review"
        / "review_patch.applied.json"
    )
    patch = json.loads(patch_path.read_text(encoding="utf-8")) if patch_path.is_file() else {}
    resolved = dict(patch.get("resolved_verdicts") or {})
    rows: list[dict[str, str]] = []
    for review in reviews:
        segment_id = str(review.get("segment_id") or "")
        verdict = str(
            review.get("verdict")
            or ("PASS" if review.get("accepted") else "BLOCK")
        )
        final_verdict = str(resolved.get(segment_id) or verdict)
        rows.append(
            {
                "segment_id": segment_id,
                "verdict": final_verdict,
                "recommended_action": str(review.get("recommended_action") or "review"),
            }
        )
    return rows


def unresolved_s3_blockers(pipeline_root: Path) -> list[dict[str, str]]:
    """Hard freeze blockers: content ``BLOCK`` only.

    ``RETRYABLE_ERROR`` is an infrastructure/auto-retry signal and is excluded from
    the S4 human queue, so it must not hard-block S7 after S4/S6 completed.
    Soft residuals are reported separately via ``soft_s3_residuals``.
    """
    return [
        item
        for item in _final_s3_verdicts(pipeline_root)
        if item["verdict"] == "BLOCK"
    ]


def soft_s3_residuals(pipeline_root: Path) -> list[dict[str, str]]:
    """Non-blocking residuals recorded into the freeze summary (e.g. RETRYABLE)."""
    return [
        item
        for item in _final_s3_verdicts(pipeline_root)
        if item["verdict"] == "RETRYABLE_ERROR"
    ]
