"""Synchronous, deterministic S7 freeze entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.build_gold import (
    build_gold,
)
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.gates import (
    soft_s3_residuals,
    unresolved_s3_blockers,
)


def _load_annotation_for_freeze(pipeline_root: Path) -> dict[str, Any]:
    """Prefer human S4 → auto S3 → normalized S2 annotation."""
    candidates = (
        pipeline_root / "s4_segment_sampling_human_review" / "human_revised_annotation.json",
        pipeline_root / "s3_segment_auto_review_revise" / "auto_revised_annotation.json",
        pipeline_root / "s2_annotation_postprocess" / "normalized_annotation.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "missing annotation for freeze "
        f"(tried S4/S3/S2 under {pipeline_root})"
    )


def _write_freeze_state(root: Path, status: str, **extra: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}
    current["stages"]["s7_freeze_publish"] = {"status": status, **extra}
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def continue_after_s6(*, movie_dir: Path, automation_smoke: bool = False) -> dict[str, Any]:
    """Freeze/publish gold from human-accepted or smoke-accepted crops."""
    pipeline_root = movie_dir / "tmp" / "pipeline"
    if not automation_smoke:
        s4_audit_path = pipeline_root / "s4_segment_sampling_human_review" / "review_audit.json"
        s4_audit = json.loads(s4_audit_path.read_text(encoding="utf-8")) if s4_audit_path.is_file() else {}
        if not s4_audit.get("human_reviewed"):
            raise ValueError("S4 human review is required before freeze (pipeline was not blocked, but freeze is)")
        s6_audit_path = pipeline_root / "s6_entities_visual_crop_human_review" / "review_audit.json"
        s6_audit = json.loads(s6_audit_path.read_text(encoding="utf-8")) if s6_audit_path.is_file() else {}
        if not s6_audit.get("human_reviewed"):
            raise ValueError("S6 human review is required before freeze")
        unresolved = unresolved_s3_blockers(pipeline_root)
        if unresolved:
            preview = ", ".join(
                f"{item['segment_id']}:{item['verdict']}" for item in unresolved[:8]
            )
            raise ValueError(
                f"S3 blockers must be cleared before freeze ({len(unresolved)} unresolved; {preview})"
            )
        soft = soft_s3_residuals(pipeline_root)
    else:
        unresolved = unresolved_s3_blockers(pipeline_root)
        soft = soft_s3_residuals(pipeline_root)
    annotation = _load_annotation_for_freeze(pipeline_root)
    accepted_path = pipeline_root / "s6_entities_visual_crop_human_review" / "accepted_crops.json"
    if not accepted_path.is_file():
        raise FileNotFoundError(f"missing accepted crops: {accepted_path}")
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    gold = build_gold(
        movie_dir=movie_dir,
        annotation=annotation,
        accepted_crops=accepted,
        automation_smoke=automation_smoke,
    )
    s7_dir = pipeline_root / "s7_freeze_publish"
    s7_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary = {
        "movie_id": movie_dir.name,
        "mode": "automation_smoke_only" if automation_smoke else "human_reviewed",
        "n_segments": sum(
            len(scene.get("visual_segments") or [])
            for scene in (annotation.get("screenplay") or {}).get("scenes") or []
        ),
        "n_accepted_crops": len(accepted),
        "n_unresolved_s3_blockers": len(unresolved),
        "n_soft_s3_residuals": len(soft),
        "soft_s3_residuals": soft,
        "gold": str(gold),
    }
    (s7_dir / "candidate_summary.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    strict_lint = (
        {
            "status": "automation_smoke_skipped_strict_human_gate",
            "reason": "skip_human/automation smoke path",
        }
        if automation_smoke
        else {
            "status": "human_reviewed_ready",
            "reason": "S3 BLOCK cleared; S4/S6 human review applied; RETRYABLE residuals are soft",
            "n_unresolved_s3_blockers": 0,
            "n_soft_s3_residuals": len(soft),
        }
    )
    (s7_dir / "strict_lint.json").write_text(
        json.dumps(strict_lint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    release_manifest = json.loads((gold / "manifest.json").read_text(encoding="utf-8"))
    release_manifest["gold"] = str(gold)
    (s7_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_freeze_state(
        pipeline_root,
        "automation_smoke" if automation_smoke else "human_reviewed",
        gold=str(gold),
        candidate_summary=str(s7_dir / "candidate_summary.json"),
        strict_lint=str(s7_dir / "strict_lint.json"),
        release_manifest=str(s7_dir / "release_manifest.json"),
    )
    return {
        "status": "automation_smoke_complete" if automation_smoke else "human_reviewed_complete",
        "movie_dir": str(movie_dir),
        "gold": str(gold),
    }
