"""Resumable S1--S7 automation entrypoint.

Human review is normally required in S4/S6.  ``--skip-human`` exists only for
automation smoke tests and writes an explicit ``automation_smoke_only`` marker
to the generated gold.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.orchestration.contracts import (
    S4_MODES,
    resolve_s4_mode,
)
from vmem_bench.annotation.pipeline.stages.s2_annotation_postprocess.materialize import (
    postprocess_annotation,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (
    DEFAULT_MAX_REVIEW_ROUNDS,
    DEFAULT_MAX_TOKENS,
    PassthroughReviewer,
    build_qwen_reviewer,
    run_auto_review,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.sampling import (
    build_sample,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.decisions import (
    apply_s4_decisions,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    FirstCandidatePicker,
    QwenCropPicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.grounding_dino import (
    GroundingDinoProposer,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.identity_consistency import (
    VlmIdentityAuditor,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.run import (
    CROP_ROUTES,
    PROPOSERS,
    run_crop_acquisition,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_concept import (
    Sam3ConceptSegmenter,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_refine import (
    Sam3BoxPointRefiner,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_grounding import (
    FullFrameGrounder,
    QwenImageGrounder,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_sam_route import (
    ROUTE_NAME as ROUTE_VLM_SAM_REFINE,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.propose_pick_route import (
    ROUTE_NAME as ROUTE_PROPOSE_AND_PICK,
)
from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.auto_accept import (
    materialize_automation_review,
)
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.freeze import (
    continue_after_s6,
)

# S4 human audit: stratified sample (not full film).
_S4_SAMPLE_MINIMUM = 3
_S4_SAMPLE_RATE = 0.01


def _vlm_output(movie_dir: Path) -> Path:
    for filename in ("vlm_output.json", "vlm_outputs.json"):
        candidate = movie_dir / filename
        if candidate.is_file():
            return candidate
    return movie_dir / "vlm_output.json"


def _maybe_read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupted stage artifacts must not crash resume.
        return None


def auto_accept_pending_s4(movie_dir: Path) -> bool:
    """Accept the current S4 queue when batch automation explicitly permits it."""
    s4_dir = movie_dir / "tmp" / "pipeline" / "s4_segment_sampling_human_review"
    audit = _maybe_read_json(s4_dir / "review_audit.json") or {}
    if not isinstance(audit, dict) or audit.get("human_reviewed"):
        return False
    queue = _maybe_read_json(s4_dir / "review_queue.json") or []
    decisions = {
        str(item["segment_id"]): {"action": "accept"}
        for item in queue
        if isinstance(item, dict) and str(item.get("segment_id") or "")
    }
    if not decisions:
        return False
    apply_s4_decisions(
        movie_dir=movie_dir,
        decisions=decisions,
        film_verdict="accept",
        reason="batch_auto_accept_s4",
    )
    return True


def infer_resume_action(movie_dir: Path) -> str:
    """Decide how to continue one movie without discarding pipeline progress.

    Returns one of:
      ``complete`` — gold / S7 already published
      ``awaiting_human`` — S4 or S6 is open for human review
      ``after_s6`` — S6 approved; freeze/publish S7
      ``after_s4`` — S4 approved; run S5+ (or retry failed/partial S5)
      ``pipeline`` — start or resume S2→S3→S4 path (S3 uses ``resume=True``)
    """
    pipeline_root = movie_dir / "tmp" / "pipeline"
    s7_dir = pipeline_root / "s7_freeze_publish"
    if (movie_dir / "gold" / "entity_registry.json").is_file() or (
        s7_dir / "release_manifest.json"
    ).is_file():
        return "complete"

    s4_audit = _maybe_read_json(
        pipeline_root / "s4_segment_sampling_human_review" / "review_audit.json"
    ) or {}
    s6_audit = _maybe_read_json(
        pipeline_root / "s6_entities_visual_crop_human_review" / "review_audit.json"
    ) or {}
    s4_human = isinstance(s4_audit, dict) and bool(s4_audit.get("human_reviewed"))
    s6_human = isinstance(s6_audit, dict) and bool(s6_audit.get("human_reviewed"))
    s4_queue = pipeline_root / "s4_segment_sampling_human_review" / "review_queue.json"
    s6_queue = pipeline_root / "s6_entities_visual_crop_human_review" / "review_queue.json"
    s5_dir = pipeline_root / "s5_entities_visual_crop_acquisition"
    s5_proposals = s5_dir / "crop_proposals.json"
    s6_ready = s5_proposals.is_file() or s6_queue.is_file()

    if s6_human:
        return "after_s6"
    if s6_ready and not s6_human:
        return "awaiting_human"
    if s4_human and not s6_ready:
        return "after_s4"
    if s4_queue.is_file() and not s4_human:
        return "awaiting_human"
    return "pipeline"


def _write_state(root: Path, stage: str, status: str, **extra: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}
    current["stages"][stage] = {"status": status, **extra}
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_annotation_for_continue(pipeline_root: Path) -> dict[str, Any]:
    """Prefer human S4 → auto S3 → normalized S2 (skip-S3 crop path)."""
    candidates = (
        pipeline_root / "s4_segment_sampling_human_review" / "human_revised_annotation.json",
        pipeline_root / "s3_segment_auto_review_revise" / "auto_revised_annotation.json",
        pipeline_root / "s2_annotation_postprocess" / "normalized_annotation.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "missing annotation for continue "
        f"(tried S4/S3/S2 under {pipeline_root})"
    )


def _annotation_source_label(pipeline_root: Path) -> str:
    if (pipeline_root / "s4_segment_sampling_human_review" / "human_revised_annotation.json").is_file():
        return "s4_human"
    if (pipeline_root / "s3_segment_auto_review_revise" / "auto_revised_annotation.json").is_file():
        return "s3_auto"
    if (pipeline_root / "s2_annotation_postprocess" / "normalized_annotation.json").is_file():
        return "s2_normalized"
    return "missing"


def _make_grounder(grounder_mode: str, grounder_base_url: str, grounder_model: str):
    if grounder_mode == "qwen":
        if not grounder_base_url:
            raise ValueError("grounder_base_url is required for qwen grounder")
        return QwenImageGrounder(base_url=grounder_base_url, model=grounder_model), Sam3BoxPointRefiner()
    return FullFrameGrounder(), None


def _run_s5(
    *,
    annotation: dict[str, Any],
    source_video: Path,
    stage_dir: Path,
    crop_route: str,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    proposer: str,
    max_tasks: int | None,
    task_mode: str = "coverage",
    segmenter: Sam3ConceptSegmenter | None = None,
    detector: GroundingDinoProposer | None = None,
) -> list[dict[str, Any]]:
    # Same VLM endpoint powers the per-entity identity-consistency gate when running
    # against a real backend; dry-run (full-frame / first) falls back to DINOv3 flags.
    identity_auditor = (
        VlmIdentityAuditor(base_url=grounder_base_url, model=grounder_model)
        if grounder_mode == "qwen" and grounder_base_url
        else None
    )
    if crop_route == ROUTE_VLM_SAM_REFINE:
        grounder, refiner = _make_grounder(grounder_mode, grounder_base_url, grounder_model)
        return run_crop_acquisition(
            annotation=annotation,
            source_video=source_video,
            stage_dir=stage_dir,
            crop_route=crop_route,
            grounder=grounder,
            refiner=refiner,
            require_sam3=grounder_mode == "qwen",
            max_tasks=max_tasks,
            task_mode=task_mode,  # type: ignore[arg-type]
            identity_auditor=identity_auditor,
        )
    if crop_route != ROUTE_PROPOSE_AND_PICK:
        raise ValueError(f"unknown crop_route: {crop_route}")
    if grounder_mode == "qwen":
        if not grounder_base_url:
            raise ValueError("grounder_base_url is required for qwen picker")
        picker = QwenCropPicker(base_url=grounder_base_url, model=grounder_model)
    else:
        picker = FirstCandidatePicker()
    if segmenter is None and proposer in ("sam3", "fusion"):
        segmenter = Sam3ConceptSegmenter()
    if detector is None and proposer in ("gdino", "fusion"):
        detector = GroundingDinoProposer()
    return run_crop_acquisition(
        annotation=annotation,
        source_video=source_video,
        stage_dir=stage_dir,
        crop_route=crop_route,
        picker=picker,
        proposer=proposer,
        segmenter=segmenter,
        detector=detector,
        max_tasks=max_tasks,
        task_mode=task_mode,  # type: ignore[arg-type]
        identity_auditor=identity_auditor,
    )


def continue_after_s4(
    *,
    movie_dir: Path,
    source_video: Path,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    skip_human: bool,
    max_tasks: int | None,
    s4_mode: str = "auto",
    crop_route: str = ROUTE_PROPOSE_AND_PICK,
    proposer: str = "sam3",
    task_mode: str = "coverage",
    s5_segmenter: Sam3ConceptSegmenter | None = None,
    s5_detector: GroundingDinoProposer | None = None,
) -> dict[str, Any]:
    """Resume from S5 after S4 human review; stop at S6 unless skip_human."""
    pipeline_root = movie_dir / "tmp" / "pipeline"
    s4_audit = pipeline_root / "s4_segment_sampling_human_review" / "review_audit.json"
    audit: dict[str, Any] = {}
    if s4_audit.is_file():
        audit = json.loads(s4_audit.read_text(encoding="utf-8"))
        if audit.get("film_verdict") == "reject_for_reannotation":
            return {
                "status": "rejected_for_reannotation",
                "movie_dir": str(movie_dir),
                "reason": audit.get("reason") or "",
            }
        if audit.get("blocks_pipeline"):
            return {
                "status": "s4_retry_requested",
                "movie_dir": str(movie_dir),
                "n_retry_requested": int(audit.get("n_retry_requested") or 0),
            }
    if s4_mode == "blocking" and not skip_human and not audit.get("human_reviewed"):
        return {
            "status": "awaiting_segment_human_review",
            "movie_dir": str(movie_dir),
            "review_dir": str(
                pipeline_root / "s4_segment_sampling_human_review"
            ),
            "s4_mode": s4_mode,
        }
    if (
        s4_mode == "blocking"
        and crop_route == ROUTE_PROPOSE_AND_PICK
        and grounder_mode != "qwen"
    ):
        raise ValueError(
            "production blocking S5 requires grounder=qwen so propose_and_pick "
            "uses the closed-set crop picker; FirstCandidatePicker is debug-only"
        )
    annotation = _load_annotation_for_continue(pipeline_root)
    s5_dir = pipeline_root / "s5_entities_visual_crop_acquisition"
    _write_state(
        pipeline_root,
        "s5_entities_visual_crop_acquisition",
        "running",
        crop_route=crop_route,
        task_mode=task_mode,
        proposer=proposer if crop_route == ROUTE_PROPOSE_AND_PICK else None,
        annotation_source=_annotation_source_label(pipeline_root),
        s4_mode=s4_mode,
    )
    try:
        proposals = _run_s5(
            annotation=annotation,
            source_video=source_video,
            stage_dir=s5_dir,
            crop_route=crop_route,
            grounder_mode=grounder_mode,
            grounder_base_url=grounder_base_url,
            grounder_model=grounder_model,
            proposer=proposer,
            max_tasks=max_tasks,
            task_mode=task_mode,
            segmenter=s5_segmenter,
            detector=s5_detector,
        )
    except Exception as exc:
        _write_state(
            pipeline_root,
            "s5_entities_visual_crop_acquisition",
            "failed",
            error_type=type(exc).__name__,
            error=str(exc)[:800],
        )
        raise
    _write_state(
        pipeline_root,
        "s5_entities_visual_crop_acquisition",
        "ok",
        n_proposals=len(proposals),
        crop_route=crop_route,
        task_mode=task_mode,
        proposer=proposer if crop_route == ROUTE_PROPOSE_AND_PICK else None,
        annotation_source=_annotation_source_label(pipeline_root),
        s4_mode=s4_mode,
    )

    s6_dir = pipeline_root / "s6_entities_visual_crop_human_review"
    if skip_human:
        accepted = materialize_automation_review(
            proposals_path=s5_dir / "crop_proposals.json",
            out_dir=s6_dir,
            accept_all=False,
        )
        _write_state(
            pipeline_root,
            "s6_entities_visual_crop_human_review",
            "automation_smoke",
            n_accepted=len(accepted),
        )
        return continue_after_s6(
            movie_dir=movie_dir,
            automation_smoke=True,
        )

    from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.review_apply import (
        ensure_s6_queue,
    )

    queue = ensure_s6_queue(movie_dir)
    _write_state(
        pipeline_root,
        "s6_entities_visual_crop_human_review",
        "awaiting_human",
        n_queue=len(queue),
    )
    return {
        "status": "awaiting_crop_human_review",
        "movie_dir": str(movie_dir),
        "review_dir": str(s6_dir),
        "n_queue": len(queue),
    }


def run_pipeline(
    *,
    movie_dir: Path,
    source_video: Path,
    reviewer_mode: str,
    reviewer_base_url: str,
    reviewer_model: str,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    skip_human: bool,
    max_tasks: int | None,
    s4_mode: str = "auto",
    max_review_rounds: int = DEFAULT_MAX_REVIEW_ROUNDS,
    max_review_workers: int | None = None,
    max_clip_workers: int | None = None,
    clip_queue_root: Path | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    crop_route: str = ROUTE_PROPOSE_AND_PICK,
    proposer: str = "sam3",
    task_mode: str = "coverage",
    resume: bool = False,
    auto_accept_s4: bool = False,
    reviewer_override: Any | None = None,
    endpoint_pool_override: Any | None = None,
    s5_segmenter: Sam3ConceptSegmenter | None = None,
    s5_detector: GroundingDinoProposer | None = None,
) -> dict[str, Any]:
    """Run S2--S7 for one movie, returning an explicit status payload.

    When ``resume=True``, S3 keeps existing ``segment_audit.jsonl`` progress
    instead of truncating and re-reviewing every segment.
    """
    pipeline_root = movie_dir / "tmp" / "pipeline"
    effective_s4_mode = resolve_s4_mode(s4_mode, skip_human=skip_human)
    input_path = _vlm_output(movie_dir)
    s2_dir = pipeline_root / "s2_annotation_postprocess"
    s2 = postprocess_annotation(input_path, s2_dir)
    _write_state(
        pipeline_root,
        "s2_annotation_postprocess",
        s2["status"],
        **{key: value for key, value in s2.items() if key != "status"},
    )
    if s2["status"] == "skipped_empty_input":
        return {"status": "skipped_empty_input", "movie_dir": str(movie_dir)}
    if s2["status"] != "ok":
        return {"status": s2["status"], "movie_dir": str(movie_dir), "stage": "s2"}

    endpoint_pool = endpoint_pool_override
    if reviewer_override is not None:
        reviewer = reviewer_override
    elif reviewer_mode == "qwen":
        if not reviewer_base_url:
            raise ValueError("reviewer_base_url is required for qwen reviewer")
        reviewer, endpoint_pool = build_qwen_reviewer(
            base_urls=reviewer_base_url, model=reviewer_model, max_tokens=max_tokens
        )
    else:
        reviewer = PassthroughReviewer()
    s3_dir = pipeline_root / "s3_segment_auto_review_revise"
    s3 = run_auto_review(
        annotation_payload=json.loads(Path(s2["normalized_annotation"]).read_text(encoding="utf-8")),
        source_video=source_video,
        stage_dir=s3_dir,
        reviewer=reviewer,
        max_review_rounds=max_review_rounds,
        endpoint_pool=endpoint_pool,
        max_workers=max_review_workers,
        max_clip_workers=max_clip_workers,
        clip_queue_root=clip_queue_root,
        resume=resume,
    )
    verdict_counts: dict[str, int] = {}
    for item in s3["reviews"]:
        verdict = str(item.get("verdict") or ("PASS" if item.get("accepted") else "WARN"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    _write_state(
        pipeline_root,
        "s3_segment_auto_review_revise",
        "ok",
        n_segments=len(s3["reviews"]),
        n_accepted=sum(1 for item in s3["reviews"] if item.get("accepted")),
        verdict_counts=verdict_counts,
        max_review_rounds=max_review_rounds,
        max_tokens=max_tokens,
        n_endpoints=(endpoint_pool.size if endpoint_pool is not None else 0),
    )
    s4_dir = pipeline_root / "s4_segment_sampling_human_review"
    # All BLOCK findings are actionable S4 decisions; WARN/PASS are sampled.
    # RETRYABLE_ERROR stays on the automatic retry surface.
    sample_seed = zlib.adler32(movie_dir.name.encode("utf-8")) & 0x7FFFFFFF
    queue = build_sample(
        list(s3["reviews"]),
        minimum=_S4_SAMPLE_MINIMUM,
        rate=_S4_SAMPLE_RATE,
        seed=sample_seed,
    )
    s4_dir.mkdir(parents=True, exist_ok=True)
    (s4_dir / "review_queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (s4_dir / "review_audit.json").write_text(
        json.dumps(
            {
                "mode": "automation_smoke_only" if skip_human else "open_for_human_review",
                "human_reviewed": bool(skip_human),
                "blocks_pipeline": False,
                "s4_mode_requested": s4_mode,
                "s4_mode_effective": effective_s4_mode,
                "minimum_segments": _S4_SAMPLE_MINIMUM,
                "sample_rate": _S4_SAMPLE_RATE,
                "n_total_segments": len(s3["reviews"]),
                "sampled_segments": len(queue),
                "queue_policy": "all_block_plus_warn_pass_sample",
                "verdict_counts": verdict_counts,
                "n_block_queued": sum(
                    1 for item in queue if str(item.get("verdict") or "") == "BLOCK"
                ),
                "n_warn_sampled": sum(
                    1 for item in queue if str(item.get("verdict") or "") == "WARN"
                ),
                "n_pass_sampled": sum(
                    1 for item in queue if str(item.get("verdict") or "") == "PASS"
                ),
                "sample_seed": sample_seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_state(
        pipeline_root,
        "s4_segment_sampling_human_review",
        "automation_smoke" if skip_human else "open_for_human_review",
        n_sampled=len(queue),
        n_total=len(s3["reviews"]),
        verdict_counts=verdict_counts,
        s4_mode_requested=s4_mode,
        s4_mode_effective=effective_s4_mode,
    )

    if effective_s4_mode == "blocking":
        if auto_accept_s4:
            auto_accept_pending_s4(movie_dir)
            return continue_after_s4(
                movie_dir=movie_dir,
                source_video=source_video,
                grounder_mode=grounder_mode,
                grounder_base_url=grounder_base_url,
                grounder_model=grounder_model,
                skip_human=skip_human,
                max_tasks=max_tasks,
                s4_mode=effective_s4_mode,
                crop_route=crop_route,
                proposer=proposer,
                task_mode=task_mode,
                s5_segmenter=s5_segmenter,
                s5_detector=s5_detector,
            )
        return {
            "status": "awaiting_segment_human_review",
            "movie_dir": str(movie_dir),
            "review_dir": str(s4_dir),
            "n_queue": len(queue),
            "s4_mode": effective_s4_mode,
        }

    return continue_after_s4(
        movie_dir=movie_dir,
        source_video=source_video,
        grounder_mode=grounder_mode,
        grounder_base_url=grounder_base_url,
        grounder_model=grounder_model,
        skip_human=skip_human,
        max_tasks=max_tasks,
        s4_mode=effective_s4_mode,
        crop_route=crop_route,
        proposer=proposer,
        task_mode=task_mode,
        s5_segmenter=s5_segmenter,
        s5_detector=s5_detector,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--reviewer", choices=("passthrough", "qwen"), default="passthrough")
    parser.add_argument(
        "--reviewer-base-url",
        default="",
        help="Qwen reviewer URL, or comma-separated H800 endpoint pool",
    )
    parser.add_argument("--reviewer-model", default="qwen3-vl-32b")
    parser.add_argument("--max-review-rounds", type=int, default=DEFAULT_MAX_REVIEW_ROUNDS)
    parser.add_argument("--max-review-workers", type=int, default=None)
    parser.add_argument("--max-clip-workers", type=int, default=None)
    parser.add_argument("--clip-queue-root", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--grounder", choices=("full-frame", "qwen"), default="full-frame")
    parser.add_argument("--grounder-base-url", default="")
    parser.add_argument("--grounder-model", default="qwen3-vl-32b")
    parser.add_argument(
        "--crop-route",
        choices=list(CROP_ROUTES),
        default=ROUTE_PROPOSE_AND_PICK,
        help="S5 ablation route: vlm_sam_refine | propose_and_pick (default: propose_and_pick)",
    )
    parser.add_argument(
        "--proposer",
        choices=list(PROPOSERS),
        default="sam3",
        help="Detector source when --crop-route propose_and_pick",
    )
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument(
        "--auto-accept-s4",
        action="store_true",
        help="accept pending and newly generated S4 queues, then continue to S5",
    )
    parser.add_argument(
        "--s4-mode",
        choices=S4_MODES,
        default="auto",
        help="S4→S5 gate: auto=blocking for production, nonblocking for skip-human debug",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--task-mode",
        choices=("coverage", "per_slot"),
        default="coverage",
        help="S5 planning: coverage library + ≤t slot bind (default), or legacy per_slot",
    )
    parser.add_argument(
        "--continue-from",
        choices=("", "after_s4", "s5_only", "after_s6"),
        default="",
        help=(
            "Resume helpers: after_s4 (S5+), s5_only (skip S3/S4; use S4>S3>S2 annotation), "
            "after_s6 (freeze)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep S3 segment_audit progress when running the full S2→S4 path",
    )
    args = parser.parse_args()
    if args.continue_from in ("after_s4", "s5_only"):
        continue_skip_human = args.skip_human or args.continue_from == "s5_only"
        result = continue_after_s4(
            movie_dir=args.movie_dir,
            source_video=args.source_video,
            grounder_mode=args.grounder,
            grounder_base_url=args.grounder_base_url,
            grounder_model=args.grounder_model,
            skip_human=continue_skip_human,
            max_tasks=args.max_tasks,
            s4_mode=resolve_s4_mode(
                args.s4_mode,
                skip_human=continue_skip_human,
            ),
            crop_route=args.crop_route,
            proposer=args.proposer,
            task_mode=args.task_mode,
        )
    elif args.continue_from == "after_s6":
        result = continue_after_s6(movie_dir=args.movie_dir, automation_smoke=False)
    else:
        result = run_pipeline(
            movie_dir=args.movie_dir,
            source_video=args.source_video,
            reviewer_mode=args.reviewer,
            reviewer_base_url=args.reviewer_base_url,
            reviewer_model=args.reviewer_model,
            grounder_mode=args.grounder,
            grounder_base_url=args.grounder_base_url,
            grounder_model=args.grounder_model,
            skip_human=args.skip_human,
            max_tasks=args.max_tasks,
            s4_mode=args.s4_mode,
            max_review_rounds=args.max_review_rounds,
            max_review_workers=args.max_review_workers,
            max_clip_workers=args.max_clip_workers,
            clip_queue_root=args.clip_queue_root,
            max_tokens=args.max_tokens,
            crop_route=args.crop_route,
            proposer=args.proposer,
            task_mode=args.task_mode,
            resume=bool(args.resume),
            auto_accept_s4=bool(args.auto_accept_s4),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()