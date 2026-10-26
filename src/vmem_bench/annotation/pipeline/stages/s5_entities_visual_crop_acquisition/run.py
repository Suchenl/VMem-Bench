"""Run S5 crop acquisition with selectable ablation routes.

Default ``task_mode=coverage``: acquire a capped per-entity visual library, prune
by attribute diversity, then expand to every present ``(chunk, entity)`` slot via
≤t ``current_appearance`` binding so S6/S7 stay slot-compatible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.coverage_expand import (
    expand_library_to_slots,
    prune_library_by_attributes,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    FirstCandidatePicker,
    QwenCropPicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.grounding_dino import (
    GroundingDinoProposer,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.identity_consistency import (
    IdentityAuditor,
    VlmIdentityAuditor,
    run_identity_consistency,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.propose_pick_route import (
    ROUTE_NAME as ROUTE_PROPOSE_AND_PICK,
    run_propose_and_pick,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_concept import (
    Sam3ConceptSegmenter,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_refine import (
    Sam3BoxPointRefiner,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CoverageCaps,
    TaskMode,
    plan_tasks,
    write_tasks,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_grounding import (
    FullFrameGrounder,
    Grounder,
    QwenImageGrounder,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_sam_route import (
    ROUTE_NAME as ROUTE_VLM_SAM_REFINE,
    run_vlm_sam_refine,
)

CROP_ROUTES = (ROUTE_VLM_SAM_REFINE, ROUTE_PROPOSE_AND_PICK)
PROPOSERS = ("sam3", "gdino", "fusion")
TASK_MODES = ("coverage", "per_slot")


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_crop_acquisition(
    *,
    annotation: dict[str, Any],
    source_video: Path,
    stage_dir: Path,
    crop_route: str = ROUTE_PROPOSE_AND_PICK,
    grounder: Grounder | None = None,
    refiner: Sam3BoxPointRefiner | None = None,
    require_sam3: bool = False,
    picker=None,
    proposer: str = "sam3",
    segmenter: Sam3ConceptSegmenter | None = None,
    detector: GroundingDinoProposer | None = None,
    max_tasks: int | None = None,
    max_candidates: int = 4,
    task_mode: TaskMode = "coverage",
    coverage_caps: CoverageCaps | None = None,
    identity_auditor: IdentityAuditor | None = None,
    identity_embedder: Any | None = None,
) -> list[dict[str, Any]]:
    """Acquire crops and write slot-complete ``crop_proposals.json``."""
    if crop_route not in CROP_ROUTES:
        raise ValueError(f"unknown crop_route: {crop_route} (expected {CROP_ROUTES})")
    if task_mode not in TASK_MODES:
        raise ValueError(f"unknown task_mode: {task_mode} (expected {TASK_MODES})")

    stage_dir.mkdir(parents=True, exist_ok=True)
    caps = coverage_caps or CoverageCaps()
    acquire_tasks, plan_stats = plan_tasks(annotation, mode=task_mode, caps=caps)
    if max_tasks is not None:
        acquire_tasks = acquire_tasks[:max_tasks]
        plan_stats = {**plan_stats, "max_tasks_applied": max_tasks, "n_acquire": len(acquire_tasks)}

    write_tasks(acquire_tasks, stage_dir / "crop_tasks.json")
    _write_json_atomic(stage_dir / "coverage_plan.json", plan_stats)
    _write_json_atomic(
        stage_dir / "route.json",
        {
            "crop_route": crop_route,
            "task_mode": task_mode,
            "proposer": proposer if crop_route == ROUTE_PROPOSE_AND_PICK else None,
            "require_sam3": require_sam3 if crop_route == ROUTE_VLM_SAM_REFINE else None,
            "coverage_caps": {
                "character": caps.character,
                "prop": caps.prop,
                "location": caps.location,
                "max_total_acquire": caps.max_total_acquire,
            },
        },
    )
    live_path = stage_dir / "crop_acquisition_live.json"
    progress_path = stage_dir / "crop_acquisition_progress.json"

    def persist_live(
        acquired_so_far: list[dict[str, Any]],
        *,
        status: str = "running",
        phase: str | None = None,
        current: dict[str, Any] | None = None,
    ) -> None:
        accepted = sum(
            1
            for proposal in acquired_so_far
            if proposal.get("accepted") and proposal.get("crop_path")
        )
        by_kind = Counter(str(p.get("kind") or "?") for p in acquired_so_far)
        by_entity: dict[str, dict[str, Any]] = {}
        for proposal in acquired_so_far:
            eid = str(proposal.get("entity_id") or proposal.get("name") or "?")
            row = by_entity.setdefault(
                eid,
                {
                    "entity_id": eid,
                    "name": str(proposal.get("name") or eid),
                    "kind": str(proposal.get("kind") or ""),
                    "done": 0,
                    "accepted": 0,
                },
            )
            row["done"] += 1
            if proposal.get("accepted") and proposal.get("crop_path"):
                row["accepted"] += 1
        last = acquired_so_far[-1] if acquired_so_far else {}
        resolved_current = current or (
            {
                "entity_id": last.get("entity_id"),
                "name": last.get("name"),
                "kind": last.get("kind"),
                "chunk_id": last.get("chunk_id"),
                "accepted": last.get("accepted"),
            }
            if last
            else None
        )
        resolved_phase = phase or (
            "completed"
            if status == "completed"
            else ("acquiring" if acquired_so_far else "planned")
        )
        _write_json_atomic(live_path, acquired_so_far)
        _write_json_atomic(
            progress_path,
            {
                "status": status,
                "phase": resolved_phase,
                "done": len(acquired_so_far),
                "total": len(acquire_tasks),
                "accepted": accepted,
                "pct": (
                    round(100.0 * len(acquired_so_far) / len(acquire_tasks), 1)
                    if acquire_tasks
                    else 0.0
                ),
                "current": resolved_current,
                "by_kind": dict(by_kind),
                "by_entity": list(by_entity.values()),
                "n_entities_touched": len(by_entity),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    persist_live([], phase="planned")

    if crop_route == ROUTE_VLM_SAM_REFINE:
        if grounder is None:
            raise ValueError("grounder is required for vlm_sam_refine")
        persist_live([], phase="loading_models")
        acquired = run_vlm_sam_refine(
            tasks=acquire_tasks,
            source_video=source_video,
            stage_dir=stage_dir,
            grounder=grounder,
            refiner=refiner,
            require_sam3=require_sam3,
        )
        persist_live(acquired, phase="acquiring")
    else:
        if picker is None:
            raise ValueError("picker is required for propose_and_pick")
        acquired_box: list[dict[str, Any]] = []

        def _on_task_start(task: Any, *, index: int, total: int) -> None:
            persist_live(
                acquired_box,
                phase="loading_models" if index == 0 else "acquiring",
                current={
                    "entity_id": getattr(task, "entity_id", None),
                    "name": getattr(task, "name", None),
                    "kind": getattr(task, "kind", None),
                    "chunk_id": getattr(task, "chunk_id", None),
                    "index": index,
                    "total": total,
                },
            )

        def _on_task_complete(proposals: list[dict[str, Any]]) -> None:
            acquired_box[:] = proposals
            persist_live(proposals, phase="acquiring")

        persist_live([], phase="loading_models")
        acquired = run_propose_and_pick(
            tasks=acquire_tasks,
            source_video=source_video,
            stage_dir=stage_dir,
            picker=picker,
            proposer=proposer,
            segmenter=segmenter,
            detector=detector,
            max_candidates=max_candidates,
            on_task_start=_on_task_start,
            on_task_complete=_on_task_complete,
        )

    # Tag acquires; prune attribute-diverse library in coverage mode.
    for proposal in acquired:
        proposal.setdefault("task_kind", "acquire")

    # WHO-consistency gate: confirm each entity's library crops share one identity
    # before diversity pruning and ≤t slot binding can propagate a mixed library.
    acquired, identity_audit = run_identity_consistency(
        acquired,
        embedder=identity_embedder,
        auditor=identity_auditor,
    )
    _write_json_atomic(stage_dir / "identity_audit.json", identity_audit)

    library = acquired
    if task_mode == "coverage":
        library = prune_library_by_attributes(acquired, caps=caps)
        (stage_dir / "crop_library.json").write_text(
            json.dumps(
                [p for p in library if p.get("accepted") and p.get("crop_path")],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        proposals = expand_library_to_slots(annotation=annotation, library_proposals=library)
        plan_stats = {
            **plan_stats,
            "n_library_accepted": sum(
                1 for p in library if p.get("accepted") and p.get("crop_path")
            ),
            "n_slot_proposals": len(proposals),
            "n_slot_bound": sum(1 for p in proposals if p.get("task_kind") == "slot_bind"),
            "n_slot_missing": sum(1 for p in proposals if not p.get("accepted")),
        }
        _write_json_atomic(stage_dir / "coverage_plan.json", plan_stats)
    else:
        proposals = acquired

    _write_json_atomic(stage_dir / "crop_proposals.json", proposals)
    persist_live(acquired, status="completed")
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument(
        "--crop-route",
        choices=CROP_ROUTES,
        default=ROUTE_PROPOSE_AND_PICK,
        help="Ablation route: VLM box+SAM refine, or detector propose + VLM pick",
    )
    parser.add_argument(
        "--task-mode",
        choices=TASK_MODES,
        default="coverage",
        help="coverage=capped library + ≤t slot bind (default); per_slot=legacy full Cartesian",
    )
    parser.add_argument("--cap-character", type=int, default=8)
    parser.add_argument("--cap-prop", type=int, default=5)
    parser.add_argument("--cap-location", type=int, default=12)
    parser.add_argument("--max-total-acquire", type=int, default=None)
    parser.add_argument("--grounder", choices=("full-frame", "qwen"), default="full-frame")
    parser.add_argument("--sam3-refine", action="store_true",
                        help="enable SAM3 refine on vlm_sam_refine (implied by --grounder qwen)")
    parser.add_argument("--require-sam3", action="store_true",
                        help="fail closed when SAM3 refine is missing/empty (vlm_sam_refine)")
    parser.add_argument(
        "--allow-vlm-bbox-fallback",
        action="store_true",
        help="on vlm_sam_refine+qwen, keep loose VLM bbox when SAM3 fails",
    )
    parser.add_argument(
        "--proposer",
        choices=PROPOSERS,
        default="sam3",
        help="Detector source for propose_and_pick",
    )
    parser.add_argument("--picker", choices=("first", "qwen"), default="first")
    parser.add_argument("--base-url", default="", help="Qwen OpenAI-compatible base URL")
    parser.add_argument("--model", default="Qwen3VL-32B-Instruct")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--no-identity-audit",
        action="store_true",
        help="disable the per-entity VLM identity-consistency gate (DINOv3-only flagging)",
    )
    parser.add_argument(
        "--identity-model",
        default="",
        help="VLM model for the identity gate (defaults to --model)",
    )
    args = parser.parse_args()

    grounder: Grounder | None = None
    refiner: Sam3BoxPointRefiner | None = None
    picker = None
    segmenter = None
    detector = None
    require_sam3 = bool(args.require_sam3)

    if args.crop_route == ROUTE_VLM_SAM_REFINE:
        if args.grounder == "qwen":
            if not args.base_url:
                parser.error("--base-url is required for --grounder qwen")
            grounder = QwenImageGrounder(base_url=args.base_url, model=args.model)
            refiner = Sam3BoxPointRefiner()
            require_sam3 = not bool(args.allow_vlm_bbox_fallback)
        else:
            grounder = FullFrameGrounder()
            if args.sam3_refine:
                refiner = Sam3BoxPointRefiner()
                require_sam3 = bool(args.require_sam3)
    else:
        if args.picker == "qwen":
            if not args.base_url:
                parser.error("--base-url is required for --picker qwen")
            picker = QwenCropPicker(base_url=args.base_url, model=args.model)
        else:
            picker = FirstCandidatePicker()
        if args.proposer in ("sam3", "fusion"):
            segmenter = Sam3ConceptSegmenter()
        if args.proposer in ("gdino", "fusion"):
            detector = GroundingDinoProposer()

    identity_auditor: IdentityAuditor | None = None
    if not args.no_identity_audit and args.base_url:
        identity_auditor = VlmIdentityAuditor(
            base_url=args.base_url,
            model=args.identity_model or args.model,
        )

    run_crop_acquisition(
        annotation=json.loads(args.annotation.read_text(encoding="utf-8")),
        source_video=args.source_video,
        stage_dir=args.stage_dir,
        crop_route=args.crop_route,
        grounder=grounder,
        refiner=refiner,
        require_sam3=require_sam3,
        picker=picker,
        proposer=args.proposer,
        segmenter=segmenter,
        detector=detector,
        max_tasks=args.max_tasks,
        max_candidates=args.max_candidates,
        task_mode=args.task_mode,
        coverage_caps=CoverageCaps(
            character=args.cap_character,
            prop=args.cap_prop,
            location=args.cap_location,
            max_total_acquire=args.max_total_acquire,
        ),
        identity_auditor=identity_auditor,
    )


if __name__ == "__main__":
    main()
