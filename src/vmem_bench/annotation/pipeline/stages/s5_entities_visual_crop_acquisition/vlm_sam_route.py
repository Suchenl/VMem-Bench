"""Route A: VLM proposes bbox+point, SAM3 refines to a tight mask box."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_qa import (
    audit_crop,
    materialize_crop,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    materialize_unmasked_companion,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.attach_attributes import (
    attach_crop_attributes,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.keyframes import (
    FrameCandidate,
    extract_candidates,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_refine import (
    Sam3BoxPointRefiner,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_grounding import (
    FullFrameGrounder,
    Grounder,
)

ROUTE_NAME = "vlm_sam_refine"


def _acquire_one(
    *,
    task: CropTask,
    source_video: Path,
    stage_dir: Path,
    grounder: Grounder,
    refiner: Sam3BoxPointRefiner | None,
    require_sam3: bool,
    candidate_count: int = 5,
    keep_count: int = 3,
) -> dict[str, Any]:
    frame_dir = stage_dir / "frames" / f"c{task.chunk_id:05d}" / task.entity_id
    candidates = extract_candidates(
        source_video=source_video,
        start_seconds=task.start_seconds,
        end_seconds=task.end_seconds,
        out_dir=frame_dir,
        candidate_count=candidate_count,
        keep_count=keep_count,
    )
    best: dict[str, Any] | None = None
    for candidate in candidates:
        proposal = _try_candidate(
            task=task,
            candidate=candidate,
            stage_dir=stage_dir,
            grounder=grounder,
            refiner=refiner,
            require_sam3=require_sam3,
        )
        if proposal is None:
            continue
        if best is None or proposal["qa"]["sharpness"] > best["qa"]["sharpness"]:
            best = proposal
    if best is None:
        return {
            **task.to_dict(),
            "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
            "route": ROUTE_NAME,
            "accepted": False,
            "reason": "no_usable_grounding",
        }
    return best


def _try_candidate(
    *,
    task: CropTask,
    candidate: FrameCandidate,
    stage_dir: Path,
    grounder: Grounder,
    refiner: Sam3BoxPointRefiner | None,
    require_sam3: bool,
) -> dict[str, Any] | None:
    # Location targets are scene context; keep full-frame without object localization.
    if task.kind == "location":
        result = FullFrameGrounder().ground(
            image=Path(candidate.path),
            frame_index=candidate.frame_index,
            entity_id=task.entity_id,
            name=task.name,
            description=task.description,
            action=task.action,
        )
    else:
        result = grounder.ground(
            image=Path(candidate.path),
            frame_index=candidate.frame_index,
            entity_id=task.entity_id,
            name=task.name,
            description=task.description,
            action=task.action,
        )
    if not result.usable:
        return None

    refined = None
    sam3_meta: dict[str, Any] | None = None
    if refiner is not None and task.kind != "location":
        try:
            refined = refiner.refine(
                image=Path(candidate.path),
                bbox_norm=result.bbox_norm,
                point_norm=result.point_norm,
            )
        except RuntimeError as exc:
            if require_sam3:
                return None
            sam3_meta = {"error": str(exc), "fallback": "vlm_bbox"}
        else:
            if refined is None:
                if require_sam3:
                    return None
                sam3_meta = {"error": "empty_mask", "fallback": "vlm_bbox"}
            else:
                sam3_meta = {
                    "score": refined.score,
                    "point_inside_mask": refined.point_inside_mask,
                }
    elif require_sam3 and task.kind != "location":
        return None

    bbox_norm = refined.bbox_norm if refined is not None else result.bbox_norm
    crop_path = (
        stage_dir / "candidates" / task.kind / task.entity_id
        / f"c{task.chunk_id:05d}_{candidate.frame_index:08d}.png"
    )
    crop_path = materialize_crop(
        frame=Path(candidate.path),
        bbox_norm=bbox_norm,
        out_path=crop_path,
        mask=refined.mask if refined is not None else None,
    )
    unmasked_crop_path = (
        materialize_unmasked_companion(
            frame=Path(candidate.path),
            bbox_norm=bbox_norm,
            crop_path=crop_path,
        )
        if refined is not None
        else None
    )
    qa = audit_crop(crop=crop_path, bbox_norm=bbox_norm, kind=task.kind)
    proposal = {
        **task.to_dict(),
        "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
        "route": ROUTE_NAME,
        "crop_path": str(crop_path),
        "unmasked_crop_path": str(unmasked_crop_path) if unmasked_crop_path else None,
        "frame_index": candidate.frame_index,
        "bbox_norm": bbox_norm,
        "point_norm": result.point_norm,
        "bbox_source": "sam3_refined" if refined is not None else "vlm_bbox",
        "grounding": result.to_dict(),
        "sam3": sam3_meta,
        "qa": qa.to_dict(),
        "accepted": qa.accepted,
    }
    return attach_crop_attributes(proposal, task=task)


def run_vlm_sam_refine(
    *,
    tasks: list[CropTask],
    source_video: Path,
    stage_dir: Path,
    grounder: Grounder,
    refiner: Sam3BoxPointRefiner | None = None,
    require_sam3: bool = False,
    candidate_count: int = 5,
    keep_count: int = 3,
) -> list[dict[str, Any]]:
    """Acquire one proposal per task via VLM grounding + optional SAM3 refine."""
    return [
        _acquire_one(
            task=task,
            source_video=source_video,
            stage_dir=stage_dir,
            grounder=grounder,
            refiner=refiner,
            require_sam3=require_sam3,
            candidate_count=candidate_count,
            keep_count=keep_count,
        )
        for task in tasks
    ]
