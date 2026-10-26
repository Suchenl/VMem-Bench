"""Route B: detector proposes crops; identity via exemplar + exclusive claim + VLM pick.

Relative to an early S5 draft that let every entity independently VLM-pick from the same
animal pool (BBB multi-critter pollution), this mirrors track_first Route B:

1. SAM3/GDINO only propose regions (category words / short names).
2. DINOv3 exemplar similarity assigns WHO when library exemplars exist.
3. Same-chunk entities claim proposals exclusively (no shared bbox races).
4. VLM closed-set pick only sees remaining unclaimed, preferably masked crops.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Protocol

from vmem_bench.common.crop_identity_gates import (
    apply_cross_entity_conflict_gate,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.attach_attributes import (
    attach_crop_attributes,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    load_crop_rgb_for_model,
    materialize_unmasked_companion,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    CropPicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_qa import (
    audit_crop,
    bbox_area_fraction,
    materialize_crop,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.exemplar_identity import (
    DEFAULT_SIM_FLOOR,
    build_exemplar_vectors,
    exclusive_assign_candidates,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.geometry import (
    dedup_by_iou,
    mask_to_bbox_norm,
    px_to_norm_bbox,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.keyframes import (
    extract_candidates,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.mask_quality import (
    assess_mask_quality,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_grounding import (
    FullFrameGrounder,
)

ROUTE_NAME = "propose_and_pick"

CHARACTER_CONCEPTS = ("person", "animal")
PROP_CONCEPTS = ("object",)

# Soft multi-object / scene-box filter for non-location entities (norm area 0-1).
_MAX_CHARACTER_BBOX_AREA = 0.42
_MIN_MASK_FILL = 0.28


class ConceptSegmenter(Protocol):
    def segment_multi(
        self, image: Path, concepts: list[str]
    ) -> dict[str, list[tuple[list[float], float, object]]]: ...


class PhraseDetector(Protocol):
    def detect_all(self, image: Path, phrase: str) -> list[tuple[list[int], float]]: ...


def _concepts_for(kind: str) -> tuple[str, ...]:
    if kind == "character":
        return CHARACTER_CONCEPTS
    if kind == "prop":
        return PROP_CONCEPTS
    return ()


def _phrases_for(task: CropTask) -> list[str]:
    """Detector phrases: name only (Pitfall #1 — never long description)."""
    phrases: list[str] = []
    if task.name.strip():
        phrases.append(task.name.strip())
    return phrases


def _mask_fill_ratio(mask: object, bbox_norm: list[int], height: int, width: int) -> float:
    import numpy as np

    y0, x0, y1, x1 = bbox_norm
    top = max(0, min(height - 1, round(y0 / 1000 * height)))
    left = max(0, min(width - 1, round(x0 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
    right = max(left + 1, min(width, round(x1 / 1000 * width)))
    array = np.asarray(mask, dtype=bool)[top:bottom, left:right]
    if array.size == 0:
        return 0.0
    return float(array.mean())


def _write_picker_feed(masked_png: Path, feed_path: Path) -> Path:
    """Composite RGBA mask crop onto white for VLM / DINOv3."""
    rgb = load_crop_rgb_for_model(masked_png)
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(feed_path, quality=95)
    return feed_path


def _collect_kind_proposals(
    *,
    kind: str,
    frame_path: Path,
    frame_index: int,
    proposer: str,
    segmenter: ConceptSegmenter | None,
    detector: PhraseDetector | None,
    phrases: list[str],
    scratch_dir: Path,
) -> list[dict[str, Any]]:
    from PIL import Image

    pil = Image.open(frame_path).convert("RGB")
    width, height = pil.size
    raw: list[dict[str, Any]] = []

    if proposer in ("sam3", "fusion") and segmenter is not None:
        concepts = list(_concepts_for(kind))
        by_concept = segmenter.segment_multi(frame_path, concepts) if concepts else {}
        for concept, instances in by_concept.items():
            for ordinal, (bbox_px, score, mask) in enumerate(instances):
                bbox_norm = (
                    mask_to_bbox_norm(mask)
                    if mask is not None
                    else px_to_norm_bbox(bbox_px, width, height)
                )
                if bbox_norm is None or len(bbox_norm) != 4:
                    continue
                x0, y0, x1, y1 = (int(v) for v in bbox_px)
                if (x1 - x0) < 16 or (y1 - y0) < 16:
                    continue
                if mask is None:
                    continue
                quality = assess_mask_quality(mask)
                if not quality.ok:
                    continue
                if kind == "character":
                    area = bbox_area_fraction(bbox_norm)
                    if area > _MAX_CHARACTER_BBOX_AREA:
                        continue
                    if _mask_fill_ratio(mask, bbox_norm, height, width) < _MIN_MASK_FILL:
                        continue
                masked_path = scratch_dir / (
                    f"f{frame_index:08d}_sam3_{concept}_{ordinal}.png"
                )
                masked_path = materialize_crop(
                    frame=frame_path,
                    bbox_norm=bbox_norm,
                    out_path=masked_path,
                    mask=mask,
                )
                feed_path = _write_picker_feed(
                    masked_path, scratch_dir / f"f{frame_index:08d}_sam3_{concept}_{ordinal}_feed.jpg"
                )
                raw.append({
                    "bbox_norm": bbox_norm,
                    "score": float(score),
                    "mask": mask,
                    "crop_path": masked_path,
                    "picker_crop": feed_path,
                    "bbox_source": "sam3_concept",
                    "concept": concept,
                    "frame_index": frame_index,
                    "frame_path": str(frame_path),
                    "mask_quality": quality.to_dict(),
                })

    if proposer in ("gdino", "fusion") and detector is not None:
        for phrase in phrases:
            for ordinal, (bbox_norm, score) in enumerate(detector.detect_all(frame_path, phrase)):
                y0, x0, y1, x1 = bbox_norm
                left = max(0, min(width - 1, round(x0 / 1000 * width)))
                top = max(0, min(height - 1, round(y0 / 1000 * height)))
                right = max(left + 1, min(width, round(x1 / 1000 * width)))
                bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
                if (right - left) < 16 or (bottom - top) < 16:
                    continue
                if kind == "character" and bbox_area_fraction(list(bbox_norm)) > _MAX_CHARACTER_BBOX_AREA:
                    continue
                crop_path = scratch_dir / (
                    f"f{frame_index:08d}_gdino_{re.sub(r'[^a-zA-Z0-9]+', '_', phrase)[:24]}_{ordinal}.png"
                )
                crop_path = materialize_crop(
                    frame=frame_path,
                    bbox_norm=list(bbox_norm),
                    out_path=crop_path,
                )
                feed_path = _write_picker_feed(
                    crop_path,
                    scratch_dir
                    / f"f{frame_index:08d}_gdino_{ordinal}_feed.jpg",
                )
                raw.append({
                    "bbox_norm": list(bbox_norm),
                    "score": float(score),
                    "mask": None,
                    "crop_path": crop_path,
                    "picker_crop": feed_path,
                    "bbox_source": "grounding_dino",
                    "grounding_phrase": phrase,
                    "frame_index": frame_index,
                    "frame_path": str(frame_path),
                })

    return dedup_by_iou(raw, iou_threshold=0.7)


def _location_proposal(
    *,
    task: CropTask,
    candidate,
    stage_dir: Path,
) -> dict[str, Any]:
    result = FullFrameGrounder().ground(
        image=Path(candidate.path),
        frame_index=candidate.frame_index,
        entity_id=task.entity_id,
        name=task.name,
        description=task.description,
        action=task.action,
    )
    crop_path = (
        stage_dir / "candidates" / task.kind / task.entity_id
        / f"c{task.chunk_id:05d}_{candidate.frame_index:08d}.png"
    )
    crop_path = materialize_crop(
        frame=Path(candidate.path),
        bbox_norm=result.bbox_norm,
        out_path=crop_path,
    )
    qa = audit_crop(crop=crop_path, bbox_norm=result.bbox_norm, kind=task.kind)
    proposal = {
        **task.to_dict(),
        "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
        "route": ROUTE_NAME,
        "crop_path": str(crop_path),
        "frame_index": candidate.frame_index,
        "bbox_norm": result.bbox_norm,
        "point_norm": result.point_norm,
        "bbox_source": "full_frame",
        "grounding": result.to_dict(),
        "sam3": None,
        "pick": None,
        "qa": qa.to_dict(),
        "accepted": qa.accepted,
    }
    return attach_crop_attributes(proposal, task=task)


def _finalize_from_candidate(
    *,
    task: CropTask,
    chosen: dict[str, Any],
    stage_dir: Path,
    proposer: str,
    pool_size: int,
    pick_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    crop_path = (
        stage_dir / "candidates" / task.kind / task.entity_id
        / f"c{task.chunk_id:05d}_{int(chosen['frame_index']):08d}.png"
    )
    crop_path = materialize_crop(
        frame=Path(chosen["frame_path"]),
        bbox_norm=chosen["bbox_norm"],
        out_path=crop_path,
        mask=chosen.get("mask"),
    )
    unmasked_crop_path = (
        materialize_unmasked_companion(
            frame=Path(chosen["frame_path"]),
            bbox_norm=chosen["bbox_norm"],
            crop_path=crop_path,
        )
        if chosen.get("mask") is not None
        else None
    )
    qa = audit_crop(crop=crop_path, bbox_norm=chosen["bbox_norm"], kind=task.kind)
    proposal = {
        **task.to_dict(),
        "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
        "route": ROUTE_NAME,
        "crop_path": str(crop_path),
        "unmasked_crop_path": str(unmasked_crop_path) if unmasked_crop_path else None,
        "frame_index": chosen["frame_index"],
        "bbox_norm": chosen["bbox_norm"],
        "point_norm": [],
        "bbox_source": chosen["bbox_source"],
        "proposer": proposer,
        "grounding": {
            "usable": True,
            "bbox_norm": chosen["bbox_norm"],
            "concept": chosen.get("concept"),
            "grounding_phrase": chosen.get("grounding_phrase"),
            "detector_score": chosen["score"],
        },
        "mask_quality": chosen.get("mask_quality"),
        "sam3": (
            {"score": chosen["score"], "concept": chosen.get("concept")}
            if chosen["bbox_source"] == "sam3_concept"
            else None
        ),
        "pick": {
            "n_proposals": pool_size,
            "candidate_sources": [],
            **(pick_meta or {}),
        },
        "qa": qa.to_dict(),
        "accepted": qa.accepted,
    }
    return attach_crop_attributes(proposal, task=task)


def _reject(task: CropTask, *, reason: str, proposer: str, **extra: Any) -> dict[str, Any]:
    return {
        **task.to_dict(),
        "representation_id": f"{task.entity_id}@c{task.chunk_id:05d}",
        "route": ROUTE_NAME,
        "accepted": False,
        "reason": reason,
        "proposer": proposer,
        **extra,
    }


def _acquire_chunk_exclusive(
    *,
    tasks: list[CropTask],
    source_video: Path,
    stage_dir: Path,
    picker: CropPicker,
    proposer: str,
    segmenter: ConceptSegmenter | None,
    detector: PhraseDetector | None,
    max_candidates: int,
    candidate_count: int,
    keep_count: int,
    exemplar_bank: dict[str, list[Path]],
    embedder: Any | None,
    sim_floor: float,
) -> list[dict[str, Any]]:
    """One shared proposal pool; exclusive identity assign for all tasks in the chunk."""
    if not tasks:
        return []
    kind = tasks[0].kind
    # Shared time span covering all entities in this chunk.
    start = min(t.start_seconds for t in tasks)
    end = max(t.end_seconds for t in tasks)
    frame_dir = stage_dir / "frames" / f"c{tasks[0].chunk_id:05d}" / "_shared"
    candidates = extract_candidates(
        source_video=source_video,
        start_seconds=start,
        end_seconds=end,
        out_dir=frame_dir,
        candidate_count=candidate_count,
        keep_count=keep_count,
    )
    scratch = stage_dir / "propose_scratch" / kind / f"c{tasks[0].chunk_id:05d}_shared"
    scratch.mkdir(parents=True, exist_ok=True)
    phrases = list(dict.fromkeys(p for task in tasks for p in _phrases_for(task)))
    pool: list[dict[str, Any]] = []
    for candidate in candidates:
        pool.extend(
            _collect_kind_proposals(
                kind=kind,
                frame_path=Path(candidate.path),
                frame_index=candidate.frame_index,
                proposer=proposer,
                segmenter=segmenter,
                detector=detector,
                phrases=phrases,
                scratch_dir=scratch,
            )
        )
    pool = dedup_by_iou(pool, iou_threshold=0.7)[: max(max_candidates * max(1, len(tasks)), max_candidates)]
    if not pool:
        return [_reject(task, reason="no_detector_proposals", proposer=proposer) for task in tasks]

    claimed: set[int] = set()
    results: dict[str, dict[str, Any]] = {}

    # 1) Exemplar-exclusive assignment when bank has anchors for any entity.
    exemplars = build_exemplar_vectors(exemplar_bank, embedder) if embedder is not None else {}
    usable_exemplars = {
        eid: vecs for eid, vecs in exemplars.items() if eid in {t.entity_id for t in tasks}
    }
    if usable_exemplars and embedder is not None:
        feed_paths = [Path(item.get("picker_crop") or item["crop_path"]) for item in pool]
        try:
            vecs = embedder.embed_batch(feed_paths)
        except Exception:
            vecs = []
        if vecs and len(vecs) == len(pool):
            assignment, _leftover = exclusive_assign_candidates(
                candidate_vecs=vecs,
                exemplars=usable_exemplars,
                entity_ids=[t.entity_id for t in tasks],
                sim_floor=sim_floor,
            )
            for task in tasks:
                hit = assignment.get(task.entity_id)
                if hit is None:
                    continue
                index, sim = hit
                claimed.add(index)
                results[task.entity_id] = _finalize_from_candidate(
                    task=task,
                    chosen=pool[index],
                    stage_dir=stage_dir,
                    proposer=proposer,
                    pool_size=len(pool),
                    pick_meta={
                        "index": index,
                        "confidence": "high" if sim >= sim_floor + 0.1 else "medium",
                        "reason": f"exemplar_sim={sim:.3f}",
                        "picker": "dinov3_exemplar",
                        "exemplar_sim": sim,
                    },
                )

    # 2) Remaining entities: VLM pick among unclaimed only (exclusive).
    for task in tasks:
        if task.entity_id in results:
            continue
        available = [i for i in range(len(pool)) if i not in claimed]
        if not available:
            results[task.entity_id] = _reject(
                task, reason="no_unclaimed_proposals", proposer=proposer, n_proposals=len(pool)
            )
            continue
        sub = [pool[i] for i in available]
        try:
            local_index = picker.pick(
                name=task.name,
                description=task.description,
                kind=task.kind,
                action="",  # do not steer with co-occurrence action text
                candidate_crops=[Path(item.get("picker_crop") or item["crop_path"]) for item in sub],
            )
        except Exception as exc:  # one picker outage must not abort the movie
            results[task.entity_id] = _reject(
                task,
                reason="picker_request_failed",
                proposer=proposer,
                picker_error=f"{type(exc).__name__}: {exc}"[:800],
                n_proposals=len(sub),
            )
            continue
        pick_result = dict(getattr(picker, "last_result", {}) or {})
        if local_index < 0 or local_index >= len(sub):
            results[task.entity_id] = _reject(
                task,
                reason="picker_rejected",
                proposer=proposer,
                n_proposals=len(sub),
                pick=pick_result,
            )
            continue
        global_index = available[local_index]
        claimed.add(global_index)
        results[task.entity_id] = _finalize_from_candidate(
            task=task,
            chosen=pool[global_index],
            stage_dir=stage_dir,
            proposer=proposer,
            pool_size=len(sub),
            pick_meta={
                "index": local_index,
                "global_index": global_index,
                "n_proposals": len(sub),
                "candidate_sources": [item["bbox_source"] for item in sub],
                "picker": pick_result.get("picker") or "vlm_exclusive",
                **{k: v for k, v in pick_result.items() if k not in {"index"}},
            },
        )

    return [results[t.entity_id] for t in tasks]


def run_propose_and_pick(
    *,
    tasks: list[CropTask],
    source_video: Path,
    stage_dir: Path,
    picker: CropPicker,
    proposer: str = "sam3",
    segmenter: ConceptSegmenter | None = None,
    detector: PhraseDetector | None = None,
    max_candidates: int = 4,
    candidate_count: int = 5,
    keep_count: int = 3,
    on_task_start: Callable[..., None] | None = None,
    on_task_complete: Callable[[list[dict[str, Any]]], None] | None = None,
    embedder: Any | None = None,
    exemplar_sim_floor: float = DEFAULT_SIM_FLOOR,
) -> list[dict[str, Any]]:
    """Acquire proposals with chunk-exclusive identity + optional DINOv3 exemplars."""
    if proposer not in ("sam3", "gdino", "fusion"):
        raise ValueError(f"unknown proposer: {proposer}")
    if proposer in ("sam3", "fusion") and segmenter is None:
        raise ValueError("segmenter is required for sam3/fusion proposer")
    if proposer in ("gdino", "fusion") and detector is None:
        raise ValueError("detector is required for gdino/fusion proposer")

    # Lazy embedder: only constructed when first accepted crop needs banking.
    active_embedder = embedder
    exemplar_bank: dict[str, list[Path]] = defaultdict(list)
    proposals: list[dict[str, Any]] = []

    def _ensure_embedder() -> Any | None:
        nonlocal active_embedder
        if active_embedder is not None:
            return active_embedder
        try:
            from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.embedding import (
                DinoV3Embedder,
            )

            active_embedder = DinoV3Embedder()
        except Exception:
            active_embedder = None
        return active_embedder

    def _emit(item: dict[str, Any], *, task: CropTask, index: int) -> None:
        if on_task_start is not None:
            on_task_start(task, index=index, total=len(tasks))
        proposals.append(item)
        if item.get("accepted") and item.get("crop_path"):
            exemplar_bank[str(item.get("entity_id") or "")].append(Path(str(item["crop_path"])))
            _ensure_embedder()
        apply_cross_entity_conflict_gate(proposals)
        if on_task_complete is not None:
            on_task_complete(proposals)

    # Walk tasks in order; flush same-chunk non-location groups together.
    buffer: list[tuple[int, CropTask]] = []
    pending_chunk: int | None = None

    def _flush() -> None:
        nonlocal buffer, pending_chunk
        if not buffer:
            return
        chunk_tasks = [task for _, task in buffer]
        indices = [idx for idx, _ in buffer]
        kind = chunk_tasks[0].kind
        if kind == "location":
            raise RuntimeError("location tasks must not enter exclusive buffer")
        chunk_results = _acquire_chunk_exclusive(
            tasks=chunk_tasks,
            source_video=source_video,
            stage_dir=stage_dir,
            picker=picker,
            proposer=proposer,
            segmenter=segmenter,
            detector=detector,
            max_candidates=max_candidates,
            candidate_count=candidate_count,
            keep_count=keep_count,
            exemplar_bank=dict(exemplar_bank),
            embedder=_ensure_embedder() if any(exemplar_bank.values()) else None,
            sim_floor=exemplar_sim_floor,
        )
        for idx, task, item in zip(indices, chunk_tasks, chunk_results):
            _emit(item, task=task, index=idx)
        buffer = []
        pending_chunk = None

    for index, task in enumerate(tasks):
        if task.kind == "location":
            _flush()
            if on_task_start is not None:
                on_task_start(task, index=index, total=len(tasks))
            frame_dir = stage_dir / "frames" / f"c{task.chunk_id:05d}" / task.entity_id
            frames = extract_candidates(
                source_video=source_video,
                start_seconds=task.start_seconds,
                end_seconds=task.end_seconds,
                out_dir=frame_dir,
                candidate_count=candidate_count,
                keep_count=keep_count,
            )
            best = max(frames, key=lambda item: item.sharpness)
            item = _location_proposal(task=task, candidate=best, stage_dir=stage_dir)
            proposals.append(item)
            apply_cross_entity_conflict_gate(proposals)
            if on_task_complete is not None:
                on_task_complete(proposals)
            continue

        if pending_chunk is None:
            pending_chunk = task.chunk_id
        if task.chunk_id != pending_chunk:
            _flush()
            pending_chunk = task.chunk_id
        buffer.append((index, task))

    _flush()
    return apply_cross_entity_conflict_gate(proposals)
