"""Shallow sample discovery and status projection for the pipeline console."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from vmem_bench.annotation.pipeline.orchestration.catalog import (
    DEFAULT_BLENDER_VIDEOS_ROOT,
    resolve_blender_source_video,
)
from vmem_bench.annotation.pipeline.orchestration.contracts import PIPELINE_STAGES, MovieManifest


DATASETS = ("BlenderOpenMovies", "LSMDC")

# Console polls /api/samples every few seconds; rebuilding 100+ KFS-backed rows
# can take 15–25s and pile up threads behind the KML AccessProxy. Cache briefly.
_SAMPLES_CACHE_LOCK = threading.Lock()
_SAMPLES_CACHE: dict[str, Any] = {"key": None, "ts": 0.0, "rows": None}
_SAMPLES_CACHE_TTL_SEC = 8.0


def memstrata_root_from_here() -> Path:
    """Return ``benchmarks/MemStrata`` from this backend module location."""
    # backend/catalog_state.py -> services -> pipeline -> annotation -> vmem_bench
    # -> src -> MemStrata
    return Path(__file__).resolve().parents[6]


def default_data_root() -> Path:
    return memstrata_root_from_here() / "data"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:  # noqa: BLE001 - malformed user artifacts are status, not fatal.
        return None


def vlm_output_path(movie_dir: Path) -> Path:
    return movie_dir / "vlm_output.json"


def _review_media_url(sample: dict[str, Any], path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    media = Path(text)
    if media.is_absolute():
        try:
            text = str(media.resolve().relative_to(memstrata_root_from_here().resolve()))
        except ValueError:
            text = str(media)
    return "/api/review/media?" + urlencode(
        {
            "dataset": sample["dataset"],
            "movie_id": sample["movie_id"],
            "path": text.replace("\\", "/"),
        }
    )


def _stage_summary(
    state: dict[str, Any] | None,
    *,
    has_vlm_output: bool = False,
) -> dict[str, Any]:
    """Project pipeline stage lamps for the console.

    Orchestrator starts at S2 and never writes ``s1_vlm_annotation`` into
    ``state.json``. When a non-empty ``vlm_output.json`` exists, project S1 as
    ``ok`` so Stage badges match the Review column / VLM column.
    """
    stages = dict((state or {}).get("stages") or {})
    if has_vlm_output and not str(stages.get("s1_vlm_annotation", {}).get("status") or ""):
        stages = {
            **stages,
            "s1_vlm_annotation": {
                "status": "ok",
                "source": "vlm_output.json",
                "projected": True,
            },
        }
    current = None
    current_status = "not_started"
    for stage in PIPELINE_STAGES:
        status = str(stages.get(stage, {}).get("status") or "")
        if status:
            current = stage
            current_status = status
    done_count = sum(1 for stage in PIPELINE_STAGES if stages.get(stage, {}).get("status"))
    return {
        "current_stage": current,
        "current_status": current_status,
        "stage_index": PIPELINE_STAGES.index(current) if current in PIPELINE_STAGES else -1,
        "stage_count": len(PIPELINE_STAGES),
        "completed_stage_count": done_count,
        "stages": stages,
    }


def _review_summary(movie_dir: Path) -> dict[str, Any]:
    pipeline_root = movie_dir / "tmp" / "pipeline"
    items = []
    for rel in (
        "s4_segment_sampling_human_review/review_queue.json",
        "s6_entities_visual_crop_human_review/review_queue.json",
    ):
        payload = _maybe_read_json(pipeline_root / rel)
        if payload is None:
            continue
        if isinstance(payload, dict):
            queue_items = payload.get("items") or payload.get("queue") or []
        elif isinstance(payload, list):
            queue_items = payload
        else:
            queue_items = []
        items.append({"path": rel, "count": len(queue_items)})
    s4_queue = pipeline_root / "s4_segment_sampling_human_review" / "review_queue.json"
    s4_audit = _maybe_read_json(pipeline_root / "s4_segment_sampling_human_review" / "review_audit.json") or {}
    s5_dir = pipeline_root / "s5_entities_visual_crop_acquisition"
    s5_proposals = s5_dir / "crop_proposals.json"
    s6_audit = _maybe_read_json(pipeline_root / "s6_entities_visual_crop_human_review" / "review_audit.json") or {}
    s3_dir = pipeline_root / "s3_segment_auto_review_revise"
    s7_dir = pipeline_root / "s7_freeze_publish"
    vlm = vlm_output_path(movie_dir)

    s1_available = vlm.is_file() and vlm.stat().st_size > 0
    s2_available = (pipeline_root / "s2_annotation_postprocess" / "normalized_annotation.json").is_file()
    s3_available = False
    if s3_dir.is_dir():
        s3_available = (
            (s3_dir / "progress.json").is_file()
            or (s3_dir / "segment_audit.jsonl").is_file()
            or (s3_dir / "auto_revised_annotation.json").is_file()
            or any(s3_dir.glob("shard_*/segment_audit.jsonl"))
        )
    s3_progress = _maybe_read_json(s3_dir / "progress.json") or {}
    if not isinstance(s3_progress, dict):
        s3_progress = {}
    s4_available = s4_queue.is_file()
    s5_available = (
        s5_proposals.is_file()
        or (s5_dir / "crop_library.json").is_file()
        or (s5_dir / "coverage_plan.json").is_file()
        or (s5_dir / "crop_tasks.json").is_file()
        or (s5_dir / "crop_acquisition_progress.json").is_file()
        or (s5_dir / "crop_acquisition_live.json").is_file()
    )
    s5_progress = _maybe_read_json(s5_dir / "crop_acquisition_progress.json") or {}
    if not isinstance(s5_progress, dict):
        s5_progress = {}
    s6_available = s5_proposals.is_file() or (
        pipeline_root / "s6_entities_visual_crop_human_review" / "review_queue.json"
    ).is_file()
    s7_available = (movie_dir / "gold" / "entity_registry.json").is_file() or (
        s7_dir / "release_manifest.json"
    ).is_file()

    s4_human_reviewed = bool(s4_audit.get("human_reviewed"))
    s6_human_reviewed = bool(s6_audit.get("human_reviewed"))
    stage_reviews = [
        {"stage": "s1_vlm_annotation", "short": "S1", "available": s1_available, "kind": "inspect", "label": "S1 VLM"},
        {"stage": "s2_annotation_postprocess", "short": "S2", "available": s2_available, "kind": "inspect", "label": "S2 结构"},
        {"stage": "s3_segment_auto_review_revise", "short": "S3", "available": s3_available, "kind": "s3", "label": "S3 审核"},
        {"stage": "s4_segment_sampling_human_review", "short": "S4", "available": s4_available, "kind": "s4", "label": "S4 审核"},
        {"stage": "s5_entities_visual_crop_acquisition", "short": "S5", "available": s5_available, "kind": "inspect", "label": "S5 Crop"},
        {"stage": "s6_entities_visual_crop_human_review", "short": "S6", "available": s6_available, "kind": "s6", "label": "S6 审核"},
        {"stage": "s7_freeze_publish", "short": "S7", "available": s7_available, "kind": "inspect", "label": "S7 Gold"},
    ]
    return {
        "queues": items,
        "total_items": sum(item["count"] for item in items),
        "awaiting_human": (s4_available and not s4_human_reviewed) or (s6_available and not s6_human_reviewed),
        "s1_available": s1_available,
        "s2_available": s2_available,
        "s3_available": s3_available,
        "s3_live_available": s3_available,
        "s3_progress": s3_progress,
        "s4_available": s4_available,
        "s5_available": s5_available,
        "s5_progress": s5_progress,
        "s6_available": s6_available,
        "s7_available": s7_available,
        "s4_human_reviewed": s4_human_reviewed,
        "s6_human_reviewed": s6_human_reviewed,
        "stage_reviews": stage_reviews,
    }


def _gold_summary(movie_dir: Path) -> dict[str, Any]:
    gold = movie_dir / "gold"
    manifest = _maybe_read_json(gold / "manifest.json")
    crop_index = _maybe_read_json(gold / "crop_index.json")
    registry = _maybe_read_json(gold / "entity_registry.json")
    crops = crop_index.get("crops", []) if isinstance(crop_index, dict) else []
    reps: list[dict[str, Any]] = []
    if isinstance(registry, dict):
        for entity in registry.get("entities", []) or []:
            if isinstance(entity, dict):
                kind = str(entity.get("kind") or "")
                for rep in entity.get("representations", []) or []:
                    if isinstance(rep, dict):
                        item = dict(rep)
                        item["_entity_kind"] = kind
                        reps.append(item)
    rejected_reps = [
        rep for rep in reps
        if isinstance(rep.get("qa"), dict) and rep["qa"].get("accepted") is False
    ]
    full_frame_non_location = [
        rep for rep in reps
        if rep.get("bbox") == [0, 0, 1000, 1000] and rep.get("_entity_kind") != "location"
    ]
    return {
        "exists": gold.is_dir(),
        "human_reviewed": bool(manifest.get("human_reviewed")) if isinstance(manifest, dict) else False,
        "automation_smoke_only": bool(manifest.get("automation_smoke_only")) if isinstance(manifest, dict) else False,
        "n_crops": len(crops),
        "n_representations": len(reps),
        "n_rejected_representations": len(rejected_reps),
        "n_full_frame_non_location": len(full_frame_non_location),
    }


def _infer_status(movie_dir: Path, state: dict[str, Any] | None) -> str:
    gold = _gold_summary(movie_dir)
    if gold["exists"] and gold["automation_smoke_only"]:
        return "automation_smoke_gold"
    if (movie_dir / "gold" / "entity_registry.json").is_file():
        return "gold_ready"
    vlm = vlm_output_path(movie_dir)
    has_vlm = vlm.is_file() and vlm.stat().st_size > 0
    summary = _stage_summary(state, has_vlm_output=has_vlm)
    if summary["current_status"] in {"awaiting_human", "automation_smoke"}:
        return summary["current_status"]
    if not vlm.is_file():
        return "s1_missing"
    if vlm.stat().st_size == 0:
        return "s1_incomplete"
    # Only orchestrator-written stages (not projected S1) mean a run started.
    written = dict((state or {}).get("stages") or {})
    if any(str(written.get(stage, {}).get("status") or "") for stage in PIPELINE_STAGES):
        return "in_progress"
    return "source_ready"


def _load_source_index(blender_index: Path | None, lsmdc_index: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {"BlenderOpenMovies": {}, "LSMDC": {}}
    if blender_index and blender_index.is_file():
        try:
            payload = _read_json(blender_index)
            storage_root = str(payload.get("storage_root") or blender_index.parent)
            videos_root = str(payload.get("videos_root") or "")
            if not videos_root:
                candidate = Path(storage_root) / "Videos"
                videos_root = str(candidate if candidate.is_dir() else DEFAULT_BLENDER_VIDEOS_ROOT)
            for item in payload.get("items", []):
                movie_id = str(item.get("id") or "")
                if movie_id:
                    normalized = dict(item)
                    normalized["_storage_root"] = storage_root
                    normalized["_videos_root"] = videos_root
                    # Local download catalogs use ``filename``; older manifests use ``file``.
                    if not normalized.get("filename") and normalized.get("file"):
                        normalized["filename"] = normalized["file"]
                    out["BlenderOpenMovies"][movie_id] = normalized
        except Exception:  # noqa: BLE001
            pass
    if lsmdc_index and lsmdc_index.is_file():
        try:
            payload = _read_json(lsmdc_index)
            for item in payload.get("movies", []):
                movie_id = str(item.get("movie_id") or "")
                if movie_id:
                    out["LSMDC"][movie_id] = dict(item)
        except Exception:  # noqa: BLE001
            pass
    return out


def _source_video(
    *,
    dataset: str,
    movie_id: str,
    movie_dir: Path,
    source_index: dict[str, dict[str, dict[str, Any]]],
    blender_index: Path | None,
) -> tuple[str, float | None, float | None, list[str]]:
    notes: list[str] = []
    if (manifest := _maybe_read_json(movie_dir / "manifest.json")) and isinstance(manifest, dict):
        src = str(manifest.get("source_video") or manifest.get("video") or "")
        if src:
            return src, _float_or_none(manifest.get("duration_sec")), _float_or_none(manifest.get("fps")), notes

    if dataset == "BlenderOpenMovies":
        # Blender: Videos/<movie_id>/<video> by folder name. Index is optional metadata.
        item = source_index.get(dataset, {}).get(movie_id)
        videos_root = Path(
            str((item or {}).get("_videos_root") or DEFAULT_BLENDER_VIDEOS_ROOT)
        )
        if not videos_root.is_dir() and blender_index is not None:
            candidate_root = Path(blender_index).parent / "Videos"
            if candidate_root.is_dir():
                videos_root = candidate_root
        filename = str((item or {}).get("filename") or (item or {}).get("file") or "")
        candidate = resolve_blender_source_video(
            movie_id=movie_id,
            filename=filename,
            videos_root=videos_root,
        )
        if not candidate.is_file() and videos_root != DEFAULT_BLENDER_VIDEOS_ROOT:
            # Retry default Videos root when index points elsewhere.
            alt = resolve_blender_source_video(
                movie_id=movie_id,
                filename=filename,
                videos_root=DEFAULT_BLENDER_VIDEOS_ROOT,
            )
            if alt.is_file():
                candidate = alt
        if item is None:
            notes.append("missing_blender_catalog_entry")
        if not candidate.is_file():
            notes.append("source_video_missing_on_disk")
            return "", None, None, notes
        return (
            str(candidate),
            _float_or_none((item or {}).get("duration_sec") or (item or {}).get("duration")),
            _float_or_none((item or {}).get("fps")),
            notes,
        )

    # LSMDC: stitched output_file from index (not Videos/<id>/ folder match).
    item = source_index.get(dataset, {}).get(movie_id)
    if not item:
        return "", None, None, ["missing_lsmdc_catalog_entry"]
    stitched = dict(item.get("stitched") or {})
    if stitched.get("status") != "complete":
        notes.append("lsmdc_stitch_incomplete")
    return (
        str(stitched.get("output_file") or ""),
        _float_or_none(stitched.get("stitched_duration_sec")),
        None,
        notes,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_sample(
    *,
    dataset: str,
    movie_dir: Path,
    source_index: dict[str, dict[str, dict[str, Any]]],
    blender_index: Path | None = None,
) -> dict[str, Any]:
    movie_id = movie_dir.name
    state_path = movie_dir / "tmp" / "pipeline" / "state.json"
    state = _maybe_read_json(state_path)
    state_dict = state if isinstance(state, dict) else None
    vlm = vlm_output_path(movie_dir)
    has_vlm = vlm.is_file() and vlm.stat().st_size > 0
    stage = _stage_summary(state_dict, has_vlm_output=has_vlm)
    source_video, duration, fps, notes = _source_video(
        dataset=dataset,
        movie_id=movie_id,
        movie_dir=movie_dir,
        source_index=source_index,
        blender_index=blender_index,
    )
    source_video_exists = bool(source_video) and Path(source_video).is_file()
    if source_video and not source_video_exists:
        notes.append("source_video_missing_on_disk")
    return {
        "dataset": dataset,
        "movie_id": movie_id,
        "movie_dir": str(movie_dir),
        "source_video": source_video,
        "source_video_exists": source_video_exists,
        "source_duration_seconds": duration,
        "source_fps": fps,
        "vlm_output": str(vlm),
        "has_vlm_output": has_vlm,
        "has_gold": (movie_dir / "gold" / "entity_registry.json").is_file(),
        "status": _infer_status(movie_dir, state_dict),
        "notes": notes,
        "review": _review_summary(movie_dir),
        "gold_summary": _gold_summary(movie_dir),
        **stage,
    }


def list_samples(
    *,
    data_root: Path,
    blender_index: Path | None = None,
    lsmdc_index: Path | None = None,
    cache_ttl_sec: float | None = _SAMPLES_CACHE_TTL_SEC,
) -> list[dict[str, Any]]:
    """Return shallow per-movie status without recursive scans."""
    data_root = Path(data_root)
    cache_key = (
        str(data_root.resolve()),
        str(blender_index or ""),
        str(lsmdc_index or ""),
    )
    now = time.monotonic()
    ttl = _SAMPLES_CACHE_TTL_SEC if cache_ttl_sec is None else float(cache_ttl_sec)
    if ttl > 0:
        with _SAMPLES_CACHE_LOCK:
            if (
                _SAMPLES_CACHE.get("key") == cache_key
                and isinstance(_SAMPLES_CACHE.get("rows"), list)
                and now - float(_SAMPLES_CACHE.get("ts") or 0.0) < ttl
            ):
                return list(_SAMPLES_CACHE["rows"])

    source_index = _load_source_index(blender_index, lsmdc_index)
    samples: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_dir = data_root / dataset
        if not dataset_dir.is_dir():
            continue
        for movie_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            samples.append(
                _build_sample(
                    dataset=dataset,
                    movie_dir=movie_dir,
                    source_index=source_index,
                    blender_index=blender_index,
                )
            )
    if ttl > 0:
        with _SAMPLES_CACHE_LOCK:
            _SAMPLES_CACHE["key"] = cache_key
            _SAMPLES_CACHE["ts"] = time.monotonic()
            _SAMPLES_CACHE["rows"] = list(samples)
    return samples


def get_sample(
    *,
    data_root: Path,
    dataset: str,
    movie_id: str,
    blender_index: Path | None = None,
    lsmdc_index: Path | None = None,
) -> dict[str, Any] | None:
    """Load one movie without scanning the full catalog (review/media hot path)."""
    if dataset not in DATASETS or not movie_id:
        return None
    movie_dir = Path(data_root) / dataset / movie_id
    if not movie_dir.is_dir():
        return None
    source_index = _load_source_index(blender_index, lsmdc_index)
    return _build_sample(
        dataset=dataset,
        movie_dir=movie_dir,
        source_index=source_index,
        blender_index=blender_index,
    )


def find_sample(samples: list[dict[str, Any]], dataset: str, movie_id: str) -> dict[str, Any] | None:
    for sample in samples:
        if sample["dataset"] == dataset and sample["movie_id"] == movie_id:
            return sample
    return None


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _stage_files(movie_dir: Path) -> list[dict[str, Any]]:
    root = movie_dir / "tmp" / "pipeline"
    rows: list[dict[str, Any]] = []
    for stage in PIPELINE_STAGES:
        stage_dir = root / stage
        if not stage_dir.is_dir():
            rows.append({"stage": stage, "exists": False, "files": []})
            continue
        files = [
            {"name": path.name, "path": str(path), "bytes": path.stat().st_size}
            for path in sorted(stage_dir.iterdir())
            if path.is_file()
        ]
        rows.append({"stage": stage, "exists": True, "path": str(stage_dir), "files": files})
    return rows


def _crop_preview_rows(sample: dict[str, Any], limit: int = 80) -> list[dict[str, Any]]:
    movie_dir = Path(str(sample["movie_dir"]))
    gold = movie_dir / "gold"
    crop_index = _maybe_read_json(gold / "crop_index.json")
    registry = _maybe_read_json(gold / "entity_registry.json")
    entity_names: dict[str, dict[str, str]] = {}
    if isinstance(registry, dict):
        for entity in registry.get("entities", []) or []:
            if isinstance(entity, dict):
                entity_id = str(entity.get("entity_id") or "")
                entity_names[entity_id] = {
                    "name": str(entity.get("name") or entity_id),
                    "kind": str(entity.get("kind") or ""),
                }
    crops = crop_index.get("crops", []) if isinstance(crop_index, dict) else []
    rows: list[dict[str, Any]] = []
    for crop in crops[:limit]:
        if not isinstance(crop, dict):
            continue
        entity_id = str(crop.get("entity_id") or "")
        entity = entity_names.get(entity_id, {})
        rel = str(crop.get("crop_path") or "")
        image_url = (
            "/api/crop"
            f"?dataset={sample['dataset']}&movie_id={sample['movie_id']}&path={rel}"
        )
        rows.append(
            {
                "crop_id": str(crop.get("crop_id") or ""),
                "entity_id": entity_id,
                "name": entity.get("name", entity_id),
                "kind": str(crop.get("kind") or entity.get("kind") or ""),
                "chunk_id": crop.get("chunk_id"),
                "frame_index": crop.get("frame_index"),
                "bbox_norm": crop.get("bbox_norm") or crop.get("bbox") or [],
                "bbox_source": str(crop.get("bbox_source") or ""),
                "crop_path": rel,
                "image_url": image_url,
            }
        )
    return rows


def sample_detail(sample: dict[str, Any]) -> dict[str, Any]:
    movie_dir = Path(str(sample["movie_dir"]))
    gold = movie_dir / "gold"
    manifest = _maybe_read_json(gold / "manifest.json") or {}
    crop_index = _maybe_read_json(gold / "crop_index.json") or {}
    registry = _maybe_read_json(gold / "entity_registry.json") or {}
    chunk_index = _maybe_read_json(gold / "chunk_index.json") or {}
    return {
        "sample": sample,
        "paths": {
            "movie_dir": str(movie_dir),
            "pipeline": str(movie_dir / "tmp" / "pipeline"),
            "gold": str(gold),
            "source_video": str(sample.get("source_video") or ""),
            "vlm_output": str(sample.get("vlm_output") or ""),
        },
        "gold": {
            "manifest": manifest,
            "summary": sample.get("gold_summary") or _gold_summary(movie_dir),
            "n_entities": len(registry.get("entities", [])) if isinstance(registry, dict) else 0,
            "n_chunks": len(chunk_index.get("chunks", [])) if isinstance(chunk_index, dict) else 0,
            "n_crops": len(crop_index.get("crops", [])) if isinstance(crop_index, dict) else 0,
            "prompts": _jsonl_count(gold / "prompts.jsonl"),
            "observations": _jsonl_count(gold / "observations.jsonl"),
            "score_context": _jsonl_count(gold / "score_context.jsonl"),
            "crop_previews": _crop_preview_rows(sample),
        },
        "review": _review_summary(movie_dir),
        "stages": _stage_files(movie_dir),
    }


def _group_s5_crops(
    sample: dict[str, Any],
    items: list[Any],
    *,
    max_per_entity: int = 16,
) -> list[dict[str, Any]]:
    """Group live/proposal crops by entity for the S5 inspect page."""
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("entity_id") or item.get("name") or "unknown")
        if eid not in groups:
            groups[eid] = {
                "entity_id": eid,
                "name": str(item.get("name") or eid),
                "kind": str(item.get("kind") or ""),
                "n_done": 0,
                "n_accepted": 0,
                "n_with_crop": 0,
                "crops": [],
            }
            order.append(eid)
        group = groups[eid]
        group["n_done"] += 1
        accepted = bool(item.get("accepted") and item.get("crop_path"))
        if accepted:
            group["n_accepted"] += 1
        crop_path = item.get("crop_path") or ""
        if crop_path:
            group["n_with_crop"] += 1
        if crop_path and len(group["crops"]) < max_per_entity:
            group["crops"].append(
                {
                    "title": str(
                        item.get("representation_id")
                        or f"c{item.get('chunk_id')}"
                        or item.get("name")
                        or ""
                    ),
                    "meta": (
                        f"chunk={item.get('chunk_id')} · accepted={item.get('accepted')} · "
                        f"{item.get('bbox_source') or item.get('task_kind') or 'acquire'}"
                    ),
                    "text": str(item.get("reason") or item.get("description") or "")[:160],
                    "accepted": accepted,
                    "image_url": _review_media_url(sample, crop_path),
                }
            )
    return [groups[eid] for eid in order]


def stage_inspect(sample: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return inspectable artifacts for any pipeline stage (S1–S7)."""
    movie_dir = Path(str(sample["movie_dir"]))
    pipeline = movie_dir / "tmp" / "pipeline"
    stage_key = str(stage or "").strip()
    aliases = {
        "s1": "s1_vlm_annotation",
        "s2": "s2_annotation_postprocess",
        "s3": "s3_segment_auto_review_revise",
        "s4": "s4_segment_sampling_human_review",
        "s5": "s5_entities_visual_crop_acquisition",
        "s6": "s6_entities_visual_crop_human_review",
        "s7": "s7_freeze_publish",
    }
    stage_name = aliases.get(stage_key, stage_key)
    if stage_name not in PIPELINE_STAGES:
        return {"ok": False, "error": f"unknown stage: {stage}", "available": False}

    review = _review_summary(movie_dir)
    stage_meta = next((row for row in review.get("stage_reviews", []) if row["stage"] == stage_name), {})
    stage_dir = pipeline / stage_name
    files = []
    if stage_dir.is_dir():
        for path in sorted(stage_dir.iterdir()):
            if path.is_file():
                files.append({"name": path.name, "bytes": path.stat().st_size, "path": str(path)})

    summary: dict[str, Any] = {}
    highlights: list[dict[str, Any]] = []
    entity_groups: list[dict[str, Any]] = []
    deep_link = ""
    q = f"dataset={sample['dataset']}&movie_id={sample['movie_id']}"

    if stage_name == "s1_vlm_annotation":
        vlm = vlm_output_path(movie_dir)
        payload = _maybe_read_json(vlm) or {}
        summary = {
            "vlm_path": str(vlm),
            "exists": vlm.is_file(),
            "bytes": vlm.stat().st_size if vlm.is_file() else 0,
            "n_characters": len(payload.get("characters") or []) if isinstance(payload, dict) else 0,
            "n_props": len(payload.get("props") or []) if isinstance(payload, dict) else 0,
            "n_locations": len(payload.get("locations") or []) if isinstance(payload, dict) else 0,
        }
        if isinstance(payload, dict):
            for group, key in (("characters", "char_id"), ("props", "prop_id"), ("locations", "loc_id")):
                for raw in (payload.get(group) or [])[:12]:
                    if isinstance(raw, dict):
                        highlights.append(
                            {
                                "title": str(raw.get("name") or raw.get(key) or ""),
                                "meta": f"{group} · {raw.get(key) or ''}",
                                "text": str(raw.get("description") or "")[:240],
                            }
                        )
    elif stage_name == "s2_annotation_postprocess":
        ann = _maybe_read_json(stage_dir / "normalized_annotation.json") or {}
        lint = _maybe_read_json(stage_dir / "structural_lint.json") or {}
        segs = 0
        for scene in ((ann.get("screenplay") or {}).get("scenes") or []) if isinstance(ann, dict) else []:
            segs += len(scene.get("visual_segments") or [])
        summary = {
            "n_characters": len(ann.get("characters") or []) if isinstance(ann, dict) else 0,
            "n_props": len(ann.get("props") or []) if isinstance(ann, dict) else 0,
            "n_locations": len(ann.get("locations") or []) if isinstance(ann, dict) else 0,
            "n_segments": segs,
            "lint_ok": lint.get("ok") if isinstance(lint, dict) else None,
            "lint_errors": len(lint.get("errors") or []) if isinstance(lint, dict) else 0,
        }
        for err in (lint.get("errors") or [])[:20] if isinstance(lint, dict) else []:
            highlights.append({"title": "lint", "meta": "error", "text": str(err)[:300]})
    elif stage_name == "s3_segment_auto_review_revise":
        progress = _maybe_read_json(stage_dir / "progress.json") or {}
        audit = stage_dir / "segment_audit.jsonl"
        n_audit = sum(1 for _ in audit.open(encoding="utf-8")) if audit.is_file() else 0
        summary = {"progress": progress, "n_audit_rows": n_audit}
        deep_link = f"/review/s3.html?{q}"
    elif stage_name == "s4_segment_sampling_human_review":
        queue = _maybe_read_json(stage_dir / "review_queue.json") or []
        audit = _maybe_read_json(stage_dir / "review_audit.json") or {}
        summary = {
            "n_queue": len(queue) if isinstance(queue, list) else 0,
            "human_reviewed": bool(audit.get("human_reviewed")),
        }
        deep_link = f"/review/s4.html?{q}"
    elif stage_name == "s5_entities_visual_crop_acquisition":
        proposals = _maybe_read_json(stage_dir / "crop_proposals.json") or []
        live = _maybe_read_json(stage_dir / "crop_acquisition_live.json") or []
        progress = _maybe_read_json(stage_dir / "crop_acquisition_progress.json") or {}
        tasks = _maybe_read_json(stage_dir / "crop_tasks.json") or []
        plan = _maybe_read_json(stage_dir / "coverage_plan.json") or {}
        # Prefer live acquires while running; switch to final proposals when present.
        visible = proposals if isinstance(proposals, list) and proposals else live
        if not isinstance(visible, list):
            visible = []
        entity_groups = _group_s5_crops(sample, visible, max_per_entity=16)
        progress_dict = progress if isinstance(progress, dict) else {}
        summary = {
            "n_proposals": len(proposals) if isinstance(proposals, list) else 0,
            "n_live_acquired": len(live) if isinstance(live, list) else 0,
            "progress": progress_dict,
            "n_tasks": len(tasks) if isinstance(tasks, list) else 0,
            "n_entity_groups": len(entity_groups),
            "coverage": plan if isinstance(plan, dict) else {},
        }
        deep_link = f"/review/s6.html?{q}" if proposals else ""
        for group in entity_groups[:24]:
            for crop in (group.get("crops") or [])[:1]:
                highlights.append(
                    {
                        "title": f"{group.get('name')} · {group.get('entity_id')}",
                        "meta": (
                            f"{group.get('kind')} · accepted {group.get('n_accepted')}/"
                            f"{group.get('n_done')}"
                        ),
                        "text": crop.get("text") or "",
                        "image_url": crop.get("image_url") or "",
                    }
                )
    elif stage_name == "s6_entities_visual_crop_human_review":
        queue = _maybe_read_json(stage_dir / "review_queue.json") or []
        audit = _maybe_read_json(stage_dir / "review_audit.json") or {}
        summary = {
            "n_queue": len(queue) if isinstance(queue, list) else 0,
            "human_reviewed": bool(audit.get("human_reviewed")),
            "accepted_count": audit.get("accepted_count"),
        }
        deep_link = f"/review/s6.html?{q}"
    elif stage_name == "s7_freeze_publish":
        gold = _gold_summary(movie_dir)
        release = _maybe_read_json(stage_dir / "release_manifest.json") or {}
        s4_audit = _maybe_read_json(
            pipeline / "s4_segment_sampling_human_review" / "review_audit.json"
        ) or {}
        s6_audit = _maybe_read_json(
            pipeline / "s6_entities_visual_crop_human_review" / "review_audit.json"
        ) or {}
        s6_accepted = _maybe_read_json(
            pipeline / "s6_entities_visual_crop_human_review" / "accepted_crops.json"
        )
        n_s6_accepted = len(s6_accepted) if isinstance(s6_accepted, list) else (
            int(s6_audit.get("accepted_count") or 0) if isinstance(s6_audit, dict) else 0
        )
        n_gold_crops = int(gold.get("n_crops") or 0) if isinstance(gold, dict) else 0
        crop_previews = _crop_preview_rows(sample, limit=240)
        by_entity: dict[str, dict[str, Any]] = {}
        for crop in crop_previews:
            entity_id = str(crop.get("entity_id") or "")
            group = by_entity.setdefault(
                entity_id,
                {
                    "entity_id": entity_id,
                    "name": crop.get("name") or entity_id,
                    "kind": crop.get("kind") or "",
                    "n_accepted": 0,
                    "n_done": 0,
                    "n_with_crop": 0,
                    "crops": [],
                },
            )
            group["n_accepted"] += 1
            group["n_done"] += 1
            group["n_with_crop"] += 1
            group["crops"].append(
                {
                    "title": crop.get("crop_id") or "",
                    "accepted": True,
                    "meta": f"chunk {crop.get('chunk_id')} · frame {crop.get('frame_index')}",
                    "text": crop.get("crop_path") or "",
                    "image_url": crop.get("image_url") or "",
                }
            )
        entity_groups = sorted(
            by_entity.values(),
            key=lambda row: (str(row.get("kind") or ""), str(row.get("name") or "")),
        )
        summary = {
            "provenance": {
                "s4_human_reviewed": bool(s4_audit.get("human_reviewed")),
                "s6_human_reviewed": bool(s6_audit.get("human_reviewed")),
                "s6_accepted_count": n_s6_accepted,
                "gold_n_crops": n_gold_crops,
                "gold_human_reviewed": bool((gold or {}).get("human_reviewed"))
                if isinstance(gold, dict)
                else False,
                "crop_drop_count": max(0, n_s6_accepted - n_gold_crops),
                "note": (
                    "S7 是冻结检视页（非 S4/S6 审核 UI）。"
                    " gold 应由 S4 修订标注 + S6 accepted_crops 物化。"
                ),
            },
            "gold": gold,
            "release_manifest_keys": list(release.keys()) if isinstance(release, dict) else [],
        }
        for group in entity_groups[:24]:
            for crop in (group.get("crops") or [])[:1]:
                highlights.append(
                    {
                        "title": f"{group.get('name')} · {group.get('entity_id')}",
                        "meta": f"{group.get('kind')} · {group.get('n_accepted')} crops in gold",
                        "text": crop.get("text") or "",
                        "image_url": crop.get("image_url") or "",
                    }
                )

    return {
        "ok": True,
        "sample": {
            "dataset": sample["dataset"],
            "movie_id": sample["movie_id"],
            "movie_dir": str(movie_dir),
            "status": sample.get("status"),
        },
        "stage": stage_name,
        "short": stage_meta.get("short") or stage_name.split("_")[0].upper(),
        "available": bool(stage_meta.get("available")),
        "kind": stage_meta.get("kind") or "inspect",
        "label": stage_meta.get("label") or stage_name,
        "stage_dir": str(stage_dir),
        "files": files,
        "summary": summary,
        "highlights": highlights,
        "entity_groups": entity_groups,
        "deep_link": deep_link,
        "stage_reviews": review.get("stage_reviews") or [],
    }


def resolve_crop_path(sample: dict[str, Any], rel: str) -> Path | None:
    movie_dir = Path(str(sample["movie_dir"])).resolve()
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None
    target = (movie_dir / "gold" / rel_path).resolve()
    allowed_root = (movie_dir / "gold" / "crops").resolve()
    if allowed_root not in target.parents:
        return None
    if target.is_file() and target.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return target
    return None


def manifest_from_sample(sample: dict[str, Any]) -> MovieManifest:
    runnable = bool(sample.get("has_vlm_output")) and bool(sample.get("source_video_exists"))
    return MovieManifest(
        dataset=str(sample["dataset"]),
        movie_id=str(sample["movie_id"]),
        movie_dir=str(sample["movie_dir"]),
        source_video=str(sample.get("source_video") or ""),
        vlm_output=str(sample.get("vlm_output") or ""),
        source_duration_seconds=_float_or_none(sample.get("source_duration_seconds")),
        source_fps=_float_or_none(sample.get("source_fps")),
        status="source_ready" if runnable else str(sample.get("status") or "source_ready"),
        notes=[str(note) for note in sample.get("notes", [])],
    )
