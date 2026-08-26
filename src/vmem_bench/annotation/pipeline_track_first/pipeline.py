"""Offline annotation pipeline orchestrator (workflow.md Phase 1, steps 1-8).

Chunks are processed causally (the registry grows chunk by chunk), but everything inside
a chunk is parallel:

- frame extraction / grounding / cropping / embedding run in thread pools;
- each QA round launches ``K = len(annotators)`` independent candidate branches in
  parallel (annotator i + verifier i on their own vLLM endpoints). The first branch that
  passes every check is accepted; otherwise the branch with the fewest failed checks
  seeds the next round's feedback. After ``qa_max_rounds`` the globally best branch is
  committed and the chunk is flagged for human review (never silently rewritten by the
  verifier — the annotator/verifier separation is what keeps errors uncorrelated).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vmem_bench.common.media import extract_frame, probe_media, sample_frame_indices
from vmem_bench.common.paths import MovieDirs, entity_asset_dir
from vmem_bench.common.schemas import ChunkAnnotations, Entity, EntityRegistry, Representation
from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
from vmem_bench.annotation.pipeline_track_first.chunking import materialize_clip, run_chunking
from vmem_bench.annotation.pipeline_track_first.events import EventLog
from vmem_bench.annotation.pipeline_track_first.consolidation import (
    Registry, consolidate_observation, present_payload, slugify, normalize_entity_name)
from vmem_bench.annotation.pipeline_track_first.drafting import build_chunk_annotation, state_events_from_draft
from vmem_bench.annotation.pipeline_track_first.interfaces import AnnotatorVlm, Grounder, ImageEmbedder, VerifierVlm

logger = logging.getLogger(__name__)


def _sample_split(start: int, end: int, k: int) -> tuple[list[int], list[int]]:
    """Two interleaved frame samples: annotator gets even slots, verifier odd slots."""
    both = sample_frame_indices(start, end, max_samples=2 * k)
    if len(both) < 2:
        return both, both
    return both[0::2], both[1::2]


def _rel(p: Path | str, out: Path) -> str:
    """Path relative to the movie output dir (keeps the registry portable)."""
    try:
        return str(Path(p).resolve().relative_to(Path(out).resolve()))
    except ValueError:
        return str(p)


def _content_score(image_path: Path) -> float:
    """Grayscale std-dev; near-black / near-uniform frames (e.g. a fade-in) score ~0."""
    import numpy as np
    from PIL import Image
    return float(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32).std())


def _bbox_area(bbox: list[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    y0, x0, y1, x1 = bbox
    return max(0, y1 - y0) * max(0, x1 - x0) / 1_000_000


def _bbox_iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = ih * iw
    if inter <= 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _bbox_center_distance(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 1.0
    acy, acx = (a[0] + a[2]) / 2000, (a[1] + a[3]) / 2000
    bcy, bcx = (b[0] + b[2]) / 2000, (b[1] + b[3]) / 2000
    return ((acy - bcy) ** 2 + (acx - bcx) ** 2) ** 0.5


def _select_consistent_detection(
    detections: list[tuple[float, int, list[int]]],
    *,
    min_frames: int,
    min_iou: float,
    max_center_distance: float,
) -> tuple[float, int, list[int]] | None:
    """Choose the best detection only if it belongs to a multi-frame spatially consistent set."""
    for anchor in sorted(detections, key=lambda d: d[0], reverse=True):
        cluster = [d for d in detections
                   if _bbox_iou(anchor[2], d[2]) >= min_iou
                   or _bbox_center_distance(anchor[2], d[2]) <= max_center_distance]
        if len(cluster) >= min_frames:
            return max(cluster, key=lambda d: d[0])
    return None


def _ground_and_crop(video_frames: dict[int, Path], ent: dict[str, str], *, tag: str,
                     crops_dir: Path, grounder: Grounder, score_threshold: float,
                     min_frames: int = 1, temporal_iou_threshold: float = 0.10,
                     temporal_center_threshold: float = 0.35) -> dict[str, Any]:
    """Locate the entity across sampled frames, crop the best detection into ``crops_dir``.

    Uses ``grounding_phrase`` (a short noun phrase) — NOT the long description, which carries
    actions/neighboring objects/moods that bias the detector onto the wrong subject in
    multi-entity frames (Pitfall_Notes root cause #1). Falls back to the entity name when no
    grounding phrase was provided (e.g. stub backends).

    Temporal consistency: a non-location entity must be detected in at least ``min_frames``
    sampled frames (best score over those) to be accepted; a one-frame high-score detection is
    usually a grounding false positive. min_frames=1 preserves the single-best-detection
    behavior. Returns ``grounding_score`` so the verifier can skip auditing high-confidence
    crops and the cover picker can prefer the clearest view."""
    from PIL import Image

    # First non-empty (after strip) of grounding_phrase -> description -> name. A bare-whitespace
    # grounding_phrase (" ") must not short-circuit the `or` chain into an empty phrase, which
    # would feed ground(frame, "") with undefined behavior (Pitfall_Notes: grounding_phrase edge).
    _cands = [(ent.get("grounding_phrase") or "").strip(),
              (ent.get("description") or "").strip(),
              str(ent["name"]).strip()]
    phrase = next((c for c in _cands if c), str(ent["name"]).strip())
    slug = "".join(c if c.isalnum() else "_" for c in ent["name"].lower()).strip("_")
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crops_dir / f"{tag}_{ent['kind']}_{slug}.jpg"

    if ent["kind"] != "location":
        # Collect detections across all sampled frames, then require temporal consistency.
        # Batch the per-entity frame sweep in one forward when the grounder supports it
        # (GroundingDino.ground_batch) — N single-image forwards per entity collapse to one
        # (Pitfall_Notes: F5 grounder batch). Stub backends without ground_batch fall back to
        # per-frame ground().
        detections: list[tuple[float, int, list[int]]] = []  # (score, frame_index, bbox)
        frame_items = list(video_frames.items())
        if hasattr(grounder, "ground_batch"):
            hits = grounder.ground_batch([fp for _, fp in frame_items], phrase)
            for (frame_index, _), hit in zip(frame_items, hits):
                if hit is None:
                    continue
                bbox, score = hit
                detections.append((score, frame_index, bbox))
        else:
            for frame_index, frame_path in frame_items:
                hit = grounder.ground(frame_path, phrase)
                if hit is None:
                    continue
                bbox, score = hit
                detections.append((score, frame_index, bbox))
        if len(detections) >= min_frames:
            best = (_select_consistent_detection(
                detections, min_frames=min_frames, min_iou=temporal_iou_threshold,
                max_center_distance=temporal_center_threshold)
                if min_frames > 1 else max(detections, key=lambda d: d[0]))
            if best is not None and best[0] >= score_threshold:
                score, frame_index, bbox = best
                pil = Image.open(video_frames[frame_index]).convert("RGB")
                w, h = pil.size
                ymin, xmin, ymax, xmax = bbox
                box_px = (max(0, int(xmin * w / 1000)), max(0, int(ymin * h / 1000)),
                          min(w, int(xmax * w / 1000)), min(h, int(ymax * h / 1000)))
                if box_px[2] > box_px[0] and box_px[3] > box_px[1]:
                    pil.crop(box_px).save(crop_path)
                    return {"crop_path": str(crop_path), "bbox": bbox,
                            "bbox_source": "grounding_dino", "frame_index": frame_index,
                            "grounding_score": float(score)}

    # location, or no confident / temporally-consistent detection: keep the most informative
    # sampled frame (skips black fade-ins / uniform frames a blind "first frame" would grab).
    frame_index = max(video_frames, key=lambda i: _content_score(video_frames[i]))
    Image.open(video_frames[frame_index]).convert("RGB").save(crop_path)
    source = "full_frame" if ent["kind"] == "location" else "vlm_fallback"
    return {"crop_path": str(crop_path), "bbox": [0, 0, 1000, 1000],
            "bbox_source": source, "frame_index": frame_index, "grounding_score": 0.0}


def _perception_gate(located: list[tuple[dict[str, Any], dict[str, Any]]],
                     config: AnnotationConfig) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]],
                                                        list[dict[str, Any]]]:
    """Block deterministic crop failures before they can mutate the identity registry."""
    blocked: set[int] = set()
    checks: list[dict[str, Any]] = []
    for i, (ent, loc) in enumerate(located):
        if ent.get("kind") == "location" or loc.get("bbox_source") != "grounding_dino":
            continue
        area = _bbox_area(loc.get("bbox", []))
        if area > config.max_non_location_bbox_area:
            blocked.add(i)
            checks.append({"check": "blocking_crop", "passed": False,
                           "name": ent.get("name", ""),
                           "detail": f"{ent.get('name', '')}: bbox covers {area:.1%} of frame"})

    for i, (ent_a, loc_a) in enumerate(located):
        if i in blocked or ent_a.get("kind") == "location" or loc_a.get("bbox_source") != "grounding_dino":
            continue
        for j, (ent_b, loc_b) in enumerate(located[i + 1:], start=i + 1):
            if j in blocked or ent_b.get("kind") == "location" or loc_b.get("bbox_source") != "grounding_dino":
                continue
            iou = _bbox_iou(loc_a.get("bbox", []), loc_b.get("bbox", []))
            if iou >= config.same_chunk_bbox_iou_threshold:
                blocked.update({i, j})
                checks.append({"check": "blocking_crop", "passed": False,
                               "name": f"{ent_a.get('name', '')}; {ent_b.get('name', '')}",
                               "detail": (f"{ent_a.get('name', '')} and {ent_b.get('name', '')} "
                                          f"share bbox IoU={iou:.3f}")})

    kept = [item for idx, item in enumerate(located) if idx not in blocked]
    return kept, checks


def _run_branch(*, branch: int, attempt: int, chunk_id: int, registry: Registry,
                frames_a: dict[int, Path], frames_b: dict[int, Path], feedback: list[str],
                prev_prompt: str, annotator: AnnotatorVlm, verifier: VerifierVlm,
                grounder: Grounder, embedder: ImageEmbedder, crops_dir: Path, out: Path,
                config: AnnotationConfig, evlog: EventLog,
                max_workers: int, chunk_video: Path | None = None) -> dict[str, Any] | None:
    """One full candidate branch: discover -> ground+embed -> draft -> verify.

    On QA retry (attempt >= 2) with ``verifier_video_for_retry`` and a video-capable verifier,
    audit the chunk video clip instead of sparse frames (sparse sampling structurally misses
    in-between actions). Falls back to frames on any video failure."""
    who = {"chunk_id": chunk_id, "attempt": attempt, "branch": branch}
    # Diversity temperature: branch 0 / attempt 1 stays deterministic (0.0) so the canonical run
    # is reproducible; redundancy branches (branch >= 1) and QA retries (attempt >= 2) sample at
    # config.diversity_temperature so parallel candidates decorrelate instead of returning
    # near-identical outputs — temperature=0 across branches defeats branches_per_chunk>1
    # (correlated errors, principle #11). Scoring is unaffected (deterministic set ops over gold).
    temp = float(getattr(config, "diversity_temperature", 0.0)) if (attempt >= 2 or branch >= 1) else 0.0
    # COW registry: entities are deep-copied (consolidation mutates entity fields), but the
    # embeddings dict is only shallow-copied — consolidation appends new keys and reads old ones,
    # it never mutates an existing embedding list, so sharing the old lists read-only is safe and
    # avoids deepcopying every embedding vector on every branch (Pitfall_Notes: deepcopy cost with
    # branches_per_chunk>1 on long videos).
    reg_try = Registry(entities=copy.deepcopy(registry.entities),
                       embeddings=dict(registry.embeddings))
    # Known-entity prior for discovery: carry static_attributes (so the VLM can reuse identities
    # by stable keys, not just by name) and cap the list to the most-recently-introduced entities
    # to keep the discovery prompt from growing without bound on long videos. Description is
    # truncated; older entities are dropped (the VLM re-names them, consolidation reconciles via
    # embedding/static-attr matching). Pitfall_Notes: prompt-bloat on long videos.
    known_limit = getattr(config, "known_entity_limit", 60)
    recent = sorted(reg_try.entities.values(), key=lambda e: e.first_chunk, reverse=True)[:known_limit]
    known = [{"name": normalize_entity_name(e.name), "kind": e.kind, "description": e.description[:120],
              "static_attributes": dict(e.static_attributes)}
             for e in recent]

    evlog.emit("role_start", role="annotator", stage="discover", **who)
    discovered = annotator.discover_entities(list(frames_a.values()), known, feedback, temperature=temp)
    if discovered:
        seen_pairs = set()
        normalized_discovered = []
        for ent in discovered:
            norm_name = normalize_entity_name(ent.get("name") or "")
            if not norm_name:
                continue
            ent["name"] = norm_name
            pair = (norm_name.lower(), ent.get("kind"))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                normalized_discovered.append(ent)
        discovered = normalized_discovered

    evlog.emit("role_end", role="annotator", stage="discover", **who,
               entities=[{"name": e["name"], "kind": e["kind"],
                          "description": e["description"]} for e in discovered])
    if not discovered:
        return None

    evlog.emit("role_start", role="annotator", stage="ground+embed", **who)
    tag = f"c{chunk_id:03d}a{attempt}b{branch}"
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        located_raw = list(pool.map(
            lambda ent: (ent, _ground_and_crop(
                frames_a, ent, tag=tag, crops_dir=crops_dir,
                grounder=grounder, score_threshold=config.grounding_score_threshold,
                min_frames=config.grounding_min_frames,
                temporal_iou_threshold=config.grounding_temporal_iou_threshold,
                temporal_center_threshold=config.grounding_temporal_center_threshold)),
            discovered))
        located, blocking_checks = _perception_gate(located_raw, config)
        if not located:
            evlog.emit("role_end", role="annotator", stage="ground+embed", **who,
                       crops=[], blocking_checks=blocking_checks)
            return {"registry": reg_try, "annotation": {"prompt": "", "present": [],
                                                        "state_events": []},
                    "checks": blocking_checks, "n_failed": len(blocking_checks),
                    "branch": branch, "attempt": attempt}
        # Embed all located crops in one batched forward pass when the backend supports it
        # (DinoV3Embedder.embed_batch). Stub backends lack embed_batch -> fall back to per-image
        # pool.map. Real batching cuts N serialized single-image GPU calls per chunk*branch to
        # one, removing the per-call lock + Python overhead and the GPU idle gap (Pitfall_Notes: F5).
        crop_paths = [Path(item[1]["crop_path"]) for item in located]
        if hasattr(embedder, "embed_batch"):
            vectors = embedder.embed_batch(crop_paths)
        else:
            vectors = list(pool.map(lambda c: embedder.embed_image(c), crop_paths))

    present = []
    for (ent, loc), vector in zip(located, vectors):
        entity, rep, is_new = consolidate_observation(
            reg_try, chunk_id=chunk_id, name=ent["name"], kind=ent["kind"],
            description=ent["description"], vector=vector,
            static_attributes=ent.get("static_attributes") or {},
            static_overlap_threshold=config.static_overlap_threshold,
            judge_same_entity=annotator.judge_same_entity,
            high_threshold=config.high_threshold, low_threshold=config.low_threshold,
            **loc)
        first = is_new or entity.first_chunk == chunk_id
        present.append(present_payload(entity, rep, first_appearance=first))
    # Inject prior_representations so the drafter can scope deprecates_representations to
    # specific historical reps rather than always defaulting to "all prior reps".
    for p in present:
        ent_obj = reg_try.entities.get(p["entity_id"])
        p["prior_representations"] = [r.representation_id for r in ent_obj.representations
                                      if r.chunk_id < chunk_id] if ent_obj else []
    evlog.emit("role_end", role="annotator", stage="ground+embed", **who,
               crops=[{"name": p["name"], "crop_path": _rel(p["crop_path"], out)} for p in present],
               blocking_checks=blocking_checks)

    evlog.emit("role_start", role="annotator", stage="draft", **who)
    draft = annotator.draft_chunk(list(frames_a.values()), present, prev_prompt, feedback, temperature=temp)
    annotation = {"prompt": str(draft.get("prompt", "")), "present": present,
                  "state_events": list(draft.get("state_events", []))}
    evlog.emit("role_end", role="annotator", stage="draft", **who,
               prompt=annotation["prompt"][:2000], state_events=annotation["state_events"])

    evlog.emit("role_start", role="verifier", stage="verify", **who)
    if chunk_video is not None and hasattr(verifier, "verify_chunk_video"):
        try:
            checks = verifier.verify_chunk_video(chunk_video, annotation, temperature=temp)
        except Exception as exc:  # noqa: BLE001 (fallback is the point)
            logger.exception("chunk %03d attempt %d branch %d video verify failed; fallback to frames",
                             chunk_id, attempt, branch)
            evlog.emit("branch_error", chunk_id=chunk_id, attempt=attempt, branch=branch,
                       error=f"video verify: {type(exc).__name__}: {exc}")
            checks = verifier.verify_chunk(list(frames_b.values()), annotation, temperature=temp)
    else:
        checks = verifier.verify_chunk(list(frames_b.values()), annotation, temperature=temp)
    checks = blocking_checks + checks
    n_failed = sum(1 for c in checks if not c["passed"])
    evlog.emit("role_end", role="verifier", stage="verify", **who, n_failed=n_failed,
               checks=[{"check": c["check"], "passed": c["passed"],
                        "detail": str(c.get("detail", ""))[:500]} for c in checks])
    return {"registry": reg_try, "annotation": annotation, "checks": checks,
            "n_failed": n_failed, "branch": branch, "attempt": attempt}


def _branch_role_pairs(*, chunk_id: int, attempt: int, branches_per_chunk: int,
                       annotators: list, verifiers: list) -> list[tuple[int, Any, Any]]:
    """Per-chunk candidate branches with endpoint-pool round-robin.

    branches_per_chunk is the per-chunk redundancy knob (default 1); the annotator/verifier
    endpoint pool is round-robined across (chunk_id, attempt, branch) so supplying more endpoints
    increases dataset-level throughput, NOT per-chunk ensemble size (Pitfall_Notes root cause #7)."""
    if branches_per_chunk < 1:
        raise ValueError(f"branches_per_chunk must be >= 1, got {branches_per_chunk}")
    na, nv = len(annotators), len(verifiers)
    pairs: list[tuple[int, Any, Any]] = []
    for b in range(branches_per_chunk):
        idx = chunk_id * branches_per_chunk + b + (attempt - 1) * branches_per_chunk
        pairs.append((b, annotators[idx % na], verifiers[idx % nv]))
    return pairs


def _only_crop_failures(checks: list[dict[str, Any]]) -> bool:
    """True iff every failed check is a crop_match check — retrying will not help (the detector
    re-grounds the same way), so we early-stop the QA loop instead of wasting VLM calls
    (Pitfall_Notes root cause #4)."""
    failed = [c for c in checks if not c.get("passed")]
    return bool(failed) and all(c.get("check") in ("crop_match", "blocking_crop") for c in failed)


def _prune_failed_crops(registry: Registry, present: list[dict[str, Any]],
                        checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove crop_match-failed representations from the registry (and drop entities that lose
    all reps). Returns the surviving present payloads (entities that still have >=1 rep). Bad
    crops must never be copied into derived/assets/ (asset-bank pollution, Pitfall_Notes).

    A crop_match check carries either a ``representation_id`` (verifier-produced checks) or a
    ``name``/detail-name (legacy/manual checks); we match on whichever is present."""
    failed_by_rep: set[str] = set()
    failed_by_name: set[str] = set()
    for c in checks:
        if c.get("check") == "crop_match" and not c.get("passed"):
            rid = c.get("representation_id", "")
            if rid:
                failed_by_rep.add(rid)
            name = c.get("name") or (c.get("detail", "").split(":", 1)[0].strip()
                                     if c.get("detail") else "")
            if name:
                failed_by_name.add(name)
    if not failed_by_rep and not failed_by_name:
        return list(present)

    surviving: list[dict[str, Any]] = []
    for p in present:
        rid = p.get("representation_id", "")
        eid = p.get("entity_id")
        name = p.get("name", "")
        is_bad = rid in failed_by_rep or name in failed_by_name
        if is_bad and eid in registry.entities:
            entity = registry.entities[eid]
            entity.representations = [r for r in entity.representations
                                      if r.representation_id != rid]
            registry.embeddings.pop(rid, None)
            if not entity.representations:
                del registry.entities[eid]
                continue
        surviving.append(p)
    return surviving


def _union_feedback(results: list[dict[str, Any]]) -> list[str]:
    """Union of all failed-check details across branches (deduped), so a retry sees every
    distinct problem rather than only the single best branch's failures."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        for c in r.get("checks", []):
            if not c.get("passed"):
                detail = f"{c.get('check')}: {c.get('detail') or 'failed'}"
                if detail not in seen:
                    seen.add(detail)
                    out.append(detail)
    return out


def _best_cover_rep(entity: Entity, embeddings: dict[str, list[float]]) -> Representation | None:
    """Pick the most representative existing committed rep as the cover: highest grounding score,
    preferring grounded crops over full-frame/vlm_fallback. Avoids a bad first crop becoming the
    permanent cover shown in human review (Pitfall_Notes root cause: blind first-rep cover)."""
    committed = [r for r in entity.representations if r.crop_path]
    if not committed:
        return None

    def _key(r: Representation) -> tuple:
        gs = float(r.qa.get("grounding_score", 0.0))
        # grounding_dino (0) > vlm_fallback (1) > full_frame (2): prefer real detections.
        source_rank = 0 if r.bbox_source == "grounding_dino" else (1 if r.bbox_source == "vlm_fallback" else 2)
        return (gs, -source_rank)

    return max(committed, key=_key)


def _fallback_chunk_annotation(*, registry: Registry, chunk_id: int, shot_span: list[int],
                               frame_span: list[int], frames: dict[int, Path],
                               crops_dir: Path, embedder: ImageEmbedder, evlog: EventLog,
                               config: AnnotationConfig) -> dict[str, Any]:
    """Deterministic fallback when every branch's discovery returned nothing: inject a single
    location entity from the most informative sampled frame so the chunk ships instead of
    crashing the whole movie run (a dark/abstract chunk still has a setting). Idempotent across
    attempts. Ponytail: full-frame fallback; upgrade path is a dedicated scene-captioning model."""
    name = f"chunk_{chunk_id:03d}_setting"
    slug = slugify(name)
    entity_id = f"loc_{slug}"
    if entity_id in registry.entities:
        entity = registry.entities[entity_id]
    else:
        entity = Entity(entity_id=entity_id, kind="location", name=name,
                        description=f"the setting of chunk {chunk_id}", first_chunk=chunk_id)
        registry.entities[entity_id] = entity
    frame_index = max(frames, key=lambda i: _content_score(frames[i])) if frames else -1
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crops_dir / f"c{chunk_id:03d}_fallback_loc_{slug}.jpg"
    rep_id = f"{entity_id}@c{chunk_id:03d}"
    if not any(r.representation_id == rep_id for r in entity.representations):
        if frame_index >= 0 and frames:
            from PIL import Image
            Image.open(frames[frame_index]).convert("RGB").save(crop_path)
        rep = Representation(representation_id=rep_id, chunk_id=chunk_id, crop_path=str(crop_path),
                            bbox=[0, 0, 1000, 1000], bbox_source="full_frame",
                            frame_index=frame_index, embedding_key=rep_id,
                            qa={"grounding_score": 0.0})
        entity.representations.append(rep)
        registry.embeddings[rep_id] = (embedder.embed_image(crop_path)
                                       if frame_index >= 0 and frames else [])
    else:
        rep = next(r for r in entity.representations if r.representation_id == rep_id)
    present = [present_payload(entity, rep, first_appearance=True)]
    annotation = {"prompt": f"The scene continues in this location (chunk {chunk_id}).",
                  "present": present, "state_events": []}
    evlog.emit("fallback_location", chunk_id=chunk_id, entity_id=entity_id)
    return {"registry": registry, "annotation": annotation, "checks": [],
            "n_failed": 0, "branch": -1, "attempt": 0}


def annotate_movie(config: AnnotationConfig, *,
                   annotator: AnnotatorVlm | None = None, verifier: VerifierVlm | None = None,
                   annotators: list[AnnotatorVlm] | None = None,
                   verifiers: list[VerifierVlm] | None = None,
                   grounder: Grounder, embedder: ImageEmbedder,
                   shots: list[tuple[int, int]] | None = None, max_workers: int = 8,
                   resume: bool = False) -> dict[str, Any]:
    annotators = annotators or ([annotator] if annotator else [])
    verifiers = verifiers or ([verifier] if verifier else [])
    if not annotators or not verifiers:
        raise ValueError("need at least one annotator and one verifier")

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # tmp/ = regenerable-from-source process artifacts (gitignored): every-branch candidate
    # crops, frame cache; assets/ holds the committed per-entity asset library.
    dirs = MovieDirs(out, write=True)
    dirs.mkdirs()
    crops_dir = dirs.candidates
    crops_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = dirs.assets
    assets_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = dirs.frames
    frames_dir.mkdir(parents=True, exist_ok=True)

    evlog = EventLog(dirs.events)
    evlog.emit("run_start", movie_id=config.movie_id, video=str(config.video),
               n_annotators=len(annotators), n_verifiers=len(verifiers), stage="chunking")

    index = run_chunking(config.video, out, min_frames=config.min_frames_per_chunk,
                         max_frames=config.max_frames_per_chunk,
                         shots=shots, sbd_method=config.sbd_method,
                         min_scene_len_sec=config.min_scene_len_sec)
    fps = float(index["fps"])
    evlog.emit("layout", n_chunks=len(index["chunks"]), fps=fps,
               layout_hash=index["layout_hash"],
               chunks=[{"chunk_id": c["chunk_id"], "frame_span": c["frame_span"]}
                       for c in index["chunks"]])

    registry = Registry()
    presence_history: dict[str, list[int]] = {}
    annotations = []
    qa_report: list[dict[str, Any]] = []
    prev_prompt = ""
    # --resume: reload registry / committed chunks / presence history from the last checkpoint
    # and skip ahead to last_chunk_id + 1. Refuse to resume across a re-chunked layout (different
    # layout_hash => chunk ids no longer line up with the saved state). Pitfall_Notes: resilience.
    resume_from_chunk = -1
    if resume:
        cp = _load_checkpoint(out)
        if cp is not None:
            if cp["layout_hash"] != index["layout_hash"]:
                logger.warning("checkpoint layout_hash mismatch (%s vs %s); ignoring checkpoint",
                               cp["layout_hash"], index["layout_hash"])
            else:
                registry = cp["registry"]
                presence_history = cp["presence_history"]
                annotations = cp["annotations"]
                prev_prompt = cp["prev_prompt"]
                resume_from_chunk = cp["last_chunk_id"]
                logger.info("resuming from chunk %d (registry=%d entities, %d chunks done)",
                            resume_from_chunk + 1, len(registry.entities), len(annotations))
                evlog.emit("resume", last_chunk_id=resume_from_chunk,
                           n_entities=len(registry.entities), n_chunks_done=len(annotations))

    for chunk in index["chunks"]:
        chunk_id = int(chunk["chunk_id"])
        if chunk_id <= resume_from_chunk:
            continue  # already committed in a prior run; skip without re-calling any VLM
        first, last = chunk["frame_span"]   # closed inclusive [first, last]
        start, end = first, last + 1        # half-open for sampling / extraction
        t0 = time.time()
        evlog.emit("chunk_start", chunk_id=chunk_id, frame_span=[first, last])
        idx_a, idx_b = _sample_split(start, end, config.max_sampled_frames)

        def _extract(i: int) -> tuple[int, Path]:
            path = frames_dir / f"f{i:07d}.jpg"
            if not path.is_file():
                extract_frame(config.video, path, frame_index=i, fps=fps)
            return i, path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            frames_a = dict(pool.map(_extract, idx_a))
            frames_b = dict(pool.map(_extract, idx_b))

        feedback: list[str] = []
        accepted: dict[str, Any] | None = None   # first fully-passing branch
        best: dict[str, Any] | None = None       # fewest failed checks across all rounds
        rounds_used = 0

        chunk_video: Path | None = None  # lazily materialized for video-verify on retry

        for attempt in range(1, config.qa_max_rounds + 1):
            rounds_used = attempt
            pairs = _branch_role_pairs(chunk_id=chunk_id, attempt=attempt,
                                       branches_per_chunk=config.branches_per_chunk,
                                       annotators=annotators, verifiers=verifiers)
            # On retry (attempt >= 2), let the verifier audit the chunk video clip if enabled
            # and the verifier backend supports video. Clips go to tmp/clips/ (gitignored,
            # never shipped), never out/chunks.
            if (attempt >= 2 and config.verifier_video_for_retry and chunk_video is None
                    and any(hasattr(v, "verify_chunk_video") for v in verifiers)):
                chunk_video = dirs.clips / f"chunk_{chunk_id:03d}.mp4"
                if not chunk_video.is_file():
                    materialize_clip(config.video, chunk_video,
                                     frame_span=chunk["frame_span"], fps=fps)
            with ThreadPoolExecutor(max_workers=max(1, config.branches_per_chunk)) as pool:
                futures = [pool.submit(
                    _run_branch, branch=b, attempt=attempt, chunk_id=chunk_id,
                    registry=registry, frames_a=frames_a, frames_b=frames_b,
                    feedback=feedback, prev_prompt=prev_prompt,
                    annotator=ann, verifier=ver,
                    grounder=grounder, embedder=embedder, crops_dir=crops_dir, out=out,
                    config=config, evlog=evlog, max_workers=max_workers,
                    chunk_video=chunk_video if attempt >= 2 else None)
                    for (b, ann, ver) in pairs]
                # A single branch raising must not abort the run: log it and drop that branch
                # (the other branch's candidate still stands; if all fail we retry this round).
                results = []
                for b, fut in enumerate(futures):
                    try:
                        results.append(fut.result())
                    except Exception as exc:  # noqa: BLE001 (branch isolation is the point)
                        logger.exception("chunk %03d attempt %d branch %d crashed", chunk_id, attempt, b)
                        evlog.emit("branch_error", chunk_id=chunk_id, attempt=attempt,
                                   branch=b, error=f"{type(exc).__name__}: {exc}")

            results = [r for r in results if r is not None]
            if not results:
                feedback = ["discovery returned no entities; every chunk has at least a location"]
                continue
            round_best = min(results, key=lambda r: r["n_failed"])
            if best is None or round_best["n_failed"] < best["n_failed"]:
                best = round_best
            if round_best["n_failed"] == 0:
                accepted = round_best
                break
            # crop-only failures cannot be repaired by re-running discover/draft (the detector
            # re-grounds the same way) -> early-stop instead of wasting VLM calls.
            if _only_crop_failures(round_best["checks"]):
                logger.info("chunk %03d attempt %d: only crop_match failures -> early stop",
                            chunk_id, attempt)
                break
            feedback = _union_feedback(results)
            logger.info("chunk %03d attempt %d: best branch %d failed %d checks: %s",
                        chunk_id, attempt, round_best["branch"], round_best["n_failed"], feedback)

        flagged = accepted is None
        chosen = accepted or best
        if chosen is None:
            # Every round returned zero entities: inject a deterministic fallback location so
            # the chunk still ships (a dark/abstract chunk should not crash the whole movie run).
            chosen = _fallback_chunk_annotation(
                registry=registry, chunk_id=chunk_id, shot_span=chunk["shot_span"],
                frame_span=chunk["frame_span"], frames=frames_a or frames_b,
                crops_dir=crops_dir, embedder=embedder, evlog=evlog, config=config)
            flagged = True
            rounds_used = rounds_used or config.qa_max_rounds
        checks = chosen["checks"]
        registry = chosen["registry"]
        annotation = chosen["annotation"]
        present = annotation["present"]
        # Prune crop_match-failed reps before committing so bad crops never enter the asset bank.
        present = _prune_failed_crops(registry, present, checks)
        annotation["present"] = present

        # Commit surviving new crops into the per-entity asset library and rewrite crop_path to a
        # portable relative path. Losing branches stay in derived/candidates. The cover is the
        # best rep of each touched entity (highest grounding score, prefer real detections), not
        # whichever crop happened to commit first.
        touched_entities: set[str] = set()
        for payload in present:
            entity = registry.entities.get(payload["entity_id"])
            if entity is None:
                continue
            for rep in entity.representations:
                if rep.chunk_id == chunk_id and Path(rep.crop_path).is_file():
                    rep.qa.update({"verified": not flagged, "rounds": rounds_used,
                                   "flagged": flagged})
                    edir = entity_asset_dir(assets_dir, entity.entity_id, entity.kind)
                    edir.mkdir(parents=True, exist_ok=True)
                    name = rep.representation_id.split("@")[-1].replace(".", "_")
                    dst = edir / f"{name}.jpg"
                    shutil.copyfile(rep.crop_path, dst)
                    rep.crop_path = _rel(dst, out)
                    touched_entities.add(entity.entity_id)
        for eid in touched_entities:
            entity = registry.entities[eid]
            edir = entity_asset_dir(assets_dir, eid, entity.kind)
            cover = _best_cover_rep(entity, registry.embeddings)
            # rep.crop_path is relative to out (rewritten during commit); resolve against out.
            cover_src = Path(out / cover.crop_path) if cover is not None else None
            if cover_src is not None and cover_src.is_file():
                shutil.copyfile(cover_src, edir / "cover.jpg")
            elif not (edir / "cover.jpg").is_file():
                first_rep = next((r for r in entity.representations
                                  if Path(out / r.crop_path).is_file()), None)
                if first_rep is not None:
                    shutil.copyfile(Path(out / first_rep.crop_path), edir / "cover.jpg")
        state_events = state_events_from_draft(registry, chunk_id, annotation["state_events"])
        present_ids = [p["entity_id"] for p in present if p["entity_id"] in registry.entities]
        first_ids = {p["entity_id"] for p in present
                     if p["first_appearance"] and p["entity_id"] in registry.entities}
        annotations.append(build_chunk_annotation(
            chunk_id=chunk_id, shot_span=chunk["shot_span"], frame_span=chunk["frame_span"],
            prompt=annotation["prompt"], present_ids=present_ids, first_ids=first_ids,
            registry=registry, presence_history=presence_history,
            has_state_event=bool(state_events)))
        for eid in present_ids:
            presence_history.setdefault(eid, []).append(chunk_id)
        prev_prompt = annotation["prompt"]
        qa_report.append({"chunk_id": chunk_id, "rounds": rounds_used, "flagged": flagged,
                          "accepted_branch": chosen.get("branch", -1),
                          "failed_checks": [c for c in checks if not c["passed"]],
                          "seconds": round(time.time() - t0, 1)})
        evlog.emit("chunk_done", chunk_id=chunk_id, rounds=rounds_used, flagged=flagged,
                   branch=chosen.get("branch", -1), n_present=len(present_ids),
                   seconds=round(time.time() - t0, 1))
        # Per-chunk registry event carries only the entities TOUCHED this chunk (delta), not the
        # whole registry every chunk — O(chunks*entities) full-dup would balloon the event log on
        # long videos (Pitfall_Notes: events.jsonl bloat). The full registry ships once in
        # registry_final at run_done and in gold/entity_registry.json.
        evlog.emit("registry", chunk_id=chunk_id, n_entities=len(registry.entities), entities=[
            {"entity_id": e.entity_id, "name": e.name, "kind": e.kind,
             "description": e.description[:300], "first_chunk": e.first_chunk,
             "n_reps": len(e.representations),
             "cover": _rel(entity_asset_dir(assets_dir, e.entity_id, e.kind) / "cover.jpg", out)}
            for e in (registry.entities[eid] for eid in touched_entities
                      if eid in registry.entities)])
        logger.info("chunk %03d done: %d present, rounds=%d, flagged=%s",
                    chunk_id, len(present_ids), rounds_used, flagged)
        # Checkpoint after every chunk so a crash at chunk N only costs redoing chunk N, not 0..N.
        # The embeddings sidecar is only written every checkpoint_embedding_interval chunks (F6);
        # checkpoint.json + checkpoint_registry.json are cheap and written every chunk, so a resume
        # always has the registry + committed annotations — only the sidecar lags, and
        # _load_checkpoint truncates the resume point to the sidecar's covered chunk when it does.
        interval = max(1, int(getattr(config, "checkpoint_embedding_interval", 1)))
        _write_checkpoint(out, index, registry, annotations, presence_history, prev_prompt,
                          last_chunk_id=chunk_id, movie_id=config.movie_id,
                          pipeline_version=config.pipeline_version, vlm_model=config.vlm_model,
                          embedder_name=config.embedder_name, layout_hash=index["layout_hash"],
                          write_embeddings=(chunk_id % interval == 0))

    summary = _persist(config, out, index, registry, annotations, qa_report)
    # Cost observability (Pitfall_Notes: cost optimization needs a feedback signal). Sum per-judger
    # token counters across all annotator + verifier judgers into one run-wide usage record. Stub
    # roles (tests) have no .judger -> getattr chain returns 0, so the signal is opt-in per backend.
    def _jt(role, attr):
        return int(getattr(getattr(role, "judger", None), attr, 0))
    total_pt = (sum(_jt(a, "total_prompt_tokens") for a in annotators)
                + sum(_jt(v, "total_prompt_tokens") for v in verifiers))
    total_ct = (sum(_jt(a, "total_completion_tokens") for a in annotators)
                + sum(_jt(v, "total_completion_tokens") for v in verifiers))
    evlog.emit("usage", prompt_tokens=total_pt, completion_tokens=total_ct,
               total_tokens=total_pt + total_ct)
    # Full registry ships ONCE at run_done (chunk-level events are deltas only). Lets a live
    # dashboard reconstruct the final state without reading every chunk's delta.
    evlog.emit("registry_final", n_entities=len(registry.entities), entities=[
        {"entity_id": e.entity_id, "name": e.name, "kind": e.kind,
         "description": e.description[:300], "first_chunk": e.first_chunk,
         "n_reps": len(e.representations),
         "cover": _rel(entity_asset_dir(assets_dir, e.entity_id, e.kind) / "cover.jpg", out)}
        for e in registry.entities.values()])
    summary["total_tokens"] = total_pt + total_ct
    # Final checkpoint so a later --resume sees the run as complete (last_chunk_id = last chunk).
    _write_checkpoint(out, index, registry, annotations, presence_history, prev_prompt,
                      last_chunk_id=int(index["chunks"][-1]["chunk_id"]),
                      movie_id=config.movie_id, pipeline_version=config.pipeline_version,
                      vlm_model=config.vlm_model, embedder_name=config.embedder_name,
                      layout_hash=index["layout_hash"])
    evlog.emit("run_done", **{k: v for k, v in summary.items() if k != "gold_dir"})
    return summary


def _write_checkpoint(out: Path, index: dict[str, Any], registry: Registry,
                     annotations: list, presence_history: dict[str, list[int]],
                     prev_prompt: str, *, last_chunk_id: int, movie_id: str,
                     pipeline_version: str, vlm_model: str, embedder_name: str,
                     layout_hash: str, write_embeddings: bool = True) -> None:
    """Persist a resumable checkpoint after every chunk_done (and once at run_done). A later
    --resume loads this and skips already-completed chunks instead of restarting the movie from
    chunk 0 — long videos no longer lose hours of VLM work to a single late crash (Pitfall_Notes:
    long-video resilience). Ponytail: JSON for registry/chunks + safetensors for embeddings (same
    shape as gold); a real DB would be overkill for an offline annotator.

    ``write_embeddings`` controls the safetensors sidecar only (the JSON files are always
    written — they are cheap). The pipeline writes the sidecar every
    ``checkpoint_embedding_interval`` chunks and once at run_done; rewriting O(embeddings)
    safetensors every chunk is the dominant checkpoint cost on long videos (Pitfall_Notes: F6).
    On --resume, if the sidecar lags behind last_chunk_id, _load_checkpoint truncates the resume
    point to the sidecar's covered chunk. Ceiling: no incremental keyset rotation, so the sidecar
    is still rewritten in full when it IS written — upgrade path is an append-only embedding store
    if chunk counts reach thousands."""
    build = MovieDirs(out, write=True).tmp
    build.mkdir(parents=True, exist_ok=True)
    payload = {"last_chunk_id": last_chunk_id, "prev_prompt": prev_prompt,
               "presence_history": presence_history,
               "chunks": [a.to_dict() for a in annotations],
               "movie_id": movie_id, "pipeline_version": pipeline_version,
               "vlm_model": vlm_model, "embedder_name": embedder_name,
               "layout_hash": layout_hash}
    (build / "checkpoint.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    er = EntityRegistry(movie_id=movie_id, entities=list(registry.entities.values()),
                        human_reviewed=False,
                        annotation_provenance={"vlm": vlm_model, "embedder": embedder_name,
                                               "pipeline_version": pipeline_version,
                                               "layout_hash": layout_hash})
    (build / "checkpoint_registry.json").write_text(
        json.dumps(er.to_dict(), ensure_ascii=False), encoding="utf-8")
    if write_embeddings and registry.embeddings:
        import numpy as np
        from safetensors.numpy import save_file
        save_file({k: np.asarray(v, dtype=np.float32) for k, v in registry.embeddings.items()},
                  str(build / "checkpoint_embeddings.safetensors"))


def _load_checkpoint(out: Path) -> dict[str, Any] | None:
    """Load a resumable checkpoint; return None when none exists. Rebuilds a Registry (entities
    + embeddings sidecar) and the committed chunk annotations so --resume can skip ahead to
    last_chunk_id + 1. Returns the layout_hash too so the caller can refuse to resume across a
    re-chunked layout (different layout_hash => chunk ids no longer line up)."""
    build = MovieDirs(out).tmp
    cp = build / "checkpoint.json"
    if not cp.is_file():
        return None
    payload = json.loads(cp.read_text(encoding="utf-8"))
    reg_path = build / "checkpoint_registry.json"
    er = EntityRegistry.from_dict(json.loads(reg_path.read_text(encoding="utf-8")))
    registry = Registry(entities={e.entity_id: e for e in er.entities})
    emb_path = build / "checkpoint_embeddings.safetensors"
    if emb_path.is_file():
        from safetensors.numpy import load_file
        registry.embeddings = {k: list(map(float, v))
                               for k, v in load_file(str(emb_path)).items()}
    chunks = ChunkAnnotations.from_dict({"movie_id": payload.get("movie_id", ""),
                                         "chunks": payload["chunks"],
                                         "human_reviewed": False})
    last_chunk_id = int(payload["last_chunk_id"])
    annotations_list = list(chunks.chunks)
    # F6: the embeddings sidecar is only written every checkpoint_embedding_interval chunks, so it
    # may lag behind last_chunk_id. Truncate the resume point to the sidecar's covered chunk and
    # drop the to-be-rerun chunks' reps from the registry so consolidate does not duplicate them
    # when those chunks re-run. Re-running <= interval chunks is cheap; resuming with missing
    # embeddings would silently degrade identity matching for the rest of the run.
    if registry.embeddings:
        def _chunk_of_key(k: str) -> int:
            tail = k.split("@c")[-1].split(".")[0]
            try:
                return int(tail)
            except ValueError:
                return -1
        max_emb_chunk = max(_chunk_of_key(k) for k in registry.embeddings)
        if max_emb_chunk < last_chunk_id:
            logger.warning("checkpoint embeddings cover up to chunk %d but last_chunk_id=%d; "
                           "truncating resume point to %d (re-running %d chunks to rebuild embeddings)",
                           max_emb_chunk, last_chunk_id, max_emb_chunk,
                           last_chunk_id - max_emb_chunk)
            last_chunk_id = max_emb_chunk
            annotations_list = [a for a in annotations_list if a.chunk_id <= max_emb_chunk]
            for ent in list(registry.entities.values()):
                ent.representations = [r for r in ent.representations if r.chunk_id <= max_emb_chunk]
                if not ent.representations:
                    del registry.entities[ent.entity_id]
    return {"last_chunk_id": last_chunk_id,
            "prev_prompt": str(payload.get("prev_prompt", "")),
            "presence_history": payload.get("presence_history", {}),
            "annotations": annotations_list,
            "registry": registry,
            "layout_hash": payload.get("layout_hash")}


def _persist(config: AnnotationConfig, out: Path, index: dict[str, Any], registry: Registry,
             annotations: list, qa_report: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = MovieDirs(out, write=True)
    gold_dir = dirs.gold
    gold_dir.mkdir(parents=True, exist_ok=True)

    entity_registry = EntityRegistry(
        movie_id=config.movie_id, entities=list(registry.entities.values()),
        human_reviewed=False,
        annotation_provenance={"vlm": config.vlm_model, "embedder": config.embedder_name,
                               "perception_backend": getattr(config, "perception_backend", ""),
                               "tracker": getattr(config, "tracker_name", ""),
                               "reid": getattr(config, "reid_name", ""),
                               "face_encoder": getattr(config, "face_encoder_name", ""),
                               "identity_resolution_mode": getattr(
                                   config, "identity_resolution_mode", "greedy"),
                               "precluster_linkage": getattr(config, "precluster_linkage", ""),
                               "roster_mode": (
                                   "seeded" if getattr(config, "roster_seed_path", None)
                                   else "proposal"),
                               "roster_seed": str(getattr(config, "roster_seed_path", "") or ""),
                               "production_mode": bool(getattr(config, "production_mode", False)),
                               "pipeline_version": config.pipeline_version,
                               "layout_hash": index["layout_hash"]})
    (gold_dir / "entity_registry.json").write_text(
        json.dumps(entity_registry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    chunk_annotations = ChunkAnnotations(movie_id=config.movie_id, chunks=annotations,
                                         human_reviewed=False)
    (gold_dir / "chunk_annotations.json").write_text(
        json.dumps(chunk_annotations.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    # Embeddings: safetensors sidecar only, never inline JSON (principle #10). Not
    # regenerable without the exact embedder, so this ships inside gold/.
    import numpy as np
    from safetensors.numpy import save_file
    if registry.embeddings:
        save_file({k: np.asarray(v, dtype=np.float32) for k, v in registry.embeddings.items()},
                  str(gold_dir / "embeddings.safetensors"))

    dirs.qa_report.parent.mkdir(parents=True, exist_ok=True)
    dirs.qa_report.write_text(
        json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_manifest(config, out, index, entity_registry, chunk_annotations)
    _write_gitignore(out)

    from vmem_bench.annotation.pipeline_track_first.review import generate_review_html
    review_path = generate_review_html(out, spot_check=config.review_spot_check,
                                       seed=config.review_seed)
    n_flagged = sum(1 for q in qa_report if q["flagged"])
    logger.info("annotation drafts written to %s (%d chunks, %d entities, %d flagged); review: %s",
                gold_dir, len(annotations), len(registry.entities), n_flagged, review_path)
    return {"gold_dir": str(gold_dir), "review_html": str(review_path),
            "n_chunks": len(annotations), "n_entities": len(registry.entities),
            "n_flagged_chunks": n_flagged, "layout_hash": index["layout_hash"]}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_manifest(config: AnnotationConfig, out: Path, index: dict[str, Any],
                    registry: EntityRegistry, chunks: ChunkAnnotations) -> None:
    """Ship-with-the-benchmark provenance: how to obtain the source video + what was built.

    The benchmark distributes only manifest + gold/; the video, clips, frames and
    crops are all re-derivable from the source video the consumer downloads themselves.
    """
    video = Path(config.video).resolve()
    info = probe_media(video)
    manifest = {
        "schema_version": "2.0.0",
        "movie_id": config.movie_id,
        "source": {
            "filename": video.name,
            "sha256": _sha256(video),
            "bytes": video.stat().st_size,
            "fps": info.fps,
            "duration_sec": round(info.duration_sec, 3),
            "width": info.width,
            "height": info.height,
            "download": {
                "dataset": config.source_dataset,
                "local_path": str(video),
                "url": config.source_url,
                "note": "Obtain this file, verify sha256, then re-derive clips via "
                        "gold/chunk_index.json (vmem_bench.annotation.pipeline_track_first.chunking.materialize_clip).",
            },
        },
        "layout": {
            "layout_hash": index["layout_hash"],
            "n_chunks": len(index["chunks"]),
            "min_frames_per_chunk": index["min_frames_per_chunk"],
            "max_frames_per_chunk": index["max_frames_per_chunk"],
            # Credits/title spans removed before chunking; evaluation MUST NOT hand these
            # frames/seconds to the SUT (they are outside every chunk's frame_span).
            "excluded_segments": index.get("excluded_segments", []),
        },
        "annotation": {
            "vlm": config.vlm_model,
            "embedder": config.embedder_name,
            "pipeline_version": config.pipeline_version,
            "roster_mode": ("seeded" if getattr(config, "roster_seed_path", None)
                            else "proposal"),
            "production_mode": bool(getattr(config, "production_mode", False)),
            "n_entities": len(registry.entities),
            "human_reviewed": registry.human_reviewed,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_gitignore(out: Path) -> None:
    """Only manifest + gold/ are versioned; everything re-derivable is ignored."""
    (out / ".gitignore").write_text(
        "# Re-derivable from the source video (see manifest.json); do not ship.\n"
        "tmp/\nassets/\nlogs/\nreview.html\nreview_patch.json\n"
        "# legacy layout names\nderived/\nbuild/\n",
        encoding="utf-8")
