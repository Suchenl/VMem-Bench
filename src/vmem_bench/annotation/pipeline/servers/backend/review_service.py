"""Review payload assembly and media resolution for S4/S6 consoles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.servers.backend.catalog_state import (
    get_sample,
    memstrata_root_from_here,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.decisions import (
    apply_s4_decisions,
    load_s4_draft,
    save_s4_draft,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.clips import (
    ensure_segment_clip,
    ensure_segment_poster,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    unmasked_companion_path,
)
from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.review_apply import (
    apply_s6_decisions,
    ensure_s6_queue,
    load_s6_draft,
)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _entity_index(annotation: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for item in annotation.get(group) or []:
            entity_id = str(item.get(id_key) or "")
            if entity_id:
                out[entity_id] = {
                    "entity_id": entity_id,
                    "kind": kind,
                    "name": str(item.get("name") or entity_id),
                    "description": str(item.get("description") or ""),
                }
    return out


def _segment_index(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        for segment in scene.get("visual_segments") or []:
            segment_id = str(segment.get("segment_id") or "")
            if segment_id:
                out[segment_id] = dict(segment)
    return out


def _s2_source_segment(
    segment_id: str, s2_segments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Resolve original S2 segment, walking hard-split parents (``_a`` / ``_b``)."""
    current = str(segment_id or "")
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        hit = s2_segments.get(current)
        if hit:
            return hit
        trimmed = re.sub(r"(?:__?[ab])$", "", current)
        if trimmed == current:
            break
        current = trimmed
    return {}


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _media_relpath(path: Path) -> str:
    """Prefer a MemStrata-root-relative path for /api/review/media URLs."""
    resolved = path.resolve()
    root = memstrata_root_from_here().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _review_media_url(sample: dict[str, Any], path: str | Path) -> str:
    from urllib.parse import urlencode

    text = str(path or "").strip()
    if not text:
        return ""
    p = Path(text)
    # Keep already-normalized relative paths as-is; only absolutize absolute inputs.
    # Path("data/...").resolve() would otherwise anchor to process CWD and break URLs.
    rel = _media_relpath(p) if p.is_absolute() else text.replace("\\", "/")
    query = urlencode(
        {
            "dataset": sample["dataset"],
            "movie_id": sample["movie_id"],
            "path": rel,
        }
    )
    return f"/api/review/media?{query}"


def _iter_image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    rows: list[Path] = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            rows.append(path)
    return rows


def _fallback_crop_path(s5_dir: Path, proposal: dict[str, Any]) -> str:
    """Pick a preview image when the proposal has no accepted crop_path."""
    existing = str(proposal.get("crop_path") or "").strip()
    if existing and Path(existing).is_file():
        return _media_relpath(Path(existing))

    entity_id = str(proposal.get("entity_id") or "")
    kind = str(proposal.get("kind") or "character")
    if not entity_id:
        return ""

    chunk_id = proposal.get("chunk_id")
    chunk_token = None
    try:
        if chunk_id is not None and str(chunk_id) != "":
            chunk_token = f"c{int(chunk_id):05d}"
    except (TypeError, ValueError):
        chunk_token = None

    ordered: list[Path] = []
    if chunk_token:
        ordered.extend(_iter_image_files(s5_dir / "propose_scratch" / kind / entity_id / chunk_token))
        cand_dir = s5_dir / "candidates" / kind / entity_id
        ordered.extend(
            path for path in _iter_image_files(cand_dir) if path.name.startswith(f"{chunk_token}_")
        )
    ordered.extend(_iter_image_files(s5_dir / "candidates" / kind / entity_id))
    # de-dupe while preserving order
    seen: set[Path] = set()
    for path in ordered:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        return _media_relpath(path)
    return ""


def resolve_review_media(sample: dict[str, Any], rel: str) -> Path | None:
    """Resolve a crop/frame path under the movie or MemStrata root."""
    if not rel or ".." in Path(rel).parts:
        return None
    movie_dir = Path(str(sample["movie_dir"])).resolve()
    memstrata_root = memstrata_root_from_here().resolve()
    candidates = [
        (movie_dir / rel).resolve(),
        (memstrata_root / rel).resolve(),
        Path(rel).resolve() if Path(rel).is_absolute() else None,
    ]
    allowed_roots = [
        (movie_dir / "tmp" / "pipeline").resolve(),
        (movie_dir / "gold" / "crops").resolve(),
        (memstrata_root / "data").resolve(),
    ]
    for target in candidates:
        if target is None or not target.is_file():
            continue
        if target.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if any(root == target or root in target.parents for root in allowed_roots):
            return target
    return None


def s4_payload(sample: dict[str, Any]) -> dict[str, Any]:
    movie_dir = Path(str(sample["movie_dir"]))
    pipeline = movie_dir / "tmp" / "pipeline"
    s4_dir = pipeline / "s4_segment_sampling_human_review"
    queue = _maybe_json(s4_dir / "review_queue.json") or []
    annotation = _maybe_json(pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json") or {}
    entities = _entity_index(annotation if isinstance(annotation, dict) else {})
    segments = _segment_index(annotation if isinstance(annotation, dict) else {})
    # Trust the on-disk stratified sample queue; do not expand to all segments.
    draft = load_s4_draft(movie_dir)
    cards = []
    for item in queue if isinstance(queue, list) else []:
        segment_id = str(item.get("segment_id") or "")
        segment = segments.get(segment_id, {})
        present = list(item.get("revised_present") or segment.get("present_entity_ids") or [])
        start = segment.get("start_seconds", item.get("start_seconds"))
        end = segment.get("end_seconds", item.get("end_seconds"))
        clip_url = ""
        poster_url = ""
        if start is not None and end is not None:
            clip_url = (
                "/api/review/segment-clip"
                f"?dataset={sample['dataset']}&movie_id={sample['movie_id']}"
                f"&segment_id={segment_id}"
            )
            poster_url = (
                "/api/review/segment-poster"
                f"?dataset={sample['dataset']}&movie_id={sample['movie_id']}"
                f"&segment_id={segment_id}"
            )
        cards.append(
            {
                "segment_id": segment_id,
                "confidence": item.get("confidence"),
                "verdict": item.get("verdict") or ("PASS" if item.get("accepted") else "WARN"),
                "findings": list(item.get("findings") or []),
                "recommended_action": item.get("recommended_action") or "spot_check",
                "risk_reasons": list(item.get("risk_reasons") or []),
                "revised_action": item.get("revised_action") or segment.get("action") or "",
                "revised_present": present,
                "present_entities": [entities.get(eid, {"entity_id": eid, "name": eid, "kind": ""}) for eid in present],
                "start_seconds": start,
                "end_seconds": end,
                "clip_url": clip_url,
                "poster_url": poster_url,
                "raw": item.get("raw") or {},
            }
        )
    audit = _maybe_json(s4_dir / "review_audit.json") or {}
    return {
        "sample": sample,
        "available": bool(cards),
        "cards": cards,
        "entities": list(entities.values()),
        "draft": draft,
        "audit": audit,
        "n_total_segments": len(segments),
        "paths": {
            "queue": str(s4_dir / "review_queue.json"),
            "annotation": str(pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json"),
        },
    }


def s6_payload(sample: dict[str, Any]) -> dict[str, Any]:
    movie_dir = Path(str(sample["movie_dir"]))
    pipeline = movie_dir / "tmp" / "pipeline"
    s5_dir = pipeline / "s5_entities_visual_crop_acquisition"
    proposals = _maybe_json(s5_dir / "crop_proposals.json") or []
    queue = ensure_s6_queue(movie_dir) if (s5_dir / "crop_proposals.json").is_file() else []
    draft = load_s6_draft(movie_dir)
    by_id = {
        str(item.get("representation_id") or ""): item
        for item in proposals if isinstance(item, dict)
    }
    cards = []
    for item in queue:
        proposal = dict(item.get("proposal") or {})
        rep_id = str(proposal.get("representation_id") or item.get("card_id") or "").removeprefix("crop:")
        if not proposal and rep_id in by_id:
            proposal = dict(by_id[rep_id])
        if not proposal and isinstance(item.get("entity_id"), str) and item["entity_id"] in by_id:
            proposal = dict(by_id[item["entity_id"]])
        crop_path = _fallback_crop_path(s5_dir, proposal)
        image_url = _review_media_url(sample, crop_path) if crop_path else ""
        qa = proposal.get("qa") if isinstance(proposal.get("qa"), dict) else {}
        accepted = proposal.get("accepted")
        if accepted is None and isinstance(qa, dict) and "accepted" in qa:
            accepted = qa.get("accepted")
        reasons: list[Any] = []
        for reason in list(item.get("reasons") or []):
            if reason and reason not in reasons:
                reasons.append(reason)
        if isinstance(proposal.get("reason"), str) and proposal["reason"] and proposal["reason"] not in reasons:
            reasons.append(proposal["reason"])
        if isinstance(qa.get("reasons"), list):
            for reason in qa["reasons"]:
                if reason and reason not in reasons:
                    reasons.append(reason)
        task_kind = str(proposal.get("task_kind") or "acquire")
        if accepted is True:
            crop_status = "accepted"
        elif accepted is False:
            crop_status = "rejected"
        elif not crop_path:
            crop_status = "missing_crop"
        else:
            crop_status = "review"
        cards.append(
            {
                "card_id": item.get("card_id") or f"crop:{rep_id}",
                "representation_id": rep_id or str(proposal.get("representation_id") or ""),
                "chunk_id": item.get("chunk_id", proposal.get("chunk_id")),
                "segment_id": item.get("segment_id", proposal.get("segment_id")),
                "entity_id": item.get("entity_id", proposal.get("entity_id")),
                "name": proposal.get("name") or item.get("entity_id"),
                "kind": proposal.get("kind") or "",
                "description": proposal.get("description") or "",
                "action": proposal.get("action") or "",
                "bbox_norm": proposal.get("bbox_norm") or [],
                "qa": qa,
                "accepted": accepted,
                "reasons": reasons,
                "recommended_action": item.get("recommended_action") or ("keep" if accepted is True else "review"),
                "review_tier": item.get("review_tier") or "spot_check",
                "task_kind": task_kind,
                "bind_source_chunk_id": proposal.get("bind_source_chunk_id"),
                "crop_status": crop_status,
                "crop_path": crop_path,
                "image_url": image_url,
                # Alternates are loaded on demand to keep the page responsive.
                "has_alternates": True,
            }
        )
    audit = _maybe_json(pipeline / "s6_entities_visual_crop_human_review" / "review_audit.json") or {}
    return {
        "sample": sample,
        "available": bool(cards) or bool(proposals),
        "cards": cards,
        "draft": draft,
        "audit": audit,
        "paths": {
            "proposals": str(s5_dir / "crop_proposals.json"),
            "queue": str(pipeline / "s6_entities_visual_crop_human_review" / "review_queue.json"),
        },
    }


def build_segment_clip(
    sample: dict[str, Any],
    segment_id: str,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Path:
    movie_dir = Path(str(sample["movie_dir"]))
    pipeline = movie_dir / "tmp" / "pipeline"
    annotation = _maybe_json(pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json") or {}
    segments = _segment_index(annotation if isinstance(annotation, dict) else {})
    segment = segments.get(segment_id) or {}
    start = start_seconds if start_seconds is not None else segment.get("start_seconds")
    end = end_seconds if end_seconds is not None else segment.get("end_seconds")
    if start is None or end is None:
        start, end = _audit_segment_window(pipeline / "s3_segment_auto_review_revise", segment_id)
    if start is None or end is None:
        # Fall back to S2 timeline for superseded parents still shown in the live feed.
        s2 = _maybe_json(pipeline / "s2_annotation_postprocess" / "normalized_annotation.json") or {}
        if isinstance(s2, dict) and isinstance(s2.get("annotation"), dict):
            s2 = s2["annotation"]
        s2_seg = _segment_index(s2 if isinstance(s2, dict) else {}).get(segment_id) or {}
        start = s2_seg.get("start_seconds") if start is None else start
        end = s2_seg.get("end_seconds") if end is None else end
    if start is None or end is None:
        raise FileNotFoundError(f"segment window not found: {segment_id}")
    source = Path(str(sample.get("source_video") or ""))
    cache = pipeline / "s4_segment_sampling_human_review" / "clips"
    return ensure_segment_clip(
        source_video=source,
        cache_dir=cache,
        segment_id=segment_id,
        start_seconds=float(start),
        end_seconds=float(end),
    )


def build_segment_poster(
    sample: dict[str, Any],
    segment_id: str,
    *,
    start_seconds: float | None = None,
) -> Path:
    movie_dir = Path(str(sample["movie_dir"]))
    pipeline = movie_dir / "tmp" / "pipeline"
    annotation = _maybe_json(pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json") or {}
    segments = _segment_index(annotation if isinstance(annotation, dict) else {})
    segment = segments.get(segment_id) or {}
    start = start_seconds if start_seconds is not None else segment.get("start_seconds")
    if start is None:
        start, _end = _audit_segment_window(pipeline / "s3_segment_auto_review_revise", segment_id)
    if start is None:
        s2 = _maybe_json(pipeline / "s2_annotation_postprocess" / "normalized_annotation.json") or {}
        if isinstance(s2, dict) and isinstance(s2.get("annotation"), dict):
            s2 = s2["annotation"]
        s2_seg = _segment_index(s2 if isinstance(s2, dict) else {}).get(segment_id) or {}
        start = s2_seg.get("start_seconds")
    if start is None:
        raise FileNotFoundError(f"segment window not found: {segment_id}")
    source = Path(str(sample.get("source_video") or ""))
    cache = pipeline / "s4_segment_sampling_human_review" / "clips"
    return ensure_segment_poster(
        source_video=source,
        cache_dir=cache,
        segment_id=segment_id,
        start_seconds=float(start),
    )


def _audit_segment_window(s3_dir: Path, segment_id: str) -> tuple[float | None, float | None]:
    audit_path = s3_dir / "segment_audit.jsonl"
    if not audit_path.is_file():
        return None, None
    latest: dict[str, Any] | None = None
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and str(item.get("segment_id") or "") == segment_id:
            latest = item
    if not latest:
        return None, None
    raw = latest.get("raw") if isinstance(latest.get("raw"), dict) else {}
    start = latest.get("start_seconds", raw.get("start_seconds"))
    end = latest.get("end_seconds", raw.get("end_seconds"))
    try:
        return (float(start) if start is not None else None, float(end) if end is not None else None)
    except (TypeError, ValueError):
        return None, None


def save_s4_partial(
    *,
    sample: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    film_verdict: str,
    reason: str,
) -> dict[str, Any]:
    return save_s4_draft(
        movie_dir=Path(str(sample["movie_dir"])),
        decisions=decisions,
        film_verdict=film_verdict,
        reason=reason,
    )


def save_s4(
    *,
    sample: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    film_verdict: str,
    reason: str,
) -> dict[str, Any]:
    return apply_s4_decisions(
        movie_dir=Path(str(sample["movie_dir"])),
        decisions=decisions,
        film_verdict=film_verdict,
        reason=reason,
    )


def accept_all_s4(sample: dict[str, Any]) -> dict[str, Any]:
    """Accept every pending S4 queue item, preserving an explicit audit trail."""
    s4_dir = (
        Path(str(sample["movie_dir"]))
        / "tmp"
        / "pipeline"
        / "s4_segment_sampling_human_review"
    )
    audit = _maybe_json(s4_dir / "review_audit.json") or {}
    queue = _maybe_json(s4_dir / "review_queue.json") or []
    if bool(audit.get("human_reviewed")):
        return {"ok": True, "status": "already_reviewed", "n_queue": len(queue)}
    decisions = {
        str(item["segment_id"]): {"action": "accept"}
        for item in queue
        if isinstance(item, dict) and str(item.get("segment_id") or "")
    }
    if not decisions:
        return {"ok": True, "status": "no_pending_s4", "n_queue": 0}
    result = save_s4(
        sample=sample,
        decisions=decisions,
        film_verdict="accept",
        reason="console_batch_accept_all_s4",
    )
    return {"status": "accepted", "n_queue": len(decisions), **result}


def _unmasked_companion(s5_dir: Path, proposal: dict[str, Any]) -> Path | None:
    """Return an existing opaque same-bbox companion for a SAM3 crop.

    Review alternate listing is read-only: never materialize missing companions
    here (that needs PIL/numpy and can 500 the S6 UI). Reconstruction belongs in S5.
    """
    del s5_dir  # kept for call-site stability / future frame-dir lookups
    configured = str(proposal.get("unmasked_crop_path") or "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    crop_path = Path(str(proposal.get("crop_path") or ""))
    source = str(proposal.get("bbox_source") or "")
    if not crop_path.is_file() or not (proposal.get("sam3") or source.startswith("sam3")):
        return None
    companion = unmasked_companion_path(crop_path)
    return companion if companion.is_file() else None


def _alternate_crops(s5_dir: Path, proposal: dict[str, Any], proposals: list[Any] | None = None) -> list[dict[str, Any]]:
    entity_id = str(proposal.get("entity_id") or "")
    kind = str(proposal.get("kind") or "character")
    if not entity_id:
        return []

    path_to_rep: dict[str, str] = {}
    for item in proposals or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("entity_id") or "") != entity_id:
            continue
        crop = str(item.get("crop_path") or "").strip().replace("\\", "/")
        rid = str(item.get("representation_id") or "")
        if not crop or not rid:
            continue
        # Proposals often store absolute paths; alternate rows use MemStrata-relative
        # media paths. Index both forms so "拉出" can find the existing card.
        path_to_rep.setdefault(crop, rid)
        try:
            path_to_rep.setdefault(_media_relpath(Path(crop)), rid)
        except OSError:
            pass

    chunk_id = proposal.get("chunk_id")
    chunk_token = None
    try:
        if chunk_id is not None and str(chunk_id) != "":
            chunk_token = f"c{int(chunk_id):05d}"
    except (TypeError, ValueError):
        chunk_token = None

    paired_unmasked = _unmasked_companion(s5_dir, proposal)
    ordered: list[Path] = [paired_unmasked] if paired_unmasked else []
    if chunk_token:
        ordered.extend(_iter_image_files(s5_dir / "propose_scratch" / kind / entity_id / chunk_token))
    ordered.extend(_iter_image_files(s5_dir / "candidates" / kind / entity_id))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_crop = str(proposal.get("crop_path") or "").strip().replace("\\", "/")
    current_keys = {current_crop} if current_crop else set()
    if current_crop:
        try:
            current_keys.add(_media_relpath(Path(current_crop)))
        except OSError:
            pass
    for path in ordered:
        if path.stem.endswith("_feed"):
            continue
        rel = _media_relpath(path)
        if rel in seen:
            continue
        # Do not offer the card's own crop as a "replacement" candidate.
        if rel in current_keys or str(path) in current_keys:
            continue
        seen.add(rel)
        row: dict[str, Any] = {
            "crop_path": rel,
            "name": path.name,
            "variant": "unmasked" if paired_unmasked and path == paired_unmasked else "candidate",
        }
        existing = path_to_rep.get(rel) or path_to_rep.get(str(path))
        if existing:
            row["existing_representation_id"] = existing
        rows.append(row)
        if len(rows) >= 12:
            break
    return rows


def s6_alternates(sample: dict[str, Any], representation_id: str) -> dict[str, Any]:
    movie_dir = Path(str(sample["movie_dir"]))
    s5_dir = movie_dir / "tmp" / "pipeline" / "s5_entities_visual_crop_acquisition"
    proposals = _maybe_json(s5_dir / "crop_proposals.json") or []
    proposal = next(
        (
            item for item in proposals
            if isinstance(item, dict) and str(item.get("representation_id") or "") == representation_id
        ),
        {},
    )
    # Human-promoted cards are not in crop_proposals; recover entity/chunk from the
    # S6 queue / draft so alternate listing still works after "拉出".
    if not isinstance(proposal, dict) or not proposal.get("entity_id"):
        queue = _maybe_json(
            movie_dir / "tmp" / "pipeline" / "s6_entities_visual_crop_human_review" / "review_queue.json"
        ) or []
        draft = load_s6_draft(movie_dir)
        decisions = draft.get("decisions") if isinstance(draft, dict) else {}
        hint: dict[str, Any] = {}
        for item in queue if isinstance(queue, list) else []:
            if not isinstance(item, dict):
                continue
            prop = item.get("proposal") if isinstance(item.get("proposal"), dict) else {}
            rid = str(prop.get("representation_id") or item.get("card_id") or "").removeprefix("crop:")
            if rid == representation_id:
                hint = {
                    "entity_id": item.get("entity_id") or prop.get("entity_id"),
                    "kind": prop.get("kind") or "character",
                    "chunk_id": item.get("chunk_id", prop.get("chunk_id")),
                    "crop_path": prop.get("crop_path"),
                }
                break
        if not hint and isinstance(decisions, dict):
            decision = decisions.get(representation_id) or {}
            prop = decision.get("proposal") if isinstance(decision.get("proposal"), dict) else {}
            replacement = decision.get("replacement") if isinstance(decision.get("replacement"), dict) else {}
            if prop or replacement:
                hint = {
                    "entity_id": prop.get("entity_id"),
                    "kind": prop.get("kind") or "character",
                    "chunk_id": prop.get("chunk_id"),
                    "crop_path": replacement.get("crop_path") or prop.get("crop_path"),
                }
        if hint.get("entity_id"):
            proposal = {**(proposal if isinstance(proposal, dict) else {}), **hint}
    return {
        "representation_id": representation_id,
        "alternates": _alternate_crops(
            s5_dir,
            proposal if isinstance(proposal, dict) else {},
            proposals if isinstance(proposals, list) else [],
        ),
    }


def _segment_id_sort_key(segment_id: str) -> tuple[int, int, str]:
    sid = str(segment_id or "")
    match = re.search(r"(\d+)\s*$", sid)
    if match:
        return (0, int(match.group(1)), sid)
    digits = "".join(ch for ch in sid if ch.isdigit())
    if digits:
        return (0, int(digits), sid)
    return (1, 10**9, sid)


def s3_live_payload(sample: dict[str, Any], *, limit: int = 200) -> dict[str, Any]:
    """Live S3 progress from incrementally written audit/progress files.

    Supports a single stage dir or sharded ``shard_*/`` workers (BDY multi-node).
    Cards are ordered by ``segment_id`` (seg_0001 …), not audit write time.
    """
    movie_dir = Path(str(sample["movie_dir"]))
    s3_dir = movie_dir / "tmp" / "pipeline" / "s3_segment_auto_review_revise"
    shard_dirs = sorted(
        path for path in s3_dir.glob("shard_*") if path.is_dir()
    ) if s3_dir.is_dir() else []
    stage_dirs = shard_dirs if shard_dirs else ([s3_dir] if s3_dir.is_dir() else [])

    reviews: list[dict[str, Any]] = []
    shard_progress: list[dict[str, Any]] = []
    done_sum = 0
    total_sum = 0
    statuses: list[str] = []
    annotation: dict[str, Any] = {}
    for stage in stage_dirs:
        prog = _maybe_json(stage / "progress.json") or {}
        if isinstance(prog, dict) and prog:
            shard_progress.append({"stage": stage.name, **prog})
            done_sum += int(prog.get("done") or 0)
            total_sum += int(prog.get("total") or 0)
            if prog.get("status"):
                statuses.append(str(prog.get("status")))
        audit_path = stage / "segment_audit.jsonl"
        if audit_path.is_file():
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("_stage", stage.name)
                    reviews.append(item)
        ann = _maybe_json(stage / "auto_revised_annotation.json")
        if isinstance(ann, dict) and ann:
            annotation = ann  # last shard wins for entity roster; cards use audit times
    if not annotation:
        annotation = _maybe_json(s3_dir / "auto_revised_annotation.json") or {}

    progress: dict[str, Any]
    if shard_dirs:
        if statuses and all(s == "done" for s in statuses):
            status = "done"
        elif any(s == "running" for s in statuses):
            status = "running"
        else:
            status = statuses[-1] if statuses else "unknown"
        progress = {
            "status": status,
            "phase": "segments",
            "done": done_sum,
            "total": total_sum,
            "shards": shard_progress,
            "n_shards": len(shard_dirs),
            "updated_at": max(
                (str(p.get("updated_at") or "") for p in shard_progress),
                default="",
            ),
        }
    else:
        progress = _maybe_json(s3_dir / "progress.json") or {}

    # Keep latest audit line per segment, then chronological by segment id.
    latest_by_segment: dict[str, dict[str, Any]] = {}
    for item in reviews:
        sid = str(item.get("segment_id") or "")
        if not sid:
            continue
        latest_by_segment[sid] = item
    feed = sorted(
        latest_by_segment.values(),
        key=lambda item: _segment_id_sort_key(str(item.get("segment_id") or "")),
    )
    if limit > 0:
        feed = feed[: max(1, limit)]
    entities = _entity_index(annotation if isinstance(annotation, dict) else {})
    segments = _segment_index(annotation if isinstance(annotation, dict) else {})
    s2 = _maybe_json(movie_dir / "tmp" / "pipeline" / "s2_annotation_postprocess" / "normalized_annotation.json") or {}
    if isinstance(s2, dict) and isinstance(s2.get("annotation"), dict):
        s2 = s2["annotation"]
    s2_segments = _segment_index(s2 if isinstance(s2, dict) else {})
    # Prefer S2 roster names so removed entities still resolve.
    for eid, meta in _entity_index(s2 if isinstance(s2, dict) else {}).items():
        entities.setdefault(eid, meta)
    cards = []
    for item in feed:
        segment_id = str(item.get("segment_id") or "")
        present = [str(x) for x in (item.get("revised_present") or [])]
        segment = segments.get(segment_id, {})
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        s2_seg = _s2_source_segment(segment_id, s2_segments)
        original_present = [str(x) for x in (s2_seg.get("present_entity_ids") or [])]
        original_action = str(s2_seg.get("action") or "")
        revised_action = str(item.get("revised_action") or "")
        original_set = set(original_present)
        revised_set = set(present)
        start = item.get("start_seconds", raw.get("start_seconds"))
        end = item.get("end_seconds", raw.get("end_seconds"))
        if start is None:
            start = segment.get("start_seconds")
        if end is None:
            end = segment.get("end_seconds")
        if start is None:
            start = s2_seg.get("start_seconds")
        if end is None:
            end = s2_seg.get("end_seconds")
        clip_url = ""
        poster_url = ""
        if start is not None and end is not None and segment_id:
            clip_url = (
                "/api/review/segment-clip"
                f"?dataset={sample['dataset']}&movie_id={sample['movie_id']}"
                f"&segment_id={segment_id}"
                f"&start_seconds={start}&end_seconds={end}"
            )
            poster_url = (
                "/api/review/segment-poster"
                f"?dataset={sample['dataset']}&movie_id={sample['movie_id']}"
                f"&segment_id={segment_id}"
                f"&start_seconds={start}"
            )
        entity_rows: list[dict[str, Any]] = []
        for eid in list(dict.fromkeys([*original_present, *present])):
            meta = entities.get(eid, {"entity_id": eid, "name": eid, "kind": "", "description": ""})
            in_orig = eid in original_set
            in_rev = eid in revised_set
            if in_orig and in_rev:
                status = "kept"  # green: existed and passed
            elif in_orig and not in_rev:
                status = "removed"  # red: existed but failed / dropped
            else:
                status = "added"  # yellow: not in S2, supplemented by S3
            entity_rows.append(
                {
                    **meta,
                    "status": status,
                    "in_original": in_orig,
                    "in_revised": in_rev,
                }
            )
        cards.append(
            {
                "segment_id": segment_id,
                "accepted": bool(item.get("accepted")),
                "verdict": str(
                    item.get("verdict")
                    or ("PASS" if item.get("accepted") else "WARN")
                ),
                "findings": list(item.get("findings") or []),
                "recommended_action": str(item.get("recommended_action") or "none"),
                "confidence": str(item.get("confidence") or ""),
                "original_present": original_present,
                "revised_present": present,
                "original_action": original_action,
                "revised_action": revised_action,
                "action_changed": original_action.strip() != revised_action.strip(),
                "risk_reasons": list(item.get("risk_reasons") or []),
                "n_rounds": int(item.get("n_rounds") or 1),
                "elapsed_seconds": _optional_float(
                    item.get("elapsed_seconds")
                    if item.get("elapsed_seconds") is not None
                    else raw.get("elapsed_seconds")
                ),
                "queue_seconds": _optional_float(
                    item.get("queue_seconds")
                    if item.get("queue_seconds") is not None
                    else raw.get("queue_seconds")
                ),
                "clip_seconds": _optional_float(
                    item.get("clip_seconds")
                    if item.get("clip_seconds") is not None
                    else raw.get("clip_seconds")
                ),
                "vlm_request_seconds": _optional_float(
                    item.get("vlm_request_seconds")
                    if item.get("vlm_request_seconds") is not None
                    else raw.get("vlm_request_seconds")
                ),
                # The standard vLLM OpenAI response does not report a
                # request-level inference duration. Keep this nullable rather
                # than presenting client request time as GPU inference time.
                "vlm_inference_seconds": _optional_float(
                    item.get("vlm_inference_seconds")
                    if item.get("vlm_inference_seconds") is not None
                    else raw.get("vlm_inference_seconds")
                ),
                "start_seconds": start,
                "end_seconds": end,
                "clip_url": clip_url,
                "poster_url": poster_url,
                "shard": str(item.get("_stage") or ""),
                "entities": entity_rows,
            }
        )
    available = bool(reviews) or bool(shard_progress) or (s3_dir / "progress.json").is_file()
    return {
        "available": available,
        "sample": {
            "dataset": sample.get("dataset"),
            "movie_id": sample.get("movie_id"),
            "movie_dir": str(movie_dir),
        },
        "progress": progress if isinstance(progress, dict) else {},
        "n_reviews": len(latest_by_segment) if reviews else 0,
        "cards": cards,
        "entities": list(entities.values()),
        "paths": {
            "stage_dir": str(s3_dir),
            "segment_audit": str(s3_dir / "segment_audit.jsonl"),
            "progress": str(s3_dir / "progress.json"),
            "annotation": str(s3_dir / "auto_revised_annotation.json"),
            "shards": [str(path) for path in shard_dirs],
        },
    }


def get_sample_or_raise(
    *,
    data_root: Path,
    blender_index: Path | None,
    lsmdc_index: Path | None,
    dataset: str,
    movie_id: str,
) -> dict[str, Any]:
    sample = get_sample(
        data_root=data_root,
        dataset=dataset,
        movie_id=movie_id,
        blender_index=blender_index,
        lsmdc_index=lsmdc_index,
    )
    if sample is None:
        raise FileNotFoundError(f"sample not found: {dataset}/{movie_id}")
    return sample


def save_s6(*, sample: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return apply_s6_decisions(movie_dir=Path(str(sample["movie_dir"])), decisions=decisions)
