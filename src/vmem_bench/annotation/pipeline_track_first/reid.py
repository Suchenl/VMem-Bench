"""Cross-shot re-identification (track-first redesign, docs/benchmark/annotation_tracking_internals.md §3.1/§3.7).

A shot yields tracklets (tracking.py); each tracklet is one object instance with a stable
appearance signature (mean of its grounded crop embeddings). Re-ID assigns a *global*
``entity_id`` by matching a tracklet's signature against existing global entities of the same
kind. Identity is decided by appearance, NOT by the VLM-proposed name -- so name noise
("white_rabbit" vs "white_rabbit_character") can no longer fragment one object into many ids
(baseline probe: 23% duplicate ids). Default bias is "split rather than merge": a false split is
repaired by a human merge patch, a false merge silently corrupts gold.

Matching is a **multi-cue fusion** (§3.7), NOT body-only:
  - ``body`` (DINOv3 crop embedding): always present, the main cue.
  - ``face`` (ArcFace on a detected face): present ONLY when a face was detected in BOTH crops
    (self-gated -- see face.py; there is no separate "is this a face?" classifier and no
    control-flow branch on style). Absent -> the cue drops out and weights renormalize.
  - ``class`` (SigLIP zero-shot class vector): optional semantic cue.
This one fused path runs identically for animation and live-action; the difference is purely
which cues happen to be present (data-driven, not ``if style == ...``). A single conservative
``reid_threshold`` gates the fused score; a strong face may rescue a low body score, but a face
alone can NEVER merge (avoids twin/look-alike over-merge).

Reuses the frozen ``Entity``/``Representation`` schema and ``Registry`` / static-attribute gate /
name normalization from consolidation.py -- no schema change (face/class vectors go to sidecar
stores on the Registry, never into JSON).
"""

from __future__ import annotations

from collections.abc import Callable

from vmem_bench.common.vecmath import cosine_similarity
from vmem_bench.common.schemas import Entity, Representation
from vmem_bench.annotation.pipeline_track_first.consolidation import (
    Registry,
    _static_compatible,
    normalize_entity_name,
)

# A strong face lets us accept a match whose body cosine sits up to this margin below
# reid_threshold (same character, different pose -> body drifts, a frontal face is far more
# discriminative). ponytail: fixed margin, adequate for a conservative rescue; upgrade path is a
# per-kind learned margin if a dataset shows body drift beyond this. A face alone still never
# merges -- the rescue requires body_cos to be within margin, not absent.
_FACE_RESCUE_BODY_MARGIN = 0.15


def fuse_similarity(cues: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted mean of present cues, renormalized over whatever is present (self-gating).

    ``cues`` maps cue name -> cosine in [-1,1] or None (absent). Absent cues (and cues with zero
    weight) contribute nothing and do not count toward the denominator, so e.g. a crop with no
    detected face fuses to exactly the body(+class) score with no penalty. Returns None only when
    *no* cue is present (nothing to compare on). Deterministic and pure -> offline unit-testable.
    """
    num = 0.0
    den = 0.0
    for name, cos in cues.items():
        if cos is None:
            continue
        w = weights.get(name, 0.0)
        if w <= 0.0:
            continue
        num += w * cos
        den += w
    return num / den if den > 0.0 else None


def _entity_signature(entity: Entity, store: dict[str, list[float]]) -> list[float] | None:
    """Mean of an entity's stored vectors in ``store`` (body/face/class share the rep-id key).

    Only reps that actually have an entry in ``store`` contribute -- so an entity whose members
    never showed a face simply has no face signature (returns None), which makes the face cue
    self-gate on the entity side too.
    """
    vecs = [store[r.embedding_key] for r in entity.representations if r.embedding_key in store]
    if not vecs:
        return None
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def _minimum_body_similarity(
    entity: Entity, vector: list[float], store: dict[str, list[float]]
) -> float | None:
    """Lowest body similarity to an entity's existing representations.

    The cluster mean can hide a single incompatible crop. This inexpensive guard
    deliberately checks the full cluster only after the mean-based re-ID gate
    accepts a candidate.
    """
    scores = [
        cosine_similarity(vector, store[rep.embedding_key])
        for rep in entity.representations
        if rep.embedding_key in store
    ]
    return min(scores) if scores else None


def reid_assign(
    registry: Registry,
    *,
    chunk_id: int,
    kind: str,
    name: str,
    description: str,
    static_attributes: dict[str, str] | None,
    signature: list[float] | None,
    crop_path: str,
    bbox: list[int],
    frame_index: int,
    grounding_score: float,
    track_id: int | None,
    reid_threshold: float,
    face_signature: list[float] | None = None,
    class_signature: list[float] | None = None,
    weights: dict[str, float] | None = None,
    face_strong: float = 0.5,
    static_overlap_threshold: float = 0.75,
    bbox_source: str = "grounding_dino",
    allowed_entity_ids: set[str] | None = None,
    cluster_min_similarity: float | None = None,
    conflict_hook: Callable[[Entity, float, float], None] | None = None,
) -> tuple[Entity, Representation, bool]:
    """Match one tracklet into the global registry; return (entity, new_rep, is_new_entity).

    Funnel per candidate entity of the same ``kind``: optional roster candidate gate ->
    static-attribute gate -> multi-cue fused
    cosine (body + optional face + optional class, §3.7). A candidate matches if its fused score
    >= ``reid_threshold``; additionally a *strong face* (``face_cos >= face_strong``) rescues a
    candidate whose body cosine is within ``_FACE_RESCUE_BODY_MARGIN`` of the threshold. The best
    matching candidate wins. When ``signature`` is None (e.g. a full-frame location with no
    grounded crop) matching falls back to name + static attributes only. A matched entity keeps
    its first non-empty description and first static attributes.
    """
    name = normalize_entity_name(name)
    static = dict(static_attributes or {})
    weights = weights or {"body": 1.0, "face": 0.6, "class": 0.3}

    matched: Entity | None = None
    best_fused = -1.0
    best_body = None
    best_face = None
    if signature is not None:
        for e in registry.entities.values():
            if e.kind != kind:
                continue
            if allowed_entity_ids is not None and e.entity_id not in allowed_entity_ids:
                continue
            if not _static_compatible(static, e.static_attributes, static_overlap_threshold):
                continue
            body_sig = _entity_signature(e, registry.embeddings)
            if body_sig is None:
                continue
            body_cos = cosine_similarity(signature, body_sig)
            face_cos = None
            if face_signature is not None:
                e_face = _entity_signature(e, registry.face_embeddings)
                if e_face is not None:
                    face_cos = cosine_similarity(face_signature, e_face)
            class_cos = None
            if class_signature is not None:
                e_class = _entity_signature(e, registry.class_embeddings)
                if e_class is not None:
                    class_cos = cosine_similarity(class_signature, e_class)
            fused = fuse_similarity(
                {"body": body_cos, "face": face_cos, "class": class_cos}, weights)
            if fused is None:
                continue
            accept = fused >= reid_threshold or (
                face_cos is not None
                and face_cos >= face_strong
                and body_cos >= reid_threshold - _FACE_RESCUE_BODY_MARGIN
            )
            if accept and cluster_min_similarity is not None:
                min_body = _minimum_body_similarity(e, signature, registry.embeddings)
                if min_body is not None and min_body < cluster_min_similarity:
                    if conflict_hook is not None:
                        conflict_hook(e, fused, min_body)
                    continue
            if accept and fused > best_fused:
                best_fused, best_body, best_face, matched = fused, body_cos, face_cos, e
    else:
        # No appearance signature: fall back to name + static gate (conservative).
        cand = registry.by_name(kind, name)
        if cand is not None and _static_compatible(
                static, cand.static_attributes, static_overlap_threshold):
            matched = cand

    is_new = matched is None
    extra_qa: dict = {}
    if not is_new:
        extra_qa["reid_score"] = round(best_fused, 4)
        if best_face is not None:
            extra_qa["face_score"] = round(best_face, 4)
    return commit_tracklet_observation(
        registry, matched, is_new, kind=kind, name=name, description=description,
        static_attributes=static, chunk_id=chunk_id, crop_path=crop_path, bbox=bbox,
        bbox_source=bbox_source, frame_index=frame_index, grounding_score=grounding_score,
        track_id=track_id, signature=signature, face_signature=face_signature,
        class_signature=class_signature, extra_qa=extra_qa)


def commit_tracklet_observation(
    registry: Registry,
    matched: Entity | None,
    is_new: bool,
    *,
    kind: str,
    name: str,
    description: str,
    static_attributes: dict[str, str] | None,
    chunk_id: int,
    crop_path: str,
    bbox: list[int],
    bbox_source: str,
    frame_index: int,
    grounding_score: float,
    track_id: int | None,
    signature: list[float] | None,
    face_signature: list[float] | None = None,
    class_signature: list[float] | None = None,
    extra_qa: dict | None = None,
) -> tuple[Entity, Representation, bool]:
    """Materialize one tracklet observation into the registry given an ALREADY-DECIDED identity.

    This is the shared tail of ``reid_assign`` (entity creation + representation bookkeeping),
    extracted so a caller that has already picked the target entity by a different route (e.g. the
    batch cluster+VLM resolution in ``identity_resolution.py``) does not have to reimplement rep-id
    minting, embedding-store bookkeeping, or the "keep first non-empty description/static attrs"
    policy. ``matched=None, is_new=True`` mints a fresh entity; otherwise appends to ``matched``.
    """
    name = normalize_entity_name(name)
    static = dict(static_attributes or {})
    if is_new:
        entity_id = registry.new_entity_id(kind, name)
        matched = Entity(entity_id=entity_id, kind=kind, name=name, description=description,
                         first_chunk=chunk_id, static_attributes=dict(static))
        registry.entities[entity_id] = matched
    else:
        assert matched is not None  # noqa: S101 -- programmer error otherwise, not a data issue
        if not matched.static_attributes and static:
            matched.static_attributes = dict(static)
        if not matched.description.strip() and description.strip():
            matched.description = description

    n_existing = sum(1 for r in matched.representations if r.chunk_id == chunk_id)
    rep_id = f"{matched.entity_id}@c{chunk_id:03d}" + (f".{n_existing}" if n_existing else "")
    qa: dict = {"grounding_score": float(grounding_score)}
    if track_id is not None:
        qa["track_id"] = int(track_id)
    qa.update(extra_qa or {})
    rep = Representation(representation_id=rep_id, chunk_id=chunk_id, crop_path=crop_path,
                         bbox=list(bbox), bbox_source=bbox_source, frame_index=frame_index,
                         embedding_key=rep_id, qa=qa)
    matched.representations.append(rep)
    if signature is not None:
        registry.embeddings[rep_id] = list(signature)
    if face_signature is not None:
        registry.face_embeddings[rep_id] = list(face_signature)
    if class_signature is not None:
        registry.class_embeddings[rep_id] = list(class_signature)
    return matched, rep, is_new
