"""Cross-chunk entity consolidation (workflow step 5).

Dual-threshold embedding match with VLM adjudication in the uncertain band; the
annotator VLM's name reuse (naming consistency in discovery) acts as a strong prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vmem_bench.common.vecmath import cosine_similarity
from vmem_bench.common.schemas import Entity, Representation

_KIND_PREFIX = {"character": "char", "location": "loc", "prop": "prop"}


def normalize_entity_name(name: str) -> str:
    if not name:
        return ""
    cleaned = name.strip()
    # 剥离带或不带括号的 category/kind 标注 (e.g. "(character)", "(location)", "(prop)")
    cleaned = re.sub(r"\s*\(\s*(character|location|prop|char|loc)\s*\)?\s*$", "", cleaned, flags=re.IGNORECASE)
    # 剥离 " - character", " - prop", " - location" 等
    cleaned = re.sub(r"\s+-(character|location|prop|char|loc)\s*$", "", cleaned, flags=re.IGNORECASE)
    # 剥离 " character", " prop", " location" 等
    cleaned = re.sub(r"\s+(character|location|prop|char|loc)\s*$", "", cleaned, flags=re.IGNORECASE)
    # 剥离末尾的 "_character", "_prop", "_location" 
    cleaned = re.sub(r"[_-](character|location|prop|char|loc)$", "", cleaned, flags=re.IGNORECASE)
    # 去除多余的双引号、单引号包裹
    cleaned = cleaned.strip('"\'')
    return cleaned.strip()


def slugify(name: str) -> str:
    norm = normalize_entity_name(name)
    slug = re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_")
    return slug or "unnamed"


@dataclass(slots=True)
class Registry:
    """In-build entity registry with embedding store (embeddings never enter JSON)."""

    entities: dict[str, Entity] = field(default_factory=dict)  # entity_id -> Entity
    embeddings: dict[str, list[float]] = field(default_factory=dict)  # embedding_key -> body vector
    # Parallel cue stores for multi-cue re-ID (annotation_tracking_internals §3.7); keyed by the same
    # representation_id / embedding_key. Absent key == that cue is missing for that rep (self-gated:
    # a crop with no detected face simply has no entry here). Never enter JSON (Principle 10).
    face_embeddings: dict[str, list[float]] = field(default_factory=dict)  # rep_id -> ArcFace vector
    class_embeddings: dict[str, list[float]] = field(default_factory=dict)  # rep_id -> SigLIP class vector

    def new_entity_id(self, kind: str, name: str) -> str:
        norm = normalize_entity_name(name)
        base = f"{_KIND_PREFIX.get(kind, kind)}_{slugify(norm)}"
        candidate, n = base, 1
        while candidate in self.entities:
            n += 1
            candidate = f"{base}_{n:02d}"
        return candidate

    def by_name(self, kind: str, name: str) -> Entity | None:
        low = normalize_entity_name(name).lower()
        matches = []
        for e in self.entities.values():
            if e.kind == kind and normalize_entity_name(e.name).lower() == low:
                matches.append(e)
        if not matches:
            return None
        # Canonical alias layer: if an older checkpoint already contains both "Rabbit" and
        # "Rabbit (character)", prefer the base slug id so new observations do not keep feeding
        # the suffix-split shard. A later review merge can clean old reps; this stops new ones.
        base = f"{_KIND_PREFIX.get(kind, kind)}_{slugify(low)}"
        return min(matches, key=lambda e: (e.entity_id != base, e.first_chunk, e.entity_id))

    def best_match(self, kind: str, vector: list[float]) -> tuple[Entity | None, float]:
        best, best_score = None, -1.0
        for e in self.entities.values():
            if e.kind != kind:
                continue
            for rep in e.representations:
                # Only grounded crops participate in identity matching: a full-frame /
                # vlm_fallback crop's embedding encodes the whole scene (background + other
                # entities), so its cosine against a real entity crop is a scene-similarity,
                # not an identity-similarity. Including it pollutes cross-chunk merges (two
                # different entities in the same scene get high scene-similarity) and spams
                # the gray-zone VLM arbiter. Identity matching is grounded-crop-only
                # (pitfalls: identity pollution via fallback embeddings).
                if rep.bbox_source != "grounding_dino":
                    continue
                emb = self.embeddings.get(rep.embedding_key)
                if emb is None:
                    continue
                score = cosine_similarity(vector, emb)
                if score > best_score:
                    best, best_score = e, score
        return best, best_score


def _static_compatible(a: dict[str, str], b: dict[str, str], threshold: float) -> bool:
    """Two stable-attribute dicts are compatible iff, over keys present in BOTH, the fraction of
    equal values is >= threshold. An empty side carries no identity info -> compatible (no gate);
    two disjoint key sets -> compatible (cannot prove a conflict). This is the static-identity
    floor of the three-level funnel (kind -> static attributes -> VLM/embedding arbitration):
    it blocks the obvious "same name, different species" over-merge that pollutes the asset bank
    (pitfalls root cause #2/#3).

    Keys and values are case-normalized (lower) before comparison: VLMs emit attribute keys
    inconsistently across chunks ('Species' vs 'species', 'PrimaryColor' vs 'primary_color' is
    handled by the caller's free-form dict but case differs), and without normalization the key
    sets look disjoint -> the gate returns True -> the static-identity floor silently fails to
    block a conflict (pitfalls: case-sensitivity hole)."""
    if not a or not b:
        return True
    an = {str(k).lower(): str(v).lower() for k, v in a.items()}
    bn = {str(k).lower(): str(v).lower() for k, v in b.items()}
    shared = set(an) & set(bn)
    if not shared:
        return True
    same = sum(1 for k in shared if an[k] == bn[k])
    return same / len(shared) >= threshold


def _best_grounded_crop(entity: Entity) -> str | None:
    """Highest-grounding-score grounded crop path of an entity, for image-image identity
    arbitration. Returns None when the entity has no grounded detection yet (only full-frame /
    vlm_fallback) — the caller then falls back to the text description. Image-image is more
    reliable than image-text for same-entity judgement (the first-appearance description may be
    partial/loose), but text is a safe fallback for first-chunk entities with no prior grounded
    crop (pitfalls: gray-zone arbitration reliability)."""
    grounded = [r for r in entity.representations
                if r.bbox_source == "grounding_dino" and r.crop_path]
    if not grounded:
        return None
    return max(grounded, key=lambda r: float(r.qa.get("grounding_score", 0.0))).crop_path


def _entity_best_grounded_score(entity: Entity, embeddings: dict[str, list[float]],
                                vector: list[float]) -> float:
    best = -1.0
    for rep in entity.representations:
        if rep.bbox_source != "grounding_dino":
            continue
        emb = embeddings.get(rep.embedding_key)
        if emb is None:
            continue
        best = max(best, cosine_similarity(vector, emb))
    return best


def _is_readable_image(path: Path) -> bool:
    """Return whether a crop can provide pixel evidence to the VLM arbiter."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def consolidate_observation(
    registry: Registry,
    *,
    chunk_id: int,
    name: str,
    kind: str,
    description: str,
    crop_path: str,
    bbox: list[int],
    bbox_source: str,
    frame_index: int,
    vector: list[float],
    judge_same_entity,  # (crop: Path, description: str, kind: str) -> bool
    high_threshold: float,
    low_threshold: float,
    static_attributes: dict[str, str] | None = None,
    static_overlap_threshold: float = 0.75,
    grounding_score: float = 0.0,
) -> tuple[Entity, Representation, bool]:
    """Match one discovered entity into the registry; return (entity, new_rep, is_new_entity).

    Three-level identity funnel (static identity vs dynamic state decoupled):
    1. kind filter (best_match is per-kind);
    2. static-attribute compatibility gate (blocks same-name / high-embedding over-merges when
       stable attributes conflict, e.g. species fox vs bird);
    3. embedding dual-threshold + VLM gray-zone arbitration (only when static attrs are
       compatible AND embedding is in the uncertain band). Default bias is "split rather than
       merge": a bad split is fixed by a human merge patch, a bad merge silently corrupts gold.
    """
    name = normalize_entity_name(name)
    static_attributes = dict(static_attributes or {})

    matched = registry.by_name(kind, name)  # naming-consistency hint from discovery
    if matched is not None and not _static_compatible(
            static_attributes, matched.static_attributes, static_overlap_threshold):
        matched = None  # name reused for a visually different identity -> do not merge
    if matched is not None and bbox_source == "grounding_dino":
        # Same name is only a hint, not identity proof. If the existing entity has grounded pixel
        # evidence, the current grounded crop must agree visually before we append it. This blocks
        # the BBB failure mode where "Brown Acorn" and squirrels receive the same high-score bbox.
        ref = _best_grounded_crop(matched)
        if ref is not None:
            score = _entity_best_grounded_score(matched, registry.embeddings, vector)
            if score >= high_threshold:
                pass
            elif score >= low_threshold:
                if not judge_same_entity(Path(crop_path), ref, kind):
                    matched = None
            else:
                # Do not ask a judge to rescue a score this low unless the current crop carries
                # readable pixels. Invalid/missing visual evidence must split conservatively.
                if (_is_readable_image(Path(crop_path))
                        and judge_same_entity(Path(crop_path), ref, kind)):
                    pass
                else:
                    matched = None
    # Embedding identity matching is grounded-crop-only on BOTH sides: the candidate side is
    # filtered in best_match, the current-observation side is gated here. A vlm_fallback /
    # full_frame crop's embedding is a scene embedding, not an entity embedding — putting it
    # through the gray-zone arbiter (image-image against a grounded crop) almost always returns
    # true (the whole frame contains the entity) and silently over-merges. So a non-grounded
    # observation falls back to name+static-attribute matching only (bias: split rather than
    # merge; a false split is repaired by a human merge patch, a false merge corrupts gold).
    if matched is None and bbox_source == "grounding_dino":
        candidate, score = registry.best_match(kind, vector)
        if candidate is not None and _static_compatible(
                static_attributes, candidate.static_attributes, static_overlap_threshold):
            if score >= high_threshold:
                matched = candidate
            elif score >= low_threshold:
                # Gray-zone arbitration: judge the current crop against the candidate's BEST
                # GROUNDED crop (image-image) when one exists, falling back to the candidate's
                # text description only when it has no prior grounded detection. Image-image is
                # more reliable than image-text: the first-appearance description may be partial
                # or loose, while a grounded crop is pixel evidence of the same identity
                # (pitfalls: gray-zone arbitration reliability).
                ref = _best_grounded_crop(candidate)
                judge_arg = ref if ref is not None else candidate.description
                if judge_same_entity(Path(crop_path), judge_arg, kind):
                    matched = candidate

    is_new = matched is None
    if is_new:
        entity_id = registry.new_entity_id(kind, name)
        matched = Entity(entity_id=entity_id, kind=kind, name=name,
                         description=description, first_chunk=chunk_id,
                         static_attributes=dict(static_attributes))
        registry.entities[entity_id] = matched
    else:
        # Fill stable attributes the first time we observe them (later chunks do not overwrite).
        if not matched.static_attributes and static_attributes:
            matched.static_attributes = dict(static_attributes)
        # Authoritative appearance description: keep the first non-empty one. A later re-discovery
        # may be shorter or from a worse angle; the first-appearance description is what gets
        # inlined into the prompt and shipped in gold. Repair of a bad first description is done
        # by human review (field_edits on entity description), not by silent overwrite.
        if not matched.description.strip() and description.strip():
            matched.description = description

    n_existing = sum(1 for r in matched.representations if r.chunk_id == chunk_id)
    rep_id = f"{matched.entity_id}@c{chunk_id:03d}" + (f".{n_existing}" if n_existing else "")
    rep = Representation(representation_id=rep_id, chunk_id=chunk_id, crop_path=crop_path,
                         bbox=list(bbox), bbox_source=bbox_source, frame_index=frame_index,
                         embedding_key=rep_id,
                         qa={"grounding_score": float(grounding_score)})
    matched.representations.append(rep)
    registry.embeddings[rep_id] = list(vector)
    return matched, rep, is_new


def present_payload(entity: Entity, rep: Representation, *, first_appearance: bool) -> dict[str, Any]:
    """Uniform per-entity payload handed to drafting / verification / observation feedback.
    Carries grounding_score + bbox_source so the verifier can skip auditing high-confidence and
    full-frame/location crops (pitfalls)."""
    return {"entity_id": entity.entity_id, "name": entity.name, "kind": entity.kind,
            "description": entity.description, "crop_path": rep.crop_path,
            "representation_id": rep.representation_id, "first_appearance": first_appearance,
            "bbox_source": rep.bbox_source,
            "bbox": list(rep.bbox),
            "grounding_score": float(rep.qa.get("grounding_score", 0.0))}
