"""Ranked identity evidence for review; this module never changes annotations."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry, normalize_entity_name
from vmem_bench.annotation.pipeline_track_first.reid import _entity_signature
from vmem_bench.common.vecmath import cosine_similarity


def _cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    return round(cosine_similarity(left, right), 4)


def _representative_crop(entity) -> str | None:
    reps = [rep for rep in entity.representations if rep.crop_path]
    if not reps:
        return None
    best = max(reps, key=lambda rep: (float(rep.qa.get("grounding_score", 0.0)),
                                      -int(rep.frame_index), rep.representation_id))
    return best.crop_path


def identity_candidates(
    registry: Registry,
    *,
    text_embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    min_signal: float = 0.5,
) -> list[dict[str, Any]]:
    """Return same-kind, review-worthy pairs with auditable independent cues.

    Candidates are deliberately broad enough to find alias splits, but are only recommendations:
    neither the registry nor any review disposition is modified.  ``text_cos`` is null when the
    optional text embedder is unavailable.  A pair is retained when its normalized names agree or
    at least one available visual/text cue reaches ``min_signal``.
    """
    entities = sorted(registry.entities.values(), key=lambda entity: entity.entity_id)
    text_vectors: dict[str, list[float]] = {}
    if text_embed_fn is not None and entities:
        vectors = text_embed_fn([f"{entity.name}. {entity.description}" for entity in entities])
        text_vectors = {entity.entity_id: vector for entity, vector in zip(entities, vectors)}

    out: list[dict[str, Any]] = []
    for left, right in combinations(entities, 2):
        if left.kind != right.kind:
            continue
        body = _cosine(_entity_signature(left, registry.embeddings),
                       _entity_signature(right, registry.embeddings))
        face = _cosine(_entity_signature(left, registry.face_embeddings),
                       _entity_signature(right, registry.face_embeddings))
        class_score = _cosine(_entity_signature(left, registry.class_embeddings),
                              _entity_signature(right, registry.class_embeddings))
        text = _cosine(text_vectors.get(left.entity_id), text_vectors.get(right.entity_id))
        same_name = normalize_entity_name(left.name).casefold() == normalize_entity_name(right.name).casefold()
        signals = [score for score in (body, face, class_score, text) if score is not None]
        if not same_name and (not signals or max(signals) < min_signal):
            continue
        support = sum(score >= min_signal for score in signals)
        recommendation = "review_merge" if same_name or support >= 2 else "review_keep_distinct"
        out.append({
            "left": left.entity_id, "right": right.entity_id, "kind": left.kind,
            "body_cos": body, "face_cos": face, "class_cos": class_score, "text_cos": text,
            "left_chunk_span": [min((r.chunk_id for r in left.representations), default=left.first_chunk),
                                max((r.chunk_id for r in left.representations), default=left.first_chunk)],
            "right_chunk_span": [min((r.chunk_id for r in right.representations), default=right.first_chunk),
                                 max((r.chunk_id for r in right.representations), default=right.first_chunk)],
            "left_representative_crop": _representative_crop(left),
            "right_representative_crop": _representative_crop(right),
            "recommendation": recommendation,
        })
    return sorted(out, key=lambda candidate: (
        candidate["recommendation"] != "review_merge",
        -max(score for score in (candidate["body_cos"], candidate["face_cos"],
                                 candidate["class_cos"], candidate["text_cos"]) if score is not None),
        candidate["left"], candidate["right"],
    ))
