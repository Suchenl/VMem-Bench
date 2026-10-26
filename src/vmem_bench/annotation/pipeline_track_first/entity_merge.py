"""Post-naming entity merge PROPOSALS (never auto-applied).

Compares same-kind entity pairs by text embedding of ``name. description`` and body
appearance signature; writes a report for human review. Identity merges stay a review
action — this module only proposes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first.reid import _entity_signature
from vmem_bench.common.vecmath import cosine_similarity


def propose_entity_merges(
    registry: Registry,
    text_embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    text_threshold: float = 0.85,
    body_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Propose same-kind merges when text cosine is high and body agrees (or is missing).

    Each proposal: ``{"keep", "merge", "text_cos", "body_cos", "kind"}``. ``keep`` is the
    entity with the smaller ``first_chunk`` (tie: lexicographic entity_id).
    """
    entities = list(registry.entities.values())
    if len(entities) < 2:
        return []
    texts = [f"{e.name}. {e.description}" for e in entities]
    text_vecs = text_embed_fn(texts)
    by_id = {e.entity_id: (e, text_vecs[i]) for i, e in enumerate(entities)}
    proposals: list[dict[str, Any]] = []
    ids = sorted(by_id)
    for i, a_id in enumerate(ids):
        a, a_tv = by_id[a_id]
        for b_id in ids[i + 1:]:
            b, b_tv = by_id[b_id]
            if a.kind != b.kind:
                continue
            text_cos = cosine_similarity(a_tv, b_tv)
            if text_cos < text_threshold:
                continue
            a_sig = _entity_signature(a, registry.embeddings)
            b_sig = _entity_signature(b, registry.embeddings)
            body_cos: float | None
            if a_sig is None or b_sig is None:
                body_cos = None
            else:
                body_cos = cosine_similarity(a_sig, b_sig)
                if body_cos < body_threshold:
                    continue
            # keep = smaller first_chunk; tie -> lexicographic id
            if (a.first_chunk, a.entity_id) <= (b.first_chunk, b.entity_id):
                keep, merge = a.entity_id, b.entity_id
            else:
                keep, merge = b.entity_id, a.entity_id
            proposals.append({
                "keep": keep,
                "merge": merge,
                "text_cos": round(text_cos, 4),
                "body_cos": (None if body_cos is None else round(body_cos, 4)),
                "kind": a.kind,
            })
    return proposals
