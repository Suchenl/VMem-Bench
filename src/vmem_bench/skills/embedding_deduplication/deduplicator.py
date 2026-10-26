"""Model-free dedup over embeddings: identity matching and non-redundant selection."""

from __future__ import annotations

from vmem_bench.common.vecmath import Vector, cosine_distance, cosine_similarity


def match_to_existing(
    query: Vector,
    candidates: list[tuple[str, Vector]],
    *,
    threshold: float = 0.6,
) -> tuple[str | None, float]:
    """Return ``(best_id, score)`` if the closest candidate is within ``threshold``."""

    best_id: str | None = None
    best_score = -1.0
    for identifier, vector in candidates:
        score = cosine_similarity(query, vector)
        if score > best_score:
            best_score = score
            best_id = identifier
    if best_id is not None and best_score >= threshold:
        return best_id, best_score
    return None, best_score


def select_non_redundant(
    vectors: list[Vector],
    *,
    max_keep: int = 5,
    min_distance: float = 0.15,
    quality: list[float] | None = None,
) -> list[int]:
    """Greedy quality-seeded farthest-point selection."""

    count = len(vectors)
    if count == 0:
        return []
    if quality is not None and len(quality) != count:
        raise ValueError("quality must align with vectors")
    if max_keep <= 0:
        return []

    order = sorted(range(count), key=lambda i: (-(quality[i] if quality else 0.0), i))
    kept: list[int] = [order[0]]
    for index in order[1:]:
        if len(kept) >= max_keep:
            break
        nearest = min(cosine_distance(vectors[index], vectors[chosen]) for chosen in kept)
        if nearest >= min_distance:
            kept.append(index)
    return sorted(kept)
