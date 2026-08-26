"""Attribute-bucket dedup (mirror of ``memstrata.lib.dedup.select_attribute_diverse``).

Keep algorithm in sync with ``memstrata.lib.dedup``. No cross-package imports.
"""

from __future__ import annotations

from vmem_bench.common.vecmath import cosine_distance


def select_attribute_diverse(
    *,
    bucket_keys: list[tuple[str, ...]],
    vectors: list[list[float]] | None = None,
    quality: list[float] | None = None,
    max_keep: int = 5,
    min_distance: float = 0.15,
) -> list[int]:
    """Prefer distinct attribute buckets, then embedding diversity.

    ``bucket_keys[i]`` is typically ``(spatial, state, shot_size, lighting)``.
    Known (non-``unknown``) labels in a bucket outrank all-unknown buckets.
    Within a bucket, keep the highest-quality index. Remaining slots are filled
    via farthest-point on embeddings when provided.
    """

    count = len(bucket_keys)
    if count == 0:
        return []
    if quality is not None and len(quality) != count:
        raise ValueError("quality must align with bucket_keys")
    if vectors is not None and len(vectors) != count:
        raise ValueError("vectors must align with bucket_keys")
    if max_keep <= 0:
        return []

    def _quality(i: int) -> float:
        return float(quality[i]) if quality else 0.0

    def _known(i: int) -> int:
        return sum(1 for part in bucket_keys[i] if part and part != "unknown")

    best_for_bucket: dict[tuple[str, ...], int] = {}
    for index in range(count):
        key = bucket_keys[index]
        prev = best_for_bucket.get(key)
        if prev is None:
            best_for_bucket[key] = index
            continue
        if (_known(index), _quality(index), -index) > (_known(prev), _quality(prev), -prev):
            best_for_bucket[key] = index

    bucket_winners = sorted(
        best_for_bucket.values(),
        key=lambda i: (-_known(i), -_quality(i), i),
    )
    kept: list[int] = []
    for index in bucket_winners:
        if len(kept) >= max_keep:
            break
        kept.append(index)

    if len(kept) >= max_keep or vectors is None:
        return sorted(kept)

    remaining = [i for i in range(count) if i not in kept]
    remaining.sort(key=lambda i: (-_quality(i), i))
    for index in remaining:
        if len(kept) >= max_keep:
            break
        nearest = min(cosine_distance(vectors[index], vectors[chosen]) for chosen in kept)
        if nearest >= min_distance:
            kept.append(index)
    return sorted(kept)


def select_angle_diverse(
    *,
    spatial_angles: list[str],
    state_angles: list[str],
    vectors: list[list[float]] | None = None,
    quality: list[float] | None = None,
    max_keep: int = 5,
    min_distance: float = 0.15,
) -> list[int]:
    """Prefer distinct ``(spatial, state)`` buckets, then embedding diversity."""

    count = len(spatial_angles)
    if count == 0:
        return []
    if len(state_angles) != count:
        raise ValueError("state_angles must align with spatial_angles")
    bucket_keys = list(zip(spatial_angles, state_angles, strict=True))
    return select_attribute_diverse(
        bucket_keys=bucket_keys,
        vectors=vectors,
        quality=quality,
        max_keep=max_keep,
        min_distance=min_distance,
    )


__all__ = ["select_angle_diverse", "select_attribute_diverse"]
