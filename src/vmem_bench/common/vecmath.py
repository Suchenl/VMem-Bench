"""Tiny vector math helpers (self-contained, no numpy dependency required)."""

from __future__ import annotations

import math

Vector = list[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must share the same dimensionality")
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)


def cosine_distance(left: Vector, right: Vector) -> float:
    return 1.0 - cosine_similarity(left, right)
