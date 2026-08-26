"""Shared bbox geometry helpers for S5 crop routes."""

from __future__ import annotations


def px_to_norm_bbox(box_px: list[float], width: int, height: int) -> list[int]:
    """Convert [x0,y0,x1,y1] pixels to [ymin,xmin,ymax,xmax] on a 0-1000 grid."""
    x0, y0, x1, y1 = box_px
    return [
        max(0, min(1000, round(y0 / height * 1000))),
        max(0, min(1000, round(x0 / width * 1000))),
        max(0, min(1000, round(y1 / height * 1000))),
        max(0, min(1000, round(x1 / width * 1000))),
    ]


def norm_to_px_xyxy(bbox_norm: list[int], width: int, height: int) -> list[float]:
    """Convert [ymin,xmin,ymax,xmax] 0-1000 to [x0,y0,x1,y1] pixels."""
    y0, x0, y1, x1 = bbox_norm
    return [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height]


def mask_to_bbox_norm(mask) -> list[int] | None:
    """Tight [ymin,xmin,ymax,xmax] 0-1000 from a bool HxW mask."""
    import numpy as np

    value = np.asarray(mask, dtype=bool)
    if not value.any():
        return None
    ys, xs = np.where(value)
    height, width = value.shape[:2]
    return [
        round(int(ys.min()) / height * 1000),
        round(int(xs.min()) / width * 1000),
        round((int(ys.max()) + 1) / height * 1000),
        round((int(xs.max()) + 1) / width * 1000),
    ]


def bbox_iou(a: list[int], b: list[int]) -> float:
    """IoU of two [ymin,xmin,ymax,xmax] boxes on the same grid."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter <= 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def dedup_by_iou(
    items: list[dict],
    *,
    iou_threshold: float = 0.7,
    score_key: str = "score",
    bbox_key: str = "bbox_norm",
) -> list[dict]:
    """Keep highest-score boxes when IoU exceeds ``iou_threshold`` (deterministic)."""
    ranked = sorted(items, key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
    kept: list[dict] = []
    for item in ranked:
        bbox = item.get(bbox_key) or []
        if len(bbox) != 4:
            continue
        if any(bbox_iou(bbox, other[bbox_key]) >= iou_threshold for other in kept):
            continue
        kept.append(item)
    return kept
