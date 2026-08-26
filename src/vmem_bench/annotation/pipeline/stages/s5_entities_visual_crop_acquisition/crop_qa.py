"""Deterministic QA and materialization helpers for current GT crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    load_crop_rgb_for_model,
    materialize_crop,
)

# Geometry gate for non-location crops:
# - reject degenerate tiny boxes
# - reject near-full-frame failures (dry-run / loose VLM / GDINO whole-shot)
# Do NOT reject legitimate close-ups (area can be 50–90% of the frame).
_MIN_ENTITY_AREA = 0.001
_NEAR_FULL_FRAME_AREA = 0.95
_NEAR_FULL_FRAME_MARGIN = 20  # norm space 0–1000

# Dark / low-information gate (non-location only): a near-black, near-flat crop
# carries no usable identity signal (e.g. a silhouette or a blob lost in shadow).
# Thresholds are deliberately conservative so a dimly-lit-but-visible subject
# (which still has facial/edge contrast, hence higher std) is NOT rejected.
_DARK_MEAN_MAX = 26.0   # mean luminance 0–255 over entity pixels
_DARK_STD_MAX = 16.0    # luminance std 0–255 over entity pixels


@dataclass(slots=True)
class CropQa:
    accepted: bool
    reasons: list[str]
    area_fraction: float
    sharpness: float
    mean_luminance: float = 0.0
    luminance_std: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def entity_luminance_stats(crop: Path) -> tuple[float, float]:
    """Mean/std luminance over entity pixels (mask alpha>0), else the whole crop.

    Measuring only the masked entity region matters: a masked crop is composited
    onto white for model feed, so a dark silhouette would otherwise look bright.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(crop)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = np.asarray(image.convert("RGBA"), dtype="float32")
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        vals = lum[alpha > 0]
    else:
        rgb = np.asarray(image.convert("RGB"), dtype="float32")
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        vals = lum.reshape(-1)
    if vals.size < 4:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())


def bbox_area_fraction(bbox_norm: list[int]) -> float:
    if len(bbox_norm) != 4:
        return 0.0
    y0, x0, y1, x1 = bbox_norm
    return max(0, y1 - y0) * max(0, x1 - x0) / 1_000_000


def is_near_full_frame(bbox_norm: list[int]) -> bool:
    """True when the box is essentially the whole frame (localization failure)."""
    if len(bbox_norm) != 4:
        return False
    y0, x0, y1, x1 = bbox_norm
    if bbox_area_fraction(bbox_norm) >= _NEAR_FULL_FRAME_AREA:
        return True
    m = _NEAR_FULL_FRAME_MARGIN
    return y0 <= m and x0 <= m and y1 >= 1000 - m and x1 >= 1000 - m


def audit_crop(*, crop: Path, bbox_norm: list[int], kind: str) -> CropQa:
    """Cheap geometry and sharpness checks; semantic checks happen in S6."""
    import numpy as np

    reasons: list[str] = []
    if len(bbox_norm) != 4:
        return CropQa(False, ["invalid_bbox"], 0.0, 0.0)
    area = bbox_area_fraction(bbox_norm)
    if kind != "location":
        if area < _MIN_ENTITY_AREA or is_near_full_frame(bbox_norm):
            reasons.append("implausible_bbox_area")
    # Composite white before sharpness so transparent regions don't dominate.
    gray = np.asarray(load_crop_rgb_for_model(crop).convert("L"), dtype="float32")
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(lap.var()) if lap.size else 0.0
    if sharpness <= 1.0:
        reasons.append("low_sharpness")
    # Near-black, near-flat entity region → no usable identity signal.
    mean_luminance, luminance_std = entity_luminance_stats(crop)
    if kind != "location" and mean_luminance < _DARK_MEAN_MAX and luminance_std < _DARK_STD_MAX:
        reasons.append("dark_low_information")
    return CropQa(not reasons, reasons, area, sharpness, mean_luminance, luminance_std)


__all__ = [
    "CropQa",
    "audit_crop",
    "entity_luminance_stats",
    "bbox_area_fraction",
    "is_near_full_frame",
    "load_crop_rgb_for_model",
    "materialize_crop",
]
