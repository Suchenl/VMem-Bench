"""Deterministic SAM3/instance mask quality gates for crop candidates.

Default crop path stays masked (RGBA). Masks that are too fragmented are
rejected before they enter the picker candidate pool — no VLM involvement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# One connected component must overwhelmingly dominate foreground.  A mask made
# of a face/hat plus detached limbs is not a usable entity crop.
_MIN_LARGEST_CC_FRAC = 0.90
# Interior holes relative to filled silhouette (Swiss-cheese bodies).
_MAX_HOLE_FRAC = 0.12
# Any second meaningful island means the crop is not one coherent entity.
_MAX_SIGNIFICANT_CCS = 1
_MIN_SIGNIFICANT_CC_FRAC = 0.02  # of total foreground


@dataclass(slots=True)
class MaskQuality:
    ok: bool
    reasons: list[str]
    fg_area: int
    n_ccs: int
    largest_cc_frac: float
    hole_frac: float
    n_significant_ccs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool_mask(mask: object) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim > 2:
        array = array.squeeze()
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    if array.dtype == np.bool_:
        return array
    return array > 0


def assess_mask_quality(mask: object) -> MaskQuality:
    """Score instance mask integrity for candidate admission."""
    from scipy import ndimage as ndi

    binary = _as_bool_mask(mask)
    fg = int(binary.sum())
    if fg <= 0:
        return MaskQuality(False, ["empty_mask"], 0, 0, 0.0, 0.0, 0)

    labeled, n_ccs = ndi.label(binary)
    sizes = ndi.sum(binary, labeled, index=range(1, n_ccs + 1)) if n_ccs else []
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    largest = float(sizes_arr.max()) if sizes_arr.size else 0.0
    largest_frac = largest / float(fg)
    significant = int((sizes_arr / float(fg) >= _MIN_SIGNIFICANT_CC_FRAC).sum()) if sizes_arr.size else 0

    filled = ndi.binary_fill_holes(binary)
    hole_pixels = int(filled.sum() - fg)
    hole_frac = float(hole_pixels) / float(max(int(filled.sum()), 1))

    reasons: list[str] = []
    if largest_frac < _MIN_LARGEST_CC_FRAC:
        reasons.append("largest_cc_too_small")
    if hole_frac > _MAX_HOLE_FRAC:
        reasons.append("interior_holes")
    if significant > _MAX_SIGNIFICANT_CCS:
        reasons.append("too_many_fragments")

    return MaskQuality(
        ok=not reasons,
        reasons=reasons,
        fg_area=fg,
        n_ccs=int(n_ccs),
        largest_cc_frac=float(largest_frac),
        hole_frac=float(hole_frac),
        n_significant_ccs=significant,
    )


def is_mask_too_fragmented(mask: object) -> bool:
    """True when the mask should not enter the crop candidate pool."""
    return not assess_mask_quality(mask).ok


__all__ = [
    "MaskQuality",
    "assess_mask_quality",
    "is_mask_too_fragmented",
]
