"""Tests for the deterministic dark / low-information crop gate in S5 crop_qa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_qa import (
    audit_crop,
    entity_luminance_stats,
)

_FULL_BBOX = [0, 0, 800, 800]  # plausible, not near-full-frame


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype("uint8"), mode="RGB").save(path)


def test_near_black_flat_crop_is_rejected(tmp_path: Path) -> None:
    crop = tmp_path / "dark.png"
    _save_rgb(crop, np.full((64, 48, 3), 8, dtype="uint8"))  # near-black, flat
    qa = audit_crop(crop=crop, bbox_norm=_FULL_BBOX, kind="character")
    assert "dark_low_information" in qa.reasons
    assert qa.accepted is False


def test_visible_textured_crop_passes(tmp_path: Path) -> None:
    crop = tmp_path / "ok.png"
    rng = np.random.default_rng(0)
    # Mid-bright, high-contrast content: plenty of identity signal.
    arr = rng.integers(40, 220, size=(64, 48, 3))
    _save_rgb(crop, arr)
    qa = audit_crop(crop=crop, bbox_norm=_FULL_BBOX, kind="character")
    assert "dark_low_information" not in qa.reasons


def test_dark_but_high_contrast_crop_passes(tmp_path: Path) -> None:
    # Dim scene but with real structure (std high) must NOT be rejected.
    crop = tmp_path / "dim.png"
    arr = np.zeros((64, 48, 3), dtype="uint8")
    arr[:, 24:, :] = 90  # half dark, half dim -> high std
    _save_rgb(crop, arr)
    qa = audit_crop(crop=crop, bbox_norm=_FULL_BBOX, kind="character")
    assert "dark_low_information" not in qa.reasons


def test_location_kind_skips_dark_gate(tmp_path: Path) -> None:
    crop = tmp_path / "night.png"
    _save_rgb(crop, np.full((64, 48, 3), 8, dtype="uint8"))
    qa = audit_crop(crop=crop, bbox_norm=_FULL_BBOX, kind="location")
    assert "dark_low_information" not in qa.reasons


def test_masked_region_luminance_ignores_white_backdrop(tmp_path: Path) -> None:
    # RGBA: near-black subject on a small alpha region; the transparent area
    # (which is composited to white for model feed) must not brighten the stat.
    crop = tmp_path / "masked.png"
    rgba = np.zeros((64, 48, 4), dtype="uint8")
    rgba[20:40, 15:30, :3] = 6      # near-black subject
    rgba[20:40, 15:30, 3] = 255     # only that box is opaque
    Image.fromarray(rgba, mode="RGBA").save(crop)
    mean_lum, std_lum = entity_luminance_stats(crop)
    assert mean_lum < 26.0 and std_lum < 16.0
    qa = audit_crop(crop=crop, bbox_norm=_FULL_BBOX, kind="character")
    assert "dark_low_information" in qa.reasons
