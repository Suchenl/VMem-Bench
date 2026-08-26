"""RGBA crop storage + white composite at model feed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    crop_to_model_data_url,
    load_crop_rgb_for_model,
    materialize_crop,
)
from memstrata.lib.media import load_crop_rgb_for_model as sut_load_crop


def test_masked_crop_saved_as_rgba_png(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(frame)
    # Full-frame mask with only a center blob true → crop bbox has transparent rim.
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:50, 40:60] = True
    out = materialize_crop(
        frame=frame,
        bbox_norm=[0, 0, 1000, 1000],
        out_path=tmp_path / "crop.jpg",  # suffix forced to .png
        mask=mask,
    )
    assert out.suffix == ".png"
    assert out.is_file()
    rgba = Image.open(out)
    assert rgba.mode == "RGBA"
    arr = np.asarray(rgba)
    assert arr[0, 0, 3] == 0
    assert arr[40, 50, 3] == 255
    assert tuple(arr[40, 50, :3].tolist()) == (10, 20, 30)


def test_model_feed_composites_white(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[1:3, 1:3, :3] = (40, 50, 60)
    rgba[1:3, 1:3, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)

    rgb = load_crop_rgb_for_model(path)
    assert rgb.mode == "RGB"
    arr = np.asarray(rgb)
    assert tuple(arr[0, 0].tolist()) == (255, 255, 255)
    assert tuple(arr[1, 1].tolist()) == (40, 50, 60)

    # SUT helper stays in sync.
    sut = sut_load_crop(path)
    assert np.array_equal(np.asarray(sut), arr)

    data_url = crop_to_model_data_url(path)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_unmasked_crop_is_opaque_png(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (50, 40), color=(1, 2, 3)).save(frame)
    out = materialize_crop(
        frame=frame,
        bbox_norm=[0, 0, 1000, 1000],
        out_path=tmp_path / "full.png",
    )
    assert out.suffix == ".png"
    image = Image.open(out)
    assert image.mode == "RGB"
