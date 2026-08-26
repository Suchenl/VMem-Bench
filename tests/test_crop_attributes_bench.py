"""Bench mirror: crop attributes + attribute dedup + S5 attach."""

from __future__ import annotations

from pathlib import Path

from vmem_bench.common.attribute_dedup import (
    select_angle_diverse,
    select_attribute_diverse,
)
from vmem_bench.common.crop_attributes import (
    CropAttributePack,
    HeuristicCropAttributeClassifier,
    Lighting,
    NullCropAttributeClassifier,
    ShotSize,
    SpatialAngle,
    StateAngle,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.attach_attributes import (
    attach_crop_attributes,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)


def test_bench_heuristic_pack() -> None:
    pack = HeuristicCropAttributeClassifier().classify(
        "/tmp/hero_side_changed_medium_night.png"
    )
    assert pack.spatial_angle == SpatialAngle.SIDE
    assert pack.state_angle == StateAngle.CHANGED
    assert pack.shot_size == ShotSize.MEDIUM
    assert pack.lighting == Lighting.NIGHT


def test_bench_attribute_dedup() -> None:
    kept = select_attribute_diverse(
        bucket_keys=[
            ("front", "default", "close_up", "day"),
            ("front", "default", "close_up", "day"),
            ("side", "default", "wide", "day"),
        ],
        quality=[0.4, 0.95, 0.7],
        max_keep=2,
    )
    assert set(kept) == {1, 2}


def test_bench_angle_wrapper() -> None:
    kept = select_angle_diverse(
        spatial_angles=["front", "side"],
        state_angles=["default", "default"],
        max_keep=2,
    )
    assert kept == [0, 1]


def test_attach_crop_attributes_null_default(tmp_path: Path) -> None:
    crop = tmp_path / "c.png"
    crop.write_bytes(b"not-a-real-image")
    task = CropTask(
        entity_id="char_lester",
        kind="character",
        name="Lester",
        description="man",
        action="standing",
        chunk_id=1,
        segment_id="seg_001",
        start_seconds=10.0,
        end_seconds=12.0,
    )
    proposal = {
        "accepted": True,
        "crop_path": str(crop),
        "frame_index": 42,
    }
    out = attach_crop_attributes(
        proposal,
        task=task,
        classifier=NullCropAttributeClassifier(),
    )
    assert "crop_attributes" in out
    attrs = out["crop_attributes"]
    assert attrs["source"] == "null"
    assert attrs["chunk_id"] == 1
    assert attrs["frame_index"] == 42
    assert attrs["spatial_angle"] == "unknown"


def test_attach_skips_rejected() -> None:
    task = CropTask(
        entity_id="x",
        kind="prop",
        name="bag",
        description="",
        action="",
        chunk_id=0,
        segment_id="seg_000",
        start_seconds=0.0,
        end_seconds=1.0,
    )
    out = attach_crop_attributes(
        {"accepted": False, "crop_path": "/tmp/x.png"},
        task=task,
        classifier=HeuristicCropAttributeClassifier(),
    )
    assert "crop_attributes" not in out


def test_pack_schema_keys_match_contract() -> None:
    keys = set(CropAttributePack().to_dict())
    assert {
        "spatial_angle",
        "state_angle",
        "shot_size",
        "lighting",
        "occlusion",
        "chunk_id",
        "frame_index",
        "seconds",
        "confidence",
        "reasoning",
        "source",
    } <= keys
