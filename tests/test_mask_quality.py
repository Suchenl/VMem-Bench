"""Tests for S5 instance-mask fragmentation gate."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    FirstCandidatePicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.mask_quality import (
    assess_mask_quality,
    is_mask_too_fragmented,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.propose_pick_route import (
    run_propose_and_pick,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.keyframes import (
    FrameCandidate,
)


def test_solid_mask_passes() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 20:60] = True
    quality = assess_mask_quality(mask)
    assert quality.ok
    assert quality.largest_cc_frac == 1.0
    assert quality.hole_frac == 0.0
    assert not is_mask_too_fragmented(mask)


def test_floating_fragment_rejected() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True  # main body
    mask[80:95, 80:95] = True  # floating foot (~22% of fg → largest ~78% may pass)
    # Make floating piece larger so largest_cc_frac drops below 0.75
    mask[70:95, 70:95] = True
    quality = assess_mask_quality(mask)
    assert not quality.ok
    assert "largest_cc_too_small" in quality.reasons or "too_many_fragments" in quality.reasons


def test_small_detached_islands_are_rejected() -> None:
    """Regression: a dominant face/hat plus floating limbs is not one crop."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:80, 0:10] = True  # 89% main connected component
    mask[85:90, 0:10] = True  # 5.5% detached island
    mask[95:100, 0:10] = True  # 5.5% detached island

    quality = assess_mask_quality(mask)

    assert not quality.ok
    assert "largest_cc_too_small" in quality.reasons
    assert "too_many_fragments" in quality.reasons


def test_interior_holes_rejected() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    mask[25:55, 25:55] = False  # big interior hole
    quality = assess_mask_quality(mask)
    assert not quality.ok
    assert "interior_holes" in quality.reasons


def test_empty_mask_rejected() -> None:
    quality = assess_mask_quality(np.zeros((16, 16), dtype=bool))
    assert not quality.ok
    assert "empty_mask" in quality.reasons


class _FragmentedSegmenter:
    def segment_multi(self, image: Path, concepts: list[str]):
        del image
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:40, 10:40] = True
        mask[70:95, 70:95] = True
        out = {c: [] for c in concepts}
        for concept in concepts:
            out[concept] = [([10.0, 10.0, 95.0, 95.0], 0.9, mask)]
        return out


class _SolidSegmenter:
    def segment_multi(self, image: Path, concepts: list[str]):
        del image
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 20:60] = True
        out = {c: [] for c in concepts}
        for concept in concepts:
            out[concept] = [([20.0, 20.0, 60.0, 60.0], 0.91, mask)]
        return out


def _write_jpeg(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(120, 80, 40)).save(path, quality=90)


def _fake_extract(monkey_frames: list[Path]):
    def _extract(*, source_video, start_seconds, end_seconds, out_dir, **_kwargs):
        del source_video, start_seconds, end_seconds
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        selected = []
        for ordinal, src in enumerate(monkey_frames):
            dst = out_dir / f"candidate_{ordinal:02d}_f{ordinal:08d}.jpg"
            dst.write_bytes(Path(src).read_bytes())
            selected.append(
                FrameCandidate(
                    frame_index=ordinal,
                    seconds=float(ordinal),
                    path=str(dst),
                    sharpness=10.0 + ordinal,
                    luminance_std=5.0,
                )
            )
        return selected

    return _extract


def _character_task() -> CropTask:
    return CropTask(
        chunk_id=0,
        segment_id="seg_1",
        entity_id="char_001",
        kind="character",
        name="Bunny",
        description="a rabbit",
        action="hops",
        start_seconds=0.0,
        end_seconds=1.0,
    )


def test_fragmented_sam3_mask_not_admitted(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    _write_jpeg(frame)
    with mock.patch(
        "vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition."
        "propose_pick_route.extract_candidates",
        _fake_extract([frame]),
    ):
        proposals = run_propose_and_pick(
            tasks=[_character_task()],
            source_video=tmp_path / "missing.mp4",
            stage_dir=tmp_path / "s5_frag",
            picker=FirstCandidatePicker(),
            proposer="sam3",
            segmenter=_FragmentedSegmenter(),
        )
    assert proposals[0]["accepted"] is False
    assert proposals[0]["reason"] == "no_detector_proposals"


def test_solid_sam3_mask_still_admitted(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    _write_jpeg(frame)
    with mock.patch(
        "vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition."
        "propose_pick_route.extract_candidates",
        _fake_extract([frame]),
    ):
        proposals = run_propose_and_pick(
            tasks=[_character_task()],
            source_video=tmp_path / "missing.mp4",
            stage_dir=tmp_path / "s5_solid",
            picker=FirstCandidatePicker(),
            proposer="sam3",
            segmenter=_SolidSegmenter(),
        )
    # Mask gate admits; crop_qa sharpness may fail on flat synthetic JPEG.
    assert proposals[0]["mask_quality"]["ok"] is True
    assert proposals[0]["bbox_source"] == "sam3_concept"
    assert Path(proposals[0]["crop_path"]).is_file()
    assert "reason" not in proposals[0] or proposals[0].get("reason") != "no_detector_proposals"
