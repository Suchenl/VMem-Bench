"""Unit tests for S5 dual crop routes (no GPU / no VLM service)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    FirstCandidatePicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_qa import (
    audit_crop,
    is_near_full_frame,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.geometry import (
    bbox_iou,
    dedup_by_iou,
    px_to_norm_bbox,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.keyframes import (
    FrameCandidate,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.propose_pick_route import (
    run_propose_and_pick,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_grounding import (
    FullFrameGrounder,
    GroundingResult,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.vlm_sam_route import (
    run_vlm_sam_refine,
)


def test_geometry_helpers() -> None:
    assert px_to_norm_bbox([100, 50, 300, 250], 1000, 500) == [100, 100, 500, 300]
    assert bbox_iou([0, 0, 500, 500], [250, 250, 750, 750]) > 0.1
    kept = dedup_by_iou(
        [
            {"bbox_norm": [0, 0, 400, 400], "score": 0.9},
            {"bbox_norm": [10, 10, 410, 410], "score": 0.5},
            {"bbox_norm": [600, 600, 900, 900], "score": 0.8},
        ],
        iou_threshold=0.5,
    )
    assert len(kept) == 2
    assert kept[0]["score"] == 0.9


def test_near_full_frame_gate_allows_closeups(tmp_path: Path) -> None:
    assert is_near_full_frame([0, 0, 1000, 1000])
    assert is_near_full_frame([5, 5, 995, 995])
    assert is_near_full_frame([0, 0, 980, 980])  # area >= 0.95
    assert not is_near_full_frame([100, 100, 900, 900])  # 64% close-up
    assert not is_near_full_frame([200, 100, 900, 800])  # ~49%

    from PIL import Image

    crop = tmp_path / "c.jpg"
    Image.new("RGB", (64, 64), color=(40, 120, 200)).save(crop)

    closeup = audit_crop(crop=crop, bbox_norm=[50, 50, 950, 950], kind="character")
    assert "implausible_bbox_area" not in closeup.reasons

    full = audit_crop(crop=crop, bbox_norm=[0, 0, 1000, 1000], kind="character")
    assert "implausible_bbox_area" in full.reasons

    loc = audit_crop(crop=crop, bbox_norm=[0, 0, 1000, 1000], kind="location")
    assert "implausible_bbox_area" not in loc.reasons


class _TightGrounder:
    def ground(self, *, image: Path, frame_index: int, entity_id: str, name: str,
               description: str, action: str) -> GroundingResult:
        del image, entity_id, name, description, action
        return GroundingResult(
            frame_index=frame_index,
            usable=True,
            bbox_norm=[200, 200, 500, 500],
            point_norm=[350, 350],
            visibility="stub",
            confidence="high",
            reason="stub tight box",
        )


class _FakeSegmenter:
    def segment_multi(self, image: Path, concepts: list[str]):
        del image
        import numpy as np

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


def test_vlm_sam_route_without_refiner(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    _write_jpeg(frame)
    with mock.patch(
        "vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition."
        "vlm_sam_route.extract_candidates",
        _fake_extract([frame]),
    ):
        proposals = run_vlm_sam_refine(
            tasks=[_character_task()],
            source_video=tmp_path / "missing.mp4",
            stage_dir=tmp_path / "s5",
            grounder=_TightGrounder(),
            refiner=None,
            require_sam3=False,
        )
    assert len(proposals) == 1
    assert proposals[0]["route"] == "vlm_sam_refine"
    assert proposals[0]["bbox_norm"] == [200, 200, 500, 500]
    assert proposals[0]["bbox_source"] == "vlm_bbox"
    assert Path(proposals[0]["crop_path"]).is_file()


def test_vlm_sam_require_sam3_skips_without_refiner(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    _write_jpeg(frame)
    with mock.patch(
        "vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition."
        "vlm_sam_route.extract_candidates",
        _fake_extract([frame]),
    ):
        proposals = run_vlm_sam_refine(
            tasks=[_character_task()],
            source_video=tmp_path / "missing.mp4",
            stage_dir=tmp_path / "s5",
            grounder=_TightGrounder(),
            refiner=None,
            require_sam3=True,
        )
    assert proposals[0]["accepted"] is False
    assert proposals[0]["reason"] == "no_usable_grounding"


def test_propose_and_pick_sam3_route(tmp_path: Path) -> None:
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
            stage_dir=tmp_path / "s5",
            picker=FirstCandidatePicker(),
            proposer="sam3",
            segmenter=_FakeSegmenter(),
        )
    assert len(proposals) == 1
    assert proposals[0]["route"] == "propose_and_pick"
    assert proposals[0]["bbox_source"] == "sam3_concept"
    assert proposals[0]["pick"]["index"] == 0
    assert Path(proposals[0]["crop_path"]).is_file()
    assert Path(proposals[0]["unmasked_crop_path"]).is_file()
    assert proposals[0]["bbox_norm"] == [200, 200, 600, 600]
    from PIL import Image

    assert Image.open(proposals[0]["crop_path"]).mode == "RGBA"
    assert Image.open(proposals[0]["unmasked_crop_path"]).mode == "RGB"


def test_full_frame_grounder_contract() -> None:
    result = FullFrameGrounder().ground(
        image=Path("."),
        frame_index=0,
        entity_id="loc_001",
        name="Room",
        description="",
        action="",
    )
    assert result.bbox_norm == [0, 0, 1000, 1000]
    assert result.visibility == "dry_run_full_frame"


if __name__ == "__main__":
    import tempfile

    test_geometry_helpers()
    test_full_frame_grounder_contract()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        test_near_full_frame_gate_allows_closeups(root / "gate")
        test_vlm_sam_route_without_refiner(root / "a")
        test_vlm_sam_require_sam3_skips_without_refiner(root / "b")
        test_propose_and_pick_sam3_route(root / "c")
    print("test_pipeline_s5_crop_routes: OK")
