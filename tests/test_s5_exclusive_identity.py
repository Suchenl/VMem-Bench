"""Tests for S5 exemplar identity + chunk-exclusive crop assignment."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_picker import (
    FirstCandidatePicker,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.exemplar_identity import (
    assign_by_exemplar,
    exclusive_assign_candidates,
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


def test_assign_by_exemplar_floor() -> None:
    exemplars = {"char_a": [[1.0, 0.0]], "char_b": [[0.0, 1.0]]}
    hit = assign_by_exemplar([0.99, 0.01], exemplars, sim_floor=0.5)
    assert hit is not None and hit[0] == "char_a"
    assert assign_by_exemplar([0.5, 0.5], exemplars, sim_floor=0.9) is None


def test_exclusive_assign_no_shared_candidate() -> None:
    exemplars = {
        "char_a": [[1.0, 0.0, 0.0]],
        "char_b": [[0.0, 1.0, 0.0]],
    }
    vecs = [
        [0.95, 0.05, 0.0],
        [0.05, 0.95, 0.0],
        [0.9, 0.1, 0.0],  # also like A but lower than first
    ]
    assignment, leftover = exclusive_assign_candidates(
        candidate_vecs=vecs,
        exemplars=exemplars,
        entity_ids=["char_a", "char_b"],
        sim_floor=0.5,
    )
    assert assignment["char_a"][0] == 0
    assert assignment["char_b"][0] == 1
    assert leftover == [2]


class _TwoAnimalSegmenter:
    def segment_multi(self, image: Path, concepts: list[str]):
        del image
        mask_a = np.zeros((100, 100), dtype=bool)
        mask_a[10:40, 10:40] = True
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_b[55:90, 55:90] = True
        out = {c: [] for c in concepts}
        for concept in concepts:
            out[concept] = [
                ([10.0, 10.0, 40.0, 40.0], 0.92, mask_a),
                ([55.0, 55.0, 90.0, 90.0], 0.91, mask_b),
            ]
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


def test_same_chunk_characters_do_not_share_bbox(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    _write_jpeg(frame)
    tasks = [
        CropTask(
            chunk_id=10,
            segment_id="seg_1",
            entity_id="char_005",
            kind="character",
            name="灰飞鼠",
            description="flying squirrel",
            action="eats",
            start_seconds=0.0,
            end_seconds=1.0,
        ),
        CropTask(
            chunk_id=10,
            segment_id="seg_1",
            entity_id="char_006",
            kind="character",
            name="灰老鼠",
            description="mouse",
            action="eats",
            start_seconds=0.0,
            end_seconds=1.0,
        ),
    ]
    with mock.patch(
        "vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition."
        "propose_pick_route.extract_candidates",
        _fake_extract([frame]),
    ):
        proposals = run_propose_and_pick(
            tasks=tasks,
            source_video=tmp_path / "missing.mp4",
            stage_dir=tmp_path / "s5_exclusive",
            picker=FirstCandidatePicker(),
            proposer="sam3",
            segmenter=_TwoAnimalSegmenter(),
            max_candidates=4,
        )
    assert len(proposals) == 2
    # Synthetic flat JPEG may fail sharpness QA; exclusivity is the contract under test.
    assert proposals[0]["bbox_source"] == "sam3_concept"
    assert proposals[1]["bbox_source"] == "sam3_concept"
    assert proposals[0]["bbox_norm"] != proposals[1]["bbox_norm"]
    assert proposals[0]["crop_path"] != proposals[1]["crop_path"]
    assert proposals[0].get("reason") != "no_detector_proposals"
    assert proposals[1].get("reason") != "no_detector_proposals"
