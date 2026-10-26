"""Unit tests for clearing S5/S6 before rerun."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.servers.backend.jobs import _clear_s5_s6


def test_clear_s5_s6_removes_dirs_and_state_entries(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    pipeline = movie / "tmp" / "pipeline"
    s4 = pipeline / "s4_segment_sampling_human_review"
    s5 = pipeline / "s5_entities_visual_crop_acquisition"
    s6 = pipeline / "s6_entities_visual_crop_human_review"
    s4.mkdir(parents=True)
    s5.mkdir(parents=True)
    s6.mkdir(parents=True)
    (s4 / "review_audit.json").write_text('{"human_reviewed": true}', encoding="utf-8")
    (s5 / "crop_proposals.json").write_text("[]", encoding="utf-8")
    (s6 / "review_queue.json").write_text("[]", encoding="utf-8")
    (pipeline / "state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "s4_segment_sampling_human_review": {"status": "human_reviewed"},
                    "s5_entities_visual_crop_acquisition": {"status": "ok"},
                    "s6_entities_visual_crop_human_review": {"status": "human_reviewed"},
                    "s7_freeze_publish": {"status": "ok"},
                }
            }
        ),
        encoding="utf-8",
    )

    cleared = _clear_s5_s6(movie)
    assert any("s5_entities_visual_crop_acquisition" in path for path in cleared)
    assert any("s6_entities_visual_crop_human_review" in path for path in cleared)
    assert not s5.exists()
    assert not s6.exists()
    assert s4.is_dir()
    state = json.loads((pipeline / "state.json").read_text(encoding="utf-8"))
    stages = state["stages"]
    assert "s4_segment_sampling_human_review" in stages
    assert "s5_entities_visual_crop_acquisition" not in stages
    assert "s6_entities_visual_crop_human_review" not in stages
    assert "s7_freeze_publish" not in stages
