"""S4 typed queue and actionable decision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.decisions import (
    apply_s4_decisions,
)
from vmem_bench.annotation.pipeline.servers.backend.review_service import (
    accept_all_s4,
)
from vmem_bench.annotation.pipeline.orchestration.orchestrator import (
    auto_accept_pending_s4,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.sampling import (
    build_sample,
)


def test_queue_contains_all_blocks_and_excludes_retryable() -> None:
    reviews = [
        {"segment_id": "b1", "verdict": "BLOCK"},
        {"segment_id": "b2", "verdict": "BLOCK"},
        {"segment_id": "w1", "verdict": "WARN"},
        {"segment_id": "p1", "verdict": "PASS"},
        {"segment_id": "r1", "verdict": "RETRYABLE_ERROR"},
    ]
    queue = build_sample(reviews, minimum=1, rate=0.2, seed=0)
    ids = {item["segment_id"] for item in queue}
    assert {"b1", "b2"}.issubset(ids)
    assert "r1" not in ids
    assert len(ids & {"w1", "p1"}) == 1


def test_nonblocking_sample_reserves_warn_and_pass_coverage() -> None:
    reviews = [
        *({"segment_id": f"w{i}", "verdict": "WARN"} for i in range(8)),
        *({"segment_id": f"p{i}", "verdict": "PASS"} for i in range(8)),
    ]
    queue = build_sample(reviews, minimum=4, rate=0, seed=0)
    verdicts = {item["verdict"] for item in queue}
    assert len(queue) == 4
    assert verdicts == {"WARN", "PASS"}


def _movie(tmp_path: Path) -> Path:
    movie = tmp_path / "movie"
    pipeline = movie / "tmp" / "pipeline"
    s3 = pipeline / "s3_segment_auto_review_revise"
    s4 = pipeline / "s4_segment_sampling_human_review"
    s3.mkdir(parents=True)
    s4.mkdir(parents=True)
    annotation = {
        "characters": [{"char_id": "char_001", "name": "大兔子"}],
        "props": [],
        "locations": [{"loc_id": "loc_001", "name": "开阔草地"}],
        "screenplay": {
            "scenes": [{
                "scene_id": "scene_1",
                "visual_segments": [{
                    "segment_id": "seg_1",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "action": "开阔草地上，大兔子站起身。",
                    "present_entity_ids": ["char_001", "loc_001"],
                }],
            }],
        },
    }
    (s3 / "auto_revised_annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )
    (s4 / "review_queue.json").write_text(
        json.dumps([{"segment_id": "seg_1", "verdict": "BLOCK"}]),
        encoding="utf-8",
    )
    return movie


def test_edit_both_applies_and_revalidates(tmp_path: Path) -> None:
    movie = _movie(tmp_path)
    result = apply_s4_decisions(
        movie_dir=movie,
        decisions={
            "seg_1": {
                "action": "edit_both",
                "present_entity_ids": ["char_001"],
                "revised_action": "大兔子站起身。",
            },
        },
    )
    assert result["n_overrides"] == 2
    revised = json.loads(Path(result["human_revised_annotation"]).read_text(encoding="utf-8"))
    segment = revised["screenplay"]["scenes"][0]["visual_segments"][0]
    assert segment["present_entity_ids"] == ["char_001"]
    assert segment["action"] == "大兔子站起身。"


def test_invalid_action_edit_is_rejected(tmp_path: Path) -> None:
    movie = _movie(tmp_path)
    with pytest.raises(ValueError, match="missing canonical names"):
        apply_s4_decisions(
            movie_dir=movie,
            decisions={
                "seg_1": {
                    "action": "edit_action",
                    "revised_action": "兔子站起身。",
                },
            },
        )


def test_request_retry_keeps_human_gate_open(tmp_path: Path) -> None:
    movie = _movie(tmp_path)
    result = apply_s4_decisions(
        movie_dir=movie,
        decisions={"seg_1": {"action": "request_retry"}},
    )
    assert result["n_retry_requested"] == 1
    audit = json.loads(
        (
            movie
            / "tmp"
            / "pipeline"
            / "s4_segment_sampling_human_review"
            / "review_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["human_reviewed"] is False
    assert audit["blocks_pipeline"] is True


def test_batch_accept_all_s4_marks_queue_human_reviewed(tmp_path: Path) -> None:
    movie = _movie(tmp_path)
    result = accept_all_s4({"movie_dir": str(movie)})

    assert result["status"] == "accepted"
    assert result["n_queue"] == 1
    audit = json.loads(
        (
            movie
            / "tmp"
            / "pipeline"
            / "s4_segment_sampling_human_review"
            / "review_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["human_reviewed"] is True
    assert audit["reason"] == "console_batch_accept_all_s4"
    assert accept_all_s4({"movie_dir": str(movie)})["status"] == "already_reviewed"


def test_auto_accept_pending_s4_marks_queue_human_reviewed(tmp_path: Path) -> None:
    movie = _movie(tmp_path)

    assert auto_accept_pending_s4(movie) is True
    audit = json.loads(
        (
            movie
            / "tmp"
            / "pipeline"
            / "s4_segment_sampling_human_review"
            / "review_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["human_reviewed"] is True
    assert audit["reason"] == "batch_auto_accept_s4"
    assert auto_accept_pending_s4(movie) is False

