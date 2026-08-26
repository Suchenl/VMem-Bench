"""Unit tests for per-movie resume action inference."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.orchestration.orchestrator import infer_resume_action


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_infer_resume_action_pipeline_when_empty(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    movie.mkdir()
    assert infer_resume_action(movie) == "pipeline"


def test_infer_resume_action_awaiting_s4(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    s4 = movie / "tmp" / "pipeline" / "s4_segment_sampling_human_review"
    _write(s4 / "review_queue.json", {"items": [{"segment_id": "seg_1"}]})
    _write(s4 / "review_audit.json", {"human_reviewed": False})
    assert infer_resume_action(movie) == "awaiting_human"


def test_infer_resume_action_after_s4(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    s4 = movie / "tmp" / "pipeline" / "s4_segment_sampling_human_review"
    _write(s4 / "review_queue.json", {"items": [{"segment_id": "seg_1"}]})
    _write(s4 / "review_audit.json", {"human_reviewed": True})
    assert infer_resume_action(movie) == "after_s4"


def test_infer_resume_action_after_s6(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    s6 = movie / "tmp" / "pipeline" / "s6_entities_visual_crop_human_review"
    _write(s6 / "review_queue.json", {"items": [{"id": "c1"}]})
    _write(s6 / "review_audit.json", {"human_reviewed": True})
    assert infer_resume_action(movie) == "after_s6"


def test_infer_resume_action_complete(tmp_path: Path) -> None:
    movie = tmp_path / "movie"
    gold = movie / "gold"
    gold.mkdir(parents=True)
    (gold / "entity_registry.json").write_text("{}", encoding="utf-8")
    assert infer_resume_action(movie) == "complete"
