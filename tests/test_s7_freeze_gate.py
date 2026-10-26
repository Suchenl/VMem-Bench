"""S7 blocker and human-review manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.build_gold import (
    build_gold,
)
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.gates import (
    soft_s3_residuals,
    unresolved_s3_blockers,
)


def _pipeline(tmp_path: Path) -> Path:
    root = tmp_path / "tmp" / "pipeline"
    s3 = root / "s3_segment_auto_review_revise"
    s3.mkdir(parents=True)
    rows = [
        {"segment_id": "seg_block", "verdict": "BLOCK", "recommended_action": "edit_action"},
        {"segment_id": "seg_warn", "verdict": "WARN", "recommended_action": "spot_check"},
        {"segment_id": "seg_retry", "verdict": "RETRYABLE_ERROR", "recommended_action": "retry"},
    ]
    (s3 / "segment_audit.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in rows),
        encoding="utf-8",
    )
    return root


def test_only_block_prevents_freeze_retryable_is_soft(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    unresolved = unresolved_s3_blockers(pipeline)
    soft = soft_s3_residuals(pipeline)
    assert {(item["segment_id"], item["verdict"]) for item in unresolved} == {
        ("seg_block", "BLOCK"),
    }
    assert {(item["segment_id"], item["verdict"]) for item in soft} == {
        ("seg_retry", "RETRYABLE_ERROR"),
    }

    s4 = pipeline / "s4_segment_sampling_human_review"
    s4.mkdir(parents=True)
    (s4 / "review_patch.applied.json").write_text(
        json.dumps({"resolved_verdicts": {"seg_block": "PASS"}}),
        encoding="utf-8",
    )
    unresolved = unresolved_s3_blockers(pipeline)
    soft = soft_s3_residuals(pipeline)
    assert unresolved == []
    assert soft == [{
        "segment_id": "seg_retry",
        "verdict": "RETRYABLE_ERROR",
        "recommended_action": "retry",
    }]


def test_gold_manifest_marks_only_real_human_review(tmp_path: Path) -> None:
    annotation = {
        "characters": [],
        "props": [],
        "locations": [],
        "screenplay": {"scenes": []},
    }
    human_movie = tmp_path / "human"
    smoke_movie = tmp_path / "smoke"
    build_gold(
        movie_dir=human_movie,
        annotation=annotation,
        accepted_crops=[],
        automation_smoke=False,
    )
    build_gold(
        movie_dir=smoke_movie,
        annotation=annotation,
        accepted_crops=[],
        automation_smoke=True,
    )
    human = json.loads((human_movie / "gold" / "manifest.json").read_text(encoding="utf-8"))
    smoke = json.loads((smoke_movie / "gold" / "manifest.json").read_text(encoding="utf-8"))
    assert human["human_reviewed"] is True
    assert smoke["human_reviewed"] is False


def test_direct_gold_build_rejects_unresolved_blocker(tmp_path: Path) -> None:
    movie = tmp_path / "blocked"
    pipeline = movie / "tmp" / "pipeline"
    s3 = pipeline / "s3_segment_auto_review_revise"
    s3.mkdir(parents=True)
    (s3 / "segment_audit.jsonl").write_text(
        json.dumps({"segment_id": "seg_1", "verdict": "BLOCK"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved S3 blockers"):
        build_gold(
            movie_dir=movie,
            annotation={
                "characters": [],
                "props": [],
                "locations": [],
                "screenplay": {"scenes": []},
            },
            accepted_crops=[],
            automation_smoke=False,
        )


def test_direct_gold_build_allows_retryable_residual(tmp_path: Path) -> None:
    movie = tmp_path / "retryable"
    pipeline = movie / "tmp" / "pipeline"
    s3 = pipeline / "s3_segment_auto_review_revise"
    s3.mkdir(parents=True)
    (s3 / "segment_audit.jsonl").write_text(
        json.dumps({"segment_id": "seg_1", "verdict": "RETRYABLE_ERROR"}) + "\n",
        encoding="utf-8",
    )
    gold = build_gold(
        movie_dir=movie,
        annotation={
            "characters": [],
            "props": [],
            "locations": [],
            "screenplay": {"scenes": []},
        },
        accepted_crops=[],
        automation_smoke=False,
    )
    assert (gold / "manifest.json").is_file()


def test_build_gold_keeps_multiple_accepted_crops_per_slot(tmp_path: Path) -> None:
    """Human add/replace can yield several accepted crops for one (chunk, entity)."""
    movie = tmp_path / "multi"
    crops_dir = tmp_path / "src_crops"
    crops_dir.mkdir()
    one = crops_dir / "a.png"
    two = crops_dir / "b.png"
    one.write_bytes(b"crop-a")
    two.write_bytes(b"crop-b")
    annotation = {
        "characters": [{
            "char_id": "char_001",
            "name": "Hero",
            "description": "Blue coat",
        }],
        "props": [],
        "locations": [],
        "screenplay": {"scenes": [{"visual_segments": [{
            "segment_id": "seg_0001",
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "action": "Hero stands.",
            "present_entity_ids": ["char_001"],
        }]}]},
    }
    accepted = [
        {
            "chunk_id": 0,
            "entity_id": "char_001",
            "kind": "character",
            "representation_id": "char_001@c00000",
            "crop_path": str(one),
            "frame_index": 0,
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "chunk_id": 0,
            "entity_id": "char_001",
            "kind": "character",
            "representation_id": "char_001@human_add_0001",
            "crop_path": str(two),
            "frame_index": 1,
            "bbox_norm": [10, 10, 90, 90],
        },
    ]
    gold = build_gold(
        movie_dir=movie,
        annotation=annotation,
        accepted_crops=accepted,
        automation_smoke=True,
    )
    index = json.loads((gold / "crop_index.json").read_text(encoding="utf-8"))
    ids = {row["representation_id"] for row in index["crops"]}
    assert ids == {"char_001@c00000", "char_001@human_add_0001"}
    assert len(list((gold / "crops").rglob("*.png"))) == 2
    chunks = json.loads((gold / "chunk_annotations.json").read_text(encoding="utf-8"))["chunks"]
    assert chunks[0]["gold_instructions"] == [
        {"entity_id": "char_001", "requirement": "introduce"}
    ]
    assert chunks[0]["forbidden"] == []
