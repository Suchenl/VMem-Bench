"""Focused contract tests for the self-contained S2 post-processing stage."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s2_annotation_postprocess import postprocess_annotation
from vmem_bench.annotation.pipeline.stages.s2_annotation_postprocess.materialize import main


def _valid_annotation() -> dict:
    return {
        "video_duration_seconds": 10.0,
        "minimum_required_segments": 1,
        "characters": [],
        "props": [],
        "locations": [
            {
                "loc_id": "loc_001",
                "name": "Room",
                "identity_scope": "scene",
                "description": "A room.",
                "first_presence_seconds": 0.0,
                "last_presence_seconds": 10.0,
            }
        ],
        "screenplay": {
            "scenes": [
                {
                    "scene_id": "scene_0001",
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "loc_id": "loc_001",
                    "visual_segments": [
                        {
                            "segment_id": "seg_0001",
                            "start_seconds": 0.0,
                            "end_seconds": 10.0,
                            "duration_seconds": 10.0,
                            "present_entity_ids": ["loc_001"],
                        }
                    ],
                }
            ]
        },
        "self_check": {"low_confidence_segment_ids": []},
        "counts": {
            "characters": 0,
            "props": 0,
            "locations": 1,
            "scenes": 1,
            "visual_segments": 1,
        },
    }


def _write_json(path: Path, annotation: dict) -> None:
    path.write_text(json.dumps(annotation), encoding="utf-8")


def _read_codes(path: Path) -> set[str]:
    return {item["code"] for item in json.loads(path.read_text(encoding="utf-8"))["errors"]}


def _read_warning_codes(path: Path) -> set[str]:
    return {item["code"] for item in json.loads(path.read_text(encoding="utf-8"))["warnings"]}


def test_materializes_normalized_annotation_and_all_check_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "s2"
    _write_json(source, _valid_annotation())

    result = postprocess_annotation(source, output)

    assert result["status"] == "ok"
    assert (output / "normalized_annotation.json").is_file()
    assert {path.name for path in output.iterdir()} == {
        "normalized_annotation.json",
        "structural_lint.json",
        "entity_checks.json",
        "segment_checks.json",
    }
    assert json.loads((output / "segment_checks.json").read_text(encoding="utf-8"))["segment_index"] == [
        {
            "end_seconds": 10.0,
            "scene_id": "scene_0001",
            "segment_id": "seg_0001",
            "start_seconds": 0.0,
        }
    ]


def test_count_mismatch_is_warning_and_synced_in_normalize(tmp_path: Path) -> None:
    source = tmp_path / "counts_off.json"
    output = tmp_path / "s2"
    annotation = deepcopy(_valid_annotation())
    annotation["counts"]["scenes"] = 10
    annotation["counts"]["visual_segments"] = 9
    _write_json(source, annotation)

    result = postprocess_annotation(source, output)

    assert result["status"] == "ok"
    assert "COUNT_MISMATCH" in _read_warning_codes(output / "structural_lint.json")
    assert not _read_codes(output / "structural_lint.json")
    assert json.loads(source.read_text(encoding="utf-8")) == annotation
    normalized = json.loads((output / "normalized_annotation.json").read_text(encoding="utf-8"))
    assert normalized["counts"] == {
        "characters": 0,
        "props": 0,
        "locations": 1,
        "scenes": 1,
        "visual_segments": 1,
    }


def test_reports_structural_entity_and_segment_failures_without_rewriting_source(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    output = tmp_path / "s2"
    annotation = deepcopy(_valid_annotation())
    annotation["counts"]["visual_segments"] = 9
    annotation["locations"][0]["first_presence_seconds"] = 12.0
    segment = annotation["screenplay"]["scenes"][0]["visual_segments"][0]
    segment["duration_seconds"] = 9.0
    segment["present_entity_ids"] = ["loc_001", "loc_001", "missing"]
    annotation["self_check"]["low_confidence_segment_ids"] = ["not_a_segment"]
    _write_json(source, annotation)

    result = postprocess_annotation(source, output)

    assert result["status"] == "invalid_structure"
    assert "COUNT_MISMATCH" in _read_warning_codes(output / "structural_lint.json")
    assert "COUNT_MISMATCH" not in _read_codes(output / "structural_lint.json")
    assert "PRESENCE_TIME_OUT_OF_RANGE" in _read_warning_codes(output / "entity_checks.json")
    segment_codes = _read_codes(output / "segment_checks.json")
    assert {"SEGMENT_DURATION_MISMATCH", "DUPLICATE_PRESENT_ENTITY_ID", "UNKNOWN_PRESENT_ENTITY_ID"} <= segment_codes
    assert "UNKNOWN_SELF_CHECK_SEGMENT" in segment_codes
    assert json.loads(source.read_text(encoding="utf-8")) == annotation
    assert (output / "normalized_annotation.json").is_file()
    normalized = json.loads((output / "normalized_annotation.json").read_text(encoding="utf-8"))
    assert normalized["counts"]["visual_segments"] == 1


def test_empty_and_non_json_inputs_have_explicit_statuses(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")

    empty_result = postprocess_annotation(empty, tmp_path / "empty-output")
    malformed_result = postprocess_annotation(malformed, tmp_path / "malformed-output")

    assert empty_result["status"] == "skipped_empty_input"
    assert malformed_result["status"] == "invalid_json"
    assert "EMPTY_INPUT" in _read_codes(tmp_path / "empty-output" / "structural_lint.json")
    assert "INVALID_JSON" in _read_codes(tmp_path / "malformed-output" / "structural_lint.json")
    assert not (tmp_path / "empty-output" / "normalized_annotation.json").exists()
    assert not (tmp_path / "malformed-output" / "normalized_annotation.json").exists()


def test_cli_and_existing_bbb_and_lsmdc_inputs_materialize_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "cli.json"
    _write_json(source, _valid_annotation())
    cli_output = tmp_path / "cli-output"

    assert main([str(source), str(cli_output)]) == 0

    data = Path(__file__).parents[1] / "data"
    for relative_path in (
        "BlenderOpenMovies/big_buck_bunny_720p/vlm_output.json",
        "LSMDC/0003_CASABLANCA/vlm_output.json",
    ):
        output = tmp_path / relative_path.replace("/", "_")
        result = postprocess_annotation(data / relative_path, output)
        assert result["status"] in {"ok", "invalid_structure"}
        assert (output / "normalized_annotation.json").is_file()
        assert (output / "structural_lint.json").is_file()
        assert (output / "entity_checks.json").is_file()
        assert (output / "segment_checks.json").is_file()
