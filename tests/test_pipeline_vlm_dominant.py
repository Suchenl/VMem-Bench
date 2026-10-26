from __future__ import annotations

import pytest

pytest.importorskip("vmem_bench.annotation.pipeline_vlm_dominant.postprocess")
from vmem_bench.annotation.pipeline_vlm_dominant.postprocess import build_gold
from vmem_bench.annotation.pipeline_vlm_dominant.run import merge_p1_response, merge_p2_response


def test_build_gold_keeps_unlisted_entity_references_for_review() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
            {"chunk_id": 1, "shot_span": [1, 1], "frame_span": [10, 19]},
        ],
    }
    model_json = {
        "counts": {"characters": 1, "props": 0, "locations": 0, "shots": 1, "state_changes": 0},
        "characters": [
            {
                "name": "Rabbit",
                "identity_scope": "individual",
                "description": "white rabbit",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 0.0,
                "appearances": [],
            }
        ],
        "props": [],
        "locations": [],
        "shots": [
            {
                "start_seconds": 11.0,
                "end_seconds": 12.0,
                "description": "Rabbit sees a mystery acorn.",
                "camera": "medium",
                "present_entity_names": ["Rabbit", "mystery_acorn"],
            }
        ],
        "state_changes": [],
    }

    registry, annotations = build_gold(
        model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
        fps_used=1.0, layout_hash="hash")

    entities = {entity.entity_id: entity for entity in registry.entities}
    assert "char_001" in entities
    assert entities["char_001"].name == "Rabbit"
    placeholder = entities["prop_001"]
    assert placeholder.name == "mystery_acorn"
    assert placeholder.first_chunk == 1
    assert "missing from the model's grouped entity table" in placeholder.description
    assert "prop_001" in annotations.chunks[1].present
    assert "prop_001" in annotations.chunks[1].first_appearances


def test_multi_stage_merges_do_not_count_nested_entity_arrays_top_level() -> None:
    roster_json = {
        "counts": {"characters": 1, "props": 1, "locations": 0},
        "characters": [
            {
                "name": "Rabbit",
                "identity_scope": "individual",
                "description": "white rabbit",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 5.0,
                "appearance_count": 2,
                "appearances": [
                    {"start_seconds": 0.0, "end_seconds": 1.0, "description": "first"},
                    {"start_seconds": 4.0, "end_seconds": 5.0, "description": "second"},
                ],
                "state_change_count": 0,
                "state_changes": [],
            }
        ],
        "props": [
            {
                "name": "Apple",
                "identity_scope": "individual",
                "description": "red apple",
                "first_appearance_seconds": 2.0,
                "last_appearance_seconds": 3.0,
                "appearance_count": 1,
                "appearances": [
                    {"start_seconds": 2.0, "end_seconds": 3.0, "description": "visible"},
                ],
                "state_change_count": 1,
                "state_changes": [
                    {
                        "seconds": 3.0,
                        "state_change_kind": "consumed",
                        "description": "eaten",
                    }
                ],
            }
        ],
        "locations": [],
    }
    shots_json = {"counts": {"shots": 1}, "shots": [{"present_entity_names": ["Rabbit"]}]}
    timeline_json = {**shots_json, "state_changes": [{"entity_name": "Apple"}]}

    assert merge_p1_response(roster_json, shots_json)["counts"] == {
        "characters": 1, "props": 1, "locations": 0, "shots": 1}
    assert merge_p2_response(roster_json, timeline_json)["counts"] == {
        "characters": 1, "props": 1, "locations": 0, "shots": 1, "state_changes": 1}


def test_build_gold_uses_entity_appearances_as_presence_evidence() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
            {"chunk_id": 1, "shot_span": [1, 1], "frame_span": [10, 19]},
        ],
    }
    model_json = {
        "counts": {"characters": 1, "props": 0, "locations": 0, "shots": 1, "state_changes": 0},
        "characters": [
            {
                "name": "Rabbit",
                "identity_scope": "individual",
                "description": "white rabbit",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 12.0,
                "appearances": [
                    {"start_seconds": 1.0, "end_seconds": 3.0, "description": "sleeping"},
                    {"start_seconds": 11.0, "end_seconds": 12.0, "description": "reappears"},
                ],
            }
        ],
        "props": [],
        "locations": [],
        "shots": [
            {
                "start_seconds": 11.0,
                "end_seconds": 12.0,
                "description": "The rabbit is visible, but the model omitted shot-level IDs.",
                "camera": "medium",
                "present_entity_names": [],
            }
        ],
        "state_changes": [],
    }

    _registry, annotations = build_gold(
        model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
        fps_used=1.0, layout_hash="hash")

    assert "char_001" in annotations.chunks[0].present
    assert "char_001" in annotations.chunks[1].present


def test_build_gold_preserves_state_change_kind_in_state_event_description() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
        ],
    }
    model_json = {
        "counts": {"characters": 0, "props": 1, "locations": 0, "shots": 1, "state_changes": 1},
        "characters": [],
        "props": [
            {
                "name": "Apple",
                "identity_scope": "individual",
                "description": "red apple",
                "first_appearance_seconds": 1.0,
                "last_appearance_seconds": 5.0,
                "appearances": [
                    {"start_seconds": 1.0, "end_seconds": 5.0, "description": "on screen"},
                ],
            }
        ],
        "locations": [],
        "shots": [
            {
                "start_seconds": 1.0,
                "end_seconds": 5.0,
                "description": "The apple is eaten.",
                "camera": "close-up",
                "present_entity_names": ["Apple"],
            }
        ],
        "state_changes": [
            {
                "seconds": 4.0,
                "entity_name": "Apple",
                "state_change_kind": "consumed",
                "description": "The apple is eaten.",
            }
        ],
    }

    registry, _annotations = build_gold(
        model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
        fps_used=1.0, layout_hash="hash")

    apple = next(entity for entity in registry.entities if entity.entity_id == "prop_001")
    assert apple.state_events[0].description == "consumed: The apple is eaten."


def test_build_gold_rejects_legacy_plural_state_change_references() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
        ],
    }
    model_json = {
        "counts": {"characters": 0, "props": 1, "locations": 0, "shots": 0, "state_changes": 1},
        "characters": [],
        "props": [
            {
                "name": "Apple",
                "identity_scope": "individual",
                "description": "red apple",
                "first_appearance_seconds": 1.0,
                "last_appearance_seconds": 1.0,
                "appearances": [],
            }
        ],
        "locations": [],
        "shots": [],
        "state_changes": [
            {
                "seconds": 4.0,
                "entity_names": ["Apple"],
                "state_change_kind": "consumed",
                "description": "The apple is eaten.",
            }
        ],
    }

    with pytest.raises(ValueError, match="requires entity_name"):
        build_gold(
            model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
            fps_used=1.0, layout_hash="hash")


def test_build_gold_reads_nested_entity_state_changes() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
        ],
    }
    model_json = {
        "counts": {"characters": 1, "props": 0, "locations": 0, "shots": 0, "state_changes": 1},
        "characters": [
            {
                "name": "Rabbit",
                "identity_scope": "individual",
                "description": "white rabbit",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 0.0,
                "appearances": [],
                "state_changes": [
                    {
                        "seconds": 6.0,
                        "state_change_kind": "appearance_changed",
                        "description": "Rabbit becomes covered in mud.",
                    }
                ],
            }
        ],
        "props": [],
        "locations": [],
        "shots": [],
    }

    registry, annotations = build_gold(
        model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
        fps_used=1.0, layout_hash="hash")

    rabbit = next(entity for entity in registry.entities if entity.entity_id == "char_001")
    assert rabbit.state_events[0].description == "appearance_changed: Rabbit becomes covered in mud."


def test_build_gold_rejects_legacy_entities_shape() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
        ],
    }
    model_json = {
        "entities": [
            {
                "entity_id": "rabbit",
                "name": "Rabbit",
                "kind": "character",
                "identity_scope": "individual",
                "description": "white rabbit",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 0.0,
                "appearances": [],
            }
        ],
        "shots": [],
        "state_changes": [],
    }

    with pytest.raises(ValueError, match="missing grouped entity arrays"):
        build_gold(
            model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
            fps_used=1.0, layout_hash="hash")


def test_build_gold_rejects_duplicate_entity_names() -> None:
    chunk_index = {
        "fps": 1.0,
        "layout_hash": "hash",
        "chunks": [
            {"chunk_id": 0, "shot_span": [0, 0], "frame_span": [0, 9]},
        ],
    }
    model_json = {
        "counts": {"characters": 1, "props": 1, "locations": 0, "shots": 0, "state_changes": 0},
        "characters": [
            {
                "name": "Apple",
                "identity_scope": "individual",
                "description": "a character named Apple",
                "first_appearance_seconds": 0.0,
                "last_appearance_seconds": 0.0,
                "appearances": [],
            }
        ],
        "props": [
            {
                "name": "Apple",
                "identity_scope": "individual",
                "description": "a red apple",
                "first_appearance_seconds": 1.0,
                "last_appearance_seconds": 1.0,
                "appearances": [],
            }
        ],
        "locations": [],
        "shots": [],
        "state_changes": [],
    }

    with pytest.raises(ValueError, match="duplicate model entity name"):
        build_gold(
            model_json, movie_id="movie", chunk_index=chunk_index, model_name="qwen",
            fps_used=1.0, layout_hash="hash")
