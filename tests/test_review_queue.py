"""CPU-only contract checks for the read-only review queue."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.review_queue import build_review_queue, write_review_queue


def test_build_queue_identity_prompt_lint_and_stable_order() -> None:
    queue = build_review_queue(
        auto_review={"must_review": ["char_a"], "queue": [{"entity_id": "char_a", "score": 2}]},
        identity_candidates=[{
            "left": "char_a", "right": "char_b", "left_chunk_span": [0, 2],
            "right_chunk_span": [2, 4], "body_cos": 0.9,
            "left_representative_crop": "assets/a.jpg", "right_representative_crop": "assets/b.jpg",
            "recommendation": "review_merge",
        }],
        qa_report=[{"chunk_id": 3, "findings": [{"code": "prompt_missing_present_entity", "entity_id": "char_b"}]}],
        strict_lint=[
            {"code": "canonical_alias_split", "severity": "error", "path": "entities[char_a]", "message": "x"},
            {"code": "canonical_alias_split", "severity": "error", "path": "entities[char_a]", "message": "x again"},
        ])
    assert queue["version"] == 1
    assert [item["id"] for item in queue["items"]] == [
        "identity:char_a--char_b", "prompt:c003", "lint:canonical_alias_split:char_a"]
    assert queue["items"][0]["evidence"]["candidate"]["body_cos"] == 0.9
    assert queue["items"][-1]["status"] == "blocked"
    assert len(queue["items"][-1]["evidence"]["violations"]) == 2
    assert queue == build_review_queue(
        auto_review={"must_review": ["char_a"], "queue": [{"entity_id": "char_a", "score": 2}]},
        identity_candidates=[queue["items"][0]["evidence"]["candidate"]],
        qa_report=[{"chunk_id": 3, "findings": [{"code": "prompt_missing_present_entity", "entity_id": "char_b"}]}],
        strict_lint=queue["items"][-1]["evidence"]["violations"])


def test_missing_sources_and_writer_are_idempotent() -> None:
    assert build_review_queue() == {"version": 1, "items": [], "summary": {
        "n_items": 0, "n_must_review": 0, "n_spot_check": 0,
        "by_kind": {"identity": 0, "state": 0, "prompt": 0, "lint": 0},
        "by_status": {"needs_review": 0, "blocked": 0}}}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "gold").mkdir()
        (root / "gold" / "entity_registry.json").write_text(json.dumps({
            "schema_version": "2.0.0", "movie_id": "m", "human_reviewed": False, "annotation_provenance": {}, "entities": []}))
        (root / "gold" / "chunk_annotations.json").write_text(json.dumps({
            "schema_version": "2.0.0", "movie_id": "m", "human_reviewed": False, "chunks": []}))
        first = write_review_queue(root)
        second = write_review_queue(root)
        assert first == second
        assert json.loads((root / "tmp" / "review_queue.json").read_text()) == first


def test_queue_groups_alias_components_and_state_events() -> None:
    queue = build_review_queue(
        identity_candidates=[
            {"left": "char_a", "right": "char_b", "left_chunk_span": [0, 0], "right_chunk_span": [1, 1]},
            {"left": "char_b", "right": "char_c", "left_chunk_span": [1, 1], "right_chunk_span": [2, 2]},
        ],
        state_events=[{"event_id": "evt_1", "entity_id": "prop_apple", "chunk_id": 2,
                       "last_chunk_id": 4, "description": "apple eaten", "deprecates": ["r0"]}],
    )
    identity = next(item for item in queue["items"] if item["kind"] == "identity")
    state = next(item for item in queue["items"] if item["kind"] == "state")
    assert identity["entity_ids"] == ["char_a", "char_b", "char_c"]
    assert len(identity["evidence"]["candidates"]) == 2
    assert state["affected_chunk_ids"] == [2, 3, 4]


def test_clean_chunks_produce_no_prompt_items() -> None:
    # A chunk with an empty findings list is clean; 64 clean chunks must yield ZERO review cards.
    queue = build_review_queue(
        qa_report=[{"chunk_id": i, "rounds": 0, "flagged": False, "findings": []}
                   for i in range(64)])
    assert queue["summary"]["n_items"] == 0


def test_state_events_grouped_per_entity() -> None:
    events = [{"event_id": f"evt_{i}", "entity_id": "char_bunny", "chunk_id": i,
               "last_chunk_id": i, "description": f"event {i}"} for i in range(5)]
    events.append({"event_id": "evt_x", "entity_id": "prop_apple", "chunk_id": 9,
                   "last_chunk_id": 9, "description": "apple eaten"})
    queue = build_review_queue(state_events=events)
    state_items = [item for item in queue["items"] if item["kind"] == "state"]
    assert len(state_items) == 2  # one card per entity, not per event
    bunny = next(item for item in state_items if item["entity_ids"] == ["char_bunny"])
    assert len(bunny["evidence"]["events"]) == 5
    assert bunny["id"] == "state:char_bunny"


def test_reversible_state_events_filtered_and_kind_lexicon() -> None:
    from vmem_bench.annotation.pipeline_track_first.drafting import split_reversible_events
    keep, filtered = split_reversible_events([
        {"entity_id": "a", "description": "The apple is eaten and destroyed"},
        {"entity_id": "b", "description": "The bird is no longer visible in the frame"},
        {"entity_id": "c", "description": "The camera pans to reveal the meadow"},
        {"entity_id": "d", "description": "The stick is snapped in half"},
    ])
    assert [e["entity_id"] for e in keep] == ["a", "d"]
    assert [e["entity_id"] for e in filtered] == ["b", "c"]

    from vmem_bench.annotation.pipeline_track_first.auto_review import kind_mixture_reasons
    from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
    from vmem_bench.common.schemas import Entity
    reg = Registry()
    reg.entities = {
        "prop_grass_field": Entity(entity_id="prop_grass_field", kind="prop",
                                   name="Grass Field", description="", first_chunk=0),
        "loc_sky_glider": Entity(entity_id="loc_sky_glider", kind="location",
                                 name="Sky Glider", description="", first_chunk=0),
        "char_rabbit": Entity(entity_id="char_rabbit", kind="character",
                              name="Rabbit", description="", first_chunk=0),
        "prop_field_with_rabbit": Entity(entity_id="prop_field_with_rabbit", kind="prop",
                                         name="Field with Rabbit", description="", first_chunk=0),
    }
    flags = kind_mixture_reasons(reg)
    assert any("prop named like a location" in f for f in flags["prop_grass_field"])
    assert any("location named like an object" in f for f in flags["loc_sky_glider"])
    assert any("name_mentions_character_word" in f for f in flags["prop_field_with_rabbit"])
    assert "char_rabbit" not in flags


def test_queue_tiering_caps_must_review_but_never_demotes_identity() -> None:
    events = [{"event_id": f"e{i}", "entity_id": f"prop_x{i}", "chunk_id": i,
               "last_chunk_id": i, "description": f"d{i}"} for i in range(20)]
    queue = build_review_queue(
        identity_candidates=[{"left": "char_a", "right": "char_b",
                              "left_chunk_span": [0, 5], "right_chunk_span": [6, 9]}],
        state_events=events, must_review_limit=5)
    assert queue["summary"]["n_must_review"] == 5
    assert queue["summary"]["n_spot_check"] == queue["summary"]["n_items"] - 5
    tiers = {item["id"]: item["review_tier"] for item in queue["items"]}
    assert tiers["identity:char_a--char_b"] == "must"  # identity always must
    assert queue["items"][0]["review_tier"] == "must"  # must cards sort first


def test_identity_resolution_findings_become_identity_cards() -> None:
    identity_resolution = {
        "group_entity_id": {"0": "char_rabbit"},
        "entity_id_by_observation": {"1": "char_rabbit", "2": "char_rabbit"},
        "observations": [{"index": 1, "name": "Rabbit", "kind": "character"},
                         {"index": 2, "name": "Rabbit", "kind": "character"}],
        "findings": [
            {"code": "roster_incomplete_unmatched_cluster", "group_index": 0,
             "members": [1, 2], "n_observations": 2, "name": "Rabbit", "kind": "character"},
            {"code": "cluster_merged_by_vlm", "kind": "character", "cluster_ids": [3, 4]},
            {"code": "unknown_future_code", "kind": "character"},  # must be silently skipped
        ],
    }
    queue = build_review_queue(identity_resolution=identity_resolution)
    ids = [item["id"] for item in queue["items"]]
    assert any(i.startswith("identity:cluster:roster_incomplete_unmatched_cluster:") for i in ids)
    assert any(i.startswith("identity:cluster:cluster_merged_by_vlm:") for i in ids)
    assert not any("unknown_future_code" in i for i in ids)
    roster_gap = next(item for item in queue["items"]
                      if item["id"].startswith("identity:cluster:roster_incomplete"))
    assert roster_gap["kind"] == "identity"
    assert roster_gap["entity_ids"] == ["char_rabbit"]
    assert roster_gap["review_tier"] == "must"       # error/gap findings are always must
    assert "Rabbit" in roster_gap["question"]
    merged = next(item for item in queue["items"]
                 if item["id"].startswith("identity:cluster:cluster_merged_by_vlm"))
    assert merged["recommended_action"] == "spot_check_cluster_merge"


def test_identity_resolution_confident_merge_is_not_forced_must() -> None:
    # A confident VLM merge confirmation (always_must=False) must NOT auto-force "must" once the
    # ordinary must_review_limit budget is already exhausted by higher-priority items.
    events = [{"event_id": f"e{i}", "entity_id": f"prop_x{i}", "chunk_id": i,
              "last_chunk_id": i, "description": f"d{i}"} for i in range(5)]
    identity_resolution = {"findings": [
        {"code": "cluster_merged_by_vlm", "kind": "prop", "cluster_ids": [0, 1]}]}
    queue = build_review_queue(state_events=events, identity_resolution=identity_resolution,
                               must_review_limit=5)
    merged = next(item for item in queue["items"]
                 if item["id"].startswith("identity:cluster:cluster_merged_by_vlm"))
    assert merged["review_tier"] == "spot_check"


def test_identity_resolution_missing_or_empty_is_a_no_op() -> None:
    assert build_review_queue(identity_resolution=None)["items"] == []
    assert build_review_queue(identity_resolution={})["items"] == []
    assert build_review_queue(identity_resolution={"findings": []})["items"] == []


def _run_all() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_review_queue: OK")


if __name__ == "__main__":
    _run_all()
