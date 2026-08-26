"""Deterministic (no-GPU, no-model) tests for the S7 freeze-layer state_events backfill.

Covers:
- ``drafting.filter_state_events`` relaxation: an explicit finite-ontology ``event_type`` (the
  VLM's ``state_change_kind``) is trusted when the description cannot be classified by the
  English regex (e.g. a Chinese corpus), while a genuine cross-ontology conflict is still rejected
  and reversible/camera prose is still dropped.
- ``s7_freeze_publish.backfill.materialize_state_events``: seconds -> chunk mapping and
  ``deprecates`` scoping to the real frozen crop ids.
- ``backfill_state_events``: end-to-end write into a fake frozen movie dir, forbidden-table
  consistency, and the idempotency guard.
"""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.drafting import filter_state_events
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.backfill import (
    backfill_state_events, materialize_state_events,
)


def test_filter_trusts_explicit_kind_when_prose_unclassifiable() -> None:
    # Chinese descriptions the English regex cannot classify; explicit kind is authoritative.
    kept, rejected = filter_state_events([
        {"entity_id": "prop_apple", "event_type": "consumed",
         "description": "consumed: 被大兔子咬食"},
        {"entity_id": "char_bunny", "event_type": "appearance_changed",
         "description": "appearance_changed: 双眼周围涂抹了黑色的伪装颜料/泥土"},
    ])
    assert [e["entity_id"] for e in kept] == ["prop_apple", "char_bunny"]
    assert rejected == []


def test_filter_rejects_genuine_cross_ontology_conflict() -> None:
    # Explicit says "consumed" but the prose clearly corroborates a *different* ontology type.
    kept, rejected = filter_state_events([
        {"entity_id": "prop_apple", "event_type": "consumed",
         "description": "The apple is destroyed and burned away."},
    ])
    assert kept == []
    assert rejected[0]["rejection_reason"] == "event_type_description_mismatch"


def test_filter_still_drops_reversible_and_out_of_ontology() -> None:
    kept, rejected = filter_state_events([
        {"entity_id": "char_bunny", "event_type": "appearance_changed",
         "description": "appearance_changed: the bunny exits the frame"},   # reversible
        {"entity_id": "prop_x", "event_type": "relocated",
         "description": "relocated: 苹果被移动到桌上"},                        # kind not in ontology
    ])
    assert kept == []
    reasons = {r["rejection_reason"] for r in rejected}
    assert reasons == {"reversible_or_camera_only", "outside_finite_state_event_ontology"}


def _fake_movie(tmp: Path) -> Path:
    """Minimal frozen movie dir: one prop with 3 crops + a consumed@later state_change."""
    movie = tmp / "movie"
    gold = movie / "gold"
    gold.mkdir(parents=True)
    (movie / "vlm_output.json").write_text(json.dumps({
        "props": [{"prop_id": "prop_apple", "name": "Apple", "state_changes": [
            {"seconds": 30.0, "state_change_kind": "consumed", "description": "被吃掉"},
        ]}],
    }, ensure_ascii=False), encoding="utf-8")
    (gold / "chunk_index.json").write_text(json.dumps({
        "chunks": [
            {"chunk_id": 0, "seconds_span": [0.0, 10.0]},
            {"chunk_id": 1, "seconds_span": [10.0, 20.0]},
            {"chunk_id": 2, "seconds_span": [20.0, 40.0]},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    (gold / "entity_registry.json").write_text(json.dumps({
        "movie_id": "m", "human_reviewed": True, "entities": [
            {"entity_id": "prop_apple", "kind": "prop", "name": "Apple", "first_chunk": 0,
             "state_events": [], "representations": [
                 {"representation_id": "prop_apple@c000", "chunk_id": 0, "crop_path": "a0.jpg"},
                 {"representation_id": "prop_apple@c001", "chunk_id": 1, "crop_path": "a1.jpg"},
                 {"representation_id": "prop_apple@c002", "chunk_id": 2, "crop_path": "a2.jpg"},
             ]},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    (gold / "chunk_annotations.json").write_text(json.dumps({
        "movie_id": "m", "human_reviewed": True, "chunks": [
            {"chunk_id": c, "present": ["prop_apple"], "first_appearances": [],
             "forbidden": [], "scenario_tags": ["none"]} for c in (0, 1, 2)
        ],
    }, ensure_ascii=False), encoding="utf-8")
    (gold / "observations.jsonl").write_text("\n".join(
        json.dumps({"chunk_id": c, "chunk_video": f"chunks/chunk_{c:03d}.mp4",
                    "observations": [], "state_events": []}) for c in (0, 1, 2)
    ) + "\n", encoding="utf-8")
    return movie


def test_materialize_maps_seconds_to_chunk_and_scopes_deprecates(tmp_path: Path) -> None:
    movie = _fake_movie(tmp_path)
    vlm = json.loads((movie / "vlm_output.json").read_text())
    er = json.loads((movie / "gold" / "entity_registry.json").read_text())
    ci = json.loads((movie / "gold" / "chunk_index.json").read_text())
    events, rejected = materialize_state_events(vlm, er["entities"], ci["chunks"])
    assert rejected == []
    ev = events["prop_apple"][0]
    assert ev["chunk_id"] == 2                       # 30.0s falls in chunk 2 ([20, 40))
    # deprecates = reps with chunk_id <= 2 -> all three crops.
    assert ev["deprecates"] == ["prop_apple@c000", "prop_apple@c001", "prop_apple@c002"]


def test_backfill_writes_registry_forbidden_and_is_idempotent(tmp_path: Path) -> None:
    movie = _fake_movie(tmp_path)
    summary = backfill_state_events(movie)
    assert summary["n_events"] == 1

    er = json.loads((movie / "gold" / "entity_registry.json").read_text())
    events = er["entities"][0]["state_events"]
    assert len(events) == 1 and events[0]["chunk_id"] == 2

    anno = json.loads((movie / "gold" / "chunk_annotations.json").read_text())
    # forbidden fires only on chunks strictly after the event chunk (event.chunk_id < chunk_id);
    # this movie has no chunk > 2, so forbidden stays empty everywhere but the state-change tag lands.
    tags_by_chunk = {c["chunk_id"]: c["scenario_tags"] for c in anno["chunks"]}
    assert "state-change" in tags_by_chunk[2]
    assert all(c["forbidden"] == [] for c in anno["chunks"])

    # Idempotency guard: a second run without overwrite is a no-op.
    again = backfill_state_events(movie)
    assert again["status"].startswith("already_has_state_events")


def test_backfill_forbidden_fires_on_later_chunk(tmp_path: Path) -> None:
    movie = _fake_movie(tmp_path)
    # Move the event earlier (chunk 0) so later chunks carry a non-empty forbidden table.
    vlm_path = movie / "vlm_output.json"
    vlm = json.loads(vlm_path.read_text())
    vlm["props"][0]["state_changes"][0]["seconds"] = 5.0     # -> chunk 0
    vlm_path.write_text(json.dumps(vlm, ensure_ascii=False), encoding="utf-8")

    backfill_state_events(movie)
    anno = json.loads((movie / "gold" / "chunk_annotations.json").read_text())
    fb = {c["chunk_id"]: [f["representation_id"] for f in c["forbidden"]] for c in anno["chunks"]}
    assert fb[0] == []                                       # event chunk itself is not forbidden
    assert fb[1] == ["prop_apple@c000"]                      # only reps with chunk_id <= 0 deprecated
    assert fb[2] == ["prop_apple@c000"]


def test_build_gold_inlines_state_events_from_state_changes(tmp_path: Path) -> None:
    """S7 ``build_gold`` materializes state_events at freeze time (same logic as the backfill),
    so newly frozen gold activates Avoidance without a separate backfill pass."""
    from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.build_gold import build_gold

    src = tmp_path / "src"
    src.mkdir()
    crops = []
    for i in range(3):
        p = src / f"c{i}.jpg"
        p.write_bytes(f"crop-{i}".encode())
        crops.append({
            "chunk_id": i, "entity_id": "prop_apple", "kind": "prop",
            "representation_id": f"prop_apple@c{i:05d}", "crop_path": str(p),
            "frame_index": i, "bbox_norm": [0, 0, 100, 100],
        })
    annotation = {
        "characters": [], "locations": [],
        "props": [{"prop_id": "prop_apple", "name": "Apple", "description": "red apple",
                   "state_changes": [
                       {"seconds": 5.0, "state_change_kind": "consumed", "description": "被吃掉"}]}],
        "screenplay": {"scenes": [{"visual_segments": [
            {"segment_id": f"seg_{i}", "start_seconds": float(i * 10),
             "end_seconds": float(i * 10 + 10), "action": f"apple {i}",
             "present_entity_ids": ["prop_apple"]} for i in range(3)
        ]}]},
    }
    movie = tmp_path / "movie"
    gold = build_gold(movie_dir=movie, annotation=annotation, accepted_crops=crops,
                      automation_smoke=True)

    er = json.loads((gold / "entity_registry.json").read_text())
    apple = [e for e in er["entities"] if e["entity_id"] == "prop_apple"][0]
    assert len(apple["state_events"]) == 1
    ev = apple["state_events"][0]
    assert ev["chunk_id"] == 0 and ev["deprecates"] == ["prop_apple@c00000"]

    chunks = json.loads((gold / "chunk_annotations.json").read_text())["chunks"]
    fb = {c["chunk_id"]: [f["representation_id"] for f in c["forbidden"]] for c in chunks}
    assert fb[0] == []                                       # event chunk itself not forbidden
    assert fb[1] == ["prop_apple@c00000"] and fb[2] == ["prop_apple@c00000"]
    assert "state-change" in {t for c in chunks if c["chunk_id"] == 0 for t in c["scenario_tags"]}


if __name__ == "__main__":
    test_filter_trusts_explicit_kind_when_prose_unclassifiable()
    test_filter_rejects_genuine_cross_ontology_conflict()
    test_filter_still_drops_reversible_and_out_of_ontology()
    test_materialize_maps_seconds_to_chunk_and_scopes_deprecates(Path("/tmp/_bbf1"))
    print("ok")
