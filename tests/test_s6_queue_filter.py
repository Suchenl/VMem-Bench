"""S6 queue filtering and decision propagation."""

from __future__ import annotations

from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.queue import (
    build_review_queue,
)
from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.review_apply import (
    expand_decisions_to_siblings,
)


def test_build_review_queue_hides_rejects_and_dedupes_same_path() -> None:
    proposals = [
        {
            "representation_id": "char_001@c00001",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 1,
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "representation_id": "char_001@c00002",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 2,
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "slot_bind",
            "bind_source_representation_id": "char_001@c00001",
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "representation_id": "char_001@c00003",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 10,
            "crop_path": "crops/b.jpg",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [500, 500, 900, 900],
        },
        {
            "representation_id": "char_001@c00004",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 4,
            "crop_path": "crops/bad.jpg",
            "accepted": False,
            "reason": "picker_rejected",
            "task_kind": "acquire",
        },
        {
            "representation_id": "char_001@c00005",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 5,
            "crop_path": None,
            "accepted": False,
            "reason": "no_library_crop_for_slot",
            "task_kind": "slot_bind",
        },
    ]
    queue = build_review_queue(proposals)
    ids = {item["proposal"]["representation_id"] for item in queue}
    assert ids == {"char_001@c00001", "char_001@c00003"}
    assert all(item["recommended_action"] == "keep" for item in queue)
    assert all(item["proposal"].get("task_kind") != "slot_bind" for item in queue)


def test_build_review_queue_dedupes_abs_rel_and_prefers_canonical() -> None:
    """human_add clones of an existing leaf must not reappear on the board."""
    proposals = [
        {
            "representation_id": "loc_001@c00003",
            "entity_id": "loc_001",
            "kind": "location",
            "chunk_id": 3,
            "crop_path": "/abs/candidates/location/loc_001/c00003_00001313.png",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "representation_id": "loc_001@human_add_0005",
            "entity_id": "loc_001",
            "kind": "location",
            "chunk_id": 26,
            "crop_path": "data/x/candidates/location/loc_001/c00003_00001313.png",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "representation_id": "loc_001@human_add_0003",
            "entity_id": "loc_001",
            "kind": "location",
            "chunk_id": 26,
            "crop_path": "data/x/candidates/location/loc_001/c00001_00000516.png",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [400, 400, 800, 800],
        },
        {
            "representation_id": "loc_001@c00001",
            "entity_id": "loc_001",
            "kind": "location",
            "chunk_id": 1,
            "crop_path": "/abs/candidates/location/loc_001/c00001_00000516.png",
            "accepted": True,
            "task_kind": "acquire",
            "bbox_norm": [400, 400, 800, 800],
        },
    ]
    queue = build_review_queue(proposals)
    ids = {item["proposal"]["representation_id"] for item in queue}
    assert ids == {"loc_001@c00001", "loc_001@c00003"}
    assert not any("@human_add_" in rid for rid in ids)


def test_build_review_queue_keeps_all_distinct_acquires() -> None:
    """Review board must not hard-cap; only path / near-identical bbox collapse."""
    proposals = []
    for i in range(12):
        proposals.append(
            {
                "representation_id": f"char_001@c{i:05d}",
                "entity_id": "char_001",
                "kind": "character",
                "chunk_id": i,
                "crop_path": f"crops/{i}.jpg",
                "accepted": True,
                "task_kind": "acquire",
                # Spread boxes so IoU stays below near-identical threshold.
                "bbox_norm": [i * 80, 0, 40 + i * 80, 40],
            }
        )
    queue = build_review_queue(proposals)
    assert len(queue) == 12
    assert all(item["proposal"].get("task_kind") == "acquire" for item in queue)


def test_expand_decisions_propagates_by_crop_path() -> None:
    proposals = [
        {
            "representation_id": "char_001@c00001",
            "entity_id": "char_001",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "acquire",
        },
        {
            "representation_id": "char_001@c00002",
            "entity_id": "char_001",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "slot_bind",
            "bind_source_representation_id": "char_001@c00001",
        },
        {
            "representation_id": "char_001@c00003",
            "entity_id": "char_001",
            "crop_path": "crops/b.jpg",
            "accepted": True,
            "task_kind": "acquire",
        },
    ]
    expanded = expand_decisions_to_siblings(
        proposals=proposals,
        decisions={"char_001@c00001": {"action": "keep", "reason": ""}},
    )
    assert expanded["char_001@c00001"]["action"] == "keep"
    assert expanded["char_001@c00002"]["action"] == "keep"
    assert "char_001@c00003" not in expanded


def test_apply_crop_reassign_updates_entity_fields() -> None:
    from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.patch import (
        apply_crop_decisions,
    )

    proposals = [
        {
            "representation_id": "alice@c00001",
            "entity_id": "alice",
            "name": "Alice",
            "kind": "character",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "acquire",
        }
    ]
    out = apply_crop_decisions(
        proposals=proposals,
        decisions={
            "alice@c00001": {
                "action": "reassign",
                "entity_id": "bob",
                "name": "Bob",
                "kind": "character",
                "from_entity_id": "alice",
                "reason": "human_reassigned",
            }
        },
    )
    assert len(out) == 1
    item = out[0]
    assert item["representation_id"] == "alice@c00001"
    assert item["entity_id"] == "bob"
    assert item["name"] == "Bob"
    assert item["reassigned_from_entity_id"] == "alice"
    assert item["accepted"] is True
    assert item["review_reason"] == "human_reassigned"


def test_expand_reassign_stays_on_source_entity() -> None:
    proposals = [
        {
            "representation_id": "alice@c00001",
            "entity_id": "alice",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "acquire",
        },
        {
            "representation_id": "alice@c00002",
            "entity_id": "alice",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "slot_bind",
            "bind_source_representation_id": "alice@c00001",
        },
        {
            "representation_id": "bob@c00009",
            "entity_id": "bob",
            "crop_path": "crops/a.jpg",
            "accepted": True,
            "task_kind": "acquire",
        },
    ]
    expanded = expand_decisions_to_siblings(
        proposals=proposals,
        decisions={
            "alice@c00001": {
                "action": "reassign",
                "entity_id": "carol",
                "name": "Carol",
                "kind": "character",
                "from_entity_id": "alice",
            }
        },
    )
    assert expanded["alice@c00001"]["action"] == "reassign"
    assert expanded["alice@c00002"]["action"] == "reassign"
    assert expanded["alice@c00002"]["entity_id"] == "carol"
    assert "bob@c00009" not in expanded
