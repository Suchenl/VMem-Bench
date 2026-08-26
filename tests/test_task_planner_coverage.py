"""Tests for S5 coverage planner + ≤t slot expansion."""

from __future__ import annotations

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.coverage_expand import (
    expand_library_to_slots,
    prune_library_by_attributes,
    resolve_current_appearance,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CoverageCaps,
    derive_tasks,
    derive_tasks_coverage,
    plan_tasks,
)


def _annotation() -> dict:
    return {
        "characters": [
            {"char_id": "char_a", "name": "Alice", "description": "woman"},
            {"char_id": "char_b", "name": "Bob", "description": "man"},
        ],
        "props": [{"prop_id": "prop_bag", "name": "Bag", "description": "bag"}],
        "locations": [{"loc_id": "loc_home", "name": "Home", "description": "house"}],
        "screenplay": {
            "scenes": [
                {
                    "visual_segments": [
                        {
                            "segment_id": "s0",
                            "start_seconds": 0.0,
                            "end_seconds": 2.0,
                            "action": "Alice enters",
                            "present_entity_ids": ["char_a", "loc_home"],
                        },
                        {
                            "segment_id": "s1",
                            "start_seconds": 2.0,
                            "end_seconds": 4.0,
                            "action": "Alice talks",
                            "present_entity_ids": ["char_a", "loc_home"],
                        },
                        {
                            "segment_id": "s2",
                            "start_seconds": 4.0,
                            "end_seconds": 6.0,
                            "action": "empty beat",
                            "present_entity_ids": ["loc_home"],
                        },
                        {
                            "segment_id": "s3",
                            "start_seconds": 6.0,
                            "end_seconds": 8.0,
                            "action": "Alice returns with bag",
                            "present_entity_ids": ["char_a", "prop_bag", "loc_home"],
                        },
                        {
                            "segment_id": "s4",
                            "start_seconds": 8.0,
                            "end_seconds": 10.0,
                            "action": "Bob arrives",
                            "present_entity_ids": ["char_a", "char_b", "loc_home"],
                        },
                        {
                            "segment_id": "s5",
                            "start_seconds": 10.0,
                            "end_seconds": 12.0,
                            "action": "group",
                            "present_entity_ids": ["char_a", "char_b", "prop_bag", "loc_home"],
                        },
                    ]
                }
            ]
        },
    }


def test_coverage_much_smaller_than_per_slot() -> None:
    ann = _annotation()
    per_slot = derive_tasks(ann)
    coverage, stats = derive_tasks_coverage(ann, caps=CoverageCaps(character=3, prop=2, location=2))
    assert stats["mode"] == "coverage"
    assert len(coverage) < len(per_slot)
    assert stats["n_per_slot_would_be"] == len(per_slot)
    # First appearance always kept.
    by_entity = {}
    for task in coverage:
        by_entity.setdefault(task.entity_id, []).append(task)
    assert by_entity["char_a"][0].chunk_id == 0
    assert by_entity["char_a"][0].reason == "first_appearance"
    # Reappearance after absence (chunk 2 missing Alice) → chunk 3 selected.
    assert any(t.chunk_id == 3 and t.reason == "reappearance" for t in by_entity["char_a"])
    assert all(task.task_kind == "acquire" for task in coverage)


def test_plan_tasks_default_coverage() -> None:
    tasks, stats = plan_tasks(_annotation())
    assert stats["mode"] == "coverage"
    assert tasks


def test_caps_respected() -> None:
    ann = _annotation()
    coverage, _ = derive_tasks_coverage(ann, caps=CoverageCaps(character=1, prop=1, location=1))
    counts: dict[str, int] = {}
    for task in coverage:
        counts[task.entity_id] = counts.get(task.entity_id, 0) + 1
    assert counts["char_a"] == 1
    assert counts["loc_home"] == 1


def test_resolve_current_appearance_le_t() -> None:
    library = [
        {"chunk_id": 0, "crop_path": "/a0.png", "crop_attributes": {"state_angle": "default"}},
        {"chunk_id": 3, "crop_path": "/a3.png", "crop_attributes": {"state_angle": "changed"}},
    ]
    hit = resolve_current_appearance(library, chunk_id=2, state="default")
    assert hit is not None and hit["chunk_id"] == 0
    hit2 = resolve_current_appearance(library, chunk_id=4, state="changed")
    assert hit2 is not None and hit2["chunk_id"] == 3


def test_expand_library_fills_slots() -> None:
    ann = _annotation()
    library = [
        {
            "chunk_id": 0,
            "entity_id": "char_a",
            "kind": "character",
            "name": "Alice",
            "description": "woman",
            "accepted": True,
            "crop_path": "/tmp/a0.png",
            "representation_id": "char_a@c00000",
            "task_kind": "acquire",
            "crop_attributes": {"state_angle": "default"},
            "bbox_norm": [1, 2, 3, 4],
            "frame_index": 10,
        },
        {
            "chunk_id": 0,
            "entity_id": "loc_home",
            "kind": "location",
            "name": "Home",
            "description": "house",
            "accepted": True,
            "crop_path": "/tmp/l0.png",
            "representation_id": "loc_home@c00000",
            "task_kind": "acquire",
        },
        {
            "chunk_id": 3,
            "entity_id": "prop_bag",
            "kind": "prop",
            "name": "Bag",
            "description": "bag",
            "accepted": True,
            "crop_path": "/tmp/p3.png",
            "representation_id": "prop_bag@c00003",
            "task_kind": "acquire",
        },
        {
            "chunk_id": 4,
            "entity_id": "char_b",
            "kind": "character",
            "name": "Bob",
            "description": "man",
            "accepted": True,
            "crop_path": "/tmp/b4.png",
            "representation_id": "char_b@c00004",
            "task_kind": "acquire",
        },
    ]
    expanded = expand_library_to_slots(annotation=ann, library_proposals=library)
    # Alice present in chunks 0,1,3,4,5 → binds for 1,3,4,5 from ≤t library
    alice = [p for p in expanded if p["entity_id"] == "char_a"]
    assert len(alice) == 5
    assert alice[0]["task_kind"] == "acquire"
    assert alice[1]["task_kind"] == "slot_bind"
    assert alice[1]["bind_source_chunk_id"] == 0
    assert alice[1]["crop_path"] == "/tmp/a0.png"
    assert alice[1]["accepted"] is True


def test_prune_library_by_attributes() -> None:
    props = [
        {
            "entity_id": "char_a",
            "kind": "character",
            "accepted": True,
            "crop_path": f"/tmp/{i}.png",
            "crop_attributes": {
                "spatial_angle": "front" if i % 2 == 0 else "side",
                "state_angle": "default",
                "shot_size": "close_up",
                "lighting": "day",
            },
            "qa": {"sharpness": float(i)},
        }
        for i in range(6)
    ]
    kept = prune_library_by_attributes(props, caps=CoverageCaps(character=2))
    accepted = [p for p in kept if p.get("accepted")]
    assert len(accepted) == 2
