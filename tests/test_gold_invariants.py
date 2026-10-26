"""Deterministic MemStrata gold/checkpoint quality gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.review import apply_patch, freeze, preview_patch
from vmem_bench.common.gold_lint import lint_annotations
from vmem_bench.common.schemas import (
    ChunkAnnotation, ChunkAnnotations, Entity, EntityRegistry, GoldInstruction,
    Representation, StateEvent,
)
from vmem_bench.publish import _check_frozen


def _entity(eid: str, *, name: str = "Bunny", kind: str = "character",
            chunk: int = 0, bbox: list[int] | None = None,
            crop_path: str | None = None) -> Entity:
    return Entity(entity_id=eid, kind=kind, name=name, description=name, first_chunk=chunk,
                  representations=[Representation(
                      representation_id=f"{eid}@c{chunk:03d}", chunk_id=chunk,
                      crop_path=crop_path or f"assets/{eid}/c{chunk:03d}.jpg",
                      bbox=bbox or [100, 100, 300, 300],
                      bbox_source="grounding_dino" if kind != "location" else "full_frame",
                      embedding_key=f"{eid}@c{chunk:03d}")])


def _chunk(present: list[str], *, prompt: str = "Bunny appears.") -> ChunkAnnotation:
    first = list(present)
    return ChunkAnnotation(
        chunk_id=0, shot_span=[0, 0], frame_span=[0, 10], prompt=prompt,
        present=present, first_appearances=first,
        gold_instructions=[GoldInstruction(entity_id=e, requirement="introduce") for e in first])


def test_lint_rejects_present_and_alias_pathologies() -> None:
    registry = EntityRegistry(movie_id="m", entities=[
        _entity("char_bunny", name="Bunny"),
        _entity("char_bunny_character", name="Bunny (character)"),
    ])
    chunks = ChunkAnnotations(movie_id="m", chunks=[
        _chunk(["char_bunny", "char_bunny", "missing"],
               prompt="The scene continues in this location (chunk 0).")
    ])
    codes = {v.code for v in lint_annotations(registry, chunks, strict_review=True)}
    assert {"duplicate_present", "unknown_present_entity", "suffix_alias_entity",
            "canonical_alias_split", "placeholder_prompt"}.issubset(codes)


def test_lint_detects_same_chunk_bbox_conflict() -> None:
    bbox = [100, 100, 500, 500]
    registry = EntityRegistry(movie_id="m", entities=[
        _entity("char_squirrel", name="Squirrel", bbox=bbox),
        _entity("prop_acorn", name="Acorn", kind="prop", bbox=bbox),
    ])
    chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_squirrel", "prop_acorn"])])
    violations = lint_annotations(registry, chunks)
    assert any(v.code == "same_chunk_bbox_conflict" for v in violations)


def test_freeze_rejects_flagged_gold_without_review_patch() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gold = root / "gold"
        gold.mkdir()
        registry = EntityRegistry(movie_id="m", entities=[_entity("char_bunny")])
        chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_bunny"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        (root / "build").mkdir()
        (root / "build" / "annotation_qa.json").write_text(
            json.dumps([{"chunk_id": 0, "flagged": True}]), encoding="utf-8")
        try:
            freeze(gold)
            assert False, "flagged unreviewed gold should not freeze"
        except ValueError as exc:
            assert "flagged_chunks" in str(exc)


def test_publish_check_runs_gold_lint() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "layout").mkdir()
        gold = root / "gold"
        gold.mkdir()
        registry = EntityRegistry(movie_id="m", human_reviewed=True, entities=[
            _entity("char_bunny", name="Bunny"),
            _entity("char_bunny_character", name="Bunny (character)"),
        ])
        chunks = ChunkAnnotations(movie_id="m", human_reviewed=True,
                                  chunks=[_chunk(["char_bunny", "char_bunny_character"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        (root / "layout" / "chunk_index.json").write_text(
            json.dumps({"schema_version": registry.schema_version, "layout_hash": "lh",
                        "chunks": [{"chunk_id": 0}]}), encoding="utf-8")
        (root / "layout" / "boundaries.csv").write_text("shot,start,end\n", encoding="utf-8")
        problems = _check_frozen(root)
        assert any("gold lint" in p and "suffix_alias_entity" in p for p in problems)


def test_lint_accepts_dual_asset_crop_prefixes() -> None:
    """Both top-level assets/ and legacy derived/assets/ crop_paths pass without warning."""
    registry = EntityRegistry(movie_id="m", entities=[
        _entity("char_bunny", name="Bunny",
                crop_path="assets/char_bunny/c000.jpg"),
        _entity("prop_apple", name="Apple", kind="prop",
                crop_path="derived/assets/prop_apple/c000.jpg"),
    ])
    chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_bunny", "prop_apple"])])
    codes = {v.code for v in lint_annotations(registry, chunks)
             if v.code == "non_asset_crop_path"}
    assert codes == set()


def test_freeze_requires_dispositions_for_auto_review_must_review() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gold = root / "gold"
        gold.mkdir()
        registry = EntityRegistry(movie_id="m", entities=[_entity("char_bunny")])
        chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_bunny"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        tmp = root / "tmp"
        tmp.mkdir()
        (tmp / "auto_review.json").write_text(json.dumps({"must_review": ["char_bunny"]}), encoding="utf-8")
        try:
            freeze(root)
            assert False, "must_review without a disposition should not freeze"
        except ValueError as exc:
            assert "lack dispositions" in str(exc)
        (tmp / "review_dispositions.json").write_text(json.dumps({
            "char_bunny": {"action": "kept_distinct", "reason": "single reviewed entity"}}),
            encoding="utf-8")
        freeze(root)


def test_apply_patch_validates_and_persists_dispositions_after_gold_update() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gold = root / "gold"
        gold.mkdir()
        registry = EntityRegistry(movie_id="m", entities=[_entity("char_a"), _entity("char_b", chunk=1)])
        chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_a", "char_b"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        patch = root / "patch.json"
        patch.write_text(json.dumps({"merges": [["char_a", "char_b"]],
            "dispositions": {"char_b": {"action": "merged", "reason": "same crop"}}}), encoding="utf-8")
        apply_patch(root, patch)
        dispositions = json.loads((root / "tmp" / "review_dispositions.json").read_text(encoding="utf-8"))
        assert dispositions["char_b"]["action"] == "merged"
        bad = root / "bad.json"
        bad.write_text(json.dumps({"drops": ["char_a"],
            "dispositions": {"char_a": {"action": "dropped", "reason": ""}}}), encoding="utf-8")
        try:
            apply_patch(root, bad)
            assert False, "empty disposition reason must be rejected"
        except ValueError as exc:
            assert "reason" in str(exc)


def test_preview_patch_is_non_mutating_and_reports_lint() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gold = root / "gold"
        gold.mkdir()
        registry = EntityRegistry(movie_id="m", entities=[_entity("char_a"), _entity("char_b", chunk=1)])
        chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["char_a", "char_b"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        before = (gold / "entity_registry.json").read_text(encoding="utf-8")
        result = preview_patch(root, {"schema_version": "2.0.0", "merges": [["char_a", "char_b"]],
                                      "dispositions": {"char_b": {"action": "merged", "reason": "same"}}})
        assert "ok" in result and "errors" in result
        assert (gold / "entity_registry.json").read_text(encoding="utf-8") == before


def test_state_event_review_rejects_event_and_records_raw_human_pair() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); gold = root / "gold"; gold.mkdir()
        entity = _entity("prop_apple", kind="prop")
        entity.state_events = [StateEvent("evt_apple", 0, "apple eaten", ["prop_apple@c000"])]
        registry = EntityRegistry(movie_id="m", entities=[entity])
        chunks = ChunkAnnotations(movie_id="m", chunks=[_chunk(["prop_apple"])])
        (gold / "entity_registry.json").write_text(json.dumps(registry.to_dict()), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
        patch = root / "state.json"
        patch.write_text(json.dumps({"state_event_reviews": {"evt_apple": {
            "action": "rejected", "reason": "only a pose change"}}}), encoding="utf-8")
        apply_patch(root, patch)
        saved = json.loads((gold / "entity_registry.json").read_text(encoding="utf-8"))
        assert saved["entities"][0]["state_events"] == []
        assert (root / "tmp" / "state_event_review_pairs.jsonl").is_file()
