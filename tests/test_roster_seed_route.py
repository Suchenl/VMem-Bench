"""Deterministic tests for the human-seeded production annotation route."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vmem_bench.annotation.pipeline_track_first.drafting import (
    filter_state_events,
)
from vmem_bench.annotation.pipeline_track_first.review_queue import build_review_queue
from vmem_bench.annotation.pipeline_track_first.roster_seed import (
    CanonicalRosterSeed, assign_closed_set, load_roster_seed)
from vmem_bench.common.gold_lint import lint_annotations
from vmem_bench.common.schemas import ChunkAnnotations, EntityRegistry


def _seed(tmp_path: Path, *, confirmed: bool = True) -> Path:
    exemplar = tmp_path / "bunny.jpg"
    exemplar.write_bytes(b"jpeg")
    path = tmp_path / "roster_seed.json"
    path.write_text(json.dumps({
        "version": 1,
        "movie_id": "movie",
        "human_confirmed": confirmed,
        "entities": [{
            "entity_id": "char_bunny",
            "name": "Bunny",
            "kind": "character",
            "identity_scope": "individual",
            "description": "Large white rabbit with long ears.",
            "grounding_phrases": ["large white rabbit"],
            "aliases": ["rabbit"],
            "exemplar_crops": ["bunny.jpg"],
            "static_attributes": {"species": "rabbit", "primary_color": "white"},
            "allowed_state_events": ["appearance_changed"],
        }, {
            "entity_id": "prop_apples",
            "name": "Apples",
            "kind": "prop",
            "identity_scope": "category",
            "description": "Red apples growing on the tree.",
            "grounding_phrases": ["red apples"],
            "aliases": ["fruit"],
            "exemplar_crops": [],
            "static_attributes": {"object_type": "fruit"},
            "allowed_state_events": [],
        }],
    }), encoding="utf-8")
    return path


def test_load_human_confirmed_seed_resolves_exemplars(tmp_path: Path) -> None:
    seed = load_roster_seed(_seed(tmp_path), expected_movie_id="movie")
    assert seed.human_confirmed is True
    assert seed.entities[0].entity_id == "char_bunny"
    assert Path(seed.entities[0].exemplar_crops[0]).is_absolute()
    assert seed.to_roster()[0]["grounding_phrase"] == "large white rabbit"


def test_production_seed_rejects_unconfirmed_and_movie_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="human_confirmed"):
        load_roster_seed(_seed(tmp_path, confirmed=False), expected_movie_id="movie")
    with pytest.raises(ValueError, match="does not match"):
        load_roster_seed(_seed(tmp_path), expected_movie_id="other")


def test_seed_rejects_lifecycle_events_for_category(tmp_path: Path) -> None:
    path = _seed(tmp_path)
    data = json.loads(path.read_text())
    data["entities"][1]["allowed_state_events"] = ["consumed"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only identity_scope='individual'"):
        load_roster_seed(path)


def test_closed_set_assignment_matches_exemplar_or_rejects_ambiguity(tmp_path: Path) -> None:
    seed = load_roster_seed(_seed(tmp_path), expected_movie_id="movie")
    matched = assign_closed_set(
        [1.0, 0.0], kind="character", phrase_owner_id="char_bunny", seed=seed,
        exemplar_embeddings={"char_bunny": [[0.99, 0.01]]},
        min_similarity=0.3, min_margin=0.04)
    assert matched.entity_id == "char_bunny" and matched.reason == "exemplar_match"
    other = replace(seed.entities[0], entity_id="char_other", name="Other")
    two_entity_seed = CanonicalRosterSeed(
        movie_id=seed.movie_id, human_confirmed=True,
        entities=(seed.entities[0], other), source_path=seed.source_path)
    rejected = assign_closed_set(
        [0.5, 0.5], kind="character", phrase_owner_id="char_bunny", seed=two_entity_seed,
        exemplar_embeddings={"char_bunny": [[1.0, 0.0]], "char_other": [[0.0, 1.0]]},
        min_similarity=0.3, min_margin=0.04)
    assert rejected.entity_id is None and rejected.reason == "ambiguous_margin"


def test_finite_state_event_ontology_and_entity_policy() -> None:
    kept, rejected = filter_state_events([
        {"entity_id": "char_bunny", "event_type": "appearance_changed",
         "description": "Bunny is permanently changed and scarred."},
        {"entity_id": "char_bunny", "event_type": "appearance_changed",
         "description": "Bunny becomes visible as the camera pans."},
        {"entity_id": "prop_apples", "event_type": "consumed",
         "description": "The apple is eaten."},
    ], allowed_by_entity={"char_bunny": {"appearance_changed"}, "prop_apples": set()})
    assert [event["entity_id"] for event in kept] == ["char_bunny"]
    assert {event["rejection_reason"] for event in rejected} == {
        "reversible_or_camera_only", "event_type_not_allowed_for_entity"}


def test_seeded_review_queue_is_one_card_per_canonical_entity() -> None:
    queue = build_review_queue(identity_resolution={
        "mode": "seeded",
        "canonical_entities": [
            {"entity_id": "char_bunny", "name": "Bunny", "kind": "character", "seen": True},
            {"entity_id": "prop_branch", "name": "Branch", "kind": "prop", "seen": False},
            {"entity_id": "loc_meadow", "name": "Meadow", "kind": "location", "seen": True},
        ],
        "findings": [{"code": "seed_entity_missing_evidence",
                      "entity_id": "prop_branch", "name": "Branch", "kind": "prop"}],
    }, surviving_ids={"char_bunny"})
    identity = [item for item in queue["items"] if item["kind"] == "identity"]
    assert [item["entity_ids"] for item in identity] == [["prop_branch"], ["char_bunny"]]
    assert identity[0]["review_tier"] == "must"
    assert identity[1]["review_tier"] == "spot_check"


def test_proposal_roster_cannot_pass_strict_freeze_lint() -> None:
    registry = EntityRegistry(
        movie_id="movie", entities=[],
        annotation_provenance={"roster_mode": "proposal", "production_mode": False})
    chunks = ChunkAnnotations(movie_id="movie", chunks=[])
    violations = lint_annotations(registry, chunks, strict_review=True)
    assert any(v.code == "unconfirmed_roster" and v.severity == "error" for v in violations)


def test_blocking_qa_findings_prevent_strict_freeze() -> None:
    registry = EntityRegistry(
        movie_id="movie", entities=[],
        annotation_provenance={"roster_mode": "seeded", "production_mode": True})
    chunks = ChunkAnnotations(movie_id="movie", chunks=[])
    violations = lint_annotations(
        registry, chunks, strict_review=True,
        qa_report=[{"chunk_id": 0, "flagged": True,
                    "findings": [{"code": "unknown_track_rejected"}]}])
    assert any(v.code == "flagged_chunks" and v.severity == "error" for v in violations)
