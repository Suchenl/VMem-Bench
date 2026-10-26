"""Tests for the S5 per-entity identity-consistency gate."""

from __future__ import annotations

from typing import Any

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.exemplar_identity import (
    exclusive_assign_candidates,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.identity_consistency import (
    IdentityGateConfig,
    run_identity_consistency,
)


class _FakeEmbedder:
    """Maps crop paths to fixed vectors; no model load."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.calls = 0

    def embed_batch(self, paths: list[Any]) -> list[list[float]]:
        self.calls += 1
        return [self.table[str(p)] for p in paths]


class _FakeAuditor:
    """Returns preconfigured verdicts and records whether it was called."""

    def __init__(self, verdicts: list[dict[str, Any]], dominant_reason: str = "majority") -> None:
        self._verdicts = verdicts
        self._dominant_reason = dominant_reason
        self.calls = 0

    def audit(self, *, name: str, description: str, kind: str, crop_paths: list[Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "dominant_reason": self._dominant_reason,
            "verdicts": self._verdicts,
            "available": True,
        }


def _prop(entity_id: str, kind: str, chunk_id: int, path: str) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "name": entity_id,
        "kind": kind,
        "chunk_id": chunk_id,
        "crop_path": path,
        "task_kind": "acquire",
        "accepted": True,
    }


def test_assign_margin_defers_ambiguous_candidate() -> None:
    exemplars = {"char_a": [[1.0, 0.0, 0.0]], "char_b": [[0.0, 1.0, 0.0]]}
    # Candidate is equidistant to both identities -> should not auto-claim either.
    assignment, leftover = exclusive_assign_candidates(
        candidate_vecs=[[0.7, 0.7, 0.0]],
        exemplars=exemplars,
        entity_ids=["char_a", "char_b"],
        sim_floor=0.5,
        assign_margin=0.06,
    )
    assert assignment == {}
    assert leftover == [0]


def test_cohesive_entity_skips_vlm() -> None:
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "a1.png": [0.99, 0.14]})
    auditor = _FakeAuditor(verdicts=[])
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 3, "a1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert auditor.calls == 0  # tight DINOv3 cluster -> no VLM, no human
    assert all(p["accepted"] for p in props)
    assert props[0]["identity_review"]["source"] == "dinov3_cohesive"
    assert summary["n_entities_cohesive_skipped"] == 1
    assert summary["n_crops_rejected"] == 0


def test_vlm_rejects_mixed_crop() -> None:
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "b1.png": [0.0, 1.0]})
    # index 1 is a different entity mixed in.
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "same_entity": True, "confidence": "high", "reason": "dominant"},
            {"index": 1, "same_entity": False, "confidence": "high", "reason": "other person"},
        ]
    )
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 5, "b1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert auditor.calls == 1
    assert props[0]["accepted"] is True
    assert props[1]["accepted"] is False
    assert props[1]["identity_review"]["source"] == "vlm_rejected"
    assert props[1]["reason"] == "identity_gate_reject"
    assert summary["n_crops_rejected"] == 1


def test_prop_low_confidence_reject_is_not_trusted() -> None:
    embedder = _FakeEmbedder({"p0.png": [1.0, 0.0], "p1.png": [0.0, 1.0]})
    # Props only reject on high confidence; a medium "different" verdict is kept.
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "same_entity": True, "confidence": "high", "reason": "dominant"},
            {"index": 1, "same_entity": False, "confidence": "medium", "reason": "maybe other"},
        ]
    )
    props = [
        _prop("prop_a", "prop", 0, "p0.png"),
        _prop("prop_a", "prop", 5, "p1.png"),
    ]
    run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert props[1]["accepted"] is True  # medium reject not trusted for props


def test_no_auditor_flags_needs_human_without_rejecting() -> None:
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "b1.png": [0.0, 1.0]})
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 5, "b1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=None)
    assert all(p["accepted"] for p in props)  # never reject on DINOv3 alone
    assert all(p["identity_review"]["needs_human"] for p in props)
    assert summary["n_crops_needs_human"] == 2
    assert summary["n_crops_rejected"] == 0


def test_vlm_trusts_keepset_even_when_medoid_rejected() -> None:
    # Mixed entity: DINOv3 medoid (idx1) is itself an intruder the VLM rejects.
    # We must still act on the VLM keep-set instead of keeping the whole mixed set.
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "a1.png": [0.9, 0.1], "b2.png": [0.0, 1.0]})
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "same_entity": False, "confidence": "high", "reason": "x"},
            {"index": 1, "same_entity": False, "confidence": "high", "reason": "y"},
            {"index": 2, "same_entity": True, "confidence": "high", "reason": "z"},
        ]
    )
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 2, "a1.png"),
        _prop("char_a", "character", 5, "b2.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert props[0]["accepted"] is False
    assert props[1]["accepted"] is False
    assert props[2]["accepted"] is True
    assert summary["n_crops_rejected"] == 2


def test_vlm_rejects_everything_defers_to_human() -> None:
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "b1.png": [0.0, 1.0]})
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "same_entity": False, "confidence": "high", "reason": "x"},
            {"index": 1, "same_entity": False, "confidence": "high", "reason": "y"},
        ]
    )
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 5, "b1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    # Nothing to anchor on -> keep all, flag human, reject nothing.
    assert all(p["accepted"] for p in props)
    assert all(p["identity_review"]["needs_human"] for p in props)
    assert summary["n_crops_rejected"] == 0


def test_not_visible_crop_dropped_not_counted_as_reject() -> None:
    # Spread embeddings so the cohesion pre-filter defers to the VLM auditor.
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "a1.png": [0.0, 1.0]})
    # idx1 has no verifiable identity view (e.g. back-of-head) though VLM guessed same.
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "identity_visible": True, "same_entity": True, "confidence": "high", "reason": "face"},
            {"index": 1, "identity_visible": False, "same_entity": True, "confidence": "low", "reason": "back of head"},
        ]
    )
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 5, "a1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert props[0]["accepted"] is True
    assert props[1]["accepted"] is False
    assert props[1]["identity_review"]["source"] == "vlm_not_visible"
    assert props[1]["identity_review"]["identity_visible"] is False
    assert props[1]["reason"] == "identity_not_visible"
    assert summary["n_crops_not_visible"] == 1
    assert summary["n_crops_rejected"] == 0


def test_all_not_visible_defers_to_human() -> None:
    embedder = _FakeEmbedder({"a0.png": [1.0, 0.0], "a1.png": [0.0, 1.0]})
    auditor = _FakeAuditor(
        verdicts=[
            {"index": 0, "identity_visible": False, "same_entity": True, "confidence": "low", "reason": "blur"},
            {"index": 1, "identity_visible": False, "same_entity": True, "confidence": "low", "reason": "dark"},
        ]
    )
    props = [
        _prop("char_a", "character", 0, "a0.png"),
        _prop("char_a", "character", 5, "a1.png"),
    ]
    _out, summary = run_identity_consistency(props, embedder=embedder, auditor=auditor)
    assert all(p["accepted"] for p in props)  # nothing usable -> keep all, human
    assert all(p["identity_review"]["needs_human"] for p in props)
    assert summary["n_crops_not_visible"] == 0
    assert summary["n_crops_rejected"] == 0


def test_location_kind_is_not_gated() -> None:
    embedder = _FakeEmbedder({"l0.png": [1.0, 0.0], "l1.png": [0.0, 1.0]})
    auditor = _FakeAuditor(verdicts=[])
    props = [
        _prop("loc_a", "location", 0, "l0.png"),
        _prop("loc_a", "location", 5, "l1.png"),
    ]
    _out, summary = run_identity_consistency(
        props, embedder=embedder, auditor=auditor, config=IdentityGateConfig()
    )
    assert auditor.calls == 0
    assert summary["n_entities_checked"] == 0
    assert all("identity_review" not in p for p in props)
