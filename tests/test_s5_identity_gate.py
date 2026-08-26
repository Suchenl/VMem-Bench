"""Staged S5/S6 identity-safety tests."""

from copy import deepcopy

import pytest

from vmem_bench.common.crop_identity_gates import (
    apply_cross_entity_conflict_gate,
)
from vmem_bench.annotation.pipeline.stages.s6_entities_visual_crop_human_review.queue import (
    build_review_queue,
)


def _proposal(entity_id: str, confidence: str = "low") -> dict:
    return {
        "chunk_id": 10,
        "segment_id": "seg_0011",
        "entity_id": entity_id,
        "kind": "character",
        "task_kind": "acquire",
        "frame_index": 2453,
        "bbox_norm": [100, 100, 400, 400],
        "crop_path": f"/tmp/{entity_id}.png",
        "grounding": {"detector_score": 0.95},
        "pick": {"confidence": confidence},
        "qa": {"accepted": True, "reasons": []},
        "accepted": True,
        "representation_id": f"{entity_id}@c00010",
    }


def test_ambiguous_same_bbox_rejects_all_entities() -> None:
    proposals = [_proposal("char_a"), _proposal("char_b")]
    gated = apply_cross_entity_conflict_gate(proposals)
    assert [item["accepted"] for item in gated] == [False, False]
    assert all(item["reason"] == "cross_entity_bbox_conflict" for item in gated)


def test_unique_high_confidence_assignment_survives() -> None:
    proposals = [_proposal("char_a", "high"), _proposal("char_b", "low")]
    gated = apply_cross_entity_conflict_gate(deepcopy(proposals))
    assert gated[0]["accepted"] is True
    assert gated[1]["accepted"] is False
    assert gated[0]["identity_gate"]["reason"] == "unique_confident_assignment"


@pytest.mark.xfail(reason="upstream: S6 hides cross_entity_bbox_conflict; fails in internal tree too", strict=False)
def test_s6_identity_conflict_is_must_review() -> None:
    proposal = _proposal("char_a")
    proposal["accepted"] = False
    proposal["reason"] = "cross_entity_bbox_conflict"
    proposal["qa"] = {
        "accepted": False,
        "reasons": ["cross_entity_bbox_conflict"],
    }
    queue = build_review_queue([proposal])
    assert queue[0]["review_tier"] == "must"
    assert queue[0]["recommended_action"] == "review"

