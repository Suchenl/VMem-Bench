"""Unit tests for S3 canonical-name coverage helpers."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (
    ACTION_ENTITY_LIST_CODA,
    ACTION_MISSING_CANONICAL_NAME,
    action_contains_canonical_name,
    action_has_entity_list_coda,
    missing_canonical_names,
    rewrite_action_canonical_mentions,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (
    PassthroughReviewer,
    QwenVideoReviewer,
    SegmentReview,
    _apply_canonical_name_gate,
    _repair_action_with_trusted_names,
    review_segment_until_accepted,
    run_auto_review,
    segment_roster,
)


def test_cjk_substring_match() -> None:
    assert action_contains_canonical_name("白兔跳上桌子。", "白兔")
    assert not action_contains_canonical_name("一只兔子跳上桌子。", "白兔")


def test_latin_casefold_and_token_boundary() -> None:
    assert action_contains_canonical_name("Blue coat Hero stands.", "Hero")
    assert not action_contains_canonical_name("The start of the scene.", "art")
    assert action_contains_canonical_name("Blue coat hero stands in a dark room.", "Room")


def test_missing_canonical_names_reports_all_entity_kinds() -> None:
    roster = {
        "char_001": {"entity_id": "char_001", "name": "Lester Burnham", "kind": "character"},
        "loc_001": {"entity_id": "loc_001", "name": "Kitchen", "kind": "location"},
    }
    missing = missing_canonical_names(
        action="A man eats an apple.",
        present_entity_ids=["char_001", "loc_001"],
        roster_by_id=roster,
    )
    assert [item["name"] for item in missing] == ["Lester Burnham", "Kitchen"]


def test_entity_list_coda_detection() -> None:
    assert action_has_entity_list_coda("兔子跑到树前，可见大兔子、粉色蝴蝶、开阔草地。")
    assert action_has_entity_list_coda("Someone enters, showing Hero and Kitchen.")
    assert action_has_entity_list_coda("出场：大兔子、粉色蝴蝶。")
    assert not action_has_entity_list_coda("开阔草地上，大兔子追赶粉色蝴蝶。")
    assert not action_has_entity_list_coda("远处可以看见大兔子正在追赶粉色蝴蝶。")


def test_rewrite_generic_mentions_and_location_naturally() -> None:
    roster = {
        "char_001": {"entity_id": "char_001", "name": "大兔子", "kind": "character"},
        "char_003": {"entity_id": "char_003", "name": "粉色蝴蝶", "kind": "character"},
        "loc_001": {"entity_id": "loc_001", "name": "开阔草地", "kind": "location"},
    }
    rewritten, changes, still = rewrite_action_canonical_mentions(
        action="兔子跑到树前，伸手去抓停在树干上的蝴蝶。",
        present_entity_ids=["char_001", "char_003", "loc_001"],
        roster_by_id=roster,
    )
    assert rewritten == "开阔草地上，大兔子跑到树前，伸手去抓停在树干上的粉色蝴蝶。"
    assert [item["operation"] for item in changes] == ["replace", "replace", "prefix"]
    assert still == []
    assert "可见" not in rewritten


def test_rewrite_cave_mentions_without_redundant_coda() -> None:
    roster = {
        "char_001": {"entity_id": "char_001", "name": "大兔子", "kind": "character"},
        "loc_002": {"entity_id": "loc_002", "name": "兔子洞穴", "kind": "location"},
    }
    rewritten, _changes, still = rewrite_action_canonical_mentions(
        action="洞内黑暗中，大兔子蜷缩着睡觉。",
        present_entity_ids=["char_001", "loc_002"],
        roster_by_id=roster,
    )
    assert rewritten == "兔子洞穴内黑暗中，大兔子蜷缩着睡觉。"
    assert still == []


def test_rewrite_group_prop_and_sky_aliases() -> None:
    roster = {
        "char_004": {"entity_id": "char_004", "name": "红松鼠", "kind": "character"},
        "char_005": {"entity_id": "char_005", "name": "灰飞鼠", "kind": "character"},
        "char_006": {"entity_id": "char_006", "name": "灰老鼠", "kind": "character"},
        "prop_001": {"entity_id": "prop_001", "name": "红色苹果", "kind": "prop"},
        "loc_004": {"entity_id": "loc_004", "name": "高空", "kind": "location"},
    }
    rewritten, _changes, still = rewrite_action_canonical_mentions(
        action="三个小动物把红苹果扔向天空。",
        present_entity_ids=[
            "char_004", "char_005", "char_006", "prop_001", "loc_004",
        ],
        roster_by_id=roster,
    )
    assert rewritten == "红松鼠、灰飞鼠和灰老鼠把红色苹果扔向高空。"
    assert still == []


def test_gate_recovers_vlm_dropped_location_instead_of_blocking() -> None:
    """A VLM paraphrase that shortens 郊区街道→街道 while adding a character must
    be deterministically canonicalized, not reverted-and-blocked (LSMDC flood)."""
    roster = [
        {"entity_id": "char_001", "name": "中年男子", "kind": "character"},
        {"entity_id": "char_002", "name": "中年女子", "kind": "character"},
        {"entity_id": "loc_003", "name": "郊区街道", "kind": "location"},
    ]
    # The VLM added both characters but dropped the verbatim location name.
    review = SegmentReview(
        segment_id="seg_0015",
        revised_present=["char_001", "char_002", "loc_003"],
        revised_action="中年女子在街道旁的车边，中年男子从房子走出后沿小径向她走来。",
        confidence="high",
        risk_reasons=[],
        raw={"accepted": True},
        accepted=True,
    )
    gated = _apply_canonical_name_gate(
        review,
        roster=roster,
        # Seed keeps the location but is itself missing 中年女子; the pre-fix
        # code reverted here and blocked on the missing character.
        previous_action="在郊区街道中，中年男子走出房子。",
    )
    assert "郊区街道" in gated.revised_action
    assert "中年男子" in gated.revised_action and "中年女子" in gated.revised_action
    assert ACTION_MISSING_CANONICAL_NAME not in gated.risk_reasons
    assert gated.accepted is True


def test_try_complete_adopts_clean_but_refuses_garbled() -> None:
    """The guarded completion recovers a clean location drop but refuses a
    garbled VLM candidate even when it would satisfy name coverage."""
    from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (
        try_complete_canonical_action,
    )

    roster_by_id = {
        "char_001": {"entity_id": "char_001", "name": "亚裔少年", "kind": "character"},
        "loc_003": {"entity_id": "loc_003", "name": "郊区街道", "kind": "location"},
    }
    present = ["char_001", "loc_003"]

    # Clean candidate that only shortened the location: recovered + adopted.
    clean = try_complete_canonical_action(
        action="亚裔少年蹲在街道旁翻看照片。",
        present_entity_ids=present,
        roster_by_id=roster_by_id,
    )
    assert clean is not None and "郊区街道" in clean

    # Garbled candidate (stray Latin filler): refused despite fixable coverage.
    garbled = try_complete_canonical_action(
        action="亚istinguished亚裔少年蹲在街道旁翻看照片。",
        present_entity_ids=present,
        roster_by_id=roster_by_id,
    )
    assert garbled is None


def test_gate_still_blocks_genuinely_absent_prop() -> None:
    """Deterministic recovery must not invent an unmentioned prop; real gaps
    still reach human review."""
    roster = [
        {"entity_id": "char_001", "name": "金发少女", "kind": "character"},
        {"entity_id": "prop_002", "name": "红玫瑰", "kind": "prop"},
    ]
    review = SegmentReview(
        segment_id="seg_0063",
        revised_present=["char_001", "prop_002"],
        revised_action="金发少女在聚光灯下，背景变暗。",
        confidence="high",
        risk_reasons=[],
        raw={"accepted": True},
        accepted=True,
    )
    gated = _apply_canonical_name_gate(review, roster=roster, previous_action=None)
    assert ACTION_MISSING_CANONICAL_NAME in gated.risk_reasons
    assert gated.accepted is False


class _MissingNameReviewer:
    def review(self, *, clip: Path, segment: dict, roster: list[dict[str, str]]) -> SegmentReview:
        del clip, roster
        present = [str(item) for item in segment.get("present_entity_ids") or []]
        return SegmentReview(
            segment_id=str(segment["segment_id"]),
            revised_present=present,
            revised_action="Someone walks into a room.",
            confidence="high",
            risk_reasons=[],
            raw={"accepted": True},
            accepted=True,
        )

    def review_first_presence(self, *, clip: Path, segment: dict, entity: dict[str, str]) -> dict:
        del clip, segment, entity
        return {"description_covered": True, "missing_visual_attributes": [], "revised_action": ""}


class _NaturalNameReviewer:
    """Round1 omits names; round2 weaves them naturally when prompted."""

    def review(self, *, clip: Path, segment: dict, roster: list[dict[str, str]]) -> SegmentReview:
        del clip, roster
        present = [str(item) for item in segment.get("present_entity_ids") or []]
        must = list(segment.get("_missing_canonical_names") or [])
        if must:
            action = "Hero walks into the Kitchen and looks around."
        else:
            action = "Someone walks into a room."
        return SegmentReview(
            segment_id=str(segment["segment_id"]),
            revised_present=present,
            revised_action=action,
            confidence="high",
            risk_reasons=[],
            raw={"accepted": True},
            accepted=True,
        )

    def review_first_presence(self, *, clip: Path, segment: dict, entity: dict[str, str]) -> dict:
        del clip, segment, entity
        return {"description_covered": True, "missing_visual_attributes": [], "revised_action": ""}


class _EntityListCodaReviewer:
    def review(self, *, clip: Path, segment: dict, roster: list[dict[str, str]]) -> SegmentReview:
        del clip, roster
        return SegmentReview(
            segment_id=str(segment["segment_id"]),
            revised_present=[str(item) for item in segment.get("present_entity_ids") or []],
            revised_action="Someone walks into a room, showing Hero and Kitchen.",
            confidence="high",
            risk_reasons=[],
            raw={"accepted": True},
            accepted=True,
        )

    def review_first_presence(self, *, clip: Path, segment: dict, entity: dict[str, str]) -> dict:
        del clip, segment, entity
        return {"description_covered": True, "missing_visual_attributes": [], "revised_action": ""}


class _SynonymReviewer(_MissingNameReviewer):
    def review(self, *, clip: Path, segment: dict, roster: list[dict[str, str]]) -> SegmentReview:
        review = super().review(clip=clip, segment=segment, roster=roster)
        review.revised_action = "A protagonist walks into an indoor room."
        return review


class _UnsafeFirstPresenceReviewer(PassthroughReviewer):
    def review_first_presence(
        self, *, clip: Path, segment: dict, entity: dict[str, str]
    ) -> dict:
        del clip, segment, entity
        return {
            "description_covered": False,
            "missing_visual_attributes": [],
            "revised_action": "Someone enters, showing Hero and Kitchen.",
        }


class _FailingFirstPresenceReviewer(PassthroughReviewer):
    def review_first_presence(
        self, *, clip: Path, segment: dict, entity: dict[str, str]
    ) -> dict:
        del clip, segment, entity
        raise TimeoutError("review endpoint timed out")


class _CanonicalActionRepairer:
    def repair_action(
        self,
        *,
        action: str,
        required_entities: list[dict[str, str]],
        retry_feedback: str = "",
    ) -> str:
        assert action == "中年男子站在门口。"
        assert [item["name"] for item in required_entities] == ["中年男子", "中年女子"]
        assert retry_feedback == ""
        return "中年男子和中年女子站在门口。"


class _NeverCalledRepairer:
    def repair_action(self, **_kwargs: object) -> str:
        raise AssertionError("fallback action should avoid an extra model call")


class _RetryingActionRepairer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def repair_action(
        self,
        *,
        action: str,
        required_entities: list[dict[str, str]],
        retry_feedback: str = "",
    ) -> str:
        del action, required_entities
        self.calls.append(retry_feedback)
        return "中年男子站在门口。" if len(self.calls) == 1 else "中年男子和中年女子站在门口。"


class _LowConfidenceDeleteQwen(QwenVideoReviewer):
    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:1/v1")
        self.prompt = ""

    def _request(self, clip, prompt, schema_name, schema):  # noqa: ANN001
        del clip, schema_name, schema
        self.prompt = prompt
        return {
            "accepted": False,
            "revised_present": [],
            "revised_action": "大兔子走出兔子洞穴。",
            "confidence": "low",
            "risk_reasons": ["present_mismatch"],
            "suggestions": [],
        }


def test_low_confidence_review_cannot_erase_seed_claims() -> None:
    reviewer = _LowConfidenceDeleteQwen()
    segment = {
        "segment_id": "seg_0001",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "action": "大兔子走出兔子洞穴。",
        "present_entity_ids": ["char_001", "loc_001"],
        "_seed_present_entity_ids": ["char_001", "loc_001"],
    }
    roster = [
        {
            "entity_id": "char_001",
            "name": "大兔子",
            "kind": "character",
            "description": "白色大兔子",
            "first_presence_seconds": "0",
            "last_presence_seconds": "1",
        },
        {
            "entity_id": "loc_001",
            "name": "兔子洞穴",
            "kind": "location",
            "description": "树根下的洞穴",
            "first_presence_seconds": "0",
            "last_presence_seconds": "1",
        },
    ]
    review = reviewer.review(clip=Path("unused.mp4"), segment=segment, roster=roster)
    assert review.revised_present == ["char_001", "loc_001"]
    assert review.raw["proposed_revised_present"] == []
    assert review.raw["presence_change_reason"] == "requires_human_confirmation"
    assert "presence_change_proposed" in review.risk_reasons
    assert "seed_claimed_present" in reviewer.prompt
    assert "只有当前片段提供明确反证时才能删除" in reviewer.prompt

    second_round = dict(segment)
    second_round["present_entity_ids"] = []
    selected = segment_roster(roster, second_round)
    assert {item["entity_id"] for item in selected} == {"char_001", "loc_001"}


def test_text_action_repair_injects_trusted_missing_name() -> None:
    review = SegmentReview(
        segment_id="seg_0011",
        revised_present=["char_001", "char_002"],
        revised_action="中年男子站在门口。",
        confidence="low",
        risk_reasons=[],
        raw={},
    )
    repaired = _repair_action_with_trusted_names(
        review,
        trusted_present_entity_ids=["char_001", "char_002"],
        roster_by_id={
            "char_001": {"entity_id": "char_001", "name": "中年男子", "kind": "character"},
            "char_002": {"entity_id": "char_002", "name": "中年女子", "kind": "character"},
        },
        repairer=_CanonicalActionRepairer(),  # type: ignore[arg-type]
    )
    assert repaired.revised_action == "中年男子和中年女子站在门口。"
    assert repaired.raw["text_action_repair"]["status"] == "accepted"


def test_action_repair_preserves_clean_prior_action_without_model_call() -> None:
    review = SegmentReview(
        segment_id="seg_0002",
        revised_present=["loc_001"],
        revised_action="航拍街道。",
        confidence="low",
        risk_reasons=[],
        raw={},
    )
    repaired = _repair_action_with_trusted_names(
        review,
        trusted_present_entity_ids=["loc_001"],
        roster_by_id={
            "loc_001": {"entity_id": "loc_001", "name": "郊区街道", "kind": "location"},
        },
        repairer=_NeverCalledRepairer(),  # type: ignore[arg-type]
        fallback_action="航拍郊区街道。",
    )
    assert repaired.revised_action == "航拍郊区街道。"
    assert repaired.raw["text_action_repair"]["status"] == "preserved_prior"


def test_trusted_greeting_repair_mentions_counterpart() -> None:
    review = SegmentReview(
        segment_id="seg_0011",
        revised_present=["char_001", "char_002"],
        revised_action="中年男子出门打招呼。",
        confidence="low",
        risk_reasons=[],
        raw={},
    )
    repaired = _repair_action_with_trusted_names(
        review,
        trusted_present_entity_ids=["char_001", "char_002"],
        roster_by_id={
            "char_001": {"entity_id": "char_001", "name": "中年男子", "kind": "character"},
            "char_002": {"entity_id": "char_002", "name": "中年女子", "kind": "character"},
        },
        repairer=_CanonicalActionRepairer(),  # type: ignore[arg-type]
    )
    assert repaired.revised_action == "中年男子出门向中年女子打招呼。"
    assert repaired.raw["text_action_repair"]["status"] == "deterministic"


def test_text_action_repair_retries_with_missing_name_feedback() -> None:
    repairer = _RetryingActionRepairer()
    review = SegmentReview(
        segment_id="seg_0011",
        revised_present=["char_001", "char_002"],
        revised_action="中年男子站在门口。",
        confidence="low",
        risk_reasons=[],
        raw={},
    )
    repaired = _repair_action_with_trusted_names(
        review,
        trusted_present_entity_ids=["char_001", "char_002"],
        roster_by_id={
            "char_001": {"entity_id": "char_001", "name": "中年男子", "kind": "character"},
            "char_002": {"entity_id": "char_002", "name": "中年女子", "kind": "character"},
        },
        repairer=repairer,  # type: ignore[arg-type]
    )
    assert repaired.revised_action == "中年男子和中年女子站在门口。"
    assert repaired.raw["text_action_repair"]["status"] == "accepted"
    assert len(repairer.calls) == 2
    assert "中年女子" in repairer.calls[1]


def test_review_loop_prefers_natural_rewrite(tmp_path: Path) -> None:
    segment = {
        "segment_id": "seg_0001",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "action": "Someone walks into a room.",
        "present_entity_ids": ["char_001", "loc_001"],
    }
    roster = [
        {"entity_id": "char_001", "name": "Hero", "kind": "character", "description": ""},
        {"entity_id": "loc_001", "name": "Kitchen", "kind": "location", "description": ""},
    ]
    review = review_segment_until_accepted(
        segment=segment,
        roster=roster,
        source_video=tmp_path / "unused.mp4",
        cache_root=tmp_path / "cache",
        worker_id="w0",
        reviewer=_NaturalNameReviewer(),
        max_review_rounds=2,
    )
    assert review.n_rounds == 2
    assert "Hero" in review.revised_action
    assert "Kitchen" in review.revised_action
    assert "出场：" not in review.revised_action
    assert "可见" not in review.revised_action
    assert review.accepted is True


def test_review_loop_keeps_missing_names_rejected_after_final_round(tmp_path: Path) -> None:
    segment = {
        "segment_id": "seg_0001",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "action": "Someone walks into a room.",
        "present_entity_ids": ["char_001", "loc_001"],
    }
    roster = [
        {"entity_id": "char_001", "name": "Hero", "kind": "character", "description": ""},
        {"entity_id": "loc_001", "name": "Kitchen", "kind": "location", "description": ""},
    ]
    review = review_segment_until_accepted(
        segment=segment,
        roster=roster,
        source_video=tmp_path / "unused.mp4",
        cache_root=tmp_path / "cache",
        worker_id="w0",
        reviewer=_MissingNameReviewer(),
        max_review_rounds=1,
    )
    assert review.revised_action == "In Kitchen, Someone walks into a room."
    assert ACTION_MISSING_CANONICAL_NAME in review.risk_reasons
    assert "max_review_rounds_exhausted" in review.risk_reasons
    assert review.accepted is False
    assert review.elapsed_seconds is not None
    assert review.elapsed_seconds >= 0.0


def test_missing_name_draft_cannot_replace_clean_prior_action(tmp_path: Path) -> None:
    segment = {
        "segment_id": "seg_0001",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "action": "Someone walks into a room.",
        "present_entity_ids": ["char_001", "loc_001"],
    }
    roster = [
        {"entity_id": "char_001", "name": "Hero", "kind": "character", "description": ""},
        {"entity_id": "loc_001", "name": "Kitchen", "kind": "location", "description": ""},
    ]
    review = review_segment_until_accepted(
        segment=segment,
        roster=roster,
        source_video=tmp_path / "unused.mp4",
        cache_root=tmp_path / "cache",
        worker_id="w0",
        reviewer=_SynonymReviewer(),
        max_review_rounds=1,
    )
    assert review.revised_action == "In Kitchen, Someone walks into a room."
    assert review.raw["rejected_missing_name_action"] == (
        "A protagonist walks into an indoor room."
    )
    assert review.accepted is False


def test_review_loop_rejects_entity_list_coda_and_preserves_prior_action(
    tmp_path: Path,
) -> None:
    segment = {
        "segment_id": "seg_0001",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "action": "Someone walks into a room.",
        "present_entity_ids": ["char_001", "loc_001"],
    }
    roster = [
        {"entity_id": "char_001", "name": "Hero", "kind": "character", "description": ""},
        {"entity_id": "loc_001", "name": "Kitchen", "kind": "location", "description": ""},
    ]
    review = review_segment_until_accepted(
        segment=segment,
        roster=roster,
        source_video=tmp_path / "unused.mp4",
        cache_root=tmp_path / "cache",
        worker_id="w0",
        reviewer=_EntityListCodaReviewer(),
        max_review_rounds=1,
    )
    assert review.revised_action == "In Kitchen, Someone walks into a room."
    assert review.raw["rejected_entity_list_action"].endswith("showing Hero and Kitchen.")
    assert ACTION_ENTITY_LIST_CODA in review.risk_reasons
    assert review.accepted is False


def test_first_presence_cannot_overwrite_action_with_entity_list_coda(
    tmp_path: Path,
) -> None:
    original_action = "Hero walks through the Kitchen."
    payload = {
        "video_duration_seconds": 1.0,
        "characters": [{
            "char_id": "char_001",
            "name": "Hero",
            "description": "A person",
            "first_presence_seconds": 0.0,
            "last_presence_seconds": 1.0,
        }],
        "props": [],
        "locations": [{
            "loc_id": "loc_001",
            "name": "Kitchen",
            "description": "An indoor kitchen",
            "first_presence_seconds": 0.0,
            "last_presence_seconds": 1.0,
        }],
        "screenplay": {
            "scenes": [{
                "scene_id": "scene_0001",
                "visual_segments": [{
                    "segment_id": "seg_0001",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "action": original_action,
                    "present_entity_ids": ["char_001", "loc_001"],
                }],
            }],
        },
    }
    result = run_auto_review(
        annotation_payload=payload,
        source_video=tmp_path / "unused.mp4",
        stage_dir=tmp_path / "s3",
        reviewer=_UnsafeFirstPresenceReviewer(),
        max_review_rounds=1,
    )
    segment = result["annotation"]["screenplay"]["scenes"][0]["visual_segments"][0]
    assert segment["action"] == original_action
    first_review = result["first_reviews"]["char_001"]
    assert first_review["revised_action"] == ""
    assert first_review["rejected_revised_action"].endswith("showing Hero and Kitchen.")
    assert ACTION_ENTITY_LIST_CODA in first_review["revision_risk_reasons"]


def test_first_presence_timeout_does_not_fail_movie(tmp_path: Path) -> None:
    payload = {
        "video_duration_seconds": 1.0,
        "characters": [{
            "char_id": "char_001",
            "name": "Hero",
            "description": "A person",
            "first_presence_seconds": 0.0,
            "last_presence_seconds": 1.0,
        }],
        "props": [],
        "locations": [],
        "screenplay": {
            "scenes": [{
                "scene_id": "scene_0001",
                "visual_segments": [{
                    "segment_id": "seg_0001",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "action": "Hero walks.",
                    "present_entity_ids": ["char_001"],
                }],
            }],
        },
    }
    stage_dir = tmp_path / "s3"
    result = run_auto_review(
        annotation_payload=payload,
        source_video=tmp_path / "unused.mp4",
        stage_dir=stage_dir,
        reviewer=_FailingFirstPresenceReviewer(),
        max_review_rounds=1,
    )
    first_review = result["first_reviews"]["char_001"]
    assert first_review["risk_reasons"] == ["first_presence_review_failed"]
    assert "TimeoutError" in first_review["error"]
    progress = json.loads((stage_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "done"


def test_review_error_tag_distinguishes_http_from_json() -> None:
    from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (
        _review_error_tag,
    )

    assert _review_error_tag("finish_reason=length; content_head='{'") == "vlm_output_truncated"
    assert (
        _review_error_tag("S3 reviewer JSON parse failed (Expecting value); finish_reason=stop")
        == "vlm_json_parse_failed"
    )
    assert (
        _review_error_tag(
            'S3 reviewer HTTP 400: {"error":{"message":"Invalid `--allowed-local-media-path`: '
            'The path ${ALLOWED_LOCAL_MEDIA_PATH:-.},/tmp does not exist."}}'
        )
        == "vlm_request_failed"
    )
    assert (
        _review_error_tag(
            "You passed 1193 input characters and requested 8192 output tokens. "
            "However, the model's context length is 8192 tokens."
        )
        == "vlm_context_overflow"
    )
