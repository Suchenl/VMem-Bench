"""Typed S3 verdict policy tests."""

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.verdicts import (
    BLOCK,
    PASS,
    RETRYABLE_ERROR,
    WARN,
    accepted_for_compatibility,
    adjudicate_review,
)


def test_deterministic_name_failure_blocks() -> None:
    verdict, findings, action = adjudicate_review(
        model_accepted=True,
        confidence="high",
        risk_reasons=["action_missing_canonical_name"],
    )
    assert verdict == BLOCK
    assert findings[0]["source"] == "deterministic"
    assert action == "edit_action"
    assert accepted_for_compatibility(verdict) is False


def test_runtime_failure_is_retryable() -> None:
    verdict, _findings, action = adjudicate_review(
        model_accepted=False,
        confidence="low",
        risk_reasons=["vlm_request_failed"],
    )
    assert verdict == RETRYABLE_ERROR
    assert action == "retry"


def test_low_confidence_disagreement_is_warn() -> None:
    verdict, findings, action = adjudicate_review(
        model_accepted=False,
        confidence="low",
        risk_reasons=["present_mismatch"],
    )
    assert verdict == WARN
    assert findings[0]["severity"] == "warn"
    assert action == "spot_check"
    assert accepted_for_compatibility(verdict) is True


def test_trusted_seed_low_confidence_disagreement_is_pass_audit() -> None:
    verdict, findings, action = adjudicate_review(
        model_accepted=False,
        confidence="low",
        risk_reasons=["presence_change_proposed"],
        raw={"seed_presence_trusted": True},
    )
    assert verdict == PASS
    assert findings[0]["severity"] == "audit"
    assert action == "spot_check"


def test_high_confidence_presence_conflict_blocks() -> None:
    verdict, _findings, action = adjudicate_review(
        model_accepted=False,
        confidence="high",
        risk_reasons=["present_mismatch"],
    )
    assert verdict == BLOCK
    assert action == "edit_present"


def test_clean_review_passes() -> None:
    verdict, findings, action = adjudicate_review(
        model_accepted=True,
        confidence="high",
        risk_reasons=[],
    )
    assert verdict == PASS
    assert findings == []
    assert action == "none"

