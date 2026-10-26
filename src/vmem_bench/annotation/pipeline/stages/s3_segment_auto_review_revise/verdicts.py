"""Typed S3 verdicts derived from deterministic gates and VLM evidence."""

from __future__ import annotations

from typing import Any

PASS = "PASS"
WARN = "WARN"
BLOCK = "BLOCK"
RETRYABLE_ERROR = "RETRYABLE_ERROR"
VERDICTS = frozenset({PASS, WARN, BLOCK, RETRYABLE_ERROR})

_DETERMINISTIC_BLOCKERS = frozenset({
    "action_missing_canonical_name",
    "action_entity_list_coda",
    "entity_empty_canonical_name",
})
_RETRYABLE_CODES = frozenset({
    "vlm_output_truncated",
    "vlm_json_parse_failed",
    "vlm_context_overflow",
    "vlm_request_failed",
    "segment_review_crashed",
})
_PRESENCE_HINTS = ("present", "entity", "roster", "identity")
_ACTION_HINTS = ("action", "canonical", "coda")


def _finding(
    code: str,
    *,
    severity: str,
    source: str,
    message: str,
    recommended_action: str,
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "source": source,
        "message": message,
        "recommended_action": recommended_action,
    }


def _recommend(findings: list[dict[str, str]]) -> str:
    actions = [item["recommended_action"] for item in findings]
    for preferred in (
        "retry",
        "edit_action",
        "edit_present",
        "spot_check",
    ):
        if preferred in actions:
            return preferred
    return "none"


def adjudicate_review(
    *,
    model_accepted: bool,
    confidence: str,
    risk_reasons: list[str],
    raw: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]], str]:
    """Return ``(verdict, findings, recommended_action)``.

    Deterministic blockers and high-confidence contradictions remain BLOCK.
    When the upstream presence labels are marked trusted, low/medium-confidence
    reviewer disagreement becomes audit evidence sampled from PASS rather than
    a human-review flood. Infrastructure failures are retryable and never
    masquerade as annotation judgments.
    """
    reasons = list(dict.fromkeys(str(item) for item in risk_reasons if str(item)))
    findings: list[dict[str, str]] = []

    for code in reasons:
        if code in _RETRYABLE_CODES or code.startswith("vlm_"):
            findings.append(
                _finding(
                    code,
                    severity="retryable",
                    source="runtime",
                    message=code,
                    recommended_action="retry",
                )
            )
        elif code in _DETERMINISTIC_BLOCKERS:
            findings.append(
                _finding(
                    code,
                    severity="block",
                    source="deterministic",
                    message=code,
                    recommended_action="edit_action",
                )
            )

    if any(item["severity"] == "retryable" for item in findings):
        return RETRYABLE_ERROR, findings, _recommend(findings)
    if any(item["severity"] == "block" for item in findings):
        return BLOCK, findings, _recommend(findings)

    semantic_reasons = [
        code
        for code in reasons
        if code not in {"max_review_rounds_exhausted", "presence_change_untrusted"}
    ]
    high_conflict = confidence == "high" and not model_accepted
    if high_conflict:
        joined = " ".join(semantic_reasons).casefold()
        if any(hint in joined for hint in _PRESENCE_HINTS):
            action = "edit_present"
        else:
            action = "edit_action"
        findings.append(
            _finding(
                semantic_reasons[0] if semantic_reasons else "high_confidence_conflict",
                severity="block",
                source="vlm",
                message="high-confidence reviewer conflict",
                recommended_action=action,
            )
        )
        return BLOCK, findings, _recommend(findings)

    if (
        bool((raw or {}).get("seed_presence_trusted"))
        and confidence in {"low", "medium"}
        and not model_accepted
    ):
        findings.append(
            _finding(
                semantic_reasons[0] if semantic_reasons else "review_inconclusive",
                severity="audit",
                source="vlm",
                message="low-confidence reviewer disagreement retained as audit evidence",
                recommended_action="spot_check",
            )
        )
        return PASS, findings, _recommend(findings)

    if not model_accepted or confidence in {"low", "medium"}:
        action = "spot_check"
        findings.append(
            _finding(
                semantic_reasons[0] if semantic_reasons else "review_inconclusive",
                severity="warn",
                source="vlm",
                message="reviewer inconclusive without a deterministic blocker",
                recommended_action=action,
            )
        )
        return WARN, findings, _recommend(findings)

    del raw
    return PASS, [], "none"


def accepted_for_compatibility(verdict: str) -> bool:
    """Old consumers treat PASS and non-blocking WARN as accepted."""
    return verdict in {PASS, WARN}

