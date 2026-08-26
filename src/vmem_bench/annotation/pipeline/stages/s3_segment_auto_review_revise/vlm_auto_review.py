"""S3 VLM review/revision for v5 visual segments.

The raw S1 JSON is never modified.  S3 writes a derived
``auto_revised_annotation.json`` after reviewing visual presence, action
coverage, and first-appearance descriptions against each segment clip.

When multiple Qwen reviewer endpoints are configured, segments are processed
from a shared work pool: a free endpoint picks the next segment, may revise
``present_entity_ids`` / ``action``, and (unless it accepts) the revised
segment is re-queued for another free reviewer until accept or
``max_review_rounds``.  Boundary re-split is suggested in the audit trail but
not auto-applied to the timeline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vmem_bench.annotation.pipeline.servers.direct_http import (
    ensure_no_proxy_env,
    ensure_no_proxy_host,
    urlopen_direct,
)
from vmem_bench.annotation.pipeline.servers.fleet.timeutil import now_beijing
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (
    ACTION_ENTITY_LIST_CODA,
    ACTION_MISSING_CANONICAL_NAME,
    ENTITY_EMPTY_CANONICAL_NAME,
    action_has_entity_list_coda,
    format_missing_names_for_prompt,
    missing_canonical_names,
    rewrite_action_canonical_mentions,
    try_complete_canonical_action,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.clip_queue import (
    ClipTask,
    SharedClipQueue,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
    ReviewerEndpointPool,
    parse_endpoint_urls,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.segment_media import (
    cleanup_cache,
    worker_clip,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.verdicts import (
    PASS,
    VERDICTS,
    accepted_for_compatibility,
    adjudicate_review,
)

# Dev-machine HTTP_PROXY must not wrap A800/H800 reviewer calls.
ensure_no_proxy_env()

# Served-model-name exposed by the vLLM fleet (Qwen3-VL-8B-Instruct). Callers
# (batch/orchestrator) override via --reviewer-model, but keep the default in
# sync with the deployed fleet so direct CLI / tooling calls do not 404.
DEFAULT_MODEL = "qwen3-vl-8b"
DEFAULT_MAX_REVIEW_ROUNDS = 2
# Fleet default max_model_len=32768; keep output budget small so video+prompt
# input still fits. Structured S3 JSON is short (action ≤220 chars).
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class SamplingProfile:
    """Decoding config for ONE reviewer role (critic vs fixer), per model family.

    NOTE (2026-07-23): earlier garbled output (男子男子, 裔→đương, traditional/
    Cyrillic leaks) was a SILENTLY CORRUPTED WEIGHT DOWNLOAD, not decoding — the
    fix is the mllm-weight-integrity pre-use gate, not these knobs. Sampling is
    split by role/model because the two roles want different behaviour:
      - critic (video review()): structured verification -> deterministic,
        reproducible; greedy for an *Instruct* model.
      - fixer (repair_action()): faithful prose rewrite -> also deterministic.
      - reasoning-tuned chat models (e.g. Qwen3.5): need their official
        non-thinking sampling; greedy temperature=0 makes them degenerate/loop.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # <=0 => disabled (not emitted)
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    enable_thinking: bool = False

    def payload(self, max_tokens: int) -> dict[str, Any]:
        """Build the vLLM request knobs; emit only non-neutral values."""
        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.top_k and self.top_k > 0:
            params["top_k"] = self.top_k
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.repetition_penalty
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        return params


# Deterministic greedy for Qwen3-VL-*-Instruct (reproducible gold annotations).
GREEDY_CRITIC = SamplingProfile()
GREEDY_FIXER = SamplingProfile()
# Reasoning-tuned chat models: official non-thinking sampling.
REASONING_CRITIC = SamplingProfile(
    temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, enable_thinking=False,
)
REASONING_FIXER = SamplingProfile(
    temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, enable_thinking=False,
)
# (critic, fixer) profile pairs, selectable by name or auto-picked from model.
SAMPLING_PRESETS: dict[str, tuple[SamplingProfile, SamplingProfile]] = {
    "qwen3vl-greedy": (GREEDY_CRITIC, GREEDY_FIXER),
    "reasoning-nothink": (REASONING_CRITIC, REASONING_FIXER),
}


def sampling_preset_for_model(model: str) -> str:
    """Pick a default sampling preset from the served model name."""
    m = model.lower()
    if "3.5" in m or "qwen3p5" in m:
        return "reasoning-nothink"
    return "qwen3vl-greedy"
# A VLM fleet may have dozens of endpoints, but ffmpeg cannot safely burst that
# many simultaneous re-encodes against one shared source video.
DEFAULT_MAX_CLIP_WORKERS = 4
DEFAULT_MAX_ACTION_REPAIR_ATTEMPTS = 3
# Keep structured fields bounded so the model cannot blow the output budget, but
# leave enough room for the action to (a) name every present entity and (b) weave
# the identifying appearance of any first-appearing entity into the sentence.
MAX_REVISED_ACTION_CHARS = 220
MAX_RISK_REASON_CHARS = 24
MAX_SUGGESTION_CHARS = 24
MAX_RISK_REASONS = 4
MAX_SUGGESTIONS = 3
# Include entities whose presence window overlaps the segment by this pad.
SEGMENT_ROSTER_PAD_SECONDS = 2.0


def _clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _clip_str_list(values: Any, *, max_chars: int, max_items: int) -> list[str]:
    out: list[str] = []
    for item in values or []:
        clipped = _clip_text(item, max_chars)
        if clipped:
            out.append(clipped)
        if len(out) >= max_items:
            break
    return list(dict.fromkeys(out))


@dataclass(slots=True)
class SegmentReview:
    segment_id: str
    revised_present: list[str]
    revised_action: str
    confidence: str
    risk_reasons: list[str]
    raw: dict[str, Any]
    accepted: bool = True
    suggestions: list[str] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    n_rounds: int = 1
    endpoints: list[str] = field(default_factory=list)
    elapsed_seconds: float | None = None
    # Client-observed timing. vLLM does not expose per-request inference
    # duration through its OpenAI-compatible response, so that field remains
    # None unless a future server-side trace explicitly populates it.
    queue_seconds: float | None = None
    clip_seconds: float | None = None
    vlm_request_seconds: float | None = None
    vlm_inference_seconds: float | None = None
    verdict: str = PASS
    findings: list[dict[str, str]] = field(default_factory=list)
    recommended_action: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _review_error_tag(err: str) -> str:
    """Map reviewer RuntimeError text to a short, accurate risk tag."""
    text = str(err or "")
    lowered = text.casefold()
    if "finish_reason=length" in lowered:
        return "vlm_output_truncated"
    if "json parse failed" in lowered or "content is not a string" in lowered:
        return "vlm_json_parse_failed"
    if (
        "output tokens" in lowered
        and ("input characters" in lowered or "max_model_len" in lowered or "context" in lowered)
    ):
        return "vlm_context_overflow"
    if (
        "http " in lowered
        or "allowed-local-media-path" in lowered
        or "badrequesterror" in lowered
        or "cannot load local files" in lowered
    ):
        return "vlm_request_failed"
    return "vlm_request_failed"


def _optional_elapsed(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _roster_by_id(roster: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        str(item.get("entity_id") or ""): {
            "entity_id": str(item.get("entity_id") or ""),
            "name": str(item.get("name") or ""),
            "kind": str(item.get("kind") or ""),
        }
        for item in roster
        if item.get("entity_id")
    }


# Minimum clip length (seconds) ffmpeg can cut into a reviewable window.
# Zero-duration / sub-frame boundary segments cannot be cut and previously
# surfaced as RETRYABLE ``segment_review_crashed`` (e.g. morevna_ep3 seg_0050).
MIN_REVIEWABLE_CLIP_SECONDS = 0.04


def _segment_duration_seconds(segment: dict[str, Any]) -> float | None:
    try:
        start = float(segment.get("start_seconds"))
        end = float(segment.get("end_seconds"))
    except (TypeError, ValueError):
        return None
    return end - start


def is_zero_duration_segment(segment: dict[str, Any]) -> bool:
    """True for a zero-duration / empty boundary segment that cannot be cut.

    Such segments carry no reviewable visual content and crash the clip path,
    so S3 auto-PASSes them deterministically (no VLM, no clip).
    """
    duration = _segment_duration_seconds(segment)
    return duration is not None and duration <= MIN_REVIEWABLE_CLIP_SECONDS


def _auto_pass_zero_duration(segment: dict[str, Any]) -> "SegmentReview":
    """Deterministic PASS for a zero-duration boundary segment (no VLM, no clip)."""
    return SegmentReview(
        segment_id=str(segment["segment_id"]),
        revised_present=[str(x) for x in (segment.get("present_entity_ids") or [])],
        revised_action=str(segment.get("action") or ""),
        confidence="high",
        risk_reasons=[],
        raw={
            "auto_pass": "zero_duration_boundary",
            "start_seconds": segment.get("start_seconds"),
            "end_seconds": segment.get("end_seconds"),
        },
        accepted=True,
        suggestions=[],
        verdict=PASS,
        recommended_action="none",
        elapsed_seconds=0.0,
    )


def _apply_canonical_name_gate(
    review: SegmentReview,
    *,
    roster: list[dict[str, str]],
    previous_action: str | None = None,
) -> SegmentReview:
    """Reject actions that omit names or append a mechanical entity list."""
    if action_has_entity_list_coda(review.revised_action):
        rejected_action = review.revised_action
        review.raw = {
            **dict(review.raw),
            "rejected_entity_list_action": rejected_action,
        }
        if previous_action is not None:
            # Do not let a rejected coda become the next round's drafting base
            # or leak into the derived annotation after the final round.
            review.revised_action = str(previous_action)
        review.risk_reasons = list(
            dict.fromkeys([*review.risk_reasons, ACTION_ENTITY_LIST_CODA])
        )
        review.accepted = False

    roster_by_id = _roster_by_id(roster)
    # A VLM paraphrase routinely drops an already-present canonical mention
    # while adding a different one — e.g. it shortens the location "郊区街道"
    # to "街道" in order to name a newly confirmed character, which then trips
    # the deterministic name gate on the location. Re-run the same
    # deterministic canonicalizer that already protects the seed action to
    # restore such safely-recoverable names (locations via prefix/fragment,
    # unambiguous generic mentions). Adopt the rewrite only when it fully
    # clears the coverage gap and stays clean — that is exactly the case where
    # recovery prevents a spurious BLOCK, and it keeps garbled drafts out of
    # gold. It never invents a missing character/prop mention, so real gaps
    # still surface as blockers below.
    recovered = try_complete_canonical_action(
        action=review.revised_action,
        present_entity_ids=list(review.revised_present),
        roster_by_id=roster_by_id,
    )
    if recovered and recovered != review.revised_action:
        review.revised_action = recovered
    missing = missing_canonical_names(
        action=review.revised_action,
        present_entity_ids=list(review.revised_present),
        roster_by_id=roster_by_id,
    )
    if (
        any(item.get("name") for item in missing)
        and previous_action is not None
        and review.revised_action != str(previous_action)
    ):
        # Do not feed a partial/synonym/emoji rewrite into the next round. Keep
        # the clean prior action and give the next reviewer the exact missing
        # names; this prevents failed drafts from compounding across rounds.
        review.raw = {
            **dict(review.raw),
            "rejected_missing_name_action": review.revised_action,
        }
        review.revised_action = str(previous_action)
        missing = missing_canonical_names(
            action=review.revised_action,
            present_entity_ids=list(review.revised_present),
            roster_by_id=roster_by_id,
        )
    repairable = [item for item in missing if item.get("name")]
    empty = [item for item in missing if not item.get("name")]
    if empty:
        review.risk_reasons = list(
            dict.fromkeys([*review.risk_reasons, ENTITY_EMPTY_CANONICAL_NAME])
        )
        review.accepted = False
    if repairable:
        review.risk_reasons = list(
            dict.fromkeys([*review.risk_reasons, ACTION_MISSING_CANONICAL_NAME])
        )
        review.raw = {
            **dict(review.raw),
            "missing_canonical_names": repairable,
        }
        # Validation must never manufacture language. The next VLM round or a
        # human reviewer resolves missing names; exhausted rounds stay rejected.
        review.accepted = False
    return review


def _apply_typed_verdict(review: SegmentReview) -> SegmentReview:
    model_accepted = bool((review.raw or {}).get("accepted", review.accepted))
    verdict, findings, recommended_action = adjudicate_review(
        model_accepted=model_accepted,
        confidence=review.confidence,
        risk_reasons=list(review.risk_reasons),
        raw=review.raw,
    )
    review.verdict = verdict
    review.findings = findings
    review.recommended_action = recommended_action
    review.accepted = accepted_for_compatibility(verdict)
    review.raw = {
        **dict(review.raw),
        "model_accepted": model_accepted,
        "verdict": verdict,
        "findings": findings,
        "recommended_action": recommended_action,
    }
    return review


class SegmentReviewer(Protocol):
    def review(
        self,
        *,
        clip: Path,
        segment: dict[str, Any],
        roster: list[dict[str, str]],
    ) -> SegmentReview: ...

    def review_first_presence(
        self,
        *,
        clip: Path,
        segment: dict[str, Any],
        entity: dict[str, str],
    ) -> dict[str, Any]: ...


def _file_url(path: Path) -> str:
    return f"file://{path.resolve()}"


class PassthroughReviewer:
    """Dependency-free reviewer used by deterministic dry-runs and tests."""

    def review(self, *, clip: Path, segment: dict[str, Any], roster: list[dict[str, str]]) -> SegmentReview:
        del clip, roster
        present = [str(item) for item in segment.get("present_entity_ids") or []]
        return SegmentReview(
            segment_id=str(segment["segment_id"]),
            revised_present=present,
            revised_action=str(segment.get("action") or ""),
            confidence="passthrough",
            risk_reasons=[],
            raw={"mode": "passthrough", "accepted": True},
            accepted=True,
        )

    def review_first_presence(
        self, *, clip: Path, segment: dict[str, Any], entity: dict[str, str]
    ) -> dict[str, Any]:
        del clip, segment, entity
        return {"description_covered": True, "missing_visual_attributes": [], "revised_action": ""}


class QwenVideoReviewer:
    """OpenAI-compatible Qwen3VL reviewer with bounded structured JSON output."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = 600,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        review_sampling: SamplingProfile | None = None,
        repair_sampling: SamplingProfile | None = None,
        fps: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        # Per-role sampling: critic (video review) vs fixer (text repair). When
        # not given, auto-pick by model family (greedy for Instruct, official
        # non-thinking for reasoning-tuned chat models).
        preset_critic, preset_fixer = SAMPLING_PRESETS[sampling_preset_for_model(model)]
        self.review_sampling = review_sampling or preset_critic
        self.repair_sampling = repair_sampling or preset_fixer
        self.fps = fps
        self.last_request_seconds: float | None = None
        ensure_no_proxy_host(self.base_url)

    def with_base_url(self, base_url: str) -> "QwenVideoReviewer":
        return QwenVideoReviewer(
            base_url=base_url,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            review_sampling=self.review_sampling,
            repair_sampling=self.repair_sampling,
            fps=self.fps,
        )

    def _request(self, clip: Path, prompt: str, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_request_seconds = None
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": _file_url(clip)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            **self.review_sampling.payload(self.max_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            # Pass only fps. do_sample_frames=true re-samples against original
            # video indices after vLLM already decoded frames and can IndexError.
            "mm_processor_kwargs": {"fps": self.fps},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_started = time.perf_counter()
        try:
            with urlopen_direct(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"S3 reviewer HTTP {exc.code}: {exc.read().decode('utf-8')[:500]}") from exc
        finally:
            # This is the client-observed HTTP request duration: it includes
            # service-side queueing, video preprocessing, inference, and
            # network transfer. It is deliberately not called inference time.
            self.last_request_seconds = time.perf_counter() - request_started
        message = body["choices"][0]["message"]
        content = message.get("content")
        finish_reason = body["choices"][0].get("finish_reason")
        if not isinstance(content, str):
            raise RuntimeError(
                f"S3 reviewer content is not a string (type={type(content).__name__}, "
                f"finish_reason={finish_reason})"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"S3 reviewer JSON parse failed ({exc}); finish_reason={finish_reason}; "
                f"content_chars={len(content)}; content_head={content[:400]!r}"
            ) from exc

    def repair_action(
        self,
        *,
        action: str,
        required_entities: list[dict[str, str]],
        retry_feedback: str = "",
    ) -> str:
        """Rewrite trusted labels into action prose without reopening visual review."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["revised_action"],
            "properties": {
                "revised_action": {
                    "type": "string",
                    "maxLength": MAX_REVISED_ACTION_CHARS,
                },
            },
        }
        prompt = (
            "你是视频标注的纯文本编辑器，不要判断视频、presence 或镜头边界。\n"
            "给定 action 和已确认出现在当前片段的实体，重写为一句紧凑自然的动作描述。"
            "必须逐字包含 required_entities 的每个 name，不能添加 roster 外实体或未给出的新事实。"
            "将交互自然写入动作（如“甲向乙打招呼”），禁止在句尾追加“可见/出场：”实体清单。\n"
            f"action={_clip_text(action, MAX_REVISED_ACTION_CHARS)}\n"
            f"required_entities={json.dumps(required_entities, ensure_ascii=False)}\n"
            f"retry_feedback={_clip_text(retry_feedback, 240)}"
        )
        self.last_request_seconds = None
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }],
            **self.repair_sampling.payload(self.max_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "canonical_action_repair", "schema": schema, "strict": True},
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_started = time.perf_counter()
        try:
            with urlopen_direct(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"S3 action repair HTTP {exc.code}: {exc.read().decode('utf-8')[:500]}"
            ) from exc
        finally:
            self.last_request_seconds = time.perf_counter() - request_started
        content = body["choices"][0]["message"].get("content")
        finish_reason = body["choices"][0].get("finish_reason")
        if not isinstance(content, str):
            raise RuntimeError(
                f"S3 action repair content is not a string (type={type(content).__name__}, "
                f"finish_reason={finish_reason})"
            )
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"S3 action repair JSON parse failed ({exc}); finish_reason={finish_reason}; "
                f"content_chars={len(content)}; content_head={content[:400]!r}"
            ) from exc
        return _clip_text(raw.get("revised_action") or "", MAX_REVISED_ACTION_CHARS)

    def review(self, *, clip: Path, segment: dict[str, Any], roster: list[dict[str, str]]) -> SegmentReview:
        local_roster = segment_roster(roster, segment)
        current_present = [str(item) for item in segment.get("present_entity_ids") or []]
        seed_present = [
            str(item)
            for item in (
                segment.get("_seed_present_entity_ids")
                or segment.get("present_entity_ids")
                or []
            )
        ]
        roster_text = json.dumps(
            _roster_prompt_view(
                local_roster,
                seed_claimed_ids=seed_present,
                current_ids=current_present,
            ),
            ensure_ascii=False,
        )
        prior_risks = [str(item) for item in (segment.get("_prior_risk_reasons") or [])]
        must_names = list(segment.get("_missing_canonical_names") or [])
        round_idx = int(segment.get("_review_round") or 1)
        review_focus = str(segment.get("_review_focus") or "full")
        focus_instruction = {
            "edit_action": (
                "本轮只修订 action：保持 current_present_proposal 不变，集中核对动作事实并自然写入"
                "全部 canonical name。\n"
            ),
            "edit_present": (
                "本轮只核验 presence：逐个候选判断 present/absent/uncertain；保留已有 action 的"
                "事实，不做无关改写。\n"
            ),
            "retry": "上一轮是运行时错误；重新完整审核，不沿用错误结论。\n",
        }.get(review_focus, "")
        prompt = (
            "你在审核一个视频 benchmark segment。只能依据当前视频片段与给定 roster。\n"
            f"review_focus={review_focus}\n"
            f"{focus_instruction}"
            "seed_claimed_present 来自上一阶段对同一片段的完整视频标注，是强先验但不是绝对真值；"
            "不要把它当成随意候选。只有当前片段提供明确反证时才能删除 seed_claimed 实体。"
            "presence_window_overlap 只是弱候选，只有画面明确支持时才能新增。不要发明 roster 外 ID。\n"
            "必须按以下顺序完成：\n"
            "1) 先核验 seed_claimed_present：若画面、原 action 或场景外观支持该实体，必须保留；"
            "证据略模糊但没有明确反证时也保留并降低 confidence，不能直接清空。尤其禁止在 action"
            " 明确描述“大兔子/紫色小鸟/兔子洞穴”等 roster 实体时返回空 revised_present。"
            "再检查 overlap 候选，只新增画面中明确存在者。\n"
            "2) 再重写 revised_action，使每个最终 present 的 character / prop / location 都以 roster"
            " 中完整的 canonical name 自然出现在动作句中。先在原 action 中寻找对应普通名词，确认是"
            "同一实体后原位替换；例如原文“兔子”对应且画面确认“大兔子”时，改成“大兔子”，"
            "不能在句尾补“大兔子”。location 应写成自然地点状语，例如“开阔草地上，……”。\n"
            "canonical name 必须从 roster 的 name 字段逐字复制，不能改成近义词、简称、别名或 emoji；"
            "例如必须写“大兔子”，不能写“大白兔”“兔子”或“🐰”。\n"
            "3) revised_action 必须是一句自包含的剧本动作行。禁止追加“可见……”“出场：……”"
            "“showing ...”或任何实体名单式尾巴；也禁止把未确认候选写进 action。\n"
            "4) 仅当 revised_present 与视频一致、其全部 canonical name 已自然进入 revised_action"
            "时设 accepted=true；否则设 accepted=false。\n"
            "输出约束（重要，必须遵守）：\n"
            f"- revised_action ≤ {MAX_REVISED_ACTION_CHARS} 字，使用 1–3 个紧凑短句，写清主要动作、"
            "实体间关系与空间进展；不要复述实体外观、服装、颜色等身份属性（首次出现的外观由后续首现"
            "阶段单独补写）；\n"
            f"- risk_reasons 最多 {MAX_RISK_REASONS} 条，每条 ≤ {MAX_RISK_REASON_CHARS} 字，用短标签"
            "（如 present_mismatch / action_vague）；\n"
            f"- suggestions 最多 {MAX_SUGGESTIONS} 条，每条 ≤ {MAX_SUGGESTION_CHARS} 字；\n"
            "- 禁止在任何字段里写长段剧情或重复粘贴 action。\n"
            "不要创建 roster 之外的 ID，不要输出资产来源、crop、bbox、评分信息。\n"
            f"segment_id={segment['segment_id']}\n"
            f"review_round={round_idx}\n"
            f"start_seconds={segment.get('start_seconds')}\n"
            f"end_seconds={segment.get('end_seconds')}\n"
            f"seed_claimed_present={json.dumps(seed_present, ensure_ascii=False)}\n"
            f"current_present_proposal={json.dumps(current_present, ensure_ascii=False)}\n"
            f"action={_clip_text(segment.get('action', ''), MAX_REVISED_ACTION_CHARS)}\n"
            f"prior_risk_reasons={json.dumps(_clip_str_list(prior_risks, max_chars=MAX_RISK_REASON_CHARS, max_items=MAX_RISK_REASONS), ensure_ascii=False)}\n"
            f"roster={roster_text}\n"
            f"required_exact_names={format_missing_names_for_prompt(must_names)}\n"
            "最终检查：逐个确认 required_exact_names 中每个 name 都原样出现在 revised_action；"
            "若列表为空，则逐个检查 revised_present 对应 roster.name。缺少任何一个都必须先重写，"
            "不得用 emoji 或近义词替代。"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "accepted",
                "revised_present",
                "revised_action",
                "confidence",
                "risk_reasons",
                "suggestions",
            ],
            "properties": {
                "accepted": {"type": "boolean"},
                "revised_present": {"type": "array", "items": {"type": "string"}},
                "revised_action": {"type": "string", "maxLength": MAX_REVISED_ACTION_CHARS},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "risk_reasons": {
                    "type": "array",
                    "maxItems": MAX_RISK_REASONS,
                    "items": {"type": "string", "maxLength": MAX_RISK_REASON_CHARS},
                },
                "suggestions": {
                    "type": "array",
                    "maxItems": MAX_SUGGESTIONS,
                    "items": {"type": "string", "maxLength": MAX_SUGGESTION_CHARS},
                },
            },
        }
        try:
            raw = self._request(clip, prompt, "segment_auto_review", schema)
        except RuntimeError as exc:
            # Truncated / malformed / HTTP errors must not kill the whole movie run.
            err = str(exc)
            tag = _review_error_tag(err)
            return SegmentReview(
                segment_id=str(segment["segment_id"]),
                revised_present=list(dict.fromkeys(current_present)),
                revised_action=str(segment.get("action") or ""),
                confidence="low",
                risk_reasons=[tag],
                raw={"error": err[:800], "accepted": False},
                accepted=False,
                suggestions=[],
            )
        allowed = {item["entity_id"] for item in local_roster}
        proposed_present = [
            entity_id for entity_id in raw.get("revised_present") or [] if entity_id in allowed
        ]
        risk_reasons = _clip_str_list(
            raw.get("risk_reasons") or [],
            max_chars=MAX_RISK_REASON_CHARS,
            max_items=MAX_RISK_REASONS,
        )
        suggestions = _clip_str_list(
            raw.get("suggestions") or [],
            max_chars=MAX_SUGGESTION_CHARS,
            max_items=MAX_SUGGESTIONS,
        )
        revised_action = _clip_text(
            str(raw.get("revised_action") or "").strip() or str(segment.get("action") or ""),
            MAX_REVISED_ACTION_CHARS,
        )
        confidence = str(raw.get("confidence") or "low")
        accepted = bool(raw.get("accepted"))
        revised = list(dict.fromkeys(proposed_present))
        if revised != list(dict.fromkeys(current_present)):
            # S3 may challenge a seed label, but it must never silently mutate it.
            # The proposal stays in the audit and a high-confidence contradiction is
            # routed to S4 for a human decision.
            raw = {
                **dict(raw),
                "proposed_revised_present": revised,
                "presence_change_reason": "requires_human_confirmation",
                "accepted": False,
            }
            revised = list(dict.fromkeys(current_present))
            accepted = False
            risk_reasons = list(
                dict.fromkeys([*risk_reasons, "presence_change_proposed"])
            )
        # Keep raw compact too — long essays previously crashed the token budget mid-JSON.
        raw = {
            **dict(raw),
            "seed_presence_trusted": True,
            "revised_action": revised_action,
            "risk_reasons": risk_reasons,
            "suggestions": suggestions,
        }
        return SegmentReview(
            segment_id=str(segment["segment_id"]),
            revised_present=list(dict.fromkeys(revised)),
            revised_action=revised_action,
            confidence=confidence,
            risk_reasons=risk_reasons,
            raw=raw,
            accepted=accepted,
            suggestions=suggestions,
            endpoints=[self.base_url],
        )

    def review_first_presence(
        self, *, clip: Path, segment: dict[str, Any], entity: dict[str, str]
    ) -> dict[str, Any]:
        confirmed_present = list(segment.get("_confirmed_present_entities") or [])
        prompt = (
            "这是实体 " + str(entity.get("name") or entity.get("entity_id") or "")
            + " 在全片首次出现的 benchmark segment。首次出现是该实体外观唯一进入 action 的时机，"
            "之后再次出现只会点名、不再复述外观。因此本轮任务：检查 action 是否已自然写入足以支持"
            "该实体 description 的可辨识外观线索（体貌、服装、颜色、显著特征等），使读者仅凭本句即可"
            "想象其长相。若不足，给出忠实于视频、自然语言的修订 action，把缺失的外观特征自然融入动作句"
            "（不要写成尾部清单或“外观：……”式补丁）。"
            f"修订后的 action 必须 ≤ {MAX_REVISED_ACTION_CHARS} 字。"
            "如果改写，必须保留 confirmed_present 中每个实体的完整 canonical name，并自然融入句子；"
            "name 必须逐字复制，禁止使用简称、近义词或 emoji；"
            "禁止追加“可见……”“出场：……”或实体名单式尾巴。"
            "不要泄漏资产来源、crop、bbox、评分信息。\n"
            f"entity_id={entity['entity_id']}\nname={entity['name']}\n"
            f"description={entity['description']}\n"
            f"confirmed_present={json.dumps(confirmed_present, ensure_ascii=False)}\n"
            f"action={segment.get('action', '')}"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["description_covered", "missing_visual_attributes", "revised_action"],
            "properties": {
                "description_covered": {"type": "boolean"},
                "missing_visual_attributes": {"type": "array", "items": {"type": "string"}},
                "revised_action": {"type": "string", "maxLength": MAX_REVISED_ACTION_CHARS},
            },
        }
        return self._request(clip, prompt, "first_presence_review", schema)


def _annotation(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept either S2's envelope or a raw v5 annotation."""
    candidate = payload.get("annotation")
    return candidate if isinstance(candidate, dict) else payload


def _segments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        for segment in scene.get("visual_segments") or []:
            output.append(segment)
    return sorted(output, key=lambda item: (float(item["start_seconds"]), str(item["segment_id"])))


def _roster(annotation: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    groups = (("characters", "char_id", "character"), ("props", "prop_id", "prop"),
              ("locations", "loc_id", "location"))
    entries: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for group, id_key, kind in groups:
        for raw in annotation.get(group) or []:
            entity_id = str(raw.get(id_key) or "")
            if not entity_id:
                continue
            entry = {
                "entity_id": entity_id,
                "kind": kind,
                "name": str(raw.get("name") or ""),
                "description": str(raw.get("description") or ""),
                "first_presence_seconds": str(raw.get("first_presence_seconds") or ""),
                "last_presence_seconds": str(raw.get("last_presence_seconds") or ""),
            }
            entries.append(entry)
            by_id[entity_id] = entry
    return entries, by_id


def _roster_prompt_view(
    roster: list[dict[str, str]],
    *,
    seed_claimed_ids: list[str] | None = None,
    current_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """Drop timing fields while distinguishing candidate source from evidence."""
    seed_claimed = {str(item) for item in (seed_claimed_ids or [])}
    current = {str(item) for item in (current_ids or [])}
    return [
        {
            "entity_id": item["entity_id"],
            "kind": item.get("kind", ""),
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "candidate_reason": (
                "seed_claimed"
                if item["entity_id"] in seed_claimed
                else (
                    "current_proposal"
                    if item["entity_id"] in current
                    else "presence_window_overlap"
                )
            ),
        }
        for item in roster
    ]


def segment_roster(
    roster: list[dict[str, str]],
    segment: dict[str, Any],
    *,
    pad_seconds: float = SEGMENT_ROSTER_PAD_SECONDS,
) -> list[dict[str, str]]:
    """Keep claimed present IDs plus entities whose presence overlaps the segment window."""
    claimed = {
        str(item)
        for item in (
            list(segment.get("_seed_present_entity_ids") or [])
            + list(segment.get("present_entity_ids") or [])
        )
        if str(item)
    }
    try:
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return [item for item in roster if item.get("entity_id") in claimed] or list(roster)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in roster:
        entity_id = str(item.get("entity_id") or "")
        if not entity_id or entity_id in seen:
            continue
        if entity_id in claimed:
            selected.append(item)
            seen.add(entity_id)
            continue
        try:
            first = float(item.get("first_presence_seconds") or "")
            last = float(item.get("last_presence_seconds") or "")
        except ValueError:
            continue
        if first - pad_seconds <= end and last + pad_seconds >= start:
            selected.append(item)
            seen.add(entity_id)
    if not selected:
        # Safety fallback: never send an empty candidate set to the reviewer.
        return list(roster)
    return selected


def _apply_review(segment: dict[str, Any], review: SegmentReview) -> None:
    segment["present_entity_ids"] = list(review.revised_present)
    segment["action"] = review.revised_action


def _repair_action_with_trusted_names(
    review: SegmentReview,
    *,
    trusted_present_entity_ids: list[str],
    roster_by_id: dict[str, dict[str, str]],
    repairer: QwenVideoReviewer,
    fallback_action: str = "",
    max_attempts: int = DEFAULT_MAX_ACTION_REPAIR_ATTEMPTS,
) -> SegmentReview:
    """Use a text-only repair only when a trusted label is absent from prose."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    missing = missing_canonical_names(
        action=review.revised_action,
        present_entity_ids=trusted_present_entity_ids,
        roster_by_id=roster_by_id,
    )
    repairable = [item for item in missing if item.get("name")]
    if not repairable:
        return review
    if fallback_action and not missing_canonical_names(
        action=fallback_action,
        present_entity_ids=trusted_present_entity_ids,
        roster_by_id=roster_by_id,
    ):
        review.revised_action = fallback_action
        review.raw = {
            **dict(review.raw),
            "text_action_repair": {
                "status": "preserved_prior",
                "candidate": fallback_action,
            },
        }
        return review
    # A common compact-action pattern is “甲打招呼” while a trusted second
    # character is present. This rewrite is factual under the seed contract,
    # preserves the original predicate, and avoids an unreliable extra VLM
    # round for the exact `seg_0011` failure shape.
    if len(repairable) == 1 and repairable[0].get("kind") == "character" and "打招呼" in review.revised_action:
        candidate = review.revised_action.replace(
            "打招呼",
            f"向{repairable[0]['name']}打招呼",
            1,
        )
        if not missing_canonical_names(
            action=candidate,
            present_entity_ids=trusted_present_entity_ids,
            roster_by_id=roster_by_id,
        ):
            review.revised_action = candidate
            review.raw = {
                **dict(review.raw),
                "text_action_repair": {
                    "status": "deterministic",
                    "required_entities": repairable,
                    "candidate": candidate,
                },
            }
            return review
    required = [
        {
            "entity_id": entity_id,
            "name": str((roster_by_id.get(entity_id) or {}).get("name") or ""),
            "kind": str((roster_by_id.get(entity_id) or {}).get("kind") or ""),
        }
        for entity_id in dict.fromkeys(trusted_present_entity_ids)
        if str((roster_by_id.get(entity_id) or {}).get("name") or "")
    ]
    attempts: list[dict[str, Any]] = []
    retry_feedback = ""
    for attempt in range(1, max_attempts + 1):
        try:
            candidate = repairer.repair_action(
                action=review.revised_action,
                required_entities=required,
                retry_feedback=retry_feedback,
            )
        except RuntimeError as exc:
            attempts.append({"attempt": attempt, "error": str(exc)[:800]})
            retry_feedback = "上次请求失败；请严格输出只含 revised_action 的合法 JSON。"
            continue

        # Deterministically complete a near-miss candidate: the VLM reliably
        # adds the missing character but frequently shortens or paraphrases a
        # location/generic mention (e.g. "郊区街道"→"街道"), which would
        # otherwise fail the name gate and reject an otherwise-good rewrite.
        # The same canonicalizer that protects the seed finishes the job
        # without another unreliable VLM round. Adopt only when it fully clears
        # coverage AND the text is clean (a garbled VLM candidate must not be
        # promoted to gold just because it happens to contain every name).
        if candidate and not action_has_entity_list_coda(candidate):
            completed = try_complete_canonical_action(
                action=candidate,
                present_entity_ids=trusted_present_entity_ids,
                roster_by_id=roster_by_id,
            )
            if completed:
                candidate = completed

        invalid_reasons: list[str] = []
        if not candidate:
            invalid_reasons.append("empty_action")
        if action_has_entity_list_coda(candidate):
            invalid_reasons.append(ACTION_ENTITY_LIST_CODA)
        remaining = missing_canonical_names(
            action=candidate,
            present_entity_ids=trusted_present_entity_ids,
            roster_by_id=roster_by_id,
        )
        if remaining:
            invalid_reasons.append(ACTION_MISSING_CANONICAL_NAME)
        attempts.append(
            {
                "attempt": attempt,
                "candidate": candidate,
                "reasons": invalid_reasons,
                "missing_names": [str(item.get("name") or "") for item in remaining],
            }
        )
        if not invalid_reasons:
            review.revised_action = candidate
            review.raw = {
                **dict(review.raw),
                "text_action_repair": {
                    "status": "accepted",
                    "required_entities": required,
                    "candidate": candidate,
                    "attempts": attempts,
                },
            }
            return review
        retry_feedback = (
            "上次候选未通过确定性校验。必须逐字包含以下缺失名称："
            f"{json.dumps([str(item.get('name') or '') for item in remaining], ensure_ascii=False)}。"
            "不得删除已有 required_entities 中的名称。"
        )
    review.raw = {
        **dict(review.raw),
        "text_action_repair": {
            "status": "rejected",
            "required_entities": required,
            "candidate": attempts[-1].get("candidate", "") if attempts else "",
            "attempts": attempts,
            "reasons": attempts[-1].get("reasons", ["request_failed"]) if attempts else ["request_failed"],
        },
    }
    return review


def review_segment_until_accepted(
    *,
    segment: dict[str, Any],
    roster: list[dict[str, str]],
    source_video: Path,
    cache_root: Path,
    worker_id: str,
    reviewer: SegmentReviewer,
    pool: ReviewerEndpointPool | None = None,
    max_review_rounds: int = DEFAULT_MAX_REVIEW_ROUNDS,
    template_reviewer: QwenVideoReviewer | None = None,
    clip_semaphore: threading.Semaphore | None = None,
    prefetched_clip: Path | None = None,
) -> SegmentReview:
    """Review one segment, revising and re-auditing until accept or round budget."""
    if max_review_rounds < 1:
        raise ValueError("max_review_rounds must be >= 1")

    rounds: list[dict[str, Any]] = []
    endpoints: list[str] = []
    last: SegmentReview | None = None
    prefer_not: str | None = None
    working = dict(segment)
    seed_present = [
        str(item) for item in (segment.get("present_entity_ids") or [])
    ]
    working["_seed_present_entity_ids"] = seed_present
    canonical_action, deterministic_rewrites, _still_missing = (
        rewrite_action_canonical_mentions(
            action=str(segment.get("action") or ""),
            present_entity_ids=seed_present,
            roster_by_id=_roster_by_id(roster),
        )
    )
    working["action"] = canonical_action
    started = time.perf_counter()
    queue_seconds = 0.0
    clip_seconds = 0.0
    vlm_request_seconds = 0.0
    has_vlm_request_timing = False

    @contextmanager
    def _round_clip(round_idx: int, timing: dict[str, float]) -> Iterator[Path]:
        if prefetched_clip is not None:
            yield prefetched_clip
            return
        with worker_clip(
            source_video=source_video,
            cache_root=cache_root,
            worker_id=f"{worker_id}-r{round_idx}",
            start_seconds=float(working["start_seconds"]),
            end_seconds=float(working["end_seconds"]),
            cut_semaphore=clip_semaphore,
            timing=timing,
        ) as clip:
            yield clip

    for round_idx in range(1, max_review_rounds + 1):
        working["_review_round"] = round_idx
        if last is not None:
            working["_prior_risk_reasons"] = list(last.risk_reasons)

        if pool is not None and template_reviewer is not None:
            # Cut the local clip before leasing a GPU endpoint. The prior order
            # held a scarce endpoint during ffmpeg work, inflating queue time.
            clip_timing: dict[str, float] = {}
            with _round_clip(round_idx, clip_timing) as clip:
                round_clip_seconds = clip_timing.get("encode_seconds", 0.0)
                clip_seconds += round_clip_seconds
                round_queue_seconds = clip_timing.get("queue_seconds", 0.0)
                queue_seconds += round_queue_seconds
                queue_started = time.perf_counter()
                with pool.lease(
                    prefer_not=prefer_not,
                    workload={
                        "segment_id": str(working.get("segment_id") or ""),
                        "stage": "s3_segment_auto_review_revise",
                        "round": round_idx,
                    },
                ) as lease:
                    endpoint_queue_seconds = time.perf_counter() - queue_started
                    round_queue_seconds += endpoint_queue_seconds
                    queue_seconds += endpoint_queue_seconds
                    active = template_reviewer.with_base_url(lease.base_url)
                    endpoints.append(lease.base_url)
                    prefer_not = lease.base_url
                    request_started = time.perf_counter()
                    last = active.review(clip=clip, segment=working, roster=roster)
                    video_request_seconds = (
                        active.last_request_seconds
                        if active.last_request_seconds is not None
                        else time.perf_counter() - request_started
                    )
                    last = _repair_action_with_trusted_names(
                        last,
                        trusted_present_entity_ids=seed_present,
                        roster_by_id=_roster_by_id(roster),
                        repairer=active,
                        fallback_action=str(working.get("action") or ""),
                    )
                    repair_status = str(
                        (last.raw.get("text_action_repair") or {}).get("status") or ""
                    )
                    repair_request_seconds = (
                        active.last_request_seconds
                        if repair_status in {"accepted", "rejected", "failed"}
                        else 0.0
                    )
                    round_vlm_request_seconds = video_request_seconds + repair_request_seconds
                    vlm_request_seconds += round_vlm_request_seconds
                    has_vlm_request_timing = True
        elif isinstance(reviewer, QwenVideoReviewer):
            endpoints.append(reviewer.base_url)
            prefer_not = reviewer.base_url
            clip_timing = {}
            with _round_clip(round_idx, clip_timing) as clip:
                round_clip_seconds = clip_timing.get("encode_seconds", 0.0)
                clip_seconds += round_clip_seconds
                round_queue_seconds = clip_timing.get("queue_seconds", 0.0)
                queue_seconds += round_queue_seconds
                request_started = time.perf_counter()
                last = reviewer.review(clip=clip, segment=working, roster=roster)
                video_request_seconds = (
                    reviewer.last_request_seconds
                    if reviewer.last_request_seconds is not None
                    else time.perf_counter() - request_started
                )
                last = _repair_action_with_trusted_names(
                    last,
                    trusted_present_entity_ids=seed_present,
                    roster_by_id=_roster_by_id(roster),
                    repairer=reviewer,
                    fallback_action=str(working.get("action") or ""),
                )
                repair_status = str(
                    (last.raw.get("text_action_repair") or {}).get("status") or ""
                )
                repair_request_seconds = (
                    reviewer.last_request_seconds
                    if repair_status in {"accepted", "rejected", "failed"}
                    else 0.0
                )
                round_vlm_request_seconds = video_request_seconds + repair_request_seconds
                vlm_request_seconds += round_vlm_request_seconds
                has_vlm_request_timing = True
        else:
            # Passthrough / unit-test reviewers: no media dependency.
            last = reviewer.review(clip=Path(), segment=working, roster=roster)
            endpoints.append(getattr(reviewer, "base_url", "local-reviewer"))
            round_queue_seconds = None
            round_clip_seconds = None
            round_vlm_request_seconds = None

        assert last is not None
        finalize = round_idx >= max_review_rounds
        last = _apply_canonical_name_gate(
            last,
            roster=roster,
            previous_action=str(working.get("action") or ""),
        )
        last = _apply_typed_verdict(last)
        rounds.append({
            "round": round_idx,
            "endpoint": endpoints[-1],
            "accepted": last.accepted,
            "confidence": last.confidence,
            "revised_present": list(last.revised_present),
            "revised_action": last.revised_action,
            "risk_reasons": list(last.risk_reasons),
            "suggestions": list(last.suggestions),
            "verdict": last.verdict,
            "findings": list(last.findings),
            "recommended_action": last.recommended_action,
            "queue_seconds": (
                round(round_queue_seconds, 2)
                if round_queue_seconds is not None
                else None
            ),
            "clip_seconds": (
                round(round_clip_seconds, 2)
                if round_clip_seconds is not None
                else None
            ),
            "vlm_request_seconds": (
                round(round_vlm_request_seconds, 2)
                if round_vlm_request_seconds is not None
                else None
            ),
            "vlm_inference_seconds": None,
            "raw": last.raw,
        })
        working["present_entity_ids"] = list(last.revised_present)
        working["action"] = last.revised_action
        working["_missing_canonical_names"] = list(
            (last.raw or {}).get("missing_canonical_names") or []
        )
        working["_review_focus"] = last.recommended_action
        if last.accepted or finalize:
            break

    assert last is not None
    if not last.accepted:
        last.risk_reasons = list(dict.fromkeys([*last.risk_reasons, "max_review_rounds_exhausted"]))
    last.rounds = rounds
    last.n_rounds = len(rounds)
    last.endpoints = endpoints
    last.elapsed_seconds = round(time.perf_counter() - started, 2)
    if has_vlm_request_timing:
        last.queue_seconds = round(queue_seconds, 2)
        last.clip_seconds = round(clip_seconds, 2)
        last.vlm_request_seconds = round(vlm_request_seconds, 2)
    last.raw = {
        **dict(last.raw),
        "rounds": rounds,
        "n_rounds": len(rounds),
        "final_accepted": last.accepted,
        "elapsed_seconds": last.elapsed_seconds,
        "queue_seconds": last.queue_seconds,
        "clip_seconds": last.clip_seconds,
        "vlm_request_seconds": last.vlm_request_seconds,
        "vlm_inference_seconds": last.vlm_inference_seconds,
        "deterministic_action_rewrites": deterministic_rewrites,
    }
    return last


def _write_progress(
    stage_dir: Path,
    *,
    status: str,
    done: int,
    total: int,
    started: float,
    latest: dict[str, Any] | None = None,
    phase: str = "segments",
) -> None:
    payload = {
        "status": status,
        "phase": phase,
        "done": done,
        "total": total,
        "elapsed_seconds": round(time.time() - started, 2),
        "updated_at": now_beijing(),
        "latest": latest or {},
    }
    (stage_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _append_audit(stage_dir: Path, review: SegmentReview, *, lock: threading.Lock) -> None:
    line = json.dumps(review.to_dict(), ensure_ascii=False) + "\n"
    with lock:
        with (stage_dir / "segment_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _write_live_annotation(stage_dir: Path, annotation: dict[str, Any], *, lock: threading.Lock) -> None:
    text = json.dumps(annotation, ensure_ascii=False, indent=2)
    with lock:
        (stage_dir / "auto_revised_annotation.json").write_text(text, encoding="utf-8")


@dataclass(slots=True)
class _PrefetchedClip:
    event: threading.Event = field(default_factory=threading.Event)
    path: Path | None = None
    error: BaseException | None = None


class _ClipPrefetcher:
    """Bounded clip producer that feeds independent VLM consumer workers."""

    def __init__(
        self,
        *,
        segments: list[dict[str, Any]],
        source_video: Path,
        cache_root: Path,
        cut_workers: int,
        buffer_size: int,
        shared_queue: SharedClipQueue | None = None,
    ) -> None:
        if cut_workers < 1 or buffer_size < 1:
            raise ValueError("cut_workers and buffer_size must be >= 1")
        self._segments = list(segments)
        self._source_video = source_video
        self._cache_root = cache_root / "prefetched"
        self._cut_workers = cut_workers
        self._shared_queue = shared_queue
        self._slots = threading.BoundedSemaphore(buffer_size)
        self._entries = {
            str(segment["segment_id"]): _PrefetchedClip()
            for segment in self._segments
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._produce, name="s3-clip-prefetch", daemon=True)
        self._thread.start()

    def _produce_one(self, segment: dict[str, Any], entry: _PrefetchedClip) -> None:
        segment_id = str(segment["segment_id"])
        try:
            ready_clip = self._cache_root / segment_id / "segment.mp4"
            if ready_clip.is_file() and ready_clip.stat().st_size > 0:
                entry.path = ready_clip
                return
            if self._shared_queue is not None:
                task_id = "clip-" + hashlib.sha256(str(ready_clip).encode("utf-8")).hexdigest()[:40]
                self._shared_queue.enqueue(
                    ClipTask(
                        task_id=task_id,
                        source_video=self._source_video,
                        output_path=ready_clip,
                        start_seconds=float(segment["start_seconds"]),
                        end_seconds=float(segment["end_seconds"]),
                        metadata={
                            "segment_id": segment_id,
                            "stage": "s3_segment_auto_review_revise",
                        },
                    )
                )
                entry.path = self._shared_queue.wait_for_ready(task_id)
                return
            timing: dict[str, float] = {}
            with worker_clip(
                source_video=self._source_video,
                cache_root=self._cache_root,
                worker_id=segment_id,
                start_seconds=float(segment["start_seconds"]),
                end_seconds=float(segment["end_seconds"]),
                timing=timing,
                remove_on_exit=False,
            ) as clip:
                entry.path = clip
        except BaseException as exc:  # noqa: BLE001 - surfaced to the matching consumer
            entry.error = exc
        finally:
            entry.event.set()

    def _produce(self) -> None:
        with ThreadPoolExecutor(max_workers=self._cut_workers) as executor:
            futures = []
            for segment in self._segments:
                while not self._slots.acquire(timeout=0.2):
                    if self._stop.is_set():
                        return
                if self._stop.is_set():
                    self._slots.release()
                    return
                entry = self._entries[str(segment["segment_id"])]
                futures.append(executor.submit(self._produce_one, segment, entry))
            for future in futures:
                future.result()

    @contextmanager
    def lease(self, segment_id: str) -> Iterator[Path]:
        entry = self._entries[str(segment_id)]
        entry.event.wait()
        try:
            if entry.error is not None:
                raise entry.error
            if entry.path is None:
                raise RuntimeError(f"prefetch produced no clip for {segment_id}")
            yield entry.path
        finally:
            if entry.path is not None:
                entry.path.unlink(missing_ok=True)
            self._slots.release()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)


def run_auto_review(
    *,
    annotation_payload: dict[str, Any],
    source_video: Path,
    stage_dir: Path,
    reviewer: SegmentReviewer,
    worker_id: str = "worker-0",
    max_segments: int | None = None,
    max_review_rounds: int = DEFAULT_MAX_REVIEW_ROUNDS,
    endpoint_pool: ReviewerEndpointPool | None = None,
    max_workers: int | None = None,
    max_clip_workers: int | None = None,
    clip_queue_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Review every segment and write a derived annotation without changing S1."""
    annotation = copy.deepcopy(_annotation(annotation_payload))
    roster, roster_by_id = _roster(annotation)
    cache_root = stage_dir / "clip_cache"
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    segments = _segments(annotation)
    if max_segments is not None:
        segments = segments[:max_segments]
    segment_by_id = {str(item["segment_id"]): item for item in segments}
    reviews_by_id: dict[str, SegmentReview] = {}
    first_reviews: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    io_lock = threading.Lock()

    template = reviewer if isinstance(reviewer, QwenVideoReviewer) else None
    pool = endpoint_pool
    if pool is None and isinstance(reviewer, QwenVideoReviewer):
        pool = ReviewerEndpointPool([reviewer.base_url])
    if pool is not None:
        # movie_dir/tmp/pipeline/s3_... → parents[2] = movie_dir
        try:
            movie_id = stage_dir.resolve().parents[2].name
        except IndexError:
            movie_id = ""
        pool._default_workload = {
            **getattr(pool, "_default_workload", {}),
            "movie_id": movie_id,
            "stage": "s3_segment_auto_review_revise",
        }

    workers = 1
    if pool is not None:
        workers = max(1, min(pool.size, max_workers or pool.size))
    if max_clip_workers is not None and max_clip_workers < 1:
        raise ValueError("max_clip_workers must be >= 1")
    clip_workers = min(
        workers,
        max_clip_workers if max_clip_workers is not None else DEFAULT_MAX_CLIP_WORKERS,
    )
    clip_semaphore = threading.BoundedSemaphore(clip_workers)

    audit_path = stage_dir / "segment_audit.jsonl"
    if resume and audit_path.is_file():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or not item.get("segment_id"):
                continue
            review = SegmentReview(
                segment_id=str(item["segment_id"]),
                revised_present=[str(x) for x in (item.get("revised_present") or [])],
                revised_action=str(item.get("revised_action") or ""),
                confidence=str(item.get("confidence") or "low"),
                risk_reasons=list(item.get("risk_reasons") or []),
                raw=dict(item.get("raw") or item),
                accepted=bool(item.get("accepted")),
                suggestions=list(item.get("suggestions") or []),
                rounds=list(item.get("rounds") or []),
                n_rounds=int(item.get("n_rounds") or 1),
                endpoints=list(item.get("endpoints") or []),
                elapsed_seconds=_optional_elapsed(item.get("elapsed_seconds")),
                verdict=str(item.get("verdict") or PASS),
                findings=list(item.get("findings") or []),
                recommended_action=str(item.get("recommended_action") or "none"),
            )
            reviews_by_id[review.segment_id] = review
            target = segment_by_id.get(review.segment_id)
            if target is not None:
                _apply_review(target, review)
        pending = [seg for seg in segments if str(seg["segment_id"]) not in reviews_by_id]
        _write_progress(
            stage_dir,
            status="running",
            done=len(reviews_by_id),
            total=len(segments),
            started=started,
            phase="segments",
        )
        _write_live_annotation(stage_dir, annotation, lock=io_lock)
    else:
        # Live artifacts: truncate audit and publish empty progress up front.
        audit_path.write_text("", encoding="utf-8")
        pending = list(segments)
        _write_progress(stage_dir, status="running", done=0, total=len(segments), started=started)
        _write_live_annotation(stage_dir, annotation, lock=io_lock)

    # Zero-duration / empty boundary segments cannot be cut into a clip; pull
    # them out of the VLM pending set and auto-PASS them below (deterministic).
    zero_duration_pending = [seg for seg in pending if is_zero_duration_segment(seg)]
    if zero_duration_pending:
        pending = [seg for seg in pending if not is_zero_duration_segment(seg)]

    shared_clip_queue = SharedClipQueue(clip_queue_root) if clip_queue_root is not None else None
    prefetcher: _ClipPrefetcher | None = None
    if template is not None and pending:
        prefetcher = _ClipPrefetcher(
            segments=pending,
            source_video=source_video,
            cache_root=cache_root,
            cut_workers=clip_workers,
            # Keep enough ready-or-inflight clips for all VLM consumers while
            # bounding shared-FS cache use to one small movie-local window.
            buffer_size=max(workers, clip_workers * 3),
            shared_queue=shared_clip_queue,
        )
        prefetcher.start()

    def _one(segment: dict[str, Any], index: int) -> SegmentReview:
        local_worker = f"{worker_id}-{index}"
        review_started = time.perf_counter()
        try:
            clip_context = (
                prefetcher.lease(str(segment["segment_id"]))
                if prefetcher is not None
                else nullcontext(None)
            )
            with clip_context as prefetched_clip:
                return review_segment_until_accepted(
                    segment=segment,
                    roster=roster,
                    source_video=source_video,
                    cache_root=cache_root,
                    worker_id=local_worker,
                    reviewer=reviewer,
                    pool=pool if template is not None else None,
                    max_review_rounds=max_review_rounds,
                    template_reviewer=template,
                    clip_semaphore=clip_semaphore,
                    prefetched_clip=prefetched_clip,
                )
        except Exception as exc:  # noqa: BLE001 - keep movie-level run alive
            crashed = SegmentReview(
                segment_id=str(segment["segment_id"]),
                revised_present=[str(x) for x in (segment.get("present_entity_ids") or [])],
                revised_action=str(segment.get("action") or ""),
                confidence="low",
                risk_reasons=["segment_review_crashed"],
                raw={"error": f"{type(exc).__name__}: {exc}"[:800]},
                accepted=False,
                suggestions=[],
                elapsed_seconds=round(time.perf_counter() - review_started, 2),
            )
            return _apply_typed_verdict(
                _apply_canonical_name_gate(
                    crashed,
                    roster=roster,
                    previous_action=str(segment.get("action") or ""),
                )
            )

    def _commit(review: SegmentReview, segment: dict[str, Any]) -> None:
        _apply_review(segment, review)
        reviews_by_id[review.segment_id] = review
        _append_audit(stage_dir, review, lock=io_lock)
        _write_live_annotation(stage_dir, annotation, lock=io_lock)
        _write_progress(
            stage_dir,
            status="running",
            done=len(reviews_by_id),
            total=len(segments),
            started=started,
            latest=review.to_dict(),
            phase="segments",
        )

    # Commit the deterministic auto-PASS reviews for zero-duration boundary
    # segments before the VLM loop so they never enter the clip/VLM path.
    for segment in zero_duration_pending:
        review = _auto_pass_zero_duration(segment)
        _commit(review, segment_by_id[review.segment_id])

    try:
        if workers == 1 or isinstance(reviewer, PassthroughReviewer):
            for index, segment in enumerate(pending):
                review = _one(segment, index)
                _commit(review, segment)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_one, segment, index): str(segment["segment_id"])
                    for index, segment in enumerate(pending)
                }
                for future in as_completed(futures):
                    review = future.result()
                    with lock:
                        target = segment_by_id[review.segment_id]
                        _commit(review, target)

        reviews = [reviews_by_id[str(item["segment_id"])] for item in segments]

        first_segment: dict[str, dict[str, Any]] = {}
        for segment in _segments(annotation):
            for entity_id in segment.get("present_entity_ids") or []:
                first_segment.setdefault(str(entity_id), segment)

        first_reviews = {}
        _write_progress(
            stage_dir,
            status="running",
            done=len(reviews),
            total=len(segments),
            started=started,
            phase="first_presence",
        )

        def _first_unchecked(
            entity_id: str,
            segment: dict[str, Any],
        ) -> tuple[str, dict[str, Any]]:
            entity = roster_by_id.get(entity_id)
            if entity is None:
                return entity_id, {}
            review_segment = dict(segment)
            review_segment["_confirmed_present_entities"] = [
                {
                    "entity_id": present_id,
                    "kind": roster_by_id[present_id].get("kind", ""),
                    "name": roster_by_id[present_id].get("name", ""),
                }
                for present_id in (segment.get("present_entity_ids") or [])
                if present_id in roster_by_id
            ]
            if isinstance(reviewer, PassthroughReviewer):
                return entity_id, reviewer.review_first_presence(
                    clip=Path(), segment=review_segment, entity=entity
                )
            if pool is not None and template is not None:
                with worker_clip(
                    source_video=source_video,
                    cache_root=cache_root,
                    worker_id=f"{worker_id}-first-{entity_id}",
                    start_seconds=float(segment["start_seconds"]),
                    end_seconds=float(segment["end_seconds"]),
                    cut_semaphore=clip_semaphore,
                ) as clip:
                    with pool.lease(
                        workload={
                            "segment_id": str(segment.get("segment_id") or ""),
                            "entity_id": entity_id,
                            "stage": "s3_first_presence",
                        }
                    ) as lease:
                        active = template.with_base_url(lease.base_url)
                        return entity_id, active.review_first_presence(
                            clip=clip, segment=review_segment, entity=entity
                        )
            with worker_clip(
                source_video=source_video,
                cache_root=cache_root,
                worker_id=f"{worker_id}-first-{entity_id}",
                start_seconds=float(segment["start_seconds"]),
                end_seconds=float(segment["end_seconds"]),
                cut_semaphore=clip_semaphore,
            ) as clip:
                return entity_id, reviewer.review_first_presence(
                    clip=clip, segment=review_segment, entity=entity
                )

        def _first(entity_id: str, segment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            try:
                return _first_unchecked(entity_id, segment)
            except Exception as exc:  # noqa: BLE001 - one endpoint must not fail the movie
                return entity_id, {
                    "description_covered": False,
                    "missing_visual_attributes": [],
                    "revised_action": "",
                    "risk_reasons": ["first_presence_review_failed"],
                    "error": f"{type(exc).__name__}: {exc}"[:800],
                }

        def _safe_first_revision(segment: dict[str, Any], result: dict[str, Any]) -> str:
            proposed = _clip_text(result.get("revised_action") or "", MAX_REVISED_ACTION_CHARS)
            if not proposed:
                return ""
            risks: list[str] = []
            if action_has_entity_list_coda(proposed):
                risks.append(ACTION_ENTITY_LIST_CODA)
            missing = missing_canonical_names(
                action=proposed,
                present_entity_ids=[
                    str(item) for item in (segment.get("present_entity_ids") or [])
                ],
                roster_by_id=roster_by_id,
            )
            if any(item.get("name") for item in missing):
                risks.append(ACTION_MISSING_CANONICAL_NAME)
            if any(not item.get("name") for item in missing):
                risks.append(ENTITY_EMPTY_CANONICAL_NAME)
            if risks:
                result["rejected_revised_action"] = proposed
                result["revision_risk_reasons"] = list(dict.fromkeys(risks))
                result["revised_action"] = ""
                return ""
            result["revised_action"] = proposed
            return proposed

        first_items = list(first_segment.items())
        if workers == 1 or isinstance(reviewer, PassthroughReviewer) or not first_items:
            for entity_id, segment in first_items:
                entity_id, result = _first(entity_id, segment)
                if not result:
                    continue
                first_reviews[entity_id] = result
                revised = _safe_first_revision(segment, result)
                if revised:
                    segment["action"] = revised
                (stage_dir / "first_presence_audit.json").write_text(
                    json.dumps(first_reviews, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _write_live_annotation(stage_dir, annotation, lock=io_lock)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_first, entity_id, segment)
                    for entity_id, segment in first_items
                ]
                for future in as_completed(futures):
                    entity_id, result = future.result()
                    if not result:
                        continue
                    with lock:
                        first_reviews[entity_id] = result
                        segment = first_segment[entity_id]
                        revised = _safe_first_revision(segment, result)
                        if revised:
                            segment["action"] = revised
                        (stage_dir / "first_presence_audit.json").write_text(
                            json.dumps(first_reviews, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        _write_live_annotation(stage_dir, annotation, lock=io_lock)
    except Exception as exc:
        _write_progress(
            stage_dir,
            status="failed",
            done=len(reviews_by_id),
            total=len(segments),
            started=started,
            phase="failed",
            latest={"error": f"{type(exc).__name__}: {exc}"[:800]},
        )
        raise
    finally:
        if prefetcher is not None:
            prefetcher.close()
        cleanup_cache(cache_root)

    reviews = [reviews_by_id[str(item["segment_id"])] for item in segments if str(item["segment_id"]) in reviews_by_id]
    # Final rewrite keeps audit ordered by timeline even after parallel completion.
    with (stage_dir / "segment_audit.jsonl").open("w", encoding="utf-8") as handle:
        for review in reviews:
            handle.write(json.dumps(review.to_dict(), ensure_ascii=False) + "\n")
    (stage_dir / "first_presence_audit.json").write_text(
        json.dumps(first_reviews, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_live_annotation(stage_dir, annotation, lock=io_lock)
    n_accepted = sum(1 for review in reviews if review.accepted)
    verdict_counts = {
        verdict: sum(1 for review in reviews if review.verdict == verdict)
        for verdict in sorted(VERDICTS)
    }
    inputs = [{
        "source_video": str(source_video),
        "source_sha256": "",
        "reviewer": type(reviewer).__name__,
        "endpoints": pool.base_urls if pool is not None else [],
        "n_endpoints": pool.size if pool is not None else 0,
        "max_workers": workers,
        "max_clip_workers": clip_workers,
        "max_review_rounds": max_review_rounds,
        "max_tokens": getattr(reviewer, "max_tokens", None),
        "n_segments": len(reviews),
        "n_accepted": n_accepted,
        "verdict_counts": verdict_counts,
        "elapsed_seconds": round(time.time() - started, 2),
    }]
    (stage_dir / "segment_inputs.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in inputs) + "\n", encoding="utf-8"
    )
    (stage_dir / "review_pool_summary.json").write_text(
        json.dumps(inputs[0], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_progress(
        stage_dir,
        status="done",
        done=len(reviews),
        total=len(segments),
        started=started,
        phase="done",
        latest=reviews[-1].to_dict() if reviews else None,
    )
    return {
        "annotation": annotation,
        "reviews": [review.to_dict() for review in reviews],
        "first_reviews": first_reviews,
        "pool_summary": inputs[0],
    }


def build_qwen_reviewer(
    *,
    base_urls: str | list[str],
    model: str = DEFAULT_MODEL,
    timeout_seconds: int = 600,
    fps: float = 2.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    sampling_preset: str | None = None,
) -> tuple[QwenVideoReviewer, ReviewerEndpointPool]:
    """Build a template reviewer plus an endpoint pool from one or more URLs.

    ``sampling_preset`` selects the (critic, fixer) profile pair from
    ``SAMPLING_PRESETS``; when omitted it is auto-picked from ``model``.
    """
    urls = parse_endpoint_urls(base_urls)
    pool = ReviewerEndpointPool(urls)
    preset = sampling_preset or sampling_preset_for_model(model)
    review_sampling, repair_sampling = SAMPLING_PRESETS[preset]
    reviewer = QwenVideoReviewer(
        base_url=urls[0],
        model=model,
        timeout_seconds=timeout_seconds,
        fps=fps,
        max_tokens=max_tokens,
        review_sampling=review_sampling,
        repair_sampling=repair_sampling,
    )
    return reviewer, pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--reviewer", choices=("passthrough", "qwen"), default="passthrough")
    parser.add_argument(
        "--base-url",
        default="",
        help="One URL, or comma-separated reviewer endpoint pool (H800 replicas)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--sampling-preset",
        choices=sorted(SAMPLING_PRESETS),
        default=None,
        help="Per-role (critic/fixer) sampling. Default: auto from --model.",
    )
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-review-rounds", type=int, default=DEFAULT_MAX_REVIEW_ROUNDS)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--max-clip-workers",
        type=int,
        default=None,
        help=f"Concurrent ffmpeg cuts (default: min(endpoint workers, {DEFAULT_MAX_CLIP_WORKERS}))",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing segment_audit.jsonl and only review missing segments",
    )
    args = parser.parse_args()
    pool: ReviewerEndpointPool | None = None
    if args.reviewer == "qwen":
        if not args.base_url:
            parser.error("--base-url is required for --reviewer qwen")
        reviewer, pool = build_qwen_reviewer(
            base_urls=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            fps=args.fps,
            max_tokens=args.max_tokens,
            sampling_preset=args.sampling_preset,
        )
    else:
        reviewer = PassthroughReviewer()
    run_auto_review(
        annotation_payload=json.loads(args.annotation.read_text(encoding="utf-8")),
        source_video=args.source_video,
        stage_dir=args.stage_dir,
        reviewer=reviewer,
        max_segments=args.max_segments,
        max_review_rounds=args.max_review_rounds,
        endpoint_pool=pool,
        max_workers=args.max_workers,
        max_clip_workers=args.max_clip_workers,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
