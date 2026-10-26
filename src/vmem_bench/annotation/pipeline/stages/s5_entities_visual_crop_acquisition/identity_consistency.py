"""S5 per-entity identity-consistency gate (library dedup for WHO, not just WHERE).

DINOv3 similarity alone is a threshold method: it separates cleanly on synthetic
footage (Blender) but collapses on real-distribution movies (LSDMC) where two
different people share clothing/lighting/shot and clear the low re-ID floor. This
gate therefore treats DINOv3 only as a cheap *triage* signal and lets a VLM make
the actual same-entity decision on the crops DINOv3 is unsure about:

1. Embed every accepted library crop of an entity (DINOv3).
2. If the crops form one tight cluster (all close to the medoid), accept them as a
   single identity with no VLM call and no human review — this is the Blender-style
   easy case and is where most of the human-review saving comes from.
3. Otherwise send *all* of the entity's crops to the VLM in one call and ask it to
   pick the dominant identity actually shown and flag every crop that does not match
   it. Flagged crops are rejected (``accepted=False``) so slot binding cannot
   propagate a mixed-identity library.
4. When no VLM auditor is available, ambiguous entities are kept but marked
   ``needs_human`` so review is spent only where DINOv3 could not vouch for identity.

The gate never rejects on DINOv3 evidence alone, and it refuses to reject a whole
entity: if the VLM cannot find a majority identity, the entity is kept and flagged
for a human instead of being silently emptied.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vmem_bench.annotation.pipeline.servers.direct_http import (
    ensure_no_proxy_env,
    ensure_no_proxy_host,
    urlopen_direct,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
    ReviewerEndpointPool,
    parse_endpoint_urls,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
    load_crop_rgb_for_model,
)
from vmem_bench.common.vecmath import cosine_similarity

ensure_no_proxy_env()


@dataclass(slots=True)
class IdentityGateConfig:
    """Tunables for the per-entity identity-consistency gate."""

    # Kinds that carry a single re-identifiable identity. ``location`` is a scene,
    # not an instance, so it is never gated here.
    apply_kinds: tuple[str, ...] = ("character", "prop")
    # DINOv3 triage: skip the VLM only when the entity is unambiguously cohesive.
    skip_vlm_medoid_floor: float = 0.55
    skip_vlm_min_pairwise: float = 0.40
    # Confidence levels at which a VLM "not the same entity" verdict is trusted to
    # reject. Characters are gated harder than props (props tolerate more visual
    # drift and are more expensive to over-reject).
    char_reject_confidences: tuple[str, ...] = ("high", "medium")
    prop_reject_confidences: tuple[str, ...] = ("high",)

    def reject_confidences(self, kind: str) -> tuple[str, ...]:
        if kind == "character":
            return self.char_reject_confidences
        return self.prop_reject_confidences


class IdentityAuditor(Protocol):
    def audit(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        crop_paths: list[Path],
    ) -> dict[str, Any]:
        """Return ``{"dominant_reason": str, "verdicts": [{index, same_entity, confidence, reason}]}``."""


# --- DINOv3 triage ---------------------------------------------------------


def _medoid_cohesion(vecs: list[list[float]]) -> tuple[int, list[float], float]:
    """Return ``(medoid_index, sim_to_medoid_per_crop, min_pairwise_sim)``."""
    n = len(vecs)
    if n == 0:
        return -1, [], 0.0
    if n == 1:
        return 0, [1.0], 1.0
    sims = [[1.0] * n for _ in range(n)]
    min_pairwise = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(vecs[i], vecs[j])
            sims[i][j] = sims[j][i] = s
            min_pairwise = min(min_pairwise, s)
    medoid = max(range(n), key=lambda i: sum(sims[i]))
    return medoid, list(sims[medoid]), min_pairwise


# --- VLM auditor -----------------------------------------------------------


class NullIdentityAuditor:
    """Dry-run auditor: never renders a verdict, forcing ambiguous entities to human."""

    def audit(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        crop_paths: list[Path],
    ) -> dict[str, Any]:
        del name, description, kind, crop_paths
        return {"dominant_reason": "null_auditor", "verdicts": [], "available": False}


class VlmIdentityAuditor:
    """One-shot multi-image same-entity auditor over an OpenAI-compatible VLM.

    All of an entity's crops are sent together; the model selects the dominant
    identity present and judges each crop against it. This is the explicit
    "are these all the same entity?" check the DINOv3-only path lacked.
    """

    def __init__(
        self,
        *,
        base_url: str | list[str],
        model: str,
        timeout_seconds: int = 300,
        max_tokens: int = 2048,
        max_retry_tokens: int = 8192,
        max_retries: int = 3,
    ) -> None:
        self._pool = ReviewerEndpointPool(parse_endpoint_urls(base_url))
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_retry_tokens = max_retry_tokens
        self.max_retries = max_retries
        self.last_result: dict[str, Any] = {}
        for url in self._pool.base_urls:
            ensure_no_proxy_host(url)

    @property
    def base_url(self) -> str:
        return self._pool.base_urls[0]

    @staticmethod
    def _data_url(image: Path) -> str:
        rgb = load_crop_rgb_for_model(image)
        rgb.thumbnail((512, 512))
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def audit(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        crop_paths: list[Path],
    ) -> dict[str, Any]:
        if len(crop_paths) < 2:
            return {"dominant_reason": "single_crop", "verdicts": [], "available": True}
        content: list[dict] = []
        for index, crop in enumerate(crop_paths):
            content.append({"type": "text", "text": f"crop_index={index}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": self._data_url(crop)},
            })
        content.append({
            "type": "text",
            "text": (
                "上面是同一条标注流水线声称属于同一个实体的多张 crop（按 crop_index 编号）。"
                "但其中可能混入了别的实体（例如同一场景里长相/服装相近的另一个人，或另一件相似道具），"
                "也可能有一些 crop 根本看不清身份（后脑勺/背影、严重模糊、过暗几乎全黑、遮挡到看不出是谁）。\n"
                "请对每张 crop 依次判断两件事：\n"
                "1) identity_visible：这张 crop 是否真的露出足以辨识身份的视图。"
                "角色需能看到正脸或清晰的侧脸/明确的面部特征；道具需能看到可辨识的关键特征。"
                "后脑勺/背影、严重模糊、过暗几乎全黑、大面积遮挡导致看不出是谁 → identity_visible=false。"
                "注意：不要仅凭发色、服装、身形或轮廓就硬判 identity_visible=true 或 same_entity=true。\n"
                "2) same_entity：先判断这些 crop 里占多数、最一致的那个实体是谁（dominant identity），"
                "再逐张判断该 crop 是否确实是这个 dominant identity 本人/本物。"
                "只依据图像本身可辨识的身份特征判断，不要因为背景、动作或景别不同就判为不同实体；"
                "同一实体的不同角度、光照、表情、远近都算 same_entity=true；"
                "只有当 crop 明显是另一个实体、或严重混入多个实体无法归属时，才判 same_entity=false。"
                "当 identity_visible=false 时，same_entity 只能凭猜测，请把 confidence 标为 low。\n"
                f"kind={kind}\nname={name}\ndescription={description}\n"
                '返回 JSON：{"dominant_reason": str, '
                '"verdicts": [{"index": int, "identity_visible": bool, "same_entity": bool, '
                '"confidence": "high|medium|low", "reason": str}]}。'
                "verdicts 必须覆盖每一个 crop_index 恰好一次。"
            ),
        })
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["dominant_reason", "verdicts"],
            "properties": {
                "dominant_reason": {"type": "string"},
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["index", "identity_visible", "same_entity", "confidence", "reason"],
                        "properties": {
                            "index": {"type": "integer"},
                            "identity_visible": {"type": "boolean"},
                            "same_entity": {"type": "boolean"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "identity_audit", "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err: Exception | None = None
        prefer_not: str | None = None
        for attempt in range(self.max_retries):
            with self._pool.lease(
                prefer_not=prefer_not,
                workload={"stage": "s5_identity_audit", "attempt": attempt + 1},
            ) as lease:
                prefer_not = lease.base_url
                request = urllib.request.Request(
                    f"{lease.base_url}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen_direct(request, timeout=self.timeout_seconds) as response:
                        body = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    last_err = RuntimeError(
                        f"identity audit HTTP {exc.code} on {lease.base_url}: "
                        f"{exc.read().decode('utf-8')[:500]}"
                    )
                    if attempt < self.max_retries - 1:
                        continue
                    raise last_err from exc
                except Exception as exc:  # endpoint disconnect/timeout: rotate and retry
                    last_err = exc
                    if attempt < self.max_retries - 1:
                        continue
                    raise RuntimeError(
                        f"identity audit request failed after {self.max_retries} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                choice = body["choices"][0]
                content_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
                try:
                    raw = json.loads(content_text)
                except json.JSONDecodeError as exc:
                    last_err = exc
                    if payload["max_tokens"] < self.max_retry_tokens:
                        payload["max_tokens"] = min(
                            self.max_retry_tokens, int(payload["max_tokens"]) * 2
                        )
                        continue
                    if attempt < self.max_retries - 1:
                        continue
                    raise RuntimeError(
                        f"identity audit returned non-JSON after {self.max_retries} attempts "
                        f"(finish_reason={finish_reason!r}): {exc}\n"
                        f"--content--\n{str(content_text)[:500]}"
                    ) from exc
                verdicts = [
                    {
                        "index": int(v["index"]),
                        "identity_visible": bool(v.get("identity_visible", True)),
                        "same_entity": bool(v["same_entity"]),
                        "confidence": str(v.get("confidence") or "low"),
                        "reason": str(v.get("reason") or ""),
                    }
                    for v in raw.get("verdicts") or []
                ]
                self.last_result = {
                    "dominant_reason": str(raw.get("dominant_reason") or ""),
                    "endpoint": lease.base_url,
                    "n_verdicts": len(verdicts),
                }
                return {
                    "dominant_reason": str(raw.get("dominant_reason") or ""),
                    "verdicts": verdicts,
                    "available": True,
                    "endpoint": lease.base_url,
                }
        raise RuntimeError(f"identity auditor exhausted retries: {last_err}")


# --- gate orchestration ----------------------------------------------------


@dataclass(slots=True)
class _EntityAudit:
    entity_id: str
    name: str
    kind: str
    n_crops: int
    decision: str
    n_rejected: int = 0
    n_not_visible: int = 0
    n_needs_human: int = 0
    medoid_min_pairwise: float | None = None
    dominant_reason: str = ""
    detail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "kind": self.kind,
            "n_crops": self.n_crops,
            "decision": self.decision,
            "n_rejected": self.n_rejected,
            "n_not_visible": self.n_not_visible,
            "n_needs_human": self.n_needs_human,
            "medoid_min_pairwise": self.medoid_min_pairwise,
            "dominant_reason": self.dominant_reason,
            "detail": self.detail,
        }


def _set_review(
    proposal: dict[str, Any],
    *,
    source: str,
    same_entity: bool,
    confidence: str,
    needs_human: bool,
    reason: str,
    sim_to_medoid: float | None = None,
    identity_visible: bool = True,
    qa_reason: str = "identity_gate_reject",
) -> None:
    proposal["identity_review"] = {
        "source": source,
        "same_entity": same_entity,
        "identity_visible": identity_visible,
        "confidence": confidence,
        "needs_human": needs_human,
        "reason": reason,
        "sim_to_medoid": sim_to_medoid,
    }
    # A crop is dropped from the library when it is a different entity OR when it
    # carries no verifiable identity view (back-of-head, blur, near-black blob):
    # such crops must not seed an identity exemplar even if the VLM guessed "same".
    if not same_entity or not identity_visible:
        proposal["accepted"] = False
        qa = dict(proposal.get("qa") or {})
        qa["accepted"] = False
        qa["reasons"] = list(dict.fromkeys([*(qa.get("reasons") or []), qa_reason]))
        proposal["qa"] = qa
        proposal["reason"] = qa_reason


def _embed(embedder: Any, paths: list[Path]) -> list[list[float]] | None:
    try:
        vecs = embedder.embed_batch(paths)
    except Exception:
        return None
    if len(vecs) != len(paths):
        return None
    return vecs


def _audit_one_entity(
    *,
    proposals: list[dict[str, Any]],
    embedder: Any | None,
    auditor: IdentityAuditor | None,
    config: IdentityGateConfig,
) -> _EntityAudit:
    first = proposals[0]
    entity_id = str(first.get("entity_id") or "")
    name = str(first.get("name") or entity_id)
    kind = str(first.get("kind") or "")
    n = len(proposals)
    audit = _EntityAudit(entity_id=entity_id, name=name, kind=kind, n_crops=n, decision="")

    if n == 1:
        _set_review(
            proposals[0],
            source="single_crop",
            same_entity=True,
            confidence="medium",
            needs_human=False,
            reason="single library crop; no intra-entity conflict to resolve",
        )
        audit.decision = "single_crop"
        return audit

    crop_paths = [Path(str(p["crop_path"])) for p in proposals]

    # DINOv3 triage (advisory only; never rejects).
    vecs = _embed(embedder, crop_paths) if embedder is not None else None
    medoid_sims: list[float] | None = None
    medoid_index: int | None = None
    if vecs is not None:
        medoid_index, medoid_sims, min_pairwise = _medoid_cohesion(vecs)
        audit.medoid_min_pairwise = round(min_pairwise, 4)
        cohesive = (
            min(medoid_sims) >= config.skip_vlm_medoid_floor
            and min_pairwise >= config.skip_vlm_min_pairwise
        )
        if cohesive:
            for proposal, sim in zip(proposals, medoid_sims):
                _set_review(
                    proposal,
                    source="dinov3_cohesive",
                    same_entity=True,
                    confidence="high",
                    needs_human=False,
                    reason="tight DINOv3 cluster; auto-confirmed same identity",
                    sim_to_medoid=round(sim, 4),
                )
            audit.decision = "cohesive_auto"
            return audit

    # Ambiguous: the VLM decides, seeing all crops at once.
    if auditor is None:
        for proposal, sim in zip(proposals, medoid_sims or [None] * n):
            _set_review(
                proposal,
                source="dinov3_flagged_no_vlm",
                same_entity=True,
                confidence="low",
                needs_human=True,
                reason="DINOv3 spread too wide and no VLM auditor available",
                sim_to_medoid=round(sim, 4) if sim is not None else None,
            )
        audit.decision = "needs_human_no_vlm"
        audit.n_needs_human = n
        return audit

    try:
        result = auditor.audit(
            name=name,
            description=str(first.get("description") or ""),
            kind=kind,
            crop_paths=crop_paths,
        )
    except Exception as exc:  # one auditor outage must not empty the entity
        for proposal, sim in zip(proposals, medoid_sims or [None] * n):
            _set_review(
                proposal,
                source="vlm_error_needs_human",
                same_entity=True,
                confidence="low",
                needs_human=True,
                reason=f"identity auditor failed: {type(exc).__name__}: {exc}"[:300],
                sim_to_medoid=round(sim, 4) if sim is not None else None,
            )
        audit.decision = "needs_human_vlm_error"
        audit.n_needs_human = n
        return audit

    audit.dominant_reason = str(result.get("dominant_reason") or "")
    reject_levels = set(config.reject_confidences(kind))
    verdict_by_index: dict[int, dict[str, Any]] = {}
    for verdict in result.get("verdicts") or []:
        idx = verdict.get("index")
        if isinstance(idx, int) and 0 <= idx < n:
            verdict_by_index[idx] = verdict

    # A crop is a "different entity" reject only when it is confidently NOT the
    # dominant identity AND its identity is actually visible (a not-visible crop
    # can't support a confident different-entity claim; it is handled separately).
    reject_indices: set[int] = set()
    not_visible_indices: set[int] = set()
    for idx, verdict in verdict_by_index.items():
        visible = bool(verdict.get("identity_visible", True))
        different = not verdict.get("same_entity", True) and str(
            verdict.get("confidence") or "low"
        ) in reject_levels
        if visible and different:
            reject_indices.add(idx)
        elif not visible:
            # No verifiable identity view (back-of-head, blur, near-black blob):
            # drop as an exemplar regardless of the model's same/different guess.
            not_visible_indices.add(idx)

    dropped = reject_indices | not_visible_indices
    usable = [i for i in range(n) if i not in dropped]
    medoid_rejected = medoid_index is not None and medoid_index in reject_indices
    if medoid_rejected:
        audit.dominant_reason = f"{audit.dominant_reason} | vlm_rejected_dinov3_medoid".strip(" |")

    # Defer the whole entity to a human only when nothing usable survives to anchor
    # identity on. We do NOT treat a rejected DINOv3-medoid as a misfire: on a
    # genuinely mixed entity the medoid is just the centre of a spread cloud and may
    # itself be an intruder, so blocking rejection there only preserves the mixed
    # library (the exact LSMDC failure). When at least one usable crop survives, the
    # VLM keep-set defines the identity and we act on it.
    if not usable:
        for proposal, sim in zip(proposals, medoid_sims or [None] * n):
            _set_review(
                proposal,
                source="vlm_inconclusive_needs_human",
                same_entity=True,
                confidence="low",
                needs_human=True,
                reason=(
                    f"VLM left no usable identity crop "
                    f"(rejected {len(reject_indices)}, not-visible {len(not_visible_indices)}, "
                    f"of {n}); deferring to human"
                ),
                sim_to_medoid=round(sim, 4) if sim is not None else None,
            )
        audit.decision = "needs_human_vlm_inconclusive"
        audit.n_needs_human = n
        return audit

    for idx, proposal in enumerate(proposals):
        verdict = verdict_by_index.get(idx)
        sim = medoid_sims[idx] if medoid_sims is not None else None
        sim = round(sim, 4) if sim is not None else None
        if idx in reject_indices:
            _set_review(
                proposal,
                source="vlm_rejected",
                same_entity=False,
                confidence=str((verdict or {}).get("confidence") or "high"),
                needs_human=False,
                reason=str((verdict or {}).get("reason") or "VLM: different entity"),
                sim_to_medoid=sim,
                qa_reason="identity_gate_reject",
            )
            audit.n_rejected += 1
        elif idx in not_visible_indices:
            _set_review(
                proposal,
                source="vlm_not_visible",
                same_entity=True,
                confidence="low",
                needs_human=False,
                reason=str((verdict or {}).get("reason") or "VLM: identity not verifiable in crop"),
                sim_to_medoid=sim,
                identity_visible=False,
                qa_reason="identity_not_visible",
            )
            audit.n_not_visible += 1
        elif verdict is None:
            # VLM skipped this index: keep but flag for a human.
            _set_review(
                proposal,
                source="vlm_missing_verdict",
                same_entity=True,
                confidence="low",
                needs_human=True,
                reason="VLM returned no verdict for this crop",
                sim_to_medoid=sim,
            )
            audit.n_needs_human += 1
        else:
            _set_review(
                proposal,
                source="vlm_confirmed",
                same_entity=True,
                confidence=str(verdict.get("confidence") or "medium"),
                needs_human=False,
                reason=str(verdict.get("reason") or "VLM: same entity"),
                sim_to_medoid=sim,
            )
    audit.decision = "vlm_resolved"
    audit.detail = list(verdict_by_index.values())
    return audit


def run_identity_consistency(
    proposals: list[dict[str, Any]],
    *,
    embedder: Any | None = None,
    auditor: IdentityAuditor | None = None,
    config: IdentityGateConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate accepted library crops so each entity holds a single verified identity.

    Mutates ``proposals`` in place (rejected crops get ``accepted=False``) and
    returns ``(proposals, audit_summary)``. Rejection decisions come only from the
    VLM auditor; DINOv3 is used solely to skip the easy cohesive entities.
    """
    config = config or IdentityGateConfig()

    # Lazily construct a DINOv3 embedder if one was not supplied.
    active_embedder = embedder
    if active_embedder is None:
        try:
            from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.embedding import (
                DinoV3Embedder,
            )

            active_embedder = DinoV3Embedder()
        except Exception:
            active_embedder = None

    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        if not proposal.get("accepted") or not proposal.get("crop_path"):
            continue
        if proposal.get("task_kind", "acquire") == "slot_bind":
            continue
        if str(proposal.get("kind") or "") not in config.apply_kinds:
            continue
        by_entity[str(proposal.get("entity_id") or "")].append(proposal)

    entity_audits: list[_EntityAudit] = []
    for entity_id, items in by_entity.items():
        # Deterministic order for stable verdict indices.
        items.sort(key=lambda p: (int(p.get("chunk_id", -1)), str(p.get("crop_path") or "")))
        entity_audits.append(
            _audit_one_entity(
                proposals=items,
                embedder=active_embedder,
                auditor=auditor,
                config=config,
            )
        )

    summary = {
        "config": {
            "apply_kinds": list(config.apply_kinds),
            "skip_vlm_medoid_floor": config.skip_vlm_medoid_floor,
            "skip_vlm_min_pairwise": config.skip_vlm_min_pairwise,
            "char_reject_confidences": list(config.char_reject_confidences),
            "prop_reject_confidences": list(config.prop_reject_confidences),
        },
        "auditor_available": auditor is not None and not isinstance(auditor, NullIdentityAuditor),
        "embedder_available": active_embedder is not None,
        "n_entities_checked": len(entity_audits),
        "n_entities_cohesive_skipped": sum(
            1 for a in entity_audits if a.decision == "cohesive_auto"
        ),
        "n_entities_vlm_resolved": sum(1 for a in entity_audits if a.decision == "vlm_resolved"),
        "n_entities_needs_human": sum(1 for a in entity_audits if a.n_needs_human > 0),
        "n_crops_rejected": sum(a.n_rejected for a in entity_audits),
        "n_crops_not_visible": sum(a.n_not_visible for a in entity_audits),
        "n_crops_needs_human": sum(a.n_needs_human for a in entity_audits),
        "per_entity": [a.to_dict() for a in entity_audits],
    }
    return proposals, summary


__all__ = [
    "IdentityAuditor",
    "IdentityGateConfig",
    "NullIdentityAuditor",
    "VlmIdentityAuditor",
    "run_identity_consistency",
]
