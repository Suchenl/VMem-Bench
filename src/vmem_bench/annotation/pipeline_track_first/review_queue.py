"""Build the read-only, reproducible human-review queue.

This module only combines existing process artifacts.  It never changes gold,
review patches, dispositions, or freeze semantics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vmem_bench.common.gold_lint import lint_movie_dir
from vmem_bench.common.paths import MovieDirs
from vmem_bench.common.schemas import EntityRegistry, ChunkAnnotations


_KIND_ORDER = {"identity": 0, "state": 1, "prompt": 2, "lint": 3}


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _span_chunks(span: Any) -> list[int]:
    if not isinstance(span, list) or len(span) != 2:
        return []
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return []
    return list(range(min(start, end), max(start, end) + 1))


def _priority(chunks: list[int]) -> int:
    """A transparent count, deliberately not a calibrated scoring-risk estimate."""
    return len(set(chunks))


def _item(*, item_id: str, kind: str, status: str, question: str,
          entity_ids: list[str], evidence: dict[str, Any],
          affected_chunk_ids: list[int], recommended_action: str) -> dict[str, Any]:
    chunks = sorted(set(affected_chunk_ids))
    return {
        "id": item_id,
        "kind": kind,
        "priority": _priority(chunks),
        "status": status,
        "question": question,
        "entity_ids": sorted(set(entity_ids)),
        "evidence": evidence,
        "affected_chunk_ids": chunks,
        "recommended_action": recommended_action,
    }


_CLUSTER_FINDING_SPEC: dict[str, tuple[str, str, bool]] = {
    # code -> (question template, recommended_action, always_must)
    # always_must=True: an evidence gap or an automated decision that failed/was vetoed -- a human
    # MUST look (mirrors why "identity"/"lint" cards are always must). always_must=False: a
    # confident VLM decision that only needs the Layer-B SAMPLED audit (Review-Human-machine
    # collaboration §三层审核策略 B) -- it competes for the must_review_limit slots like any other
    # residual finding instead of automatically blocking every run.
    "roster_incomplete_unmatched_cluster": (
        "{name}（{kind}）从未匹配到 roster 中的任何角色/道具，是否是标注遗漏的新实体？",
        "review_roster_gap", True),
    "cluster_verify_error": (
        "VLM 簇内校验调用失败，{name} 已保守拆成单例，是否需要人工核实？",
        "review_cluster_verify_failure", True),
    "cluster_merge_error": (
        "VLM 跨簇合并调用失败（kind={kind}），同类未做跨短语合并，是否需要人工核实？",
        "review_cluster_merge_failure", True),
    "cluster_merge_static_veto": (
        "VLM 判定应合并（kind={kind}），但静态属性冲突已否决该合并，是否需要人工复核？",
        "review_static_veto", True),
    "large_precluster_sampled": (
        "{name} 的候选簇成员数超过 VLM 校验预算，仅按抽样代表 crop 判定为同一个体，是否需要补充核实？",
        "review_large_cluster_sample", True),
    "cluster_merge_budget_truncated": (
        "kind={kind} 的候选簇数超过跨簇合并预算，部分簇未参与合并判定，是否需要人工检查遗漏？",
        "review_merge_budget", True),
    "cluster_split_by_vlm": (
        "VLM 判定 {name} 所在候选簇混有不同个体并已拆分，抽样确认拆分是否正确？",
        "spot_check_cluster_split", False),
    "cluster_merged_by_vlm": (
        "VLM 判定以下候选簇（kind={kind}）为同一个体并已合并，抽样确认合并是否正确？",
        "spot_check_cluster_merge", False),
}


def _identity_resolution_items(identity_resolution: dict[str, Any] | None) -> list[dict[str, Any]]:
    """cluster_vlm mode's identity_resolution.json findings -> identity-kind review cards.

    Deliberately reuses ``kind="identity"`` (not a separate top-level kind): these ARE identity
    decisions, just sourced from batch cluster+VLM resolution instead of pairwise embedding
    candidates -- one unified queue, no second UI paradigm for the reviewer to learn. Each item
    additionally sets ``always_must`` so ``build_review_queue`` can give error/veto/gap findings
    the same "always surfaced" treatment as pairwise identity cards, while confident VLM
    merge/split decisions compete for the ordinary must-review budget like any spot-check."""
    identity_resolution = identity_resolution or {}
    if identity_resolution.get("mode") == "seeded":
        items: list[dict[str, Any]] = []
        findings = [f for f in identity_resolution.get("findings") or [] if isinstance(f, dict)]
        missing_by_id = {
            str(f.get("entity_id") or ""): f
            for f in findings if f.get("code") == "seed_entity_missing_evidence"
        }
        unknown = [f for f in findings if f.get("code") == "unknown_track_rejected"]
        for entity in identity_resolution.get("canonical_entities") or []:
            if not isinstance(entity, dict) or entity.get("kind") == "location":
                continue
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            missing = missing_by_id.get(entity_id)
            entry = _item(
                item_id=f"identity:canonical:{entity_id}", kind="identity",
                status="needs_review",
                question=(f"{entity.get('name') or entity_id} 没有任何跟踪证据，seed 或感知是否遗漏？"
                          if missing else
                          f"{entity.get('name') or entity_id} 的全片证据是否均属于该 canonical 实体？"),
                entity_ids=[entity_id],
                evidence={"canonical_entity": entity, "finding": missing} if missing
                else {"canonical_entity": entity, "source": "human_confirmed_roster"},
                affected_chunk_ids=[],
                recommended_action=("fix_seed_or_tracking" if missing
                                    else "spot_check_canonical_entity"))
            entry["always_must"] = bool(missing)
            entry["always_spot"] = not bool(missing)
            entry["retain_if_missing"] = bool(missing)
            items.append(entry)
        if unknown:
            chunks = sorted({
                int(f.get("chunk_id", -1)) for f in unknown
                if isinstance(f.get("chunk_id"), (int, float)) and int(f["chunk_id"]) >= 0
            })
            entry = _item(
                item_id="identity:canonical:unknown_tracks", kind="identity",
                status="needs_review",
                question=f"{len(unknown)} 个显著 track 未匹配 canonical roster，是否需要补 seed？",
                entity_ids=[], evidence={"findings": unknown},
                affected_chunk_ids=chunks, recommended_action="review_unknown_tracks")
            entry["always_must"] = True
            items.append(entry)
        return items
    findings = identity_resolution.get("findings") or []
    entity_by_obs = {str(k): v for k, v in
                     (identity_resolution.get("entity_id_by_observation") or {}).items()}
    group_entity_id = {str(k): v for k, v in
                      (identity_resolution.get("group_entity_id") or {}).items()}
    obs_by_index = {str(o.get("index")): o for o in identity_resolution.get("observations") or []}

    def _entities_for(finding: dict[str, Any]) -> list[str]:
        ids: set[str] = set()
        for member in finding.get("members") or finding.get("original_members") or []:
            eid = entity_by_obs.get(str(member))
            if eid:
                ids.add(eid)
        gi = finding.get("group_index")
        if gi is not None and str(gi) in group_entity_id:
            ids.add(group_entity_id[str(gi)])
        return sorted(ids)

    def _name_for(entity_ids: list[str], finding: dict[str, Any]) -> str:
        if finding.get("name"):
            return str(finding["name"])
        for member in finding.get("members") or finding.get("original_members") or []:
            obs = obs_by_index.get(str(member))
            if obs and obs.get("name"):
                return str(obs["name"])
        return entity_ids[0] if entity_ids else "未知实体"

    items: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or "")
        spec = _CLUSTER_FINDING_SPEC.get(code)
        if spec is None:
            continue  # unrecognized/future finding code -> no card rather than a garbled one
        template, recommended_action, always_must = spec
        entity_ids = _entities_for(finding)
        name = _name_for(entity_ids, finding)
        question = template.format(name=name, kind=finding.get("kind", ""))
        suffix = "--".join(entity_ids) if entity_ids else (
            "--".join(str(c) for c in finding.get("cluster_ids") or []) or code)
        entry = _item(
            item_id=f"identity:cluster:{code}:{suffix}", kind="identity", status="needs_review",
            question=question, entity_ids=entity_ids,
            evidence={"finding": finding, "source": "identity_resolution"},
            affected_chunk_ids=[], recommended_action=recommended_action)
        entry["always_must"] = always_must
        items.append(entry)
    return items


def build_review_queue(
    *,
    auto_review: dict[str, Any] | None = None,
    identity_candidates: list[dict[str, Any]] | None = None,
    qa_report: list[dict[str, Any]] | None = None,
    strict_lint: list[dict[str, Any]] | None = None,
    state_events: list[dict[str, Any]] | None = None,
    identity_adjudication: dict[str, Any] | None = None,
    identity_resolution: dict[str, Any] | None = None,
    surviving_ids: set[str] | None = None,
    ignored_identity_ids: set[str] | None = None,
    must_review_limit: int = 10,
) -> dict[str, Any]:
    """Return a deterministic queue from JSON-compatible source artifacts.

    ``surviving_ids``, when given, drops identity cards that reference entities no longer in
    gold (e.g. merged away by the three-vote auto-merge) — a card about a dead id is pure noise."""
    auto_review = auto_review or {}
    identity_candidates = identity_candidates or []
    qa_report = qa_report or []
    strict_lint = strict_lint or []
    state_events = state_events or []
    ignored_identity_ids = set(ignored_identity_ids or ())
    # VLM same-individual groups (review-only recommendation): entity_id -> its suggested group.
    suggestion_by_entity: dict[str, list[str]] = {}
    for group in (identity_adjudication or {}).get("groups", []):
        ids = [str(entity_id) for entity_id in group.get("entity_ids", [])]
        if len(ids) > 1:
            for entity_id in ids:
                suggestion_by_entity[entity_id] = ids
    items: list[dict[str, Any]] = []
    auto_by_entity = {
        str(entry.get("entity_id")): entry
        for entry in auto_review.get("queue", [])
        if isinstance(entry, dict) and entry.get("entity_id")
    }
    must_review = {str(entity_id) for entity_id in auto_review.get("must_review", [])}

    # Connected alias candidates are one human decision surface, not repeated A-B/B-C cards.
    candidate_by_id: dict[str, dict[str, Any]] = {}
    neighbors: dict[str, set[str]] = defaultdict(set)
    for candidate in identity_candidates:
        if not isinstance(candidate, dict):
            continue
        left, right = str(candidate.get("left") or ""), str(candidate.get("right") or "")
        if not left or not right:
            continue
        if left in ignored_identity_ids or right in ignored_identity_ids:
            continue
        key = f"{min(left, right)}--{max(left, right)}"
        candidate_by_id[key] = candidate
        neighbors[left].add(right); neighbors[right].add(left)
    visited: set[str] = set()
    for start in sorted(neighbors):
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            entity_id = stack.pop()
            if entity_id in component:
                continue
            component.add(entity_id); visited.add(entity_id)
            stack.extend(neighbors[entity_id] - component)
        entity_ids = sorted(component)
        candidates = [candidate for candidate in candidate_by_id.values()
                      if candidate.get("left") in component and candidate.get("right") in component]
        chunks = [chunk for candidate in candidates for chunk in (
            _span_chunks(candidate.get("left_chunk_span")) + _span_chunks(candidate.get("right_chunk_span")))]
        evidence = {"candidates": candidates}
        if len(candidates) == 1:
            evidence["candidate"] = candidates[0]
        auto_evidence = {eid: auto_by_entity[eid] for eid in entity_ids if eid in auto_by_entity}
        if auto_evidence:
            evidence["auto_review"] = auto_evidence
        suggested = {tuple(suggestion_by_entity[eid]) for eid in entity_ids
                     if eid in suggestion_by_entity}
        if suggested:
            evidence["machine_suggestion"] = {
                "same_individual_groups": [list(g) for g in sorted(suggested)],
                "source": "vlm_identity_adjudication"}
        recommendation = (str(candidates[0].get("recommendation") or "review_identity")
                          if len(candidates) == 1 else "review_identity_component")
        if suggested:
            recommendation = "merge_per_machine_suggestion"
        items.append(_item(
            item_id=f"identity:{'--'.join(entity_ids)}", kind="identity", status="needs_review",
            question=(f"{entity_ids[0]} 与 {entity_ids[1]} 是否为同一实体？" if len(entity_ids) == 2
                      else f"这 {len(entity_ids)} 个实体是否属于同一别名簇？"), entity_ids=entity_ids,
            evidence=evidence, affected_chunk_ids=chunks, recommended_action=recommendation))

    # VLM-suggested same-individual groups not already covered by an embedding-candidate card
    # get their own decision card — visual adjudication finds exactly the splits whose embedding
    # similarity collapsed, so requiring an embedding candidate first would hide them.
    covered = [set(item["entity_ids"]) for item in items]
    seen_groups: set[tuple[str, ...]] = set()
    for group in (identity_adjudication or {}).get("groups", []):
        ids = tuple(sorted(str(e) for e in group.get("entity_ids", [])
                           if str(e) not in ignored_identity_ids))
        if len(ids) < 2 or ids in seen_groups:
            continue
        seen_groups.add(ids)
        if any(set(ids) <= c for c in covered):
            continue
        items.append(_item(
            item_id=f"identity:vlm:{'--'.join(ids)}", kind="identity", status="needs_review",
            question=f"VLM 判定这 {len(ids)} 个实体为同一个体，是否合并？",
            entity_ids=list(ids),
            evidence={"machine_suggestion": {
                "same_individual_groups": [list(ids)],
                "species": group.get("species", ""),
                "source": "vlm_identity_adjudication"}},
            affected_chunk_ids=[], recommended_action="merge_per_machine_suggestion"))

    candidate_entity_ids = {eid for item in items for eid in item["entity_ids"]}
    for entity_id in sorted(must_review - candidate_entity_ids - ignored_identity_ids):
        entry = auto_by_entity.get(entity_id, {"entity_id": entity_id})
        recommended = str(entry.get("recommendation") or "review_identity")
        question = (f"{entity_id} 疑似检测噪声（单证据、<1s 屏幕时间），是否删除？"
                    if recommended == "candidate_drop"
                    else f"{entity_id} 的身份与证据是否一致？")
        items.append(_item(
            item_id=f"identity:entity:{entity_id}", kind="identity", status="needs_review",
            question=question, entity_ids=[entity_id],
            evidence={"auto_review": entry}, affected_chunk_ids=[],
            recommended_action=recommended))

    # Codes that record an automatic action (FYI provenance), not a pending human decision;
    # they must never spawn review cards.
    informational_codes = {"state_event_filtered_reversible", "unknown_track_ignored"}

    def _needs_human(finding: dict[str, Any]) -> bool:
        code = str(finding.get("code") or "")
        if code in informational_codes:
            return False
        # A prompt that fails semantic coverage of a PROP/LOCATION is low-stakes (scenery is
        # rarely load-bearing for identity metrics); only characters earn a human card.
        if code == "prompt_missing_present_entity":
            return str(finding.get("entity_id") or "").startswith("char_")
        return True

    findings_by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in qa_report:
        if not isinstance(record, dict) or "chunk_id" not in record:
            continue
        findings = [f for f in record.get("findings") or []
                    if isinstance(f, dict) and _needs_human(f)]
        # A clean chunk is NOT a review item: only actual QA findings ask for human eyes.
        if not findings:
            continue
        try:
            chunk_id = int(record["chunk_id"])
        except (TypeError, ValueError):
            continue
        findings_by_chunk[chunk_id].extend(findings)
    for chunk_id, findings in sorted(findings_by_chunk.items()):
        entity_ids = [str(f["entity_id"]) for f in findings if f.get("entity_id")]
        codes = sorted({str(f.get("code") or "qa_finding") for f in findings})
        items.append(_item(
            item_id=f"prompt:c{chunk_id:03d}", kind="prompt", status="needs_review",
            question=f"Chunk {chunk_id} 的 prompt 是否覆盖已确定实体并与证据一致？",
            entity_ids=entity_ids, evidence={"findings": findings, "codes": codes},
            affected_chunk_ids=[chunk_id], recommended_action="review_prompt"))

    # One card per entity, not per event: a reviewer judges an entity's state timeline as a
    # whole, and 70+ single-event cards is exactly the queue bloat this module must avoid.
    events_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in state_events:
        if isinstance(event, dict) and event.get("event_id"):
            events_by_entity[str(event.get("entity_id") or "")].append(event)
    for entity_id, events in sorted(events_by_entity.items()):
        chunks = [chunk for ev in events for chunk in range(
            int(ev.get("chunk_id", 0)), int(ev.get("last_chunk_id", ev.get("chunk_id", 0))) + 1)]
        first_desc = str(events[0].get("description") or events[0]["event_id"])
        question = (f"状态事件“{first_desc}”是否不可逆？" if len(events) == 1 else
                    f"{entity_id or '未知实体'} 的 {len(events)} 个状态事件是否均为不可逆变化？")
        items.append(_item(
            item_id=f"state:{entity_id or 'unknown'}", kind="state", status="needs_review",
            question=question, entity_ids=[entity_id] if entity_id else [],
            evidence={"events": events},
            affected_chunk_ids=chunks, recommended_action="review_state_event"))

    grouped_lint: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for violation in strict_lint:
        if not isinstance(violation, dict) or violation.get("severity", "error") != "error":
            continue
        code = str(violation.get("code") or "lint_error")
        path = str(violation.get("path") or "")
        entity_id = ""
        if path.startswith("entities[") and "]" in path:
            entity_id = path[len("entities["):path.index("]")]
        grouped_lint[(code, entity_id or path)].append(violation)
    for (code, group), violations in sorted(grouped_lint.items()):
        entity_ids = [group] if group and not group.startswith("chunks[") else []
        chunks: list[int] = []
        for violation in violations:
            path = str(violation.get("path") or "")
            if path.startswith("chunks[") and "]" in path:
                try:
                    chunks.append(int(path[len("chunks["):path.index("]")]))
                except ValueError:
                    pass
        suffix = group or "global"
        items.append(_item(
            item_id=f"lint:{code}:{suffix}", kind="lint", status="blocked",
            question=f"修复严格 lint 错误：{code}", entity_ids=entity_ids,
            evidence={"violations": violations}, affected_chunk_ids=chunks,
            recommended_action="fix_lint"))

    items.extend(_identity_resolution_items(identity_resolution))

    if surviving_ids is not None:
        items = [item for item in items
                 if item["kind"] != "identity"
                 or item.get("retain_if_missing")
                 or all(eid in surviving_ids for eid in item["entity_ids"])]

    items.sort(key=lambda item: (-item["priority"], _KIND_ORDER[item["kind"]], item["id"]))

    # Tiering (design principle #11 + triad): humans MUST decide identity/lint plus the
    # highest-impact residuals up to ``must_review_limit``; everything else is a spot-check
    # sample that does not block freezing. Identity/lint cards are never demoted by default —
    # they are the decisions only a human can make — UNLESS an item explicitly overrides via
    # ``always_must`` (identity_resolution's confident VLM merge/split confirmations set this to
    # False so they compete for the ordinary budget like any other Layer-B sampled audit item;
    # see Review-Human-machine_collaboration.md 三层审核策略).
    must = 0
    for item in items:
        item.pop("retain_if_missing", None)
        forced = item.pop("always_must", None)
        force_spot = bool(item.pop("always_spot", False))
        is_must = forced if forced is not None else item["kind"] in ("identity", "lint")
        if not force_spot and (is_must or must < must_review_limit):
            item["review_tier"] = "must"
            must += 1
        else:
            item["review_tier"] = "spot_check"
    items.sort(key=lambda item: (item["review_tier"] != "must",
                                 -item["priority"], _KIND_ORDER[item["kind"]], item["id"]))
    summary = {
        "n_items": len(items),
        "n_must_review": sum(item["review_tier"] == "must" for item in items),
        "n_spot_check": sum(item["review_tier"] == "spot_check" for item in items),
        "by_kind": {kind: sum(item["kind"] == kind for item in items) for kind in _KIND_ORDER},
        "by_status": {status: sum(item["status"] == status for item in items)
                      for status in ("needs_review", "blocked")},
    }
    return {"version": 1, "items": items, "summary": summary}


def write_review_queue(out: Path) -> dict[str, Any]:
    """Load available artifacts, lint existing gold strictly, and atomically write the queue."""
    dirs = MovieDirs(Path(out), write=True)
    read_dirs = MovieDirs(Path(out))
    lint: list[dict[str, Any]] = []
    state_events: list[dict[str, Any]] = []
    surviving_ids: set[str] | None = None
    ignored_identity_ids: set[str] = set()
    if read_dirs.registry_json.is_file() and read_dirs.annotations_json.is_file():
        lint = [violation.to_dict() for violation in lint_movie_dir(out, strict_review=True)]
        registry = EntityRegistry.from_dict(_read_json(read_dirs.registry_json, {}))
        chunks = ChunkAnnotations.from_dict(_read_json(read_dirs.annotations_json, {}))
        last_chunk_id = max((chunk.chunk_id for chunk in chunks.chunks), default=0)
        surviving_ids = {entity.entity_id for entity in registry.entities}
        ignored_identity_ids = {
            entity.entity_id for entity in registry.entities if entity.kind == "location"}
        for entity in registry.entities:
            for event in entity.state_events:
                state_events.append({**event.to_dict(), "entity_id": entity.entity_id,
                                     "last_chunk_id": last_chunk_id})
    queue = build_review_queue(
        auto_review=_read_json(read_dirs.auto_review_json, {}),
        identity_candidates=_read_json(read_dirs.identity_candidates, []),
        qa_report=_read_json(read_dirs.qa_report, []), strict_lint=lint,
        state_events=state_events,
        identity_adjudication=_read_json(read_dirs.tmp / "identity_adjudication.json", {}),
        identity_resolution=_read_json(read_dirs.tmp / "identity_resolution.json", {}),
        surviving_ids=surviving_ids, ignored_identity_ids=ignored_identity_ids)
    dirs.tmp.mkdir(parents=True, exist_ok=True)
    tmp = dirs.review_queue.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dirs.review_queue)
    return queue


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the read-only MemStrata review queue")
    parser.add_argument("--out", type=Path, required=True, help="annotated movie output directory")
    args = parser.parse_args(argv)
    print(json.dumps(write_review_queue(args.out)["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
