"""Gold materialization: instructions, forbidden tables, scenario tags (workflow steps 6-7).

Everything derivable deterministically IS derived deterministically (cheapest-reliable-tool):
scenario tags, gold instructions, and forbidden tables are set operations over the registry
and presence history; the VLM only drafts prompts and proposes state events.
"""

from __future__ import annotations

import re

from vmem_bench.common.schemas import (
    ChunkAnnotation, ForbiddenRep, GoldInstruction, StateEvent,
)
from vmem_bench.common.vecmath import cosine_similarity
from vmem_bench.annotation.pipeline_track_first.consolidation import Registry

MULTI_INSTANCE_SIM = 0.75

# Visibility/camera/screen-entry phrasings describe REVERSIBLE observations; the state-event
# contract admits only irreversible appearance/existence changes. Filtering these by pattern is
# deterministic, so bogus events never reach gold nor cost a human review card.
_REVERSIBLE_EVENT_PATTERNS = re.compile(
    r"(no longer (visible|in view|seen)"
    r"|out of (view|frame|sight)"
    r"|exits?\b|leaves? the (frame|scene|shot|view)"
    r"|enters? the (frame|scene|shot|view)"
    r"|camera (pans?|tilts?|zooms?|cuts?|moves?|tracks?)"
    r"|scene transitions?|transitions? (from|to)\b"
    r"|is introduced|first appear|becomes? visible|comes? into view"
    r"|glides? (into|out of)|(flies|moves|walks|runs|hops) (into|out of|away|off))",
    re.IGNORECASE)

STATE_EVENT_TYPES = (
    "destroyed",
    "consumed",
    "broken",
    "acquired",
    "attached",
    "detached",
    "appearance_changed",
)
_STATE_EVENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "destroyed": re.compile(r"\b(destroyed|killed|dead|dies|disintegrat|burned away)\b", re.I),
    "consumed": re.compile(r"\b(eaten|ate|consumed|swallowed|devoured)\b", re.I),
    "broken": re.compile(r"\b(broken|snapped|shattered|torn|split in two)\b", re.I),
    "acquired": re.compile(r"\b(acquired|picked up|takes possession|now owns)\b", re.I),
    "attached": re.compile(r"\b(attached|fastened|tied|embedded|impaled|affixed)\b", re.I),
    "detached": re.compile(r"\b(detached|removed|separated|no longer attached|cut free)\b", re.I),
    "appearance_changed": re.compile(
        r"\b(transformed|permanently changed|irreversibly changed|scarred|painted|burned|melted)\b",
        re.I),
}


def infer_state_event_type(description: str) -> str | None:
    """Map one description to the finite lifecycle ontology, or reject it."""
    if _REVERSIBLE_EVENT_PATTERNS.search(description):
        return None
    for event_type, pattern in _STATE_EVENT_PATTERNS.items():
        if pattern.search(description):
            return event_type
    return None


def filter_state_events(
    drafted: list[dict],
    *,
    allowed_by_entity: dict[str, set[str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Keep only finite-ontology events allowed for the affected canonical entity.

    Rejected records are returned with ``rejection_reason`` so QA/review can surface the exact
    contract failure.  This is intentionally strict: an uncertain event must not deprecate valid
    memory representations.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for raw in drafted:
        item = dict(raw)
        description = str(item.get("description") or "")
        explicit = str(item.get("event_type") or "").strip()
        inferred = infer_state_event_type(description)
        event_type = explicit if explicit in STATE_EVENT_TYPES else inferred
        reason = ""
        if _REVERSIBLE_EVENT_PATTERNS.search(description):
            reason = "reversible_or_camera_only"
        elif event_type is None:
            reason = "outside_finite_state_event_ontology"
        elif explicit and inferred is not None and explicit != inferred:
            # An explicit finite-ontology ``event_type`` (the VLM's structured
            # ``state_change_kind``) is authoritative. The description regex can only (a) reject
            # reversible/camera prose or (b) corroborate the kind; when it *cannot classify* the
            # prose (``inferred is None`` — e.g. a non-English corpus), the explicit kind is trusted
            # rather than vetoed. Only a genuine conflict (a *different* inferred ontology type)
            # is a mismatch, so a valid VLM-tagged event is never dropped just because the prose
            # regex is English-only.
            reason = "event_type_description_mismatch"
        else:
            entity_id = str(item.get("entity_id") or "")
            if allowed_by_entity is not None:
                allowed = allowed_by_entity.get(entity_id, set())
                if event_type not in allowed:
                    reason = "event_type_not_allowed_for_entity"
        if reason:
            item["rejection_reason"] = reason
            rejected.append(item)
        else:
            item["event_type"] = event_type
            kept.append(item)
    return kept, rejected


def split_reversible_events(drafted: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split VLM-drafted state events into (kept, filtered-as-reversible) by description."""
    keep, filtered = filter_state_events(drafted)
    return keep, filtered


def gold_instructions_for(present_ids: list[str], first_ids: set[str]) -> list[GoldInstruction]:
    return [GoldInstruction(entity_id=eid,
                            requirement="introduce" if eid in first_ids else "continuity")
            for eid in present_ids]


def materialize_forbidden(registry: Registry, chunk_id: int) -> list[ForbiddenRep]:
    """F_active(t): representations deprecated by events with event.chunk_id < t (§4)."""
    forbidden: list[ForbiddenRep] = []
    for entity in registry.entities.values():
        for event in entity.state_events:
            if event.chunk_id < chunk_id:
                forbidden += [ForbiddenRep(representation_id=rid, reason=event.event_id)
                              for rid in event.deprecates]
    return forbidden


def scenario_tags_for(chunk_id: int, present_ids: list[str], first_ids: set[str],
                      presence_history: dict[str, list[int]], registry: Registry,
                      has_state_event: bool) -> list[str]:
    """Deterministic scenario tagging over presence history + registry."""
    tags: set[str] = set()
    for eid in present_ids:
        if eid in first_ids:
            continue
        history = [c for c in presence_history.get(eid, []) if c < chunk_id]
        if history and max(history) < chunk_id - 1:  # was away >= 1 chunk and returned
            entity = registry.entities[eid]
            tags.add("scene-return" if entity.kind == "location" else "re-appearance")
    by_kind: dict[str, list[str]] = {}
    for eid in present_ids:
        by_kind.setdefault(registry.entities[eid].kind, []).append(eid)
    for kind, ids in by_kind.items():
        if kind == "location" or len(ids) < 2:
            continue
        vecs = []
        for eid in ids:
            reps = registry.entities[eid].representations
            if not reps:
                continue
            rep_vecs = [registry.embeddings[r.embedding_key] for r in reps
                        if r.embedding_key in registry.embeddings]
            if not rep_vecs:
                continue
            # ponytail: mean of all rep embeddings is the entity representative (not reps[0],
            # which lets a single outlier crop dominate multi-instance detection). Ceiling: the
            # 0.75 cosine threshold is not calibrated on the DINOv3-vits16 distribution; upgrade
            # path is medoid + a threshold calibrated on a small labeled same/different set.
            dim = len(rep_vecs[0])
            mean = [sum(v[j] for v in rep_vecs) / len(rep_vecs) for j in range(dim)]
            vecs.append(mean)
        if any(cosine_similarity(a, b) >= MULTI_INSTANCE_SIM
               for i, a in enumerate(vecs) for b in vecs[i + 1:]):
            tags.add("multi-instance")
    if has_state_event:
        tags.add("state-change")
    return sorted(tags) or ["none"]


def state_events_from_draft(registry: Registry, chunk_id: int,
                            drafted: list[dict[str, str]], *,
                            fps: float | None = None,
                            frame_span: tuple[int, int] | None = None) -> list[StateEvent]:
    """Attach VLM-drafted state events to their entities.

    deprecates resolution: if the drafter named specific rep ids in ``deprecates_representations``,
    only those (filtered to this entity's reps up to this chunk) are deprecated — this lets an
    event scope itself to appearance-superseded reps rather than the whole history (avoids
    over-deprecating still-valid appearance references when only position changed). If the
    drafter left it empty, ALL prior reps of that entity up to this chunk are deprecated (the
    default for "the entity is destroyed / completely changed")."""
    events: list[StateEvent] = []
    for i, item in enumerate(drafted):
        entity = registry.entities.get(item.get("entity_id", ""))
        if entity is None:
            continue
        named = [str(x) for x in item.get("deprecates_representations", []) if x]
        if named:
            valid_rep_ids = {r.representation_id for r in entity.representations
                             if r.chunk_id <= chunk_id}
            deprecates = [rid for rid in named if rid in valid_rep_ids]
        else:
            deprecates = [r.representation_id for r in entity.representations
                          if r.chunk_id <= chunk_id]
        # Advisory sub-chunk timing (Q3): the drafter's best-effort event_frame -> frame_index +
        # seconds. NEVER scored (chunk_id is authoritative); clamped to the chunk's frame span so a
        # hallucinated index can't leave the chunk. null when absent/invalid or fps/span unknown.
        frame_index, seconds = None, None
        raw = item.get("event_frame")
        if raw is not None and frame_span is not None:
            try:
                fi = int(raw)
                frame_index = min(max(fi, frame_span[0]), frame_span[1])
                if fps and fps > 0:
                    seconds = round(frame_index / fps, 2)
            except (TypeError, ValueError):
                frame_index = None
        event = StateEvent(
            event_id=f"evt_c{chunk_id:03d}_{entity.entity_id}" + (f".{i}" if i else ""),
            chunk_id=chunk_id, description=str(item.get("description", "")),
            deprecates=deprecates, frame_index=frame_index, seconds=seconds)
        entity.state_events.append(event)
        events.append(event)
    return events


def build_chunk_annotation(*, chunk_id: int, shot_span: list[int], frame_span: list[int],
                           prompt: str, present_ids: list[str], first_ids: set[str],
                           registry: Registry, presence_history: dict[str, list[int]],
                           has_state_event: bool) -> ChunkAnnotation:
    return ChunkAnnotation(
        chunk_id=chunk_id, shot_span=list(shot_span), frame_span=list(frame_span),
        prompt=prompt, present=list(present_ids), first_appearances=sorted(first_ids),
        gold_instructions=gold_instructions_for(present_ids, first_ids),
        forbidden=materialize_forbidden(registry, chunk_id),
        scenario_tags=scenario_tags_for(chunk_id, present_ids, first_ids,
                                        presence_history, registry, has_state_event))
