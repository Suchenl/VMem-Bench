"""Text<->text semantic helpers (Q2), backed by Qwen3-Embedding-4B via a resident service.

CLIP/SigLIP text encoders are trained for image-caption matching (short, weak on sentence
semantics) -> they are the WRONG tool for text<->text. A dedicated text embedder (Qwen3-Embedding-4B)
is used here for two gaps the track-first pipeline currently leaves open:

  A. roster semantic de-dup -- ``merge_roster`` only merges by exact normalized name, so "grey
     rabbit" / "the bunny" / "Big Buck Bunny" stay 3 separate roster entries (=> 3 grounding
     phrases => parallel tracks => re-ID has to clean up). A second pass merges same-kind entries
     whose name+description embeddings are close AND whose static_attributes do not conflict.
  B. prompt completeness / naming consistency -- track-first dropped the VLM verifier, so nothing
     guards principle #9 ("every present entity is narrated in the chunk prompt"). A cheap,
     reproducible check flags chunks where a present entity's name/description is not semantically
     present in the prompt.

Both are PURE given an injected ``embed_fn`` (maps texts -> vectors), so they unit-test offline
without a GPU/service. Determinism: fixed thresholds, no VLM, no randomness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vmem_bench.common.vecmath import cosine_similarity

EmbedFn = Callable[[list[str]], list[list[float]]]


def _entry_text(entry: dict) -> str:
    """Canonical text of a roster/entity dict for semantic comparison: name + description."""
    name = (entry.get("name") or "").strip()
    desc = (entry.get("description") or "").strip()
    return f"{name}. {desc}".strip()


def _static_conflict(a: dict, b: dict) -> bool:
    """True if two static_attributes dicts disagree on any shared key (identity funnel rule)."""
    return any(k in b and str(a[k]).lower() != str(b[k]).lower() for k in a)


def semantic_dedup(entries: Sequence[dict], embed_fn: EmbedFn, *, threshold: float = 0.82,
                   ) -> list[dict]:
    """Second-pass merge of roster entries by name+description embedding cosine (per kind).

    Input = output of ``merge_roster`` (already exact-name merged). Two entries of the SAME kind
    merge when their text cosine >= ``threshold`` and their static_attributes do not conflict; the
    earlier (first-seen) entry absorbs the later one (keeps its name/grounding_phrase; fills empty
    description; unions non-conflicting static_attributes). Order-stable, deterministic.

    Returns the reduced list. No-op (returns list(entries)) when <2 entries."""
    items = list(entries)
    if len(items) < 2:
        return items
    vecs = embed_fn([_entry_text(e) for e in items])
    keep: list[dict] = []
    keep_vecs: list[list[float]] = []
    for ent, vec in zip(items, vecs):
        target = -1
        for i, (k, kv) in enumerate(zip(keep, keep_vecs)):
            if k["kind"] != ent["kind"]:
                continue
            if _static_conflict(k.get("static_attributes") or {}, ent.get("static_attributes") or {}):
                continue
            if cosine_similarity(vec, kv) >= threshold:
                target = i
                break
        if target < 0:
            keep.append(dict(ent))
            keep_vecs.append(vec)
        else:
            cur = keep[target]
            if not (cur.get("description") or "").strip() and (ent.get("description") or "").strip():
                cur["description"] = ent["description"].strip()
            merged_static = dict(cur.get("static_attributes") or {})
            for k, v in (ent.get("static_attributes") or {}).items():
                merged_static.setdefault(k, v)
            cur["static_attributes"] = merged_static
    return keep


def prompt_completeness(items: Sequence[tuple[str, str]], prompt: str, embed_fn: EmbedFn, *,
                        threshold: float = 0.5) -> dict:
    """Score whether each present entity is semantically covered by the chunk prompt (gap B).

    ``items`` = list of (entity_id, entity_text) where entity_text is e.g. "name. description".
    Returns ``{"scores": {eid: cosine}, "flagged": [eid...], "threshold": threshold}``; an entity
    is flagged when max-sim(entity_text vs prompt) < threshold (=> route the chunk to human review;
    never auto-edits the prompt). Empty ``items`` or empty ``prompt`` -> empty result. Deterministic.
    """
    if not items or not (prompt or "").strip():
        return {"scores": {}, "flagged": [], "threshold": threshold}
    texts = [prompt] + [t for _eid, t in items]
    vecs = embed_fn(texts)
    pvec = vecs[0]
    scores: dict[str, float] = {}
    flagged: list[str] = []
    for (eid, _text), vec in zip(items, vecs[1:]):
        s = cosine_similarity(pvec, vec)
        scores[eid] = s
        if s < threshold:
            flagged.append(eid)
    return {"scores": scores, "flagged": flagged, "threshold": threshold}
