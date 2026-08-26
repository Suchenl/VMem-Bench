"""Exemplar identity assignment for S5 (owned copy of track_first Route B idea).

SAM3/GDINO only propose regions; WHO each crop is comes from DINOv3 similarity to
already-accepted exemplar crops. Unmatched candidates stay unassigned for VLM fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vmem_bench.common.vecmath import cosine_similarity

DEFAULT_SIM_FLOOR = 0.28
# A candidate only auto-claims an entity when it is closer to that entity's
# exemplars than to the runner-up entity by this cosine margin. Without it, on
# real-distribution footage two look-alike identities trade crops around the
# low sim_floor and the library ends up mixed. Ambiguous candidates below the
# margin are deliberately left unassigned so the VLM pick/audit can decide.
DEFAULT_ASSIGN_MARGIN = 0.06


def rank_by_exemplar(
    candidate_vec: list[float],
    exemplars: dict[str, list[list[float]]],
) -> list[tuple[str, float]]:
    """All entities ranked by max cosine of ``candidate_vec`` to their exemplars."""
    ranked: list[tuple[str, float]] = []
    for entity_id, vecs in exemplars.items():
        if not vecs:
            continue
        if vecs and not isinstance(vecs[0], (list, tuple)):
            vecs = [vecs]  # type: ignore[list-item]
        sim = max(
            (cosine_similarity(list(candidate_vec), list(v)) for v in vecs),
            default=-2.0,
        )
        ranked.append((entity_id, sim))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def assign_by_exemplar(
    candidate_vec: list[float],
    exemplars: dict[str, list[list[float]]],
    *,
    sim_floor: float = DEFAULT_SIM_FLOOR,
) -> tuple[str, float] | None:
    """Argmax cosine over exemplar vectors; None when nothing clears the floor."""
    ranked = rank_by_exemplar(candidate_vec, exemplars)
    if not ranked:
        return None
    best_id, best_sim = ranked[0]
    if best_sim < sim_floor:
        return None
    return best_id, best_sim


def exclusive_assign_candidates(
    *,
    candidate_vecs: list[list[float]],
    exemplars: dict[str, list[list[float]]],
    entity_ids: list[str],
    sim_floor: float = DEFAULT_SIM_FLOOR,
    assign_margin: float = DEFAULT_ASSIGN_MARGIN,
) -> tuple[dict[str, tuple[int, float]], list[int]]:
    """Greedy exclusive assignment: highest sim first, one candidate per entity.

    A candidate is only eligible to claim its best entity when that entity wins
    over the runner-up entity by ``assign_margin`` cosine; otherwise the match is
    too ambiguous to bank and the candidate is left as leftover for VLM review.

    Returns ``{entity_id: (cand_index, sim)}`` and leftover candidate indices.
    """
    wanted = set(entity_ids)
    scored: list[tuple[float, str, int]] = []
    for index, vec in enumerate(candidate_vecs):
        ranked = [(eid, sim) for eid, sim in rank_by_exemplar(vec, exemplars) if eid in wanted]
        if not ranked:
            continue
        entity_id, sim = ranked[0]
        if sim < sim_floor:
            continue
        runner_up = ranked[1][1] if len(ranked) > 1 else -2.0
        if sim - runner_up < assign_margin:
            # Too close to a second identity to bank safely; defer to VLM.
            continue
        scored.append((sim, entity_id, index))
    scored.sort(reverse=True)
    claimed_entity: set[str] = set()
    claimed_cand: set[int] = set()
    assignment: dict[str, tuple[int, float]] = {}
    for sim, entity_id, index in scored:
        if entity_id in claimed_entity or index in claimed_cand:
            continue
        claimed_entity.add(entity_id)
        claimed_cand.add(index)
        assignment[entity_id] = (index, sim)
    leftover = [i for i in range(len(candidate_vecs)) if i not in claimed_cand]
    return assignment, leftover


def build_exemplar_vectors(
    bank: dict[str, list[Path]],
    embedder: Any,
) -> dict[str, list[list[float]]]:
    """entity_id -> list of DINOv3 vectors from accepted library crops."""
    out: dict[str, list[list[float]]] = {}
    for entity_id, paths in bank.items():
        vecs: list[list[float]] = []
        for path in paths:
            if path.is_file():
                vecs.append(embedder.embed_image(path))
        if vecs:
            out[entity_id] = vecs
    return out
