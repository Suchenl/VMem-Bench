"""Global cast-roster keyframe selection (track-first redesign §3.1a).

Pick a small, diverse, movie-wide set of frames to feed the roster-discovery VLM: too many frames
blow the token budget and induce hallucination; too few miss late-appearing entities. All of the
selection is deterministic + cheap (cheapest-reliable-tool): per-shot candidate sampling -> per-shot
representative (medoid / centroid-nearest real frame) -> farthest-point sampling in embedding space
down to a global budget. The VLM only sees the final K frames. Embeddings are injected so the
selection math is unit-testable without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vmem_bench.common.vecmath import cosine_similarity


def shot_candidate_indices(shots: Sequence[tuple[int, int]], *, fps: float,
                           candidate_fps: float) -> list[list[int]]:
    """Per shot, evenly sample candidate absolute frame indices at ~``candidate_fps``.

    ``shots`` are closed-inclusive ``(first_frame, last_frame)``. Always yields >=1 frame per
    non-empty shot (its midpoint). Deterministic."""
    step = max(1, int(round(fps / candidate_fps))) if candidate_fps > 0 else 1
    out: list[list[int]] = []
    for first, last in shots:
        if last < first:
            out.append([])
            continue
        idxs = list(range(first, last + 1, step))
        if not idxs:
            idxs = [(first + last) // 2]
        out.append(idxs)
    return out


def _centroid_nearest(vectors: Sequence[Sequence[float]]) -> int:
    """Index of the real vector closest to the set centroid (a medoid-lite representative)."""
    dim = len(vectors[0])
    centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    return max(range(len(vectors)), key=lambda i: cosine_similarity(vectors[i], centroid))


def _most_novel(vectors: Sequence[Sequence[float]], picked: Sequence[int]) -> tuple[int, float]:
    """Unpicked index farthest from the picked set (min cosine distance), with that distance."""
    best_i, best_d = -1, -2.0
    for i in range(len(vectors)):
        if i in picked:
            continue
        # distance to the picked set = 1 - max similarity to any picked point (min distance).
        d = 1.0 - max(cosine_similarity(vectors[i], vectors[p]) for p in picked)
        if d > best_d:
            best_d, best_i = d, i
    return best_i, best_d


def farthest_point_sample(vectors: Sequence[Sequence[float]], k: int) -> list[int]:
    """Greedy farthest-point sampling in cosine space -> ``k`` diverse indices (sorted).

    Seeds with the centroid-nearest point (deterministic), then repeatedly adds the point with the
    largest cosine-distance to the already-picked set. Returns all indices when ``k`` >= n."""
    n = len(vectors)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    picked = [_centroid_nearest(vectors)]
    while len(picked) < k:
        best_i, _best_d = _most_novel(vectors, picked)
        picked.append(best_i)
    return sorted(picked)


def farthest_point_sample_adaptive(vectors: Sequence[Sequence[float]], *, min_k: int, max_k: int,
                                   novelty_threshold: float) -> list[int]:
    """Coverage-driven FPS: grow the picked set while the most novel remaining point still has
    cosine distance >= ``novelty_threshold`` to the set (a top-p-style stop on residual novelty),
    but never below ``min_k`` nor above ``max_k`` picks. Deterministic; sorted indices."""
    n = len(vectors)
    max_k = min(max_k, n)
    min_k = max(1, min(min_k, max_k))
    if n == 0 or max_k <= 0:
        return []
    picked = [_centroid_nearest(vectors)]
    while len(picked) < max_k:
        best_i, best_d = _most_novel(vectors, picked)
        if best_i < 0 or (len(picked) >= min_k and best_d < novelty_threshold):
            break
        picked.append(best_i)
    return sorted(picked)


def representative_indices(vectors: Sequence[Sequence[float]], k: int) -> list[int]:
    """``k`` representatives of a set: centroid-nearest for k<=1, else farthest-point sampling."""
    if not vectors:
        return []
    if k <= 1:
        return [_centroid_nearest(vectors)]
    return farthest_point_sample(vectors, k)


def select_roster_keyframes(per_shot_indices: Sequence[Sequence[int]],
                            embed_indices: Callable[[list[int]], list[list[float]]],
                            *, per_shot_k: int, budget: int, budget_max: int | None = None,
                            novelty_threshold: float = 0.0, min_ratio: float = 0.0,
                            total_frames: int = 0) -> list[int]:
    """Movie-wide keyframe budget: per-shot representative(s) -> global FPS.

    With only ``budget`` (legacy), FPS cuts to exactly that fixed count. With ``budget_max``, the
    count adapts to the film instead: ``budget`` (raised to ``min_ratio * total_frames`` when
    larger) is the floor, ``budget_max`` the cap, and between them selection keeps adding frames
    only while residual visual novelty stays >= ``novelty_threshold``. Longer/more varied films
    therefore get more keyframes; short single-set films stop at the floor.

    ``embed_indices`` maps a list of absolute frame indices to their embeddings (injected: the
    orchestrator extracts+DINOv3-embeds; tests fake it). Returns sorted absolute frame indices.
    Deterministic given the same embeddings."""
    import math
    reps: list[int] = []
    for shot_idxs in per_shot_indices:
        idxs = list(shot_idxs)
        if not idxs:
            continue
        if len(idxs) == 1:
            reps.append(idxs[0])
            continue
        vecs = embed_indices(idxs)
        for j in representative_indices(vecs, per_shot_k):
            reps.append(idxs[j])
    reps = sorted(set(reps))
    if budget_max is None:
        if len(reps) <= budget:
            return reps
        rep_vecs = embed_indices(reps)
        return sorted(reps[j] for j in farthest_point_sample(rep_vecs, budget))
    floor = max(int(budget), math.ceil(min_ratio * total_frames) if total_frames > 0 else 0)
    cap = max(floor, int(budget_max))
    if len(reps) <= floor:
        return reps
    rep_vecs = embed_indices(reps)
    picked = farthest_point_sample_adaptive(rep_vecs, min_k=floor, max_k=cap,
                                            novelty_threshold=novelty_threshold)
    return sorted(reps[j] for j in picked)


# Head nouns that are OBJECTS in frame, never narrative stages. A "location" whose grounding
# phrase ends in one of these is a roster VLM mislabel (BBB v10: loc_weathered_tree_branch,
# loc_sun_dappled_tree_trunk, 92-rep loc_lush_canopy_overhang). Deterministic guard: the prompt
# alone does not restrain an 8B model reliably.
_OBJECT_NOT_PLACE_WORDS = {
    "trunk", "branch", "stump", "log", "rock", "boulder", "stone", "cloud", "sky",
    "canopy", "leaf", "leaves", "bush", "flower", "mushroom", "glider", "overhang",
}


def demote_object_locations(roster: Sequence[dict]) -> tuple[list[dict], list[str]]:
    """Reclassify 'location' entries whose head noun is an in-frame object to 'prop'.

    Returns (updated roster, demoted names). Demoted entries then face the story-prop gate like
    any other prop (a background trunk is dropped; a story-relevant one survives). Deterministic,
    unit-testable."""
    out: list[dict] = []
    demoted: list[str] = []
    for entry in roster:
        e = dict(entry)
        if e.get("kind") == "location":
            phrase = str(e.get("grounding_phrase") or e.get("name") or "")
            words = [w for w in phrase.replace("_", " ").lower().split() if w]
            if words and words[-1] in _OBJECT_NOT_PLACE_WORDS:
                e["kind"] = "prop"
                demoted.append(str(e.get("name") or phrase))
        out.append(e)
    return out, demoted


def merge_roster(batches: Sequence[Sequence[dict]]) -> list[dict]:
    """Merge per-batch roster discoveries by (normalized name, kind); first non-empty wins for
    description/grounding_phrase, static attributes are unioned. Deterministic order (first-seen)."""
    from vmem_bench.annotation.pipeline_track_first.consolidation import normalize_entity_name
    merged: dict[tuple[str, str], dict] = {}
    for batch in batches:
        for ent in batch:
            name = normalize_entity_name(ent.get("name") or "")
            kind = ent.get("kind") or ""
            if not name or not kind:
                continue
            key = (name.lower(), kind)
            if key not in merged:
                merged[key] = {"name": name, "kind": kind,
                               "grounding_phrase": (ent.get("grounding_phrase") or name).strip(),
                               "description": (ent.get("description") or "").strip(),
                               "static_attributes": dict(ent.get("static_attributes") or {})}
            else:
                cur = merged[key]
                if not cur["description"] and ent.get("description"):
                    cur["description"] = ent["description"].strip()
                for k, v in (ent.get("static_attributes") or {}).items():
                    cur["static_attributes"].setdefault(k, v)
    return list(merged.values())
