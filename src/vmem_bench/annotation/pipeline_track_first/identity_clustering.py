"""Deterministic pre-clustering for cross-shot identity resolution (VLM-primary redesign).

Replaces the online greedy nearest-neighbor assignment in ``reid.py::reid_assign`` (accept the
first/best entity whose FUSED score clears a single global ``reid_threshold``) with an offline,
batch, transitivity-aware step: cluster ALL of a movie's tracklets (within one kind/identity_group
bucket) at once, then let a VLM adjudicate each candidate cluster (see ``identity_resolution.py``).
This module only produces the CANDIDATE clusters; it never decides gold identity by itself.

Why not simple connected components (single-link clustering)
--------------------------------------------------------------
Building a graph with one edge per pair whose similarity clears a threshold, then taking connected
components, is single-link clustering: transitivity chains through the WEAKEST edge. Given a noisy,
low-instance-discriminative embedding (the diagnosed root cause -- DINOv3/SigLIP are semantic, not
identity, embeddings), a single spurious "bridge" pair (e.g. a bad-angle rabbit crop that happens to
look 0.50 similar to a squirrel crop) silently drags two genuinely different individuals into one
cluster, and a third weak bridge can chain in a fourth. This is invisible to a human reviewer who
only sees clean crop pairs later.

complete-link (this module's default) and average-link avoid chaining: a merge is only allowed when
the WORST (complete) or MEAN (average) pairwise similarity across the two clusters-to-merge clears
the threshold, not just the best one. This formalizes the ad-hoc ``cluster_min_similarity`` guard
already present in ``reid.py`` (checked post-hoc, only against the mean-based accept) as the PRIMARY
clustering criterion. Bias stays "split rather than merge": under-merging (more, smaller candidate
clusters) is cheaply repaired downstream by the VLM cross-cluster merge pass; over-merging (one
cluster silently mixing individuals) is the dangerous failure this module exists to avoid.

Deterministic, pure, CPU-only, no VLM/GPU -- see tests/test_identity_clustering.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Linkage = str  # "complete" | "average"


def cluster_by_linkage(
    n: int,
    similarity: Callable[[int, int], float | None],
    *,
    threshold: float,
    linkage: Linkage = "complete",
    compatible: Callable[[int, int], bool] | None = None,
) -> list[list[int]]:
    """Agglomerative clustering of ``range(n)`` using complete-link or average-link linkage.

    ``similarity(i, j)`` returns a cosine-like score in [-1, 1] or ``None`` when the pair carries
    no comparable evidence (e.g. one side has no signature) -- ``None`` pairs are excluded from the
    linkage aggregate, they are never treated as -1 (a missing cue must not veto a merge that other
    cues support). ``compatible(i, j)``, when given, is a HARD gate checked on every cross-pair of a
    candidate merge (e.g. static-attribute compatibility): if any cross-pair is incompatible, the
    merge is blocked outright regardless of similarity score (same "worst pair wins" spirit as
    complete-link, but for a categorical gate instead of a continuous score).

    Repeatedly merges the pair of clusters with the highest linkage score, stopping when the best
    remaining score is below ``threshold`` (or undefined -- no comparable pair). Ties break on the
    lexicographically smallest pair of member-index tuples, so results are reproducible.

    Returns clusters as sorted lists of original indices, sorted by their smallest member (stable,
    deterministic order across runs on the same input).

    Complexity: O(n^2) pairwise cache + O(k^3) merge search over k <= n clusters. Fine at the scale
    this module targets (tens to a few hundred tracklets per kind/identity_group bucket per movie).
    """
    if n <= 0:
        return []
    if n == 1:
        return [[0]]

    # Cache pairwise similarity/compatibility once; clusters shrink but the raw pair values do not.
    sim_cache: dict[tuple[int, int], float | None] = {}
    compat_cache: dict[tuple[int, int], bool] = {}

    def _sim(i: int, j: int) -> float | None:
        key = (i, j) if i < j else (j, i)
        if key not in sim_cache:
            sim_cache[key] = similarity(key[0], key[1])
        return sim_cache[key]

    def _compat(i: int, j: int) -> bool:
        if compatible is None:
            return True
        key = (i, j) if i < j else (j, i)
        if key not in compat_cache:
            compat_cache[key] = bool(compatible(key[0], key[1]))
        return compat_cache[key]

    def _linkage_score(a: Sequence[int], b: Sequence[int]) -> float | None:
        scores = [s for i in a for j in b if (s := _sim(i, j)) is not None]
        if not scores:
            return None
        return min(scores) if linkage == "complete" else sum(scores) / len(scores)

    def _fully_compatible(a: Sequence[int], b: Sequence[int]) -> bool:
        return all(_compat(i, j) for i in a for j in b)

    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best_score = None
        best_pair: tuple[int, int] | None = None
        for a_idx in range(len(clusters)):
            for b_idx in range(a_idx + 1, len(clusters)):
                a, b = clusters[a_idx], clusters[b_idx]
                if not _fully_compatible(a, b):
                    continue
                score = _linkage_score(a, b)
                if score is None or score < threshold:
                    continue
                tie_key = (min(a), min(b))
                if best_score is None or score > best_score or (
                        score == best_score and tie_key < best_pair):  # type: ignore[operator]
                    best_score, best_pair = score, tie_key
                    best_a_idx, best_b_idx = a_idx, b_idx
        if best_pair is None:
            break
        merged = sorted(clusters[best_a_idx] + clusters[best_b_idx])
        clusters = [c for idx, c in enumerate(clusters) if idx not in (best_a_idx, best_b_idx)]
        clusters.append(merged)

    clusters.sort(key=lambda c: c[0])
    return clusters


def normalize_partition(groups: object, n: int) -> list[list[int]]:
    """Coerce a (possibly malformed) VLM-style grouping into a total partition of ``range(n)``.

    Clamps to valid, unseen indices; any index the model omitted (or double-used, or invented out
    of range) becomes its own singleton. Deterministic and total -- shared by ``verify_cluster`` and
    the cross-cluster merge pass so both degrade the same way on a malformed model response."""
    seen: set[int] = set()
    out: list[list[int]] = []
    for group in (groups if isinstance(groups, list) else []):
        clean = []
        for idx in (group if isinstance(group, list) else []):
            if isinstance(idx, (int, float)) and 0 <= int(idx) < n and int(idx) not in seen:
                clean.append(int(idx))
                seen.add(int(idx))
        if clean:
            out.append(sorted(clean))
    for idx in range(n):
        if idx not in seen:
            out.append([idx])
    return sorted(out)
