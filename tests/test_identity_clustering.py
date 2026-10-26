"""Offline unit checks for the deterministic identity pre-clustering algorithm.

No GPU / VLM: pure similarity-matrix inputs. Run:
    cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_identity_clustering.py
"""

from __future__ import annotations

from vmem_bench.annotation.pipeline_track_first.identity_clustering import cluster_by_linkage, normalize_partition


def _matrix_similarity(matrix: list[list[float]]):
    def sim(i: int, j: int) -> float | None:
        return matrix[i][j]
    return sim


# --- basic clustering --------------------------------------------------------------------------

def test_two_tight_clusters_separate() -> None:
    # 0,1 tight (0.9); 2,3 tight (0.85); cross-pairs weak (0.1) -> two clusters, no cross-merge.
    m = [
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.85],
        [0.1, 0.1, 0.85, 1.0],
    ]
    clusters = cluster_by_linkage(4, _matrix_similarity(m), threshold=0.5, linkage="complete")
    assert sorted(clusters) == [[0, 1], [2, 3]]


def test_all_similar_merge_into_one() -> None:
    m = [[1.0 if i == j else 0.9 for j in range(3)] for i in range(3)]
    clusters = cluster_by_linkage(3, _matrix_similarity(m), threshold=0.5, linkage="complete")
    assert clusters == [[0, 1, 2]]


def test_below_threshold_stays_singletons() -> None:
    m = [[1.0 if i == j else 0.2 for j in range(3)] for i in range(3)]
    clusters = cluster_by_linkage(3, _matrix_similarity(m), threshold=0.5, linkage="complete")
    assert clusters == [[0], [1], [2]]


def test_n_le_1_edge_cases() -> None:
    assert cluster_by_linkage(0, lambda i, j: 1.0, threshold=0.5) == []
    assert cluster_by_linkage(1, lambda i, j: 1.0, threshold=0.5) == [[0]]


# --- chaining: the core reason complete-link beats connected components ------------------------

def test_complete_link_resists_chaining_single_link_would_not() -> None:
    """0<->1 strong (0.9, e.g. two rabbit crops), 2<->3 strong (0.85, two fox crops), but a single
    noisy bridge 1<->2 = 0.55 clears a naive connected-components threshold of 0.5 and would chain
    all four into one cluster. Complete-link requires the WORST cross-pair (e.g. 0<->3 = 0.05) to
    also clear the threshold, so it correctly keeps the two individuals apart."""
    m = [
        [1.0, 0.90, 0.10, 0.05],
        [0.90, 1.0, 0.55, 0.10],
        [0.10, 0.55, 1.0, 0.85],
        [0.05, 0.10, 0.85, 1.0],
    ]
    # Sanity: naive connected-components AT THE SAME THRESHOLD would chain everything, because
    # every step of the chain (0-1, 1-2, 2-3) individually clears 0.5.
    assert m[0][1] >= 0.5 and m[1][2] >= 0.5 and m[2][3] >= 0.5
    clusters = cluster_by_linkage(4, _matrix_similarity(m), threshold=0.5, linkage="complete")
    assert sorted(clusters) == [[0, 1], [2, 3]]


def test_average_link_is_looser_than_complete_but_still_resists_bad_chains() -> None:
    # Same bridge scenario; average-link considers the MEAN of cross-pairs (0.10+0.55+0.90+0.05)/...
    # still well below 0.5 for the 0-1 vs 2-3 merge, so it also keeps them separate here.
    m = [
        [1.0, 0.90, 0.10, 0.05],
        [0.90, 1.0, 0.55, 0.10],
        [0.10, 0.55, 1.0, 0.85],
        [0.05, 0.10, 0.85, 1.0],
    ]
    clusters = cluster_by_linkage(4, _matrix_similarity(m), threshold=0.5, linkage="average")
    assert sorted(clusters) == [[0, 1], [2, 3]]


def test_average_link_merges_when_complete_link_would_not() -> None:
    # Cross-pairs are moderate (0.45) but not uniformly bad; average of {0.45,0.5,0.4,0.55}=0.475
    # is just under a 0.45 threshold... construct so average clears 0.45 while the worst pair (0.30)
    # does not -- demonstrates the two linkages genuinely differ, not just a relabeling.
    m = [
        [1.0, 0.9, 0.55, 0.30],
        [0.9, 1.0, 0.50, 0.60],
        [0.55, 0.50, 1.0, 0.85],
        [0.30, 0.60, 0.85, 1.0],
    ]
    complete = cluster_by_linkage(4, _matrix_similarity(m), threshold=0.45, linkage="complete")
    average = cluster_by_linkage(4, _matrix_similarity(m), threshold=0.45, linkage="average")
    assert sorted(complete) == [[0, 1], [2, 3]]  # worst pair 0<->3=0.30 blocks the merge
    assert average == [[0, 1, 2, 3]]              # mean of cross-pairs clears 0.45


# --- missing evidence (None) never vetoes, never fabricates a match ----------------------------

def test_none_similarity_excluded_from_linkage_not_treated_as_veto() -> None:
    # 0<->1 has NO comparable evidence (None, e.g. no face signature on either side) but body-cue
    # equivalents (here just the matrix) are strong for every OTHER pair -> None must not block the
    # merge; it is simply excluded from the aggregate.
    def sim(i: int, j: int) -> float | None:
        pairs = {(0, 1): None, (0, 2): 0.9, (1, 2): 0.9}
        return pairs.get((min(i, j), max(i, j)))
    clusters = cluster_by_linkage(3, sim, threshold=0.5, linkage="complete")
    assert clusters == [[0, 1, 2]]


def test_all_none_similarity_cannot_merge() -> None:
    clusters = cluster_by_linkage(3, lambda i, j: None, threshold=0.5, linkage="complete")
    assert clusters == [[0], [1], [2]]


# --- hard compatibility gate (e.g. static-attribute conflict) ----------------------------------

def test_compatible_gate_blocks_merge_despite_high_similarity() -> None:
    # 0 and 1 have a hard conflict (e.g. species fox vs bird); 0-2 and 1-2 are fine. 0 and 2 still
    # merge on similarity alone, but 1 can never join that cluster (the gate checks EVERY cross
    # pair of a candidate merge, so 1 joining {0,2} would still hit the 0-1 conflict) -- it stays
    # its own cluster despite a 0.95 similarity to everything.
    def sim(i: int, j: int) -> float | None:
        return 0.95
    def compat(i: int, j: int) -> bool:
        return not (i == 0 and j == 1)
    clusters = cluster_by_linkage(3, sim, threshold=0.5, linkage="complete", compatible=compat)
    assert sorted(clusters) == [[0, 2], [1]]


# --- determinism / tie-breaking -----------------------------------------------------------------

def test_deterministic_across_repeated_calls() -> None:
    m = [
        [1.0, 0.7, 0.7, 0.1],
        [0.7, 1.0, 0.7, 0.1],
        [0.7, 0.7, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ]
    results = {tuple(map(tuple, cluster_by_linkage(4, _matrix_similarity(m), threshold=0.5)))
              for _ in range(20)}
    assert len(results) == 1  # same input -> identical output every time


# --- normalize_partition -------------------------------------------------------------------------

def test_normalize_partition_fills_missing_as_singletons() -> None:
    assert normalize_partition([[0, 1]], 3) == [[0, 1], [2]]


def test_normalize_partition_drops_duplicates_and_out_of_range() -> None:
    # index 1 appears twice (second occurrence dropped); index 5 is out of range (dropped);
    # "not a list" garbage group ignored entirely.
    assert normalize_partition([[0, 1], [1, 5], "garbage", [2]], 3) == [[0, 1], [2]]


def test_normalize_partition_non_list_input_is_all_singletons() -> None:
    assert normalize_partition(None, 3) == [[0], [1], [2]]
    assert normalize_partition("not a list", 2) == [[0], [1]]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
