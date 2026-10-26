"""Offline unit checks for VLM-primary batch identity resolution (identity_resolution.py).

No GPU: real DINOv3-shaped vectors are replaced by small synthetic ones; the VLM is a fake role
object (mirrors the FakeJudger pattern in test_pipeline_track_first.py /
test_annotation_fixes.py). Run:
    cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_identity_resolution.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first.identity_resolution import (
    TrackletObservation, bucket_by_group, commit_groups_to_registry, merge_clusters,
    precluster_bucket, resolve_identities, roster_completeness_findings, verify_clusters)


def _obs(index: int, *, kind: str = "character", name: str = "rabbit", signature=None,
        identity_group: str = "rabbit", chunk_id: int = 0, frame_index: int = 0,
        grounding_score: float = 0.5, crop_path: str = "", static=None,
        roster_matched: bool = True, track_id: int | None = None) -> TrackletObservation:
    return TrackletObservation(
        index=index, kind=kind, name=name, description="", static_attributes=static or {},
        signature=signature, crop_path=crop_path or f"crop_{index}.jpg", bbox=[0, 0, 100, 100],
        frame_index=frame_index, chunk_id=chunk_id, grounding_score=grounding_score,
        track_id=track_id if track_id is not None else index, bbox_source="tracker",
        identity_group=identity_group, roster_matched=roster_matched)


WEIGHTS = {"body": 1.0, "face": 0.0, "class": 0.0}


# --- bucketing -----------------------------------------------------------------------------------

def test_bucket_by_group_splits_kind_and_identity_group() -> None:
    obs = [
        _obs(0, kind="character", identity_group="rabbit"),
        _obs(1, kind="character", identity_group="rabbit"),
        _obs(2, kind="character", identity_group="fox"),
        _obs(3, kind="prop", identity_group="apple"),
    ]
    buckets = bucket_by_group(obs)
    assert buckets[("character", "rabbit")] == [0, 1]
    assert buckets[("character", "fox")] == [2]
    assert buckets[("prop", "apple")] == [3]


# --- preclustering (reuses fuse_similarity + _static_compatible) -----------------------------

def test_precluster_bucket_groups_similar_signatures() -> None:
    obs = [
        _obs(0, signature=[1.0, 0.0]),
        _obs(1, signature=[0.95, 0.05]),   # close to 0 -> same cluster
        _obs(2, signature=[0.0, 1.0]),     # orthogonal -> different cluster
    ]
    clusters = precluster_bucket(obs, [0, 1, 2], weights=WEIGHTS, threshold=0.8)
    assert sorted(clusters) == [[0, 1], [2]]


def test_precluster_bucket_static_conflict_blocks_merge() -> None:
    obs = [
        _obs(0, signature=[1.0, 0.0], static={"species": "fox"}),
        _obs(1, signature=[0.99, 0.01], static={"species": "bird"}),  # conflicting species
    ]
    clusters = precluster_bucket(obs, [0, 1], weights=WEIGHTS, threshold=0.5,
                                 static_overlap_threshold=0.75)
    assert sorted(clusters) == [[0], [1]]


def test_precluster_bucket_missing_signature_stays_singleton() -> None:
    obs = [_obs(0, signature=None), _obs(1, signature=[1.0, 0.0])]
    clusters = precluster_bucket(obs, [0, 1], weights=WEIGHTS, threshold=0.5)
    assert sorted(clusters) == [[0], [1]]


# --- VLM cluster verification --------------------------------------------------------------------

class _FakeVerifyVlm:
    """Returns a scripted verdict per call, in call order (for deterministic assertions)."""

    def __init__(self, verdicts: list[dict]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[list] = []

    def verify_cluster(self, crops):
        self.calls.append(list(crops))
        return self._verdicts.pop(0)


def test_verify_clusters_skips_singletons_no_vlm_call() -> None:
    obs = [_obs(0)]
    vlm = _FakeVerifyVlm([])
    refined, findings = verify_clusters(obs, [[0]], judge_vlm=vlm)
    assert refined == [[0]]
    assert findings == []
    assert vlm.calls == []


def test_verify_clusters_coherent_keeps_cluster() -> None:
    obs = [_obs(0), _obs(1)]
    vlm = _FakeVerifyVlm([{"coherent": True, "subgroups": []}])
    refined, findings = verify_clusters(obs, [[0, 1]], judge_vlm=vlm)
    assert refined == [[0, 1]]
    assert findings == []
    assert len(vlm.calls[0]) == 2


def test_verify_clusters_splits_on_incoherent_verdict() -> None:
    obs = [_obs(0, signature=[1.0, 0.0]), _obs(1, signature=[0.0, 1.0]),
           _obs(2, signature=[1.0, 0.0])]
    vlm = _FakeVerifyVlm([{"coherent": False, "subgroups": [[0, 2], [1]]}])
    refined, findings = verify_clusters(obs, [[0, 1, 2]], judge_vlm=vlm)
    assert sorted(refined) == [[0, 2], [1]]
    assert any(f["code"] == "cluster_split_by_vlm" for f in findings)


def test_verify_clusters_large_cluster_samples_and_reattaches_excluded() -> None:
    # 5 members, cap=3: a diverse subset is sent to the VLM. Split verdict on the SAMPLE only;
    # the excluded member is deterministically reattached by nearest body cosine.
    obs = [
        _obs(0, signature=[1.0, 0.0], grounding_score=0.9),
        _obs(1, signature=[0.99, 0.01], grounding_score=0.1),  # likely excluded (low score, near 0)
        _obs(2, signature=[0.0, 1.0], grounding_score=0.9),
        _obs(3, signature=[0.98, 0.02], grounding_score=0.8),
        _obs(4, signature=[0.01, 0.99], grounding_score=0.7),
    ]
    vlm = _FakeVerifyVlm([{"coherent": False, "subgroups": [[0, 1], [2]]}])
    refined, findings = verify_clusters(obs, [[0, 1, 2, 3, 4]], judge_vlm=vlm, max_crops=3)
    all_members = sorted(i for group in refined for i in group)
    assert all_members == [0, 1, 2, 3, 4]  # nobody dropped
    # member 1 (near [1,0]) must land in the "rabbit-ish" subgroup, not the "fox-ish" one.
    rabbit_group = next(g for g in refined if 0 in g)
    assert 1 in rabbit_group
    assert any(f["code"] == "cluster_split_by_vlm" for f in findings)


def test_verify_clusters_vlm_error_falls_back_to_singletons_and_flags() -> None:
    class _Boom:
        def verify_cluster(self, crops):
            raise RuntimeError("endpoint down")
    refined, findings = verify_clusters([_obs(0), _obs(1)], [[0, 1]], judge_vlm=_Boom())
    assert sorted(refined) == [[0], [1]]
    assert findings[0]["code"] == "cluster_verify_error"


# --- VLM cross-cluster merge --------------------------------------------------------------------

class _FakeMergeVlm:
    def __init__(self, groups: list[list[int]]) -> None:
        self._groups = groups
        self.merge_calls = 0

    def group_same_individuals(self, labeled, crops):
        self.merge_calls += 1
        return self._groups


def test_merge_clusters_merges_different_phrase_groups_same_kind() -> None:
    # Two clusters under different identity_group phrases ("white_rabbit" vs "big_buck_bunny")
    # that the VLM says are the same individual -> merged into one final group.
    obs = [_obs(0, identity_group="white_rabbit"), _obs(1, identity_group="big_buck_bunny")]
    vlm = _FakeMergeVlm([[0, 1]])
    final, findings = merge_clusters(obs, [[0], [1]], judge_vlm=vlm, kinds=("character",))
    assert final == [[0, 1]]
    assert any(f["code"] == "cluster_merged_by_vlm" for f in findings)
    assert vlm.merge_calls == 1


def test_merge_clusters_static_conflict_vetoes_vlm_merge() -> None:
    obs = [_obs(0, identity_group="a", static={"species": "fox"}),
          _obs(1, identity_group="b", static={"species": "bird"})]
    vlm = _FakeMergeVlm([[0, 1]])  # VLM says merge, but static attrs conflict
    final, findings = merge_clusters(obs, [[0], [1]], judge_vlm=vlm, kinds=("character",))
    assert sorted(final) == [[0], [1]]
    assert any(f["code"] == "cluster_merge_static_veto" for f in findings)


def test_merge_clusters_single_cluster_no_vlm_call() -> None:
    obs = [_obs(0)]
    vlm = _FakeMergeVlm([])
    final, findings = merge_clusters(obs, [[0]], judge_vlm=vlm, kinds=("character",))
    assert final == [[0]]
    assert vlm.merge_calls == 0


def test_merge_clusters_locations_never_merged_kind_not_in_list() -> None:
    obs = [_obs(0, kind="location", identity_group="a"), _obs(1, kind="location", identity_group="b")]
    vlm = _FakeMergeVlm([[0, 1]])
    final, findings = merge_clusters(obs, [[0], [1]], judge_vlm=vlm, kinds=("character", "prop"))
    assert sorted(final) == [[0], [1]]
    assert vlm.merge_calls == 0


# --- roster completeness -------------------------------------------------------------------------

def test_roster_completeness_flags_unmatched_large_group() -> None:
    obs = [_obs(i, roster_matched=False) for i in range(3)]
    findings = roster_completeness_findings(obs, [[0, 1, 2]], min_observations=3)
    assert findings[0]["code"] == "roster_incomplete_unmatched_cluster"


def test_roster_completeness_ignores_small_group() -> None:
    obs = [_obs(i, roster_matched=False) for i in range(2)]
    assert roster_completeness_findings(obs, [[0, 1]], min_observations=3) == []


def test_roster_completeness_ignores_matched_group() -> None:
    obs = [_obs(i, roster_matched=True) for i in range(3)]
    assert roster_completeness_findings(obs, [[0, 1, 2]], min_observations=3) == []


# --- end-to-end orchestration ---------------------------------------------------------------------

def test_resolve_identities_end_to_end_precluster_verify_merge() -> None:
    obs = [
        _obs(0, identity_group="white_rabbit", signature=[1.0, 0.0], grounding_score=0.9),
        _obs(1, identity_group="white_rabbit", signature=[0.98, 0.02], grounding_score=0.8),
        _obs(2, identity_group="big_buck_bunny", signature=[0.95, 0.05], grounding_score=0.7),
        _obs(3, identity_group="red_fox", signature=[0.0, 1.0], grounding_score=0.9),
    ]

    class _Vlm(_FakeVerifyVlm, _FakeMergeVlm):
        def __init__(self):
            # cluster [0,1] (the two white_rabbit observations) verifies as coherent.
            _FakeVerifyVlm.__init__(self, [{"coherent": True, "subgroups": []}])
            _FakeMergeVlm.__init__(self, [[0, 1]])  # the two clusters ARE the same individual

    vlm = _Vlm()
    result = resolve_identities(obs, judge_vlm=vlm, weights=WEIGHTS, precluster_threshold=0.8)
    # rabbit + bunny clusters merge into one final entity; fox stays separate.
    assert sorted(sorted(g) for g in result.final_groups) == [[0, 1, 2], [3]]


def test_resolve_identities_threshold_for_bucket_overrides_per_group() -> None:
    # Two identity_groups with the SAME moderate similarity (cos=0.8): the "anchored" group gets a
    # permissive override threshold (0.5, clears) while the other keeps the strict global default
    # (0.9, does not clear) -- demonstrates per-bucket threshold override actually takes effect.
    obs = [
        _obs(0, identity_group="anchored:bunny", signature=[1.0, 0.0]),
        _obs(1, identity_group="anchored:bunny", signature=[0.8, 0.6]),  # cos == 0.8 with obs 0
        _obs(2, identity_group="plain:fox", signature=[1.0, 0.0]),
        _obs(3, identity_group="plain:fox", signature=[0.8, 0.6]),
    ]
    vlm = _FakeVerifyVlm([{"coherent": True, "subgroups": []}])  # only the anchored pair clusters

    def threshold_for(_kind: str, group: str) -> float:
        return 0.5 if group.startswith("anchored:") else 0.9

    result = resolve_identities(obs, judge_vlm=vlm, weights=WEIGHTS, precluster_threshold=0.9,
                                threshold_for_bucket=threshold_for)
    groups = sorted(sorted(g) for g in result.final_groups)
    assert [0, 1] in groups            # anchored pair merged under the permissive override
    assert [2] in groups and [3] in groups  # plain pair stayed apart under the strict default


def test_resolve_identities_roster_incomplete_finding_surfaces() -> None:
    # All 3 share one identity_group -> one precluster of size 3 -> ONE VLM verify call (coherent),
    # and only one cluster of this kind -> merge_clusters never calls group_same_individuals.
    obs = [_obs(i, identity_group="mystery", roster_matched=False,
                signature=[1.0, 0.0]) for i in range(3)]
    vlm = _FakeVerifyVlm([{"coherent": True, "subgroups": []}])
    result = resolve_identities(obs, judge_vlm=vlm, weights=WEIGHTS, precluster_threshold=0.5,
                                roster_completeness_min_observations=3)
    assert any(f["code"] == "roster_incomplete_unmatched_cluster" for f in result.findings)


# --- registry commit (thin last mile) -------------------------------------------------------------

def test_commit_groups_to_registry_creates_one_entity_per_group() -> None:
    with tempfile.TemporaryDirectory() as d:
        registry = Registry()
        obs = [
            _obs(0, name="rabbit", signature=[1.0, 0.0], chunk_id=0, frame_index=0,
                crop_path=str(Path(d) / "a.jpg")),
            _obs(1, name="rabbit", signature=[0.99, 0.01], chunk_id=1, frame_index=10,
                crop_path=str(Path(d) / "b.jpg")),
            _obs(2, name="fox", signature=[0.0, 1.0], chunk_id=0, frame_index=5,
                crop_path=str(Path(d) / "c.jpg")),
        ]
        mapping, rep_ids = commit_groups_to_registry(registry, obs, [[0, 1], [2]])
        assert len(mapping) == 2
        rabbit_entity = registry.entities[mapping[0]]
        assert len(rabbit_entity.representations) == 2
        assert rabbit_entity.representations[0].qa["cluster_group_index"] == 0
        fox_entity = registry.entities[mapping[1]]
        assert len(fox_entity.representations) == 1
        assert rabbit_entity.entity_id != fox_entity.entity_id
        # every observation index maps to the representation_id it was committed as.
        assert set(rep_ids) == {0, 1, 2}
        assert rep_ids[0] in {r.representation_id for r in rabbit_entity.representations}
        assert rep_ids[2] in {r.representation_id for r in fox_entity.representations}


def test_round_robin_role_cycles_across_underlying_roles() -> None:
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import RoundRobinRole

    class _Counter:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.calls = 0

        def verify_cluster(self, crops):
            self.calls += 1
            return {"tag": self.tag, "coherent": True, "subgroups": []}

    a, b, c = _Counter("a"), _Counter("b"), _Counter("c")
    pool = RoundRobinRole([a, b, c])
    tags = [pool.verify_cluster([])["tag"] for _ in range(7)]
    assert tags == ["a", "b", "c", "a", "b", "c", "a"]
    assert (a.calls, b.calls, c.calls) == (3, 2, 2)


def test_round_robin_role_thread_safe_under_concurrency() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import RoundRobinRole

    class _Counter:
        def __init__(self) -> None:
            self.calls = 0

        def verify_cluster(self, crops):
            self.calls += 1
            return {"coherent": True, "subgroups": []}

    roles = [_Counter() for _ in range(4)]
    pool = RoundRobinRole(roles)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: pool.verify_cluster([]), range(200)))
    assert sum(r.calls for r in roles) == 200
    assert all(r.calls == 50 for r in roles)  # perfectly even under concurrency, no lost/dup calls


def test_round_robin_role_rejects_empty_pool() -> None:
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import RoundRobinRole
    try:
        RoundRobinRole([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
