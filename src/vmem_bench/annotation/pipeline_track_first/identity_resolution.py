"""VLM-primary batch identity resolution (replaces the online greedy reid_assign hot path).

Root-cause redesign (see docs/benchmark/annotation_tracking_internals.md "identity resolution v2" and
Pitfall_Notes.md): cross-shot identity for characters/props is a CLOSED-SET classification problem
(the cast roster is already known) that was being modeled as OPEN-SET online clustering with a
single global embedding threshold -- a mismatch that produced ~23% duplicate ids on the pre-redesign
baseline and kept failing after track-first (sam3_exemplar_bbb probe: argmax exemplar cosine picks
the wrong species with a ~0.1 margin). This module flips which signal is authoritative:

    1. Bucket tracklet observations by (kind, identity_group) -- same restriction the old
       ``allowed_entity_ids`` used (a detector phrase is a cheap, stable identity prior).
    2. Deterministic pre-cluster each bucket (identity_clustering.cluster_by_linkage,
       complete/average-link -- NOT simple connected components, to avoid single-link chaining
       through one noisy embedding pair).
    3. VLM verifies each multi-member candidate cluster (AUTHORITATIVE, not gray-zone fallback):
       does every crop show the same individual? May split further.
    4. VLM cross-cluster merge WITHIN one kind, across identity_group buckets -- catches the
       classic phrase-fragmentation case ("white_rabbit" vs "big_buck_bunny" naming the same
       recurring character under different roster phrases).
    5. Roster-completeness check: a final group with substantial screen time whose tracklets never
       matched a roster phrase is a candidate MISSED roster entry -- flagged, not silently
       force-classified into the nearest (possibly wrong) roster candidate.

Bias stays "split rather than merge" throughout: an under-split is cheaply repaired by step 4; an
over-merge silently corrupts gold and is invisible to a reviewer looking at clean crops later.

This module never touches a ``Registry`` or does file I/O -- it is pure data in (tracklet
observations) / data out (final groups + findings), so it is unit-testable with a fake VLM role and
no GPU (tests/test_identity_resolution.py). ``commit_groups_to_registry`` (bottom of file) is the
thin, separately-testable last mile that materializes a decided grouping into a Registry via
``reid.commit_tracklet_observation``.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry, _static_compatible
from vmem_bench.annotation.pipeline_track_first.identity_clustering import cluster_by_linkage, normalize_partition
from vmem_bench.annotation.pipeline_track_first.reid import commit_tracklet_observation, fuse_similarity
from vmem_bench.common.schemas import Entity
from vmem_bench.common.vecmath import cosine_similarity


@dataclass(slots=True)
class TrackletObservation:
    """One tracklet's contribution to identity resolution (mirrors ``reid_assign``'s parameters).

    ``index`` is this observation's position in the caller's full observation list -- clusters and
    findings reference observations by this index, never by object identity, so the whole pipeline
    stays plain-data and easy to snapshot/test."""

    index: int
    kind: str
    name: str
    description: str
    static_attributes: dict[str, str]
    signature: list[float] | None
    crop_path: str
    bbox: list[int]
    frame_index: int
    chunk_id: int
    grounding_score: float
    track_id: int | None
    bbox_source: str
    identity_group: str
    roster_matched: bool
    face_signature: list[float] | None = None
    class_signature: list[float] | None = None


@dataclass(slots=True)
class IdentityResolution:
    """Result of ``resolve_identities``: a final partition of observation indices + diagnostics."""

    final_groups: list[list[int]] = field(default_factory=list)
    # group index (position in final_groups) -> diagnostic dict (method/confidence/notes).
    group_provenance: dict[int, dict[str, Any]] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)


# --- bucketing --------------------------------------------------------------------------------

def bucket_by_group(observations: Sequence[TrackletObservation]) -> dict[tuple[str, str], list[int]]:
    """Group observation indices by (kind, identity_group), preserving input order within a bucket."""
    buckets: dict[tuple[str, str], list[int]] = {}
    for obs in observations:
        buckets.setdefault((obs.kind, obs.identity_group), []).append(obs.index)
    return buckets


# --- pre-clustering (deterministic) -----------------------------------------------------------

def _pairwise_similarity(observations: Sequence[TrackletObservation], weights: dict[str, float],
                         a: int, b: int) -> float | None:
    obs_a, obs_b = observations[a], observations[b]
    if obs_a.signature is None or obs_b.signature is None:
        return None
    body = cosine_similarity(obs_a.signature, obs_b.signature)
    face = None
    if obs_a.face_signature is not None and obs_b.face_signature is not None:
        face = cosine_similarity(obs_a.face_signature, obs_b.face_signature)
    klass = None
    if obs_a.class_signature is not None and obs_b.class_signature is not None:
        klass = cosine_similarity(obs_a.class_signature, obs_b.class_signature)
    return fuse_similarity({"body": body, "face": face, "class": klass}, weights)


def _static_compatible_pair(observations: Sequence[TrackletObservation], threshold: float,
                            a: int, b: int) -> bool:
    return _static_compatible(observations[a].static_attributes, observations[b].static_attributes,
                              threshold)


def precluster_bucket(observations: Sequence[TrackletObservation], indices: Sequence[int], *,
                      weights: dict[str, float], threshold: float, linkage: str = "complete",
                      static_overlap_threshold: float = 0.75) -> list[list[int]]:
    """Cluster one (kind, identity_group) bucket's tracklets into candidate identity groups.

    Operates on a LOCAL index space (0..len(indices)-1) internally and maps back to the caller's
    original observation indices in the return value, so ``identity_clustering`` stays agnostic of
    what an "observation" is."""
    local = list(indices)
    if not local:
        return []

    def sim(i: int, j: int) -> float | None:
        return _pairwise_similarity(observations, weights, local[i], local[j])

    def compat(i: int, j: int) -> bool:
        return _static_compatible_pair(observations, static_overlap_threshold, local[i], local[j])

    local_clusters = cluster_by_linkage(len(local), sim, threshold=threshold, linkage=linkage,
                                       compatible=compat)
    return [[local[i] for i in cluster] for cluster in local_clusters]


# --- representative / diverse crop selection --------------------------------------------------

def _pick_representative(observations: Sequence[TrackletObservation], indices: Sequence[int]) -> int:
    """Highest-grounding-score member of a group; ties broken by smallest index (deterministic)."""
    return max(indices, key=lambda i: (observations[i].grounding_score, -i))


def _pick_diverse_subset(observations: Sequence[TrackletObservation], indices: Sequence[int],
                         limit: int) -> list[int]:
    """Greedy max-min-distance subset (mirrors pipeline_track_first._diverse_reps), deterministic.

    Anchors on the highest-grounding-score member, then repeatedly adds the member maximizing its
    minimum visual distance to the already-picked set. Falls back to index order when no member has
    a usable signature (distance undefined)."""
    pool = sorted(indices, key=lambda i: (-observations[i].grounding_score, i))
    if limit <= 0 or not pool:
        return []
    chosen = [pool.pop(0)]
    while pool and len(chosen) < limit:
        def score(i: int) -> tuple[float, float, int]:
            sig = observations[i].signature
            if sig is None:
                return (-1.0, observations[i].grounding_score, -i)
            dists = [1.0 - cosine_similarity(sig, observations[c].signature)
                    for c in chosen if observations[c].signature is not None]
            diversity = min(dists) if dists else -1.0
            return (diversity, observations[i].grounding_score, -i)
        best = max(pool, key=score)
        chosen.append(best)
        pool.remove(best)
    return chosen


# --- VLM cluster verification (authoritative) --------------------------------------------------

def verify_clusters(observations: Sequence[TrackletObservation], clusters: Sequence[list[int]], *,
                    judge_vlm: Any, out_root: Path | None = None, max_crops: int = 8,
                    max_workers: int = 8) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """VLM-verify every multi-member candidate cluster; return refined clusters + findings.

    Singleton clusters skip the VLM call (nothing to verify). Clusters larger than ``max_crops``
    send a diverse SUBSET to the VLM (still an authoritative decision on the sample, per
    ``docs`` -- excluded members are deterministically reattached to the nearest resulting subgroup
    by body cosine) and get a ``large_precluster_sampled`` finding so a reviewer knows full coverage
    was not achieved. Independent clusters are verified concurrently (extreme parallelization)."""
    findings: list[dict[str, Any]] = []
    to_verify = [c for c in clusters if len(c) > 1]
    passthrough = [c for c in clusters if len(c) <= 1]

    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() or out_root is None else out_root / p

    def _one(cluster: list[int]) -> list[list[int]]:
        sample = (cluster if len(cluster) <= max_crops
                 else _pick_diverse_subset(observations, cluster, max_crops))
        crops = [_resolve(observations[i].crop_path) for i in sample]
        try:
            result = judge_vlm.verify_cluster(crops)
        except Exception as exc:  # noqa: BLE001 -- a verification failure must not corrupt/hide
            # identity; fall back to the conservative default (every member its own singleton) and
            # surface the failure as a finding for human attention.
            findings.append({"code": "cluster_verify_error", "members": list(cluster),
                             "detail": str(exc)})
            return [[i] for i in cluster]
        if bool(result.get("coherent")) and len(sample) == len(cluster):
            return [list(cluster)]
        if bool(result.get("coherent")):
            findings.append({"code": "large_precluster_sampled", "members": list(cluster),
                             "sampled": list(sample)})
            return [list(cluster)]
        # Split: map the VLM's subgroup indices (0-based over `sample`) back to observation
        # indices, then deterministically reattach any excluded member to its nearest subgroup.
        n = len(sample)
        partition = normalize_partition(result.get("subgroups"), n)
        subgroups = [[sample[i] for i in group] for group in partition]
        excluded = [i for i in cluster if i not in sample]
        for member in excluded:
            sig = observations[member].signature
            if sig is None or not any(observations[g[0]].signature is not None for g in subgroups):
                subgroups[0].append(member)
                continue
            best_group, best_score = 0, -2.0
            for gi, group in enumerate(subgroups):
                sims = [cosine_similarity(sig, observations[m].signature) for m in group
                       if observations[m].signature is not None]
                if sims and max(sims) > best_score:
                    best_group, best_score = gi, max(sims)
            subgroups[best_group].append(member)
        if len(subgroups) > 1 or len(sample) < len(cluster):
            findings.append({"code": "cluster_split_by_vlm", "original_members": list(cluster),
                             "subgroups": [list(g) for g in subgroups]})
        return [sorted(g) for g in subgroups]

    refined: list[list[int]] = [list(c) for c in passthrough]
    if to_verify:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(to_verify)))) as pool:
            for groups in pool.map(_one, to_verify):
                refined.extend(groups)
    refined.sort(key=lambda c: min(c))
    # Deterministic order regardless of which worker thread finished first (correctness of the
    # partition never depends on thread completion order, but a stable diagnostic ordering does).
    findings.sort(key=lambda f: (str(f.get("code")), min(f.get("members") or f.get("original_members") or [0])))
    return refined, findings


# --- VLM cross-cluster merge (authoritative, catches phrase-fragmentation) --------------------

def merge_clusters(observations: Sequence[TrackletObservation], clusters: Sequence[list[int]], *,
                   judge_vlm: Any, kinds: tuple[str, ...] = ("character", "prop"),
                   out_root: Path | None = None, max_images: int = 24,
                   static_overlap_threshold: float = 0.75, max_workers: int = 4,
                   ) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """VLM-merge candidate clusters of the SAME kind that show one individual under different
    ``identity_group`` phrases (the "white_rabbit" vs "big_buck_bunny" fragmentation case).

    One VLM call per kind (parallel across kinds), showing one representative crop per cluster.
    A static-attribute conflict between two clusters vetoes the merge regardless of the VLM verdict
    (same hard-gate spirit as the rest of re-ID). Clusters of a kind not in ``kinds`` pass through
    unmerged (e.g. locations use a separate scene-clustering path, never this module)."""
    findings: list[dict[str, Any]] = []
    by_kind: dict[str, list[int]] = {}  # kind -> list of cluster indices (into `clusters`)
    for ci, cluster in enumerate(clusters):
        kind = observations[cluster[0]].kind
        by_kind.setdefault(kind, []).append(ci)

    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() or out_root is None else out_root / p

    def _static_of(cluster: list[int]) -> dict[str, str]:
        for i in cluster:
            if observations[i].static_attributes:
                return observations[i].static_attributes
        return {}

    def _merge_one_kind(kind: str) -> tuple[str, list[int], dict[int, int]]:
        """Returns (kind, cluster_ids considered, {cluster_id -> final_group_id})."""
        all_ids = sorted(by_kind.get(kind, []), key=lambda ci: (-len(clusters[ci]), ci))
        cluster_ids = all_ids[:max_images]
        if len(all_ids) > max_images:
            findings.append({"code": "cluster_merge_budget_truncated", "kind": kind,
                             "n_clusters": len(all_ids), "max_images": max_images,
                             "excluded_cluster_ids": all_ids[max_images:]})
        if len(cluster_ids) < 2:
            return kind, cluster_ids, {}
        labeled = [(f"cluster_{ci}", observations[clusters[ci][0]].name) for ci in cluster_ids]
        crops = [_resolve(observations[_pick_representative(observations, clusters[ci])].crop_path)
                for ci in cluster_ids]
        try:
            groups = judge_vlm.group_same_individuals(labeled, crops)
        except Exception as exc:  # noqa: BLE001 -- merge failure must degrade to "no merges", not
            # crash the run; every cluster stays its own final entity.
            findings.append({"code": "cluster_merge_error", "kind": kind, "detail": str(exc)})
            return kind, cluster_ids, {}
        partition = normalize_partition(groups, len(cluster_ids))
        mapping: dict[int, int] = {}
        for group in partition:
            member_cluster_ids = [cluster_ids[i] for i in group]
            if len(member_cluster_ids) > 1:
                # Hard veto: a static-attribute conflict between any two clusters in this VLM
                # group blocks the merge for the WHOLE group (complete-link spirit, categorical).
                statics = [_static_of(clusters[ci]) for ci in member_cluster_ids]
                conflict = any(
                    not _static_compatible(statics[a], statics[b], static_overlap_threshold)
                    for a in range(len(statics)) for b in range(a + 1, len(statics)))
                if conflict:
                    findings.append({"code": "cluster_merge_static_veto", "kind": kind,
                                     "cluster_ids": member_cluster_ids})
                    for ci in member_cluster_ids:
                        mapping[ci] = ci
                    continue
                findings.append({"code": "cluster_merged_by_vlm", "kind": kind,
                                 "cluster_ids": member_cluster_ids})
            for ci in member_cluster_ids:
                mapping.setdefault(ci, member_cluster_ids[0])
        return kind, cluster_ids, mapping

    active_kinds = [k for k in kinds if len(by_kind.get(k, [])) >= 2]
    kind_mappings: dict[int, int] = {}
    if active_kinds:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(active_kinds)))) as pool:
            for _kind, _cluster_ids, mapping in pool.map(_merge_one_kind, active_kinds):
                kind_mappings.update(mapping)

    final_by_root: dict[int, list[int]] = {}
    for ci, cluster in enumerate(clusters):
        root = kind_mappings.get(ci, ci)
        final_by_root.setdefault(root, []).extend(cluster)
    final_groups = [sorted(members) for members in final_by_root.values()]
    final_groups.sort(key=lambda g: min(g))
    findings.sort(key=lambda f: (str(f.get("code")), str(f.get("kind") or "")))
    return final_groups, findings


# --- roster completeness ------------------------------------------------------------------------

def roster_completeness_findings(observations: Sequence[TrackletObservation],
                                 final_groups: Sequence[list[int]], *,
                                 min_observations: int = 3) -> list[dict[str, Any]]:
    """Flag final groups whose tracklets never matched a roster phrase and that have non-trivial
    evidence (>= ``min_observations`` tracklet observations) -- a likely MISSED roster entry, not
    silently force-classified into whatever candidate happened to be nearest."""
    findings = []
    for gi, group in enumerate(final_groups):
        if len(group) < min_observations:
            continue
        if any(observations[i].roster_matched for i in group):
            continue
        findings.append({"code": "roster_incomplete_unmatched_cluster", "group_index": gi,
                         "members": list(group), "n_observations": len(group),
                         "name": observations[group[0]].name, "kind": observations[group[0]].kind})
    return findings


# --- top-level orchestration ---------------------------------------------------------------------

def resolve_identities(
    observations: Sequence[TrackletObservation], *, judge_vlm: Any,
    weights: dict[str, float], precluster_threshold: float, linkage: str = "complete",
    static_overlap_threshold: float = 0.75, out_root: Path | None = None,
    verify_max_crops: int = 8, merge_max_images: int = 24,
    merge_kinds: tuple[str, ...] = ("character", "prop"),
    roster_completeness_min_observations: int = 3, max_workers: int = 8,
    threshold_for_bucket: Any = None,
) -> IdentityResolution:
    """Full batch identity resolution: precluster -> VLM verify -> VLM cross-cluster merge ->
    roster-completeness check. See module docstring for the 5-step rationale.

    ``threshold_for_bucket``, when given, is a ``(kind, identity_group) -> float`` callable that
    overrides ``precluster_threshold`` for specific buckets (e.g. an exemplar-anchored character
    phrase reconciling viewpoints of ONE individual merges permissively, mirroring the old
    ``anchored_reid_threshold`` -- see pipeline_track_first.py)."""
    observations = list(observations)
    findings: list[dict[str, Any]] = []
    buckets = bucket_by_group(observations)
    preclusters: list[list[int]] = []
    for (kind, group), indices in buckets.items():
        threshold = (threshold_for_bucket(kind, group) if threshold_for_bucket is not None
                    else precluster_threshold)
        preclusters.extend(precluster_bucket(
            observations, indices, weights=weights, threshold=threshold,
            linkage=linkage, static_overlap_threshold=static_overlap_threshold))

    verified, verify_findings = verify_clusters(
        observations, preclusters, judge_vlm=judge_vlm, out_root=out_root,
        max_crops=verify_max_crops, max_workers=max_workers)
    findings.extend(verify_findings)

    final_groups, merge_findings = merge_clusters(
        observations, verified, judge_vlm=judge_vlm, kinds=merge_kinds, out_root=out_root,
        max_images=merge_max_images, static_overlap_threshold=static_overlap_threshold,
        max_workers=max_workers)
    findings.extend(merge_findings)

    findings.extend(roster_completeness_findings(
        observations, final_groups, min_observations=roster_completeness_min_observations))

    provenance = {gi: {"method": "cluster_vlm", "n_observations": len(group),
                       "n_preclusters_merged": sum(
                           1 for c in verified if set(c) & set(group))}
                 for gi, group in enumerate(final_groups)}
    return IdentityResolution(final_groups=final_groups, group_provenance=provenance,
                             findings=findings)


# --- registry commit (thin last mile; separately testable) --------------------------------------

def commit_groups_to_registry(
    registry: Registry, observations: Sequence[TrackletObservation],
    final_groups: Sequence[list[int]],
) -> tuple[dict[int, str], dict[int, str]]:
    """Materialize a decided grouping into ``registry`` via ``reid.commit_tracklet_observation``.

    Members within a group commit in (chunk_id, frame_index) order so the first-created rep is a
    stable, deterministic "founding" observation (mirrors the old online order-dependence, but now
    the GROUPING itself no longer depends on temporal order -- only which member happens to mint the
    entity vs. append to it does, and that has no scoring effect).

    Returns ``(group_entity_id, rep_id_by_observation)``: the first maps group index (position in
    ``final_groups``) to the entity_id it became; the second maps each observation's ``index`` to
    the ``representation_id`` it was committed as, so a caller that tracks tracklet-specific data
    (e.g. frame spans for presence, §3.4) not carried by ``TrackletObservation`` can zip it back in
    without this module needing to know about that data."""
    group_entity_id: dict[int, str] = {}
    rep_id_by_observation: dict[int, str] = {}
    for gi, group in enumerate(final_groups):
        ordered = sorted(group, key=lambda i: (observations[i].chunk_id,
                                              observations[i].frame_index))
        entity: Entity | None = None
        for i in ordered:
            obs = observations[i]
            entity, rep, _is_new = commit_tracklet_observation(
                registry, entity, entity is None, kind=obs.kind, name=obs.name,
                description=obs.description, static_attributes=obs.static_attributes,
                chunk_id=obs.chunk_id, crop_path=obs.crop_path, bbox=obs.bbox,
                bbox_source=obs.bbox_source, frame_index=obs.frame_index,
                grounding_score=obs.grounding_score, track_id=obs.track_id,
                signature=obs.signature, face_signature=obs.face_signature,
                class_signature=obs.class_signature,
                extra_qa={"cluster_group_index": gi})
            rep_id_by_observation[obs.index] = rep.representation_id
        if entity is not None:
            group_entity_id[gi] = entity.entity_id
    return group_entity_id, rep_id_by_observation
