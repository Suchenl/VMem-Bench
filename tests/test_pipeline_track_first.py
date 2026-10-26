"""Offline unit checks for the track-first orchestrator's deterministic core + roster selection.

No GPU / VLM / ffmpeg: covers the identity-independent logic that decides gold correctness ---
frame->chunk mapping, deterministic presence + first-appearance (tracklet/​shot span ∩ chunk span,
§3.4), roster keyframe selection (FPS/medoid/dedup, §3.1a), and roster batch merge.

Run: cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_pipeline_track_first.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first import (
    cluster_scene_locations, entity_time_metadata, frame_to_chunk_fn, is_closeup_shot, merge_spans, presence_for_chunks,
    reslug_entities, shots_from_boundaries_csv, should_reuse_location_without_tracklets,
    _present_payloads)
from vmem_bench.annotation.pipeline_track_first.crop_classify import reassign_label
from vmem_bench.annotation.pipeline_track_first.entity_merge import propose_entity_merges
from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.common.schemas import Entity, Representation
from vmem_bench.annotation.pipeline_track_first import roster as R


_CHUNKS = [
    {"chunk_id": 0, "frame_span": [0, 99]},
    {"chunk_id": 1, "frame_span": [100, 199]},
    {"chunk_id": 2, "frame_span": [200, 299]},
]


# --- Q3 time metadata ----------------------------------------------------------------------

def test_merge_spans_merges_overlap_and_touch() -> None:
    assert merge_spans([(0, 10), (11, 20), (30, 40)]) == [[0, 20], [30, 40]]
    assert merge_spans([(5, 9), (0, 6)]) == [[0, 9]]
    assert merge_spans([]) == []


def test_entity_time_metadata_computes_absence_and_screentime() -> None:
    # present 0..23 and 120..143 at 24 fps: two 1s spans, 96-frame gap (=4s) between them.
    tm = entity_time_metadata([(0, 23), (120, 143)], fps=24.0)
    assert tm["presence_spans"] == [[0, 23], [120, 143]]
    assert tm["first_frame"] == 0 and tm["first_seconds"] == 0.0
    assert tm["last_frame"] == 143 and tm["last_seconds"] == round(143 / 24, 2)
    assert tm["screen_time_seconds"] == round(48 / 24, 2)      # 2 seconds on screen
    assert tm["max_absence_frames"] == 96 and tm["max_absence_seconds"] == 4.0


def test_entity_time_metadata_single_span_zero_absence() -> None:
    tm = entity_time_metadata([(10, 20)], fps=10.0)
    assert tm["max_absence_frames"] == 0 and tm["max_absence_seconds"] == 0.0
    assert entity_time_metadata([], fps=10.0)["first_frame"] is None


# --- frame -> chunk ------------------------------------------------------------------------

def test_frame_to_chunk_maps_and_clamps() -> None:
    f = frame_to_chunk_fn(_CHUNKS)
    assert f(0) == 0 and f(99) == 0 and f(100) == 1 and f(250) == 2
    assert f(-5) == 0        # below range -> first chunk
    assert f(10_000) == 2    # above range -> last chunk (rounding overshoot)


# --- deterministic presence / first appearance (§3.4) --------------------------------------

def test_presence_spans_intersect_chunks() -> None:
    spans = {
        "char_rabbit": [(50, 250)],           # spans chunks 0,1,2
        "prop_apple": [(210, 220)],           # only chunk 2
        "loc_meadow": [(0, 99), (200, 299)],  # chunks 0 and 2 (a returning location)
    }
    present, first = presence_for_chunks(_CHUNKS, spans)
    assert present[0] == ["char_rabbit", "loc_meadow"]
    assert present[1] == ["char_rabbit"]
    assert present[2] == ["char_rabbit", "loc_meadow", "prop_apple"]
    assert first == {"char_rabbit": 0, "loc_meadow": 0, "prop_apple": 2}


def test_presence_first_appearance_is_earliest_chunk() -> None:
    spans = {"e": [(150, 160), (250, 260)]}  # appears in chunk 1 first, then 2
    present, first = presence_for_chunks(_CHUNKS, spans)
    assert first["e"] == 1
    assert present[0] == [] and present[1] == ["e"] and present[2] == ["e"]


def test_presence_no_overlap_absent() -> None:
    # a one-frame tracklet outside every chunk span (shouldn't happen, but must not crash/insert)
    present, first = presence_for_chunks(_CHUNKS, {"ghost": [(500, 500)]})
    assert all("ghost" not in v for v in present.values())
    assert "ghost" not in first


# --- boundaries.csv parsing ----------------------------------------------------------------

def test_shots_from_boundaries_csv() -> None:
    with tempfile.TemporaryDirectory() as d:
        layout = Path(d) / "layout"
        layout.mkdir()
        (layout / "boundaries.csv").write_text(
            "shot_idx,start_frame,last_frame\n0,0,49\n1,50,120\n2,121,200\n", encoding="utf-8")
        assert shots_from_boundaries_csv(Path(d)) == [(0, 49), (50, 120), (121, 200)]


# --- roster keyframe selection (§3.1a) -----------------------------------------------------

def test_shot_candidate_indices_step() -> None:
    # fps=24, candidate_fps=2 -> step 12; shot [0,47] -> 0,12,24,36; empty shot -> []
    out = R.shot_candidate_indices([(0, 47), (100, 100), (5, 4)], fps=24.0, candidate_fps=2.0)
    assert out[0] == [0, 12, 24, 36]
    assert out[1] == [100]
    assert out[2] == []


def test_farthest_point_sample_prefers_diverse() -> None:
    # 3 orthogonal + 1 near-duplicate of the first; picking 3 must avoid the duplicate.
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.99, 0.01, 0]]
    picked = set(R.farthest_point_sample(vecs, 3))
    assert len(picked) == 3
    assert not ({0, 3} <= picked)  # never both a vector and its near-duplicate


def test_representative_indices_centroid_for_k1() -> None:
    # centroid ~ mean; the point nearest the centroid is the middle-ish one.
    vecs = [[1, 0], [0.8, 0.6], [0, 1]]
    assert R.representative_indices(vecs, 1) == [1]


def test_select_roster_keyframes_respects_budget() -> None:
    # 5 shots, 1 candidate each -> 5 reps; budget 3 -> FPS to 3, sorted absolute indices.
    per_shot = [[10], [20], [30], [40], [50]]
    def embed(idxs):  # noqa: ANN001
        return [[float(i), 0.0] for i in idxs]  # collinear -> FPS still deterministic
    out = R.select_roster_keyframes(per_shot, embed, per_shot_k=1, budget=3)
    assert len(out) == 3 and out == sorted(out)
    assert set(out) <= {10, 20, 30, 40, 50}


def test_select_roster_keyframes_adaptive_stops_on_low_novelty() -> None:
    # 3 visual clusters, each duplicated: once one frame per cluster is picked, residual novelty
    # collapses to ~0, so the adaptive budget stops at 3 even though the cap allows 6.
    per_shot = [[10], [20], [30], [40], [50], [60]]
    cluster = {10: [1, 0, 0], 40: [1, 0, 0], 20: [0, 1, 0], 50: [0, 1, 0],
               30: [0, 0, 1], 60: [0, 0, 1]}
    def embed(idxs):  # noqa: ANN001
        return [cluster[i] for i in idxs]
    out = R.select_roster_keyframes(per_shot, embed, per_shot_k=1, budget=2, budget_max=6,
                                    novelty_threshold=0.5)
    assert len(out) == 3
    assert len({tuple(cluster[i]) for i in out}) == 3  # one frame per cluster


def test_select_roster_keyframes_adaptive_enforces_floor_and_ratio() -> None:
    # All frames near-identical -> novelty ~0 everywhere, yet the floor must still be honored;
    # min_ratio raises the floor above the legacy budget (0.005 * 1000 frames = 5).
    per_shot = [[i * 10] for i in range(1, 9)]
    def embed(idxs):  # noqa: ANN001
        return [[1.0, float(i) * 1e-4] for i in idxs]
    out = R.select_roster_keyframes(per_shot, embed, per_shot_k=1, budget=2, budget_max=8,
                                    novelty_threshold=0.5, min_ratio=0.005, total_frames=1000)
    assert len(out) == 5  # ceil(0.005 * 1000), not the legacy budget of 2


def test_demote_object_locations_guard() -> None:
    roster = [
        {"name": "grassy meadow", "kind": "location", "grounding_phrase": "grassy meadow"},
        {"name": "tree trunk", "kind": "location", "grounding_phrase": "sunlit tree trunk"},
        {"name": "canopy", "kind": "location", "grounding_phrase": "lush canopy overhang"},
        {"name": "red apple", "kind": "prop", "grounding_phrase": "red apple"},
        {"name": "white rabbit", "kind": "character", "grounding_phrase": "white rabbit"},
    ]
    out, demoted = R.demote_object_locations(roster)
    kinds = {e["name"]: e["kind"] for e in out}
    assert kinds["grassy meadow"] == "location"  # real narrative stage survives
    assert kinds["tree trunk"] == "prop" and kinds["canopy"] == "prop"
    assert kinds["red apple"] == "prop" and kinds["white rabbit"] == "character"
    assert sorted(demoted) == ["canopy", "tree trunk"]


def test_merge_roster_dedupes_and_unions_static() -> None:
    b1 = [{"name": "Rabbit", "kind": "character", "grounding_phrase": "grey rabbit",
           "description": "", "static_attributes": {"species": "rabbit"}}]
    b2 = [{"name": "rabbit", "kind": "character", "grounding_phrase": "grey rabbit",
           "description": "a chubby grey rabbit", "static_attributes": {"primary_color": "grey"}},
          {"name": "Apple", "kind": "prop", "grounding_phrase": "red apple",
           "description": "a red apple", "static_attributes": {}}]
    merged = R.merge_roster([b1, b2])
    assert len(merged) == 2  # Rabbit merged across batches, Apple new
    rab = next(m for m in merged if m["kind"] == "character")
    assert rab["description"] == "a chubby grey rabbit"  # filled from later non-empty
    assert rab["static_attributes"] == {"species": "rabbit", "primary_color": "grey"}


# --- reslug / closeup / reassign / merge proposals -----------------------------------------

def test_reslug_entities_renames_ids_reps_dirs() -> None:
    with tempfile.TemporaryDirectory() as d:
        assets = Path(d) / "assets"
        old_dir = assets / "characters" / "char_setting"
        old_dir.mkdir(parents=True)
        (old_dir / "c000.jpg").write_bytes(b"x")
        reg = Registry()
        eid = "char_setting"
        rid = f"{eid}@c000"
        ent = Entity(entity_id=eid, kind="character", name="Big Buck Bunny",
                     description="a grey rabbit", first_chunk=0,
                     representations=[Representation(
                         representation_id=rid, chunk_id=0,
                         crop_path=f"assets/characters/{eid}/c000.jpg",
                         embedding_key=rid)])
        reg.entities[eid] = ent
        reg.embeddings[rid] = [1.0, 0.0]
        reg.face_embeddings[rid] = [0.5, 0.5]
        idmap = reslug_entities(reg, assets)
        assert idmap == {eid: "char_big_buck_bunny"}
        assert "char_big_buck_bunny" in reg.entities
        assert eid not in reg.entities
        new_ent = reg.entities["char_big_buck_bunny"]
        assert new_ent.entity_id == "char_big_buck_bunny"
        assert new_ent.representations[0].representation_id == "char_big_buck_bunny@c000"
        assert new_ent.representations[0].embedding_key == "char_big_buck_bunny@c000"
        assert new_ent.representations[0].crop_path == "assets/characters/char_big_buck_bunny/c000.jpg"
        assert "char_big_buck_bunny@c000" in reg.embeddings
        assert "char_big_buck_bunny@c000" in reg.face_embeddings
        assert rid not in reg.embeddings
        assert (assets / "characters" / "char_big_buck_bunny" / "c000.jpg").is_file()
        assert not old_dir.exists()


def test_reslug_entities_adopts_existing_target_dir() -> None:
    """Resume leftover: assets/{new_id}/ already exists -> adopt new_id (do not keep provisional)."""
    with tempfile.TemporaryDirectory() as d:
        assets = Path(d) / "assets"
        old_dir = assets / "characters" / "char_setting"
        new_dir = assets / "characters" / "char_big_buck_bunny"
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        (new_dir / "c000.jpg").write_bytes(b"new")
        (old_dir / "c001.jpg").write_bytes(b"old")
        reg = Registry()
        eid = "char_setting"
        rid = f"{eid}@c000"
        ent = Entity(entity_id=eid, kind="character", name="Big Buck Bunny",
                     description="", first_chunk=0,
                     representations=[Representation(
                         representation_id=rid, chunk_id=0,
                         crop_path="/abs/missing/derived/candidates/x.jpg",
                         embedding_key=rid)])
        reg.entities[eid] = ent
        reg.embeddings[rid] = [1.0, 0.0]
        idmap = reslug_entities(reg, assets)
        assert idmap == {eid: "char_big_buck_bunny"}
        assert eid not in reg.entities
        new_ent = reg.entities["char_big_buck_bunny"]
        assert new_ent.representations[0].crop_path == "assets/characters/char_big_buck_bunny/c000.jpg"
        assert (new_dir / "c001.jpg").is_file()  # merged from old_dir

    with tempfile.TemporaryDirectory() as d:
        assets = Path(d) / "assets"
        assets.mkdir()
        reg = Registry()
        # Already at desired slug (name matches id) so it stays; second Rabbit collides -> _02
        taken = Entity(entity_id="char_rabbit", kind="character", name="Rabbit",
                       description="", first_chunk=0, representations=[])
        reg.entities["char_rabbit"] = taken
        e2 = Entity(entity_id="char_tmp", kind="character", name="Rabbit",
                    description="", first_chunk=1, representations=[])
        reg.entities["char_tmp"] = e2
        e3 = Entity(entity_id="char_weird", kind="character", name="!!!",
                    description="", first_chunk=2, representations=[])
        reg.entities["char_weird"] = e3
        idmap = reslug_entities(reg, assets)
        assert idmap["char_tmp"] == "char_rabbit_02"
        assert "char_weird" not in idmap  # unnamed slug keeps old id
        assert "char_weird" in reg.entities
        assert "char_rabbit_02" in reg.entities
        assert "char_rabbit" in reg.entities


def test_is_closeup_shot_thresholds() -> None:
    # area 500*500/1e6 = 0.25 -> not closeup at 0.45; area 800*800/1e6 = 0.64 -> closeup
    assert is_closeup_shot([[100, 100, 600, 600]], coverage_threshold=0.45) is False
    assert is_closeup_shot([[100, 100, 900, 900]], coverage_threshold=0.45) is True
    assert is_closeup_shot([], coverage_threshold=0.45) is False
    assert is_closeup_shot([[0, 0, 100, 100], [0, 0, 900, 900]],
                           coverage_threshold=0.45) is True


def test_untracked_location_continuation_uses_previous_scene_signature() -> None:
    eid = "loc_meadow"
    rid = f"{eid}@s000"
    location = Entity(
        entity_id=eid, kind="location", name="Meadow", description="", first_chunk=0,
        representations=[Representation(
            representation_id=rid, chunk_id=0, crop_path="", embedding_key=rid)])
    embeddings = {rid: [1.0, 0.0]}

    assert should_reuse_location_without_tracklets(
        [0.9, 0.1], location, embeddings, similarity_threshold=0.48)
    assert not should_reuse_location_without_tracklets(
        [0.0, 1.0], location, embeddings, similarity_threshold=0.48)
    assert not should_reuse_location_without_tracklets(
        [1.0, 0.0, 0.0], location, embeddings, similarity_threshold=0.48)


def test_cluster_scene_locations_uses_visual_centroids() -> None:
    clusters = cluster_scene_locations(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
        similarity_threshold=0.8)
    assert clusters == [[0, 1], [2, 3]]


def test_reassign_label_margin_logic() -> None:
    ranked = [("rabbit", 0.8), ("fox", 0.5), ("apple", 0.1)]
    assert reassign_label(ranked, "fox", 0.15) == "rabbit"   # 0.8-0.5=0.3 >= 0.15
    assert reassign_label(ranked, "fox", 0.40) == "fox"      # 0.3 < 0.40
    assert reassign_label(ranked, "rabbit", 0.01) == "rabbit"  # already top
    assert reassign_label([], "fox", 0.15) == "fox"
    # current missing from ranked -> cur_prob=0.0
    assert reassign_label([("rabbit", 0.2)], "fox", 0.15) == "rabbit"
    assert reassign_label([("rabbit", 0.1)], "fox", 0.15) == "fox"


def test_propose_entity_merges_thresholds_and_keep() -> None:
    reg = Registry()
    a = Entity(entity_id="char_a", kind="character", name="Bunny",
               description="grey rabbit", first_chunk=0,
               representations=[Representation(
                   representation_id="char_a@c000", chunk_id=0, crop_path="a.jpg",
                   embedding_key="char_a@c000")])
    b = Entity(entity_id="char_b", kind="character", name="Bunny",
               description="grey rabbit", first_chunk=2,
               representations=[Representation(
                   representation_id="char_b@c000", chunk_id=2, crop_path="b.jpg",
                   embedding_key="char_b@c000")])
    c = Entity(entity_id="prop_x", kind="prop", name="Apple",
               description="red", first_chunk=0,
               representations=[Representation(
                   representation_id="prop_x@c000", chunk_id=0, crop_path="c.jpg",
                   embedding_key="prop_x@c000")])
    reg.entities = {"char_a": a, "char_b": b, "prop_x": c}
    # unit-norm: a and b identical text+body; c orthogonal
    reg.embeddings = {
        "char_a@c000": [1.0, 0.0],
        "char_b@c000": [1.0, 0.0],
        "prop_x@c000": [0.0, 1.0],
    }

    def fake_text_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if "Bunny" in t:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    props = propose_entity_merges(reg, fake_text_embed, text_threshold=0.85, body_threshold=0.5)
    assert len(props) == 1
    assert props[0]["keep"] == "char_a" and props[0]["merge"] == "char_b"
    assert props[0]["kind"] == "character"
    assert props[0]["text_cos"] >= 0.85
    assert props[0]["body_cos"] >= 0.5

    # below body threshold -> no proposal
    reg.embeddings["char_b@c000"] = [0.0, 1.0]
    assert propose_entity_merges(reg, fake_text_embed, text_threshold=0.85,
                                 body_threshold=0.5) == []


def test_identity_resolution_mode_gates_the_two_paths() -> None:
    """Both cluster_vlm (default) and greedy (fallback) code paths must exist in stage 4, gated
    by config.identity_resolution_mode -- a cheap source-level regression guard (mirrors
    test_auto_review_disabled_by_config's inspect.getsource pattern) that the greedy fallback was
    not accidentally deleted during the identity-resolution-v2 redesign."""
    import inspect
    from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
    from vmem_bench.annotation import pipeline_track_first as ptf

    assert AnnotationConfig(video=Path("v.mp4"), out_dir=Path("o"),
                            movie_id="m").identity_resolution_mode == "cluster_vlm"
    src = inspect.getsource(ptf.annotate_movie_track_first)
    assert "use_cluster_vlm" in src
    assert "identity_resolution.resolve_identities" in src
    assert "identity_resolution.commit_groups_to_registry" in src
    assert "reid_assign(" in src  # greedy fallback path still present


def test_write_identity_resolution_artifact_round_trips() -> None:
    import json
    from vmem_bench.annotation.pipeline_track_first import identity_resolution as ir
    from vmem_bench.annotation.pipeline_track_first import _write_identity_resolution_artifact

    obs = [ir.TrackletObservation(
        index=0, kind="character", name="Bunny", description="", static_attributes={},
        signature=[1.0, 0.0], crop_path="a.jpg", bbox=[0, 0, 10, 10], frame_index=0, chunk_id=0,
        grounding_score=0.9, track_id=0, bbox_source="tracker", identity_group="bunny",
        roster_matched=True)]
    resolution = ir.IdentityResolution(final_groups=[[0]], group_provenance={0: {}},
                                       findings=[{"code": "roster_incomplete_unmatched_cluster"}])
    with tempfile.TemporaryDirectory() as d:
        tmp_dir = Path(d) / "tmp"
        _write_identity_resolution_artifact(tmp_dir, obs, resolution, {0: "char_bunny"},
                                            {0: "char_bunny"})
        payload = json.loads((tmp_dir / "identity_resolution.json").read_text(encoding="utf-8"))
        assert payload["mode"] == "cluster_vlm"
        assert payload["group_entity_id"] == {"0": "char_bunny"}
        assert payload["entity_id_by_observation"] == {"0": "char_bunny"}
        assert payload["observations"][0]["name"] == "Bunny"
        assert payload["findings"][0]["code"] == "roster_incomplete_unmatched_cluster"


def test_auto_review_disabled_by_config() -> None:
    """config.auto_review=False skips the post-persist call (flag / source gate check)."""
    import inspect
    from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
    from vmem_bench.annotation import pipeline_track_first as ptf

    cfg = AnnotationConfig(video=Path("v.mp4"), out_dir=Path("o"), movie_id="m", auto_review=False)
    assert cfg.auto_review is False
    assert AnnotationConfig(video=Path("v.mp4"), out_dir=Path("o"), movie_id="m").auto_review is True
    src = inspect.getsource(ptf.annotate_movie_track_first)
    assert "config.auto_review" in src
    assert "run_auto_review" in src


def test_present_payloads_prefer_current_diverse_crops_and_mark_history() -> None:
    reg = Registry()
    reps = [
        Representation("char_bunny@old", 0, "old.jpg", embedding_key="char_bunny@old"),
        Representation("char_bunny@c001a", 1, "a.jpg", frame_index=10,
                       embedding_key="char_bunny@c001a", qa={"grounding_score": 0.9},
                       bbox_source="grounding_dino"),
        Representation("char_bunny@c001b", 1, "b.jpg", frame_index=20,
                       embedding_key="char_bunny@c001b", qa={"grounding_score": 0.8},
                       bbox_source="grounding_dino"),
        Representation("char_bunny@c001c", 1, "c.jpg", frame_index=30,
                       embedding_key="char_bunny@c001c", qa={"grounding_score": 0.7},
                       bbox_source="grounding_dino"),
    ]
    reg.entities = {"char_bunny": Entity("char_bunny", "character", "Bunny", "grey", 0, reps)}
    reg.embeddings = {r.embedding_key: v for r, v in zip(reps, [[1, 0], [1, 0], [0, 1], [-1, 0]])}
    payloads, findings = _present_payloads(reg, ["char_bunny"], set(), 1,
                                            per_entity_limit=3, total_limit=3)
    assert findings == []
    assert [c["representation_id"] for c in payloads[0]["crops"]] == [
        "char_bunny@c001a", "char_bunny@c001c", "char_bunny@c001b"]
    assert not any(c["continuity_reference"] for c in payloads[0]["crops"])

    # A continuation can fall back to one explicitly-marked historical reference.
    payloads, findings = _present_payloads(reg, ["char_bunny"], set(), 2,
                                            per_entity_limit=3, total_limit=3)
    assert findings == [] and len(payloads[0]["crops"]) == 1
    assert payloads[0]["crops"][0]["continuity_reference"] is True


def test_present_payloads_reports_first_entity_without_current_crop() -> None:
    reg = Registry()
    rep = Representation("char_bunny@old", 0, "old.jpg", embedding_key="char_bunny@old")
    reg.entities = {"char_bunny": Entity("char_bunny", "character", "Bunny", "grey", 1, [rep])}
    reg.embeddings = {rep.embedding_key: [1, 0]}
    payloads, findings = _present_payloads(reg, ["char_bunny"], {"char_bunny"}, 1)
    assert [p["entity_id"] for p in payloads] == ["char_bunny"]
    assert payloads[0]["crops"] == []
    assert {f["code"] for f in findings} == {"first_missing_current_crop", "present_missing_crop"}


def test_present_payloads_global_budget_is_deterministic() -> None:
    reg = Registry()
    for eid in ("char_a", "char_b"):
        reps = [Representation(f"{eid}@c001.{i}", 1, f"{eid}{i}.jpg",
                               embedding_key=f"{eid}@c001.{i}", qa={"grounding_score": 1.0})
                for i in range(3)]
        reg.entities[eid] = Entity(eid, "character", eid, "", 1, reps)
        reg.embeddings.update({r.embedding_key: [1.0, float(i)] for i, r in enumerate(reps)})
    payloads, _ = _present_payloads(reg, ["char_a", "char_b"], {"char_b"}, 1,
                                    per_entity_limit=3, total_limit=3)
    assert [p["entity_id"] for p in payloads] == ["char_a", "char_b"]
    assert payloads[0]["crops"] == []  # still present in the authoritative roster
    assert len(payloads[1]["crops"]) == 3  # first appearance receives scarce visual budget


def test_drafter_sends_entity_evidence_crops_with_sampled_frames() -> None:
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import AnnotatorRole

    class FakeJudger:
        def __init__(self):
            self.messages = None

        def _call_api(self, messages, _schema, **_kwargs):  # noqa: ANN001
            self.messages = messages
            return {"prompt": "Bunny appears.", "state_events": []}

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        frame, crop_a, crop_b = root / "frame.jpg", root / "a.jpg", root / "b.jpg"
        for path in (frame, crop_a, crop_b):
            path.write_bytes(b"not-a-real-jpeg-but-encodable")
        judger = FakeJudger()
        role = AnnotatorRole(judger)
        role.draft_chunk([frame], [{"entity_id": "char_bunny", "name": "Bunny",
            "kind": "character", "description": "grey rabbit", "representation_id": "r0",
            "prior_representations": [], "first_appearance": True,
            "crops": [{"representation_id": "r0", "crop_path": str(crop_a)},
                      {"representation_id": "r1", "crop_path": str(crop_b)}]}], "", [])
        content = judger.messages[0]["content"]
        assert len([part for part in content if part["type"] == "image_url"]) == 3


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
