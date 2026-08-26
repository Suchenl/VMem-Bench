"""Offline unit checks for the track-first core (tracking.py + reid.py). No VLM / GPU / network.

Proves the root-cause fix from docs/design/bench/track_first_redesign.md:
- intra-shot tracking links a moving object into ONE tracklet and keeps co-occurring
  same-class instances apart (multi-instance), tolerates short misses, drops 1-frame FPs;
- cross-shot re-ID decides identity by APPEARANCE, not by the VLM name -> different names with
  the same appearance merge (kills the 'white_rabbit' vs 'white_rabbit_character' split), while
  same name with conflicting appearance / static attributes stays split.

Run: cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_tracking_reid.py
"""

from __future__ import annotations

from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet, iou, track_shot
from vmem_bench.annotation.pipeline_track_first.reid import reid_assign, fuse_similarity
from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first.face import _largest_face
from vmem_bench.annotation.pipeline_track_first.perception.sam3_track import mask_to_bbox
from vmem_bench.annotation.pipeline_track_first.crop_classify import audit_registry_crops, nearest_prototype
from vmem_bench.common.schemas import Representation


# --- tracking -------------------------------------------------------------------------------

def _det(f, box, phrase="grey rabbit", score=0.9, emb=None):
    return Detection(frame_index=f, bbox=box, score=score, phrase=phrase, embedding=emb)


def test_iou_basic() -> None:
    assert iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    assert 0.0 < iou([0, 0, 100, 100], [0, 50, 100, 150]) < 1.0


def test_single_object_moving_becomes_one_tracklet() -> None:
    frames = [[_det(0, [100, 100, 300, 300])],
              [_det(1, [110, 110, 310, 310])],
              [_det(2, [120, 120, 320, 320])]]
    tracks = track_shot(frames, iou_threshold=0.3, min_len=2)
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 3
    assert tracks[0].frame_span == (0, 2)


def test_two_same_class_instances_stay_separate() -> None:
    # multi-instance: two grey rabbits far apart, both present every frame -> two tracklets.
    frames = [[_det(0, [0, 0, 100, 100]), _det(0, [0, 800, 100, 900])],
              [_det(1, [5, 5, 105, 105]), _det(1, [5, 805, 105, 905])]]
    tracks = track_shot(frames, iou_threshold=0.3, min_len=2)
    assert len(tracks) == 2
    assert all(len(t.detections) == 2 for t in tracks)


def test_miss_tolerance_bridges_one_gap() -> None:
    frames = [[_det(0, [100, 100, 300, 300])],
              [],  # object missing this frame
              [_det(2, [110, 110, 310, 310])]]
    tracks = track_shot(frames, iou_threshold=0.3, max_miss=1, min_len=2)
    assert len(tracks) == 1
    assert [d.frame_index for d in tracks[0].detections] == [0, 2]


def test_single_frame_false_positive_dropped() -> None:
    frames = [[_det(0, [100, 100, 300, 300])], [], []]
    assert track_shot(frames, min_len=2) == []


def test_appearance_gate_blocks_low_cosine_match() -> None:
    a, b = [1.0, 0.0], [0.0, 1.0]  # orthogonal -> cosine 0
    frames = [[_det(0, [100, 100, 300, 300], emb=a)],
              [_det(1, [110, 110, 310, 310], emb=b)]]  # high IoU but wrong appearance
    tracks = track_shot(frames, iou_threshold=0.3, appearance_gate=0.5, min_len=1)
    assert len(tracks) == 2  # gate refused the merge -> two separate 1-frame tracks


def test_next_track_id_offset() -> None:
    frames = [[_det(0, [0, 0, 100, 100])], [_det(1, [5, 5, 105, 105])]]
    tracks = track_shot(frames, next_track_id=42, min_len=2)
    assert tracks[0].track_id == 42


def test_tracklet_mean_embedding() -> None:
    t = Tracklet(0, "x", [_det(0, [0, 0, 1, 1], emb=[0.0, 2.0]),
                          _det(1, [0, 0, 1, 1], emb=[2.0, 0.0])])
    assert t.mean_embedding() == [1.0, 1.0]


def test_predict_bbox_constant_velocity() -> None:
    from vmem_bench.annotation.pipeline_track_first.tracking import predict_bbox
    # single box predicts itself; two boxes extrapolate constant velocity (clamped 0-1000).
    assert predict_bbox(Tracklet(0, "x", [_det(0, [10, 10, 20, 20])])) == [10, 10, 20, 20]
    t = Tracklet(0, "x", [_det(0, [0, 0, 100, 100]), _det(1, [10, 10, 110, 110])])
    assert predict_bbox(t) == [20, 20, 120, 120]


def test_two_stage_low_score_recovers_but_never_seeds() -> None:
    # A low-score (< high_score) detection extends an existing track (ByteTrack recovery) ...
    frames = [[_det(0, [100, 100, 300, 300], score=0.9)],
              [_det(1, [110, 110, 310, 310], score=0.2)]]
    tracks = track_shot(frames, iou_threshold=0.3, min_len=2, high_score=0.5)
    assert len(tracks) == 1 and len(tracks[0].detections) == 2
    # ... but a lone low-score detection never STARTS a track.
    only_low = [[_det(0, [100, 100, 300, 300], score=0.2)],
                [_det(1, [110, 110, 310, 310], score=0.2)]]
    assert track_shot(only_low, min_len=1, high_score=0.5) == []


# --- re-ID ----------------------------------------------------------------------------------

def _assign(reg, chunk, name, sig, kind="character", static=None, thr=0.55, track_id=0):
    return reid_assign(reg, chunk_id=chunk, kind=kind, name=name, description=f"{name} desc",
                       static_attributes=static, signature=sig, crop_path=f"{name}_{chunk}.jpg",
                       bbox=[0, 0, 500, 500], frame_index=chunk * 100, grounding_score=0.7,
                       track_id=track_id, reid_threshold=thr)


def test_reid_merges_different_names_same_appearance() -> None:
    # THE fragmentation fix: identity by appearance, not by VLM name.
    reg = Registry()
    e1, _, new1 = _assign(reg, 0, "White Rabbit", [1.0, 0.0, 0.0])
    e2, _, new2 = _assign(reg, 1, "Bunny", [0.98, 0.02, 0.0])  # different name, same look
    assert new1 is True and new2 is False
    assert e1.entity_id == e2.entity_id
    assert len(reg.entities) == 1
    assert len(e2.representations) == 2


def test_reid_suffix_noise_merges() -> None:
    reg = Registry()
    e1, _, _ = _assign(reg, 0, "white rabbit", [1.0, 0.0])
    e2, _, new = _assign(reg, 1, "white rabbit character", [0.99, 0.01])
    assert new is False and e1.entity_id == e2.entity_id == "char_white_rabbit"


def test_reid_splits_same_name_different_appearance() -> None:
    reg = Registry()
    _assign(reg, 0, "Rabbit", [1.0, 0.0, 0.0])
    _, _, new = _assign(reg, 1, "Rabbit", [0.0, 1.0, 0.0])  # same name, orthogonal look
    assert new is True
    assert len(reg.entities) == 2


def test_reid_static_conflict_blocks_merge() -> None:
    reg = Registry()
    _assign(reg, 0, "Critter", [1.0, 0.0], static={"species": "rabbit"})
    _, _, new = _assign(reg, 1, "Critter", [1.0, 0.0], static={"species": "bird"})
    assert new is True  # identical appearance but species conflict -> keep split
    assert len(reg.entities) == 2


def test_reid_roster_gate_blocks_cross_phrase_merge() -> None:
    reg = Registry()
    first, _, _ = _assign(reg, 0, "Purple Bird", [1.0, 0.0])
    other, _, new = reid_assign(
        reg, chunk_id=1, kind="character", name="Grey Rabbit", description="",
        static_attributes=None, signature=[1.0, 0.0], crop_path="rabbit.jpg",
        bbox=[0, 0, 500, 500], frame_index=100, grounding_score=0.7, track_id=1,
        reid_threshold=0.55, allowed_entity_ids=set())
    assert new is True and other.entity_id != first.entity_id


def test_reid_cluster_minimum_similarity_blocks_hidden_mixture() -> None:
    reg = Registry()
    first, _, _ = _assign(reg, 0, "Rabbit", [1.0, 0.0])
    _assign(reg, 1, "Rabbit", [0.8, 0.6])  # joins the same cluster at the standard threshold
    conflicts = []
    split, _, new = reid_assign(
        reg, chunk_id=2, kind="character", name="Rabbit", description="",
        static_attributes=None, signature=[0.8, 0.6], crop_path="rabbit_2.jpg",
        bbox=[0, 0, 500, 500], frame_index=200, grounding_score=0.7, track_id=2,
        reid_threshold=0.55, cluster_min_similarity=0.85,
        conflict_hook=lambda entity, fused, minimum: conflicts.append(
            (entity.entity_id, fused, minimum)))
    assert new is True and split.entity_id != first.entity_id
    assert conflicts and conflicts[0][0] == first.entity_id


def test_reid_location_without_signature_matches_by_name() -> None:
    reg = Registry()
    e1, _, _ = reid_assign(reg, chunk_id=0, kind="location", name="Meadow", description="d",
                           static_attributes=None, signature=None, crop_path="m0.jpg",
                           bbox=[0, 0, 1000, 1000], frame_index=0, grounding_score=0.0,
                           track_id=None, reid_threshold=0.55, bbox_source="full_frame")
    e2, _, new = reid_assign(reg, chunk_id=3, kind="location", name="Meadow", description="d",
                             static_attributes=None, signature=None, crop_path="m3.jpg",
                             bbox=[0, 0, 1000, 1000], frame_index=300, grounding_score=0.0,
                             track_id=None, reid_threshold=0.55, bbox_source="full_frame")
    assert new is False and e1.entity_id == e2.entity_id


def test_reid_rep_qa_carries_track_and_reid_score() -> None:
    reg = Registry()
    _assign(reg, 0, "Fox", [1.0, 0.0], track_id=7)
    e, rep, _ = _assign(reg, 1, "Fox", [1.0, 0.0], track_id=9)
    assert rep.qa["track_id"] == 9
    assert "reid_score" in rep.qa and rep.qa["reid_score"] >= 0.99
    assert rep.bbox_source == "grounding_dino"


# --- multi-cue fusion / face self-gating (§3.7) --------------------------------------------

def test_fuse_absent_face_equals_body_only() -> None:
    # A crop with no detected face must fuse to exactly the body score (no penalty, self-gating).
    w = {"body": 1.0, "face": 0.6, "class": 0.3}
    assert fuse_similarity({"body": 0.8, "face": None, "class": None}, w) == 0.8


def test_fuse_renormalizes_over_present_cues() -> None:
    w = {"body": 1.0, "face": 1.0}
    # (1.0*0.4 + 1.0*0.8) / (1.0 + 1.0) = 0.6
    assert abs(fuse_similarity({"body": 0.4, "face": 0.8}, w) - 0.6) < 1e-9


def test_fuse_none_when_no_cue_present() -> None:
    assert fuse_similarity({"body": None, "face": None}, {"body": 1.0}) is None


def _assign_face(reg, chunk, name, sig, face_sig, thr=0.55, face_strong=0.5):
    return reid_assign(reg, chunk_id=chunk, kind="character", name=name, description=f"{name} d",
                       static_attributes=None, signature=sig, face_signature=face_sig,
                       crop_path=f"{name}_{chunk}.jpg", bbox=[0, 0, 500, 500],
                       frame_index=chunk * 100, grounding_score=0.7, track_id=0,
                       reid_threshold=thr, face_strong=face_strong)


def test_face_cue_records_face_score() -> None:
    reg = Registry()
    _assign_face(reg, 0, "Alice", [1.0, 0.0], [1.0, 0.0])
    e, rep, new = _assign_face(reg, 1, "Alice", [0.99, 0.01], [0.98, 0.02])
    assert new is False and "face_score" in rep.qa and rep.qa["face_score"] >= 0.99


def test_strong_face_rescues_low_body() -> None:
    # body cosine ~0.45 (below 0.55 threshold) but a strong frontal face (cos~1) rescues the match.
    reg = Registry()
    _assign_face(reg, 0, "Bob", [1.0, 0.0], [1.0, 0.0])
    # body drifts to ~0.44 (within 0.15 margin of 0.55), face stays identical.
    _, _, new = _assign_face(reg, 1, "Bob", [0.44, 0.898], [1.0, 0.0], thr=0.55, face_strong=0.5)
    assert new is False  # rescued by strong face


def test_face_alone_cannot_merge_when_body_contradicts() -> None:
    # Same face similarity but body is orthogonal (cos 0, far below threshold-margin) -> NO merge
    # (guards against twins/look-alikes; a face alone never merges).
    reg = Registry()
    _assign_face(reg, 0, "Cara", [1.0, 0.0], [1.0, 0.0])
    _, _, new = _assign_face(reg, 1, "Cara", [0.0, 1.0], [1.0, 0.0], thr=0.55, face_strong=0.5)
    assert new is True  # body contradicts -> stays split despite identical face


# --- per-crop QA (§3.3) --------------------------------------------------------------------

def test_nearest_prototype_argmax() -> None:
    protos = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    eid, score = nearest_prototype([0.9, 0.1], protos)
    assert eid == "a" and score > 0.98
    assert nearest_prototype([1.0], {}) == (None, -1.0)


def _add_bad_rep(reg, entity, rep_id, vec) -> None:
    entity.representations.append(Representation(
        representation_id=rep_id, chunk_id=9, crop_path=f"{rep_id}.jpg", bbox=[0, 0, 1, 1],
        bbox_source="grounding_dino", frame_index=9, embedding_key=rep_id, qa={}))
    reg.embeddings[rep_id] = list(vec)


def test_crop_audit_flags_misassigned_rep() -> None:
    # Build two visually distinct multi-rep characters, then graft a B-looking crop onto A.
    reg = Registry()
    _assign(reg, 0, "A", [1.0, 0.0, 0.0]); _assign(reg, 1, "A", [0.98, 0.02, 0.0])
    _assign(reg, 0, "B", [0.0, 1.0, 0.0]); _assign(reg, 1, "B", [0.02, 0.98, 0.0])
    a = next(e for e in reg.entities.values() if e.name == "A")
    _add_bad_rep(reg, a, "char_a@c009", [0.0, 1.0, 0.0])  # looks like B
    flagged = audit_registry_crops(reg, margin=0.05)
    assert "char_a@c009" in flagged
    # A's own good crops are NOT flagged.
    assert not any(rid.startswith("char_a@c00") and rid != "char_a@c009" for rid in flagged)


def test_crop_audit_singleton_kind_not_flagged() -> None:
    reg = Registry()
    _assign(reg, 0, "Solo", [1.0, 0.0])  # only one character entity -> nothing to confuse with
    assert audit_registry_crops(reg, margin=0.05) == []


# --- geometry helpers ----------------------------------------------------------------------

def test_largest_face_picks_biggest_area() -> None:
    class F:
        def __init__(self, bbox):
            self.bbox = bbox
    small, big = F([0, 0, 10, 10]), F([0, 0, 100, 100])
    assert _largest_face([small, big]) is big
    assert _largest_face([]) is None


def test_mask_to_bbox_tight_and_normalized() -> None:
    import numpy as np
    m = np.zeros((100, 100), dtype=bool)
    m[10:30, 20:60] = True  # rows 10..29, cols 20..59
    assert mask_to_bbox(m) == [100, 200, 300, 600]  # y0,x0,y1,x1 -> /100*1000
    assert mask_to_bbox(np.zeros((10, 10), dtype=bool)) is None


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
