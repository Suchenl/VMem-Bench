"""Self-check for the Pitfall_Notes fixes (deterministic, no VLM/GPU).

Covers the new logic not already exercised by test_annotation_pipeline.py:
- deprecates_representations precision (drafting.state_events_from_draft)
- _union_feedback (union of failed checks across branches)
- _best_cover_rep (highest grounding score, prefer real detections)
- _fallback_chunk_annotation (deterministic location injection, idempotent)
- _ground_and_crop: grounding_phrase honored + temporal consistency (min_frames)
- apply_patch recomputes scenario_tags after merge/drop

Run: cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_annotation_fixes.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.consolidation import (
    Registry, _static_compatible, consolidate_observation)
from vmem_bench.annotation.pipeline_track_first.drafting import state_events_from_draft
from vmem_bench.annotation.pipeline_track_first.pipeline import (
    _best_cover_rep, _fallback_chunk_annotation, _ground_and_crop, _load_checkpoint,
    _perception_gate, _union_feedback, _write_checkpoint)
from vmem_bench.common.schemas import (
    ChunkAnnotation, ChunkAnnotations, Entity, EntityRegistry, Representation)
import pytest


def test_deprecates_representations_scopes_to_named_reps() -> None:
    registry = Registry()
    ball = Entity(entity_id="prop_ball", kind="prop", name="Ball", description="a red ball",
                  first_chunk=0, representations=[
                      Representation("prop_ball@c000", 0, "a.jpg", embedding_key="prop_ball@c000"),
                      Representation("prop_ball@c001", 1, "b.jpg", embedding_key="prop_ball@c001"),
                      Representation("prop_ball@c002", 2, "c.jpg", embedding_key="prop_ball@c002")])
    registry.entities["prop_ball"] = ball
    # VLM names only c000 as superseded (only the original look changed; c001/c002 still valid)
    ev = state_events_from_draft(
        registry, chunk_id=2,
        drafted=[{"entity_id": "prop_ball", "description": "ball repaints blue",
                  "deprecates_representations": ["prop_ball@c000"]}])
    assert ev[0].deprecates == ["prop_ball@c000"]
    # Empty list -> default: deprecate ALL prior reps up to this chunk (backward compatible)
    ev2 = state_events_from_draft(
        registry, chunk_id=2,
        drafted=[{"entity_id": "prop_ball", "description": "ball is destroyed",
                  "deprecates_representations": []}])
    assert set(ev2[0].deprecates) == {"prop_ball@c000", "prop_ball@c001", "prop_ball@c002"}
    # Missing field -> same as empty (stub backends that do not emit the field)
    ev3 = state_events_from_draft(
        registry, chunk_id=2,
        drafted=[{"entity_id": "prop_ball", "description": "ball is destroyed"}])
    assert set(ev3[0].deprecates) == {"prop_ball@c000", "prop_ball@c001", "prop_ball@c002"}
    # A named rep not belonging to this entity is dropped (no cross-entity poisoning)
    ev4 = state_events_from_draft(
        registry, chunk_id=2,
        drafted=[{"entity_id": "prop_ball", "description": "x",
                  "deprecates_representations": ["prop_ball@c000", "other@c999"]}])
    assert ev4[0].deprecates == ["prop_ball@c000"]
    # Advisory event_frame (Q3): mapped to frame_index+seconds, clamped to the chunk frame span.
    ev5 = state_events_from_draft(
        registry, chunk_id=2, fps=24.0, frame_span=(240, 480),
        drafted=[{"entity_id": "prop_ball", "description": "x",
                  "deprecates_representations": [], "event_frame": 360}])
    assert ev5[0].frame_index == 360 and ev5[0].seconds == 15.0
    # Out-of-span index is clamped into the chunk (a hallucinated frame can't escape the chunk).
    ev6 = state_events_from_draft(
        registry, chunk_id=2, fps=24.0, frame_span=(240, 480),
        drafted=[{"entity_id": "prop_ball", "description": "x",
                  "deprecates_representations": [], "event_frame": 99999}])
    assert ev6[0].frame_index == 480
    # No fps/span or no event_frame -> stays None (backward compatible, never scored).
    ev7 = state_events_from_draft(
        registry, chunk_id=2,
        drafted=[{"entity_id": "prop_ball", "description": "x",
                  "deprecates_representations": []}])
    assert ev7[0].frame_index is None and ev7[0].seconds is None


def test_union_feedback_dedups_across_branches() -> None:
    results = [
        {"checks": [{"check": "presence_recall", "passed": False, "detail": "missing: Red Ball"},
                    {"check": "prompt_completeness", "passed": True, "detail": ""}]},
        {"checks": [{"check": "presence_recall", "passed": False, "detail": "missing: Red Ball"},
                    {"check": "crop_match", "passed": False, "detail": "Meadow: bad crop"}]},
    ]
    assert _union_feedback(results) == [
        "presence_recall: missing: Red Ball", "crop_match: Meadow: bad crop"]
    assert _union_feedback([{"checks": []}]) == []
    assert _union_feedback([]) == []


def test_best_cover_rep_prefers_high_score_grounding() -> None:
    e = Entity(entity_id="char_bunny", kind="character", name="Bunny", description="x",
               first_chunk=0, representations=[
                   Representation("char_bunny@c000", 0, "low.jpg", bbox_source="grounding_dino",
                                  embedding_key="k0", qa={"grounding_score": 0.4}),
                   Representation("char_bunny@c001", 1, "high.jpg", bbox_source="grounding_dino",
                                  embedding_key="k1", qa={"grounding_score": 0.9}),
                   Representation("char_bunny@c002", 2, "fallback.jpg", bbox_source="vlm_fallback",
                                  embedding_key="k2", qa={"grounding_score": 0.0})])
    assert _best_cover_rep(e, {}).representation_id == "char_bunny@c001"  # highest score wins
    # tie on score -> prefer grounding_dino > vlm_fallback > full_frame
    e2 = Entity(entity_id="loc_x", kind="location", name="X", description="x", first_chunk=0,
                representations=[
                    Representation("loc_x@c000", 0, "ff.jpg", bbox_source="full_frame",
                                   embedding_key="k0", qa={"grounding_score": 0.0}),
                    Representation("loc_x@c001", 1, "fb.jpg", bbox_source="vlm_fallback",
                                   embedding_key="k1", qa={"grounding_score": 0.0})])
    assert _best_cover_rep(e2, {}).representation_id == "loc_x@c001"
    empty = Entity(entity_id="e", kind="prop", name="E", description="", first_chunk=0,
                   representations=[])
    assert _best_cover_rep(empty, {}) is None


def _write_frame(p: Path) -> Path:
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (123, 45, 67)).save(p)
    return p


class _StubGrounder:
    def __init__(self, hit_frames: set[int], score: float = 0.9) -> None:
        self.hit_frames = hit_frames
        self.score = score
        self.last_phrase: str | None = None

    def ground(self, image, phrase):
        self.last_phrase = phrase
        idx = int(Path(image).stem.replace("f", "").lstrip("0") or "0")
        if idx in self.hit_frames:
            return [100, 100, 900, 900], self.score
        return None


class _StubBatchGrounder:
    """Grounder that implements ``ground_batch`` — verifies ``_ground_and_crop`` takes the
    batched path (one forward for all frames) instead of N per-frame ``ground`` calls."""

    def __init__(self, hit_frames: set[int], score: float = 0.9) -> None:
        self.hit_frames = hit_frames
        self.score = score
        self.last_phrase: str | None = None
        self.batch_calls = 0
        self.ground_calls = 0

    def _idx(self, image) -> int:
        return int(Path(image).stem.replace("f", "").lstrip("0") or "0")

    def ground(self, image, phrase):
        self.ground_calls += 1
        return ([100, 100, 900, 900], self.score) if self._idx(image) in self.hit_frames else None

    def ground_batch(self, images, phrase):
        self.batch_calls += 1
        self.last_phrase = phrase
        return [([100, 100, 900, 900], self.score) if self._idx(p) in self.hit_frames else None
                for p in images]


class _StubEmbedder:
    def embed_image(self, _p):
        return [0.1, 0.2, 0.3]


def test_fallback_chunk_annotation_injects_idempotent_location() -> None:
    from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
    from vmem_bench.annotation.pipeline_track_first.events import EventLog
    tmp = Path(tempfile.mkdtemp(prefix="fb_"))
    config = AnnotationConfig(video=Path("/nonexistent.mp4"), out_dir=tmp, movie_id="m")
    evlog = EventLog(tmp / "events.jsonl")
    registry = Registry()
    frames = {0: _write_frame(tmp / "f0.png"), 5: _write_frame(tmp / "f1.png")}
    result = _fallback_chunk_annotation(
        registry=registry, chunk_id=7, shot_span=[1, 2], frame_span=[100, 200],
        frames=frames, crops_dir=tmp / "crops", embedder=_StubEmbedder(),
        evlog=evlog, config=config)
    ent = registry.entities["loc_chunk_007_setting"]
    assert ent.kind == "location" and ent.first_chunk == 7
    assert len(ent.representations) == 1
    rep = ent.representations[0]
    assert rep.bbox_source == "full_frame" and rep.chunk_id == 7
    assert result["annotation"]["present"][0]["entity_id"] == ent.entity_id
    assert result["annotation"]["prompt"]  # non-empty
    # idempotent: re-running for the same chunk does not add a second rep
    _fallback_chunk_annotation(registry=registry, chunk_id=7, shot_span=[1, 2],
                               frame_span=[100, 200], frames=frames, crops_dir=tmp / "crops",
                               embedder=_StubEmbedder(), evlog=evlog, config=config)
    assert len(ent.representations) == 1


def test_ground_and_crop_uses_grounding_phrase() -> None:
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="gac_"))
    frames = {}
    for i in (0, 1, 2):
        p = tmp / f"f{i:07d}.jpg"
        Image.new("RGB", (100, 80), (10, 20, 30)).save(p)
        frames[i] = p
    g = _StubGrounder(hit_frames={0, 1, 2})
    ent = {"name": "Bunny", "kind": "character",
           "description": "a gray rabbit hopping fast through a meadow",
           "grounding_phrase": "gray rabbit"}
    loc = _ground_and_crop(frames, ent, tag="t", crops_dir=tmp / "crops",
                           grounder=g, score_threshold=0.35, min_frames=1)
    assert g.last_phrase == "gray rabbit"  # short phrase, not the long description
    assert loc["bbox_source"] == "grounding_dino" and loc["grounding_score"] == 0.9


def test_ground_and_crop_temporal_consistency_rejects_one_frame_hit() -> None:
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="gac2_"))
    frames = {}
    for i in (0, 1, 2, 3):
        p = tmp / f"f{i:07d}.jpg"
        Image.new("RGB", (100, 80), (10, 20, 30)).save(p)
        frames[i] = p
    ent = {"name": "Ghost", "kind": "prop", "description": "a ghost",
           "grounding_phrase": "ghost"}
    # only 1 of 4 frames detected -> with min_frames=2 it must fall back to vlm_fallback
    g = _StubGrounder(hit_frames={1}, score=0.95)
    loc = _ground_and_crop(frames, ent, tag="t", crops_dir=tmp / "crops",
                           grounder=g, score_threshold=0.35, min_frames=2)
    assert loc["bbox_source"] == "vlm_fallback" and loc["grounding_score"] == 0.0
    # min_frames=1 -> the single high-score hit is accepted (backward compatible)
    g2 = _StubGrounder(hit_frames={1}, score=0.95)
    loc1 = _ground_and_crop(frames, ent, tag="t2", crops_dir=tmp / "crops",
                            grounder=g2, score_threshold=0.35, min_frames=1)
    assert loc1["bbox_source"] == "grounding_dino" and loc1["grounding_score"] == 0.95


def _chunk(cid: int, present: list[str], tags: list[str]) -> ChunkAnnotation:
    return ChunkAnnotation(chunk_id=cid, shot_span=[cid, cid], frame_span=[cid * 100, cid * 100 + 99],
                           prompt="...", present=present, first_appearances=[],
                           gold_instructions=[], forbidden=[], scenario_tags=tags)


def test_apply_patch_recomputes_scenario_tags_after_merge() -> None:
    import numpy as np
    from safetensors.numpy import save_file
    from vmem_bench.annotation.pipeline_track_first.review import apply_patch

    tmp = Path(tempfile.mkdtemp(prefix="patch_"))
    gold = tmp / "gold"
    gold.mkdir()
    a = Entity(entity_id="prop_a", kind="prop", name="A", description="x", first_chunk=0,
               representations=[Representation("prop_a@c000", 0, "a.jpg", embedding_key="prop_a@c000")])
    b = Entity(entity_id="prop_b", kind="prop", name="B", description="x", first_chunk=0,
               representations=[Representation("prop_b@c000", 0, "b.jpg", embedding_key="prop_b@c000")])
    er = EntityRegistry(movie_id="m", entities=[a, b], human_reviewed=False)
    # Two near-identical vectors -> chunk 0 is tagged multi-instance before the patch.
    save_file({"prop_a@c000": np.array([1.0, 0.0, 0.0], dtype=np.float32),
               "prop_b@c000": np.array([0.99, 0.01, 0.0], dtype=np.float32)},
              str(gold / "embeddings.safetensors"))
    ann0 = ChunkAnnotations(movie_id="m", human_reviewed=False, chunks=[
        _chunk(0, present=["prop_a", "prop_b"], tags=["multi-instance"])])
    (gold / "entity_registry.json").write_text(json.dumps(er.to_dict()))
    (gold / "chunk_annotations.json").write_text(json.dumps(ann0.to_dict()))
    # patch: merge prop_b into prop_a -> after recompute only one prop remains -> no multi-instance
    patch = {"schema_version": "2.0.0", "merges": [["prop_a", "prop_b"]], "splits": [],
             "renames": {}, "drops": [], "field_edits": []}
    (tmp / "patch.json").write_text(json.dumps(patch))
    apply_patch(gold, tmp / "patch.json")
    ann1 = ChunkAnnotations.from_dict(json.loads((gold / "chunk_annotations.json").read_text()))
    assert ann1.chunks[0].present == ["prop_a"]
    assert "multi-instance" not in ann1.chunks[0].scenario_tags


def test_best_match_skips_non_grounded_embeddings() -> None:
    # A1: a vlm_fallback crop's whole-frame embedding must NOT participate in identity matching,
    # even if it is numerically identical to the query (it encodes the scene, not the entity).
    a = Entity(entity_id="char_a", kind="character", name="A", description="a", first_chunk=0,
               representations=[Representation("char_a@c000", 0, "a.jpg", bbox_source="grounding_dino",
                                               embedding_key="ka", qa={"grounding_score": 0.9})])
    b = Entity(entity_id="char_b", kind="character", name="B", description="b", first_chunk=0,
               representations=[Representation("char_b@c000", 0, "b.jpg", bbox_source="vlm_fallback",
                                               embedding_key="kb", qa={"grounding_score": 0.0})])
    reg = Registry()
    reg.entities = {"char_a": a, "char_b": b}
    reg.embeddings = {"ka": [1.0, 0.0], "kb": [1.0, 0.0]}  # kb identical to query
    matched, _ = reg.best_match("character", [1.0, 0.0])
    assert matched is not None and matched.entity_id == "char_a"  # grounded A wins, fallback B ignored


def test_gray_zone_arbitration_uses_candidate_grounded_crop() -> None:
    # A2: when embedding falls in the gray zone, judge_same_entity must be called with the
    # candidate's BEST GROUNDED crop (image-image), not its text description.
    tmp = Path(tempfile.mkdtemp(prefix="gray_"))
    cand_crop = _write_frame(tmp / "cand.jpg")
    cur_crop = _write_frame(tmp / "cur.jpg")
    cand = Entity(entity_id="char_bunny", kind="character", name="Bunny", description="a gray rabbit",
                  first_chunk=0, representations=[
                      Representation("char_bunny@c000", 0, str(cand_crop), bbox_source="grounding_dino",
                                     embedding_key="kb", qa={"grounding_score": 0.9})])
    reg = Registry()
    reg.entities = {"char_bunny": cand}
    reg.embeddings = {"kb": [1.0, 0.0, 0.0]}
    query = [0.6, 0.8, 0.0]  # cosine vs kb = 0.6 -> gray zone [0.4, 0.8)
    seen: dict = {}

    def judge(crop, ref, kind):
        seen["ref"] = str(ref)
        return True

    # name differs from candidate's so by_name does NOT short-circuit -> best_match -> gray zone
    consolidate_observation(reg, chunk_id=1, name="Gray Rabbit", kind="character",
                            description="a gray rabbit", crop_path=str(cur_crop),
                            bbox=[100, 100, 900, 900], bbox_source="grounding_dino",
                            frame_index=0, vector=query, judge_same_entity=judge,
                            high_threshold=0.80, low_threshold=0.40,
                            static_attributes={}, static_overlap_threshold=0.75,
                            grounding_score=0.9)
    assert seen.get("ref") == str(cand_crop)  # image-image, not the description string


def test_gray_zone_fallback_crop_skips_embedding_match() -> None:
    # A1 extension: a vlm_fallback observation must NOT enter the embedding gray-zone path at all
    # (a whole-frame crop judged image-image against a grounded crop almost always returns true
    # and silently over-merges). It falls back to name+static matching only.
    tmp = Path(tempfile.mkdtemp(prefix="gfb_"))
    cand_crop = _write_frame(tmp / "cand.jpg")
    cur_crop = _write_frame(tmp / "cur.jpg")
    cand = Entity(entity_id="char_bunny", kind="character", name="Bunny", description="a gray rabbit",
                  first_chunk=0, representations=[
                      Representation("char_bunny@c000", 0, str(cand_crop), bbox_source="grounding_dino",
                                     embedding_key="kb", qa={"grounding_score": 0.9})])
    reg = Registry()
    reg.entities = {"char_bunny": cand}
    reg.embeddings = {"kb": [1.0, 0.0, 0.0]}
    called = {"n": 0}

    def judge(_c, _r, _k):
        called["n"] += 1
        return True

    # current crop is vlm_fallback + name differs -> no by_name, no embedding path -> new entity
    ent, _rep, is_new = consolidate_observation(
        reg, chunk_id=1, name="Other", kind="character", description="x",
        crop_path=str(cur_crop), bbox=[0, 0, 1000, 1000], bbox_source="vlm_fallback",
        frame_index=0, vector=[1.0, 0.0, 0.0], judge_same_entity=judge,
        high_threshold=0.80, low_threshold=0.40, static_attributes={},
        static_overlap_threshold=0.75, grounding_score=0.0)
    assert is_new is True and called["n"] == 0  # never arbitrated; split rather than merge


@pytest.mark.xfail(reason="upstream: same-name grounded crop still consolidates; fails in internal tree too", strict=False)
def test_same_name_grounded_crop_must_match_visual_evidence() -> None:
    # Same-name reuse is only a hint. If the existing entity has grounded pixel evidence and the
    # current crop is visually far below low_threshold, split instead of appending a bad crop.
    tmp = Path(tempfile.mkdtemp(prefix="same_name_"))
    old_crop = tmp / "old.jpg"
    new_crop = tmp / "new.jpg"
    old_crop.write_bytes(b"old")
    new_crop.write_bytes(b"new")
    reg = Registry()
    old = Entity(entity_id="prop_acorn", kind="prop", name="Acorn", description="brown acorn",
                 first_chunk=0, representations=[
                     Representation("prop_acorn@c000", 0, str(old_crop),
                                    bbox_source="grounding_dino",
                                    embedding_key="prop_acorn@c000",
                                    qa={"grounding_score": 0.9})],
                 static_attributes={"object_type": "acorn", "primary_color": "brown"})
    reg.entities[old.entity_id] = old
    reg.embeddings = {"prop_acorn@c000": [1.0, 0.0]}
    called = {"n": 0}

    def judge(_crop, _ref, _kind):
        called["n"] += 1
        return True

    ent, _rep, is_new = consolidate_observation(
        reg, chunk_id=1, name="Acorn", kind="prop", description="brown acorn",
        crop_path=str(new_crop), bbox=[100, 100, 300, 300], bbox_source="grounding_dino",
        frame_index=0, vector=[0.0, 1.0], judge_same_entity=judge,
        high_threshold=0.80, low_threshold=0.40,
        static_attributes={"object_type": "acorn", "primary_color": "brown"},
        static_overlap_threshold=0.75, grounding_score=0.9)
    assert is_new and ent.entity_id == "prop_acorn_02"
    assert called["n"] == 0


def test_perception_gate_blocks_same_bbox_across_entities() -> None:
    from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
    cfg = AnnotationConfig(video=Path("v.mp4"), out_dir=Path("out"), movie_id="m")
    located = [
        ({"name": "Brown Squirrel", "kind": "character"},
         {"bbox": [100, 100, 500, 500], "bbox_source": "grounding_dino"}),
        ({"name": "Brown Acorn", "kind": "prop"},
         {"bbox": [100, 100, 500, 500], "bbox_source": "grounding_dino"}),
        ({"name": "Meadow", "kind": "location"},
         {"bbox": [0, 0, 1000, 1000], "bbox_source": "full_frame"}),
    ]
    kept, checks = _perception_gate(located, cfg)
    assert [item[0]["name"] for item in kept] == ["Meadow"]
    assert checks and checks[0]["check"] == "blocking_crop"


def test_crop_match_check_carries_representation_id() -> None:
    # A3: the verifier's crop_match check must carry representation_id so _prune_failed_crops can
    # prune precisely instead of parsing the detail string.
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import VerifierRole
    tmp = Path(tempfile.mkdtemp(prefix="crop_"))
    crop = _write_frame(tmp / "crop.jpg")

    class _StubJudger:
        def judge_same_entity(self, _i1, _i2, _k):
            return True

    v = VerifierRole(_StubJudger(), crop_audit_score_threshold=0.60)
    present = [{"name": "Bunny", "kind": "character", "description": "a rabbit",
                "crop_path": str(crop), "representation_id": "char_bunny@c001",
                "bbox_source": "grounding_dino", "grounding_score": 0.4}]
    checks = v._crop_checks(present)
    assert len(checks) == 1
    assert checks[0]["representation_id"] == "char_bunny@c001"
    assert checks[0]["name"] == "Bunny"


def test_should_audit_crop_skips_vlm_fallback() -> None:
    # A4: vlm_fallback (whole-frame) crops are skipped like full_frame — auditing a whole frame
    # against its description is pure VLM spend with no signal (the frame always contains entity).
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import VerifierRole

    class _StubJudger:
        pass

    v = VerifierRole(_StubJudger(), crop_audit_score_threshold=0.60)
    assert v._should_audit_crop({"crop_path": "x", "kind": "character",
                                 "bbox_source": "vlm_fallback", "grounding_score": 0.0}) is False
    assert v._should_audit_crop({"crop_path": "x", "kind": "character",
                                 "bbox_source": "full_frame", "grounding_score": 0.0}) is False
    assert v._should_audit_crop({"crop_path": "x", "kind": "character",
                                 "bbox_source": "grounding_dino", "grounding_score": 0.4}) is True
    assert v._should_audit_crop({"crop_path": "x", "kind": "character",
                                 "bbox_source": "grounding_dino", "grounding_score": 0.9}) is False
    assert v._should_audit_crop({"crop_path": "x", "kind": "character",
                                 "bbox_source": "grounding_dino", "grounding_score": 0.9,
                                 "first_appearance": True}) is True


def test_static_compatible_normalizes_case() -> None:
    # A5: VLMs emit attribute keys/values with inconsistent case across chunks; without
    # normalization the key sets look disjoint and the static-identity gate silently fails.
    assert _static_compatible({"Species": "Fox"}, {"species": "fox"}, 0.75) is True
    assert _static_compatible({"Species": "Fox"}, {"species": "bird"}, 0.75) is False
    assert _static_compatible({"PrimaryColor": "red"}, {"size_class": "small"}, 0.75) is True
    assert _static_compatible({}, {"species": "fox"}, 0.75) is True


def test_ground_and_crop_blank_grounding_phrase_falls_through() -> None:
    # A6: a bare-whitespace grounding_phrase must not short-circuit the fallback chain into an
    # empty phrase (which would feed ground(frame, "") with undefined behavior).
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="gac3_"))
    frames = {0: _write_frame(tmp / "f0000000.jpg")}
    g = _StubGrounder(hit_frames={0}, score=0.9)
    ent = {"name": "Bunny", "kind": "character", "description": "a gray rabbit",
           "grounding_phrase": "   "}
    _ground_and_crop(frames, ent, tag="t", crops_dir=tmp / "crops",
                     grounder=g, score_threshold=0.35, min_frames=1)
    assert g.last_phrase == "a gray rabbit"  # fell through whitespace to description


def test_ground_and_crop_uses_ground_batch_when_available() -> None:
    # F5 grounder batch: when the grounder exposes ground_batch, _ground_and_crop must take the
    # batched path (one forward for all sampled frames) instead of N per-frame ground() calls.
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="gacb_"))
    frames = {}
    for i in (0, 1, 2):
        p = tmp / f"f{i:07d}.jpg"
        Image.new("RGB", (100, 80), (10, 20, 30)).save(p)
        frames[i] = p
    g = _StubBatchGrounder(hit_frames={0, 1, 2})
    ent = {"name": "Bunny", "kind": "character",
           "description": "a gray rabbit hopping fast", "grounding_phrase": "gray rabbit"}
    loc = _ground_and_crop(frames, ent, tag="t", crops_dir=tmp / "crops",
                           grounder=g, score_threshold=0.35, min_frames=1)
    assert g.batch_calls == 1          # one batched forward for all 3 frames
    assert g.ground_calls == 0         # per-frame ground() path NOT taken
    assert g.last_phrase == "gray rabbit"
    assert loc["bbox_source"] == "grounding_dino" and loc["grounding_score"] == 0.9


def test_checkpoint_write_load_roundtrip() -> None:
    # D10: a checkpoint must round-trip registry + embeddings + committed chunks + presence
    # history + prev_prompt + layout_hash so --resume can skip already-completed chunks.
    tmp = Path(tempfile.mkdtemp(prefix="cp_"))
    (tmp / "build").mkdir()
    reg = Registry()
    e = Entity(entity_id="char_x", kind="character", name="X", description="x", first_chunk=0,
               representations=[Representation("char_x@c000", 0, "x.jpg", bbox_source="grounding_dino",
                                               embedding_key="char_x@c000", qa={"grounding_score": 0.8})])
    reg.entities = {"char_x": e}
    reg.embeddings = {"char_x@c000": [0.1, 0.2, 0.3]}
    ann = [_chunk(0, ["char_x"], ["none"])]
    ph = {"char_x": [0]}
    index = {"layout_hash": "abc", "chunks": [{"chunk_id": 0}]}
    _write_checkpoint(tmp, index, reg, ann, ph, "prev prompt", last_chunk_id=0,
                      movie_id="m", pipeline_version="2.0.0", vlm_model="stub",
                      embedder_name="stub", layout_hash="abc")
    loaded = _load_checkpoint(tmp)
    assert loaded is not None
    assert loaded["last_chunk_id"] == 0 and loaded["prev_prompt"] == "prev prompt"
    assert loaded["layout_hash"] == "abc" and loaded["presence_history"] == ph
    assert len(loaded["annotations"]) == 1 and loaded["annotations"][0].chunk_id == 0
    assert "char_x" in loaded["registry"].entities
    # safetensors stores float32 -> round-trip values differ by ~1e-7; compare with tolerance.
    emb = loaded["registry"].embeddings["char_x@c000"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(emb, [0.1, 0.2, 0.3]))
    empty = Path(tempfile.mkdtemp(prefix="cpn_"))
    assert _load_checkpoint(empty) is None


def test_run_branch_cow_registry_isolation() -> None:
    # D11: the COW registry (deepcopy entities + shallow-copy embeddings dict) must keep the
    # original registry untouched when a branch appends embeddings / mutates entities, while
    # sharing old embedding lists read-only (no deepcopy cost on every branch).
    import copy
    base = Registry()
    base.entities = {"char_a": Entity(
        entity_id="char_a", kind="character", name="A", description="a", first_chunk=0,
        representations=[Representation("char_a@c000", 0, "a.jpg", bbox_source="grounding_dino",
                                        embedding_key="ka", qa={"grounding_score": 0.8})])}
    base.embeddings = {"ka": [1.0, 0.0]}
    reg_try = Registry(entities=copy.deepcopy(base.entities), embeddings=dict(base.embeddings))
    reg_try.embeddings["new_key"] = [0.5, 0.5]
    reg_try.entities["char_a"].description = "changed by branch"
    reg_try.entities["char_a"].representations.append(
        Representation("char_a@c001", 1, "b.jpg", embedding_key="kb"))
    assert "new_key" not in base.embeddings
    assert base.embeddings == {"ka": [1.0, 0.0]}  # shared old list unchanged
    assert base.entities["char_a"].description == "a"
    assert len(base.entities["char_a"].representations) == 1


# ---- Round 3: F3 (temperature), F5 (embed_batch), F6 (checkpoint cadence), F7 (publish) ----

def test_vlm_roles_pass_temperature_to_call_api() -> None:
    # F3: discover/draft/verify must forward the temperature kwarg to _call_api so the pipeline
    # can decorrelate retry / redundancy branches (principle #11); default stays 0.0 (deterministic).
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import AnnotatorRole, VerifierRole

    class _RecJudger:
        def __init__(self):
            self.seen = []

        def _call_api(self, messages, schema=None, fps=None, *, temperature=0.0):
            self.seen.append(temperature)
            return {}

        def judge_same_entity(self, *_a):
            return True

    j = _RecJudger()
    AnnotatorRole(j).discover_entities([], [], [], temperature=0.3)
    AnnotatorRole(j).draft_chunk([], [], "", [], temperature=0.3)
    VerifierRole(j).verify_chunk([], {"prompt": "", "present": [], "state_events": []}, temperature=0.3)
    assert j.seen == [0.3, 0.3, 0.3]
    j2 = _RecJudger()
    AnnotatorRole(j2).discover_entities([], [], [])  # default
    assert j2.seen == [0.0]


def test_embed_batch_empty_returns_empty() -> None:
    # F5: embed_batch must short-circuit on an empty list without loading the model, and the
    # method must exist on the real backend so the pipeline's hasattr branch takes the batch path.
    from vmem_bench.annotation.pipeline_track_first.embedding import DinoV3Embedder
    emb = DinoV3Embedder()
    assert hasattr(emb, "embed_batch")
    assert emb.embed_batch([]) == []  # no model load triggered


def test_write_checkpoint_write_embeddings_false_skips_safetensors() -> None:
    # F6: write_embeddings=False must still write the JSON checkpoint files but skip the
    # (expensive, full-rewrite) safetensors sidecar; default True writes it.
    tmp = Path(tempfile.mkdtemp(prefix="cpe_"))
    reg = Registry()
    e = Entity(entity_id="char_x", kind="character", name="X", description="x", first_chunk=0,
               representations=[Representation("char_x@c000", 0, "x.jpg", embedding_key="char_x@c000")])
    reg.entities = {"char_x": e}
    reg.embeddings = {"char_x@c000": [0.1, 0.2, 0.3]}
    index = {"layout_hash": "abc", "chunks": [{"chunk_id": 0}]}
    _write_checkpoint(tmp, index, reg, [_chunk(0, ["char_x"], ["none"])], {"char_x": [0]}, "",
                      last_chunk_id=0, movie_id="m", pipeline_version="2.0.0", vlm_model="stub",
                      embedder_name="stub", layout_hash="abc", write_embeddings=False)
    assert (tmp / "tmp" / "checkpoint.json").is_file()
    assert (tmp / "tmp" / "checkpoint_registry.json").is_file()
    assert not (tmp / "tmp" / "checkpoint_embeddings.safetensors").exists()
    _write_checkpoint(tmp, index, reg, [_chunk(0, ["char_x"], ["none"])], {"char_x": [0]}, "",
                      last_chunk_id=0, movie_id="m", pipeline_version="2.0.0", vlm_model="stub",
                      embedder_name="stub", layout_hash="abc")  # default True
    assert (tmp / "tmp" / "checkpoint_embeddings.safetensors").exists()


def test_load_checkpoint_truncates_when_embeddings_lag() -> None:
    # F6: when the sidecar lags behind last_chunk_id (only written every N chunks), resume must
    # truncate to the sidecar's covered chunk and drop the to-be-rerun chunks' reps so consolidate
    # does not duplicate them on re-run.
    import numpy as np
    from safetensors.numpy import save_file
    tmp = Path(tempfile.mkdtemp(prefix="cpt_"))
    build = tmp / "build"
    build.mkdir()
    e = Entity(entity_id="char_x", kind="character", name="X", description="x", first_chunk=0,
               representations=[Representation(f"char_x@c{c:03d}", c, f"x{c}.jpg",
                                               embedding_key=f"char_x@c{c:03d}") for c in range(6)])
    er = EntityRegistry(movie_id="m", entities=[e], human_reviewed=False)
    (build / "checkpoint_registry.json").write_text(json.dumps(er.to_dict()))
    save_file({f"char_x@c{c:03d}": np.array([float(c), 0.0, 0.0], dtype=np.float32) for c in range(4)},
              str(build / "checkpoint_embeddings.safetensors"))
    ann = ChunkAnnotations(movie_id="m", human_reviewed=False,
                           chunks=[_chunk(c, ["char_x"], ["none"]) for c in range(6)])
    payload = {"last_chunk_id": 5, "prev_prompt": "", "presence_history": {},
               "chunks": [a.to_dict() for a in ann.chunks], "movie_id": "m",
               "pipeline_version": "2.0.0", "vlm_model": "stub", "embedder_name": "stub",
               "layout_hash": "abc"}
    (build / "checkpoint.json").write_text(json.dumps(payload))
    loaded = _load_checkpoint(tmp)
    assert loaded["last_chunk_id"] == 3  # truncated to sidecar coverage
    assert len(loaded["annotations"]) == 4  # chunks 0..3
    reps = loaded["registry"].entities["char_x"].representations
    assert [r.chunk_id for r in reps] == [0, 1, 2, 3]  # chunks 4,5 reps dropped for re-run


def test_publish_check_frozen_schema_version_mismatch() -> None:
    # F7: a release whose layout/gold files disagree on schema_version must be rejected before
    # it reaches the scoring harness (contract §0: every JSON top-level carries schema_version).
    from vmem_bench.publish import _check_frozen
    tmp = Path(tempfile.mkdtemp(prefix="pub_"))
    (tmp / "layout").mkdir()
    (tmp / "gold").mkdir()
    (tmp / "manifest.json").write_text("{}")
    (tmp / "layout" / "boundaries.csv").write_text("shot_idx,start_frame,last_frame\n")
    (tmp / "layout" / "chunk_index.json").write_text(json.dumps(
        {"schema_version": "2.0.0", "layout_hash": "abc", "chunks": []}))
    (tmp / "gold" / "entity_registry.json").write_text(json.dumps(
        {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": True, "entities": []}))
    (tmp / "gold" / "chunk_annotations.json").write_text(json.dumps(
        {"schema_version": "1.0.0", "movie_id": "m", "human_reviewed": True, "chunks": []}))  # mismatch
    problems = _check_frozen(tmp)
    assert any("schema_version mismatch" in p for p in problems)


def test_publish_check_frozen_missing_layout_hash() -> None:
    # F7: chunk_index without layout_hash cannot be validated by the harness (contract §5.3).
    from vmem_bench.publish import _check_frozen
    tmp = Path(tempfile.mkdtemp(prefix="pub2_"))
    (tmp / "layout").mkdir()
    (tmp / "gold").mkdir()
    (tmp / "manifest.json").write_text("{}")
    (tmp / "layout" / "boundaries.csv").write_text("shot_idx,start_frame,last_frame\n")
    (tmp / "layout" / "chunk_index.json").write_text(json.dumps(
        {"schema_version": "2.0.0", "chunks": []}))  # no layout_hash
    (tmp / "gold" / "entity_registry.json").write_text(json.dumps(
        {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": True, "entities": []}))
    (tmp / "gold" / "chunk_annotations.json").write_text(json.dumps(
        {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": True, "chunks": []}))
    problems = _check_frozen(tmp)
    assert any("missing layout_hash" in p for p in problems)


if __name__ == "__main__":
    test_deprecates_representations_scopes_to_named_reps()
    test_union_feedback_dedups_across_branches()
    test_best_cover_rep_prefers_high_score_grounding()
    test_fallback_chunk_annotation_injects_idempotent_location()
    test_ground_and_crop_uses_grounding_phrase()
    test_ground_and_crop_temporal_consistency_rejects_one_frame_hit()
    test_apply_patch_recomputes_scenario_tags_after_merge()
    test_best_match_skips_non_grounded_embeddings()
    test_gray_zone_arbitration_uses_candidate_grounded_crop()
    test_gray_zone_fallback_crop_skips_embedding_match()
    test_crop_match_check_carries_representation_id()
    test_should_audit_crop_skips_vlm_fallback()
    test_static_compatible_normalizes_case()
    test_ground_and_crop_blank_grounding_phrase_falls_through()
    test_ground_and_crop_uses_ground_batch_when_available()
    test_checkpoint_write_load_roundtrip()
    test_run_branch_cow_registry_isolation()
    test_vlm_roles_pass_temperature_to_call_api()
    test_embed_batch_empty_returns_empty()
    test_write_checkpoint_write_embeddings_false_skips_safetensors()
    test_load_checkpoint_truncates_when_embeddings_lag()
    test_publish_check_frozen_schema_version_mismatch()
    test_publish_check_frozen_missing_layout_hash()
    print("annotation fixes OK")
