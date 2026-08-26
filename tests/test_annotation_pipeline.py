"""Self-check for the offline annotation pipeline.

Part 1: unit checks of the deterministic logic (chunk aggregation, forbidden
materialization, scenario tags). Part 2: end-to-end run on a tiny synthetic video with
deterministic stub backends (no GPU, no VLM service), exercising the QA retry loop,
gold persistence, review page, patch application and freezing.

Run: cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_annotation_pipeline.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path
import pytest
from unittest import mock

from vmem_bench.annotation.pipeline_track_first.chunking import ChunkSpec, aggregate_shots, layout_hash, shots_from_boundaries
from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
from vmem_bench.annotation.pipeline_track_first.consolidation import Registry, consolidate_observation
from vmem_bench.annotation.pipeline_track_first.drafting import materialize_forbidden, scenario_tags_for
from vmem_bench.annotation.pipeline_track_first.pipeline import (
    _branch_role_pairs,
    _only_crop_failures,
    _prune_failed_crops,
    annotate_movie,
)
from vmem_bench.annotation.pipeline_track_first.review import apply_patch, freeze
from vmem_bench.annotation.pipeline_track_first.vlm_roles import VerifierRole
from vmem_bench.common.schemas import ChunkAnnotations, Entity, EntityRegistry, Representation, StateEvent
from vmem_bench.judger.vlm import VlmJudger


# ---------------------------------------------------------------------- unit checks

def test_aggregate_shots() -> None:
    # frame_span is closed inclusive [first, last]: adjacent chunks never share a frame number.
    chunks = aggregate_shots([(0, 30), (30, 60), (60, 130), (130, 140)],
                             min_frames=48, max_frames=120)
    assert [c.frame_span for c in chunks] == [[0, 59], [60, 139]]  # 60 | 70+10(邻并) frames
    assert [c.shot_span for c in chunks] == [[0, 1], [2, 3]]
    sizes = [c.frame_span[1] - c.frame_span[0] + 1 for c in chunks]  # +1: closed interval
    assert all(48 <= s <= 120 for s in sizes), sizes
    # Oversized shot splits evenly inside the shot into in-range parts.
    chunks = aggregate_shots([(0, 250)], min_frames=48, max_frames=100)
    assert [c.frame_span for c in chunks] == [[0, 83], [84, 167], [168, 249]]
    # Trailing undersized fragment merges backward when the merge fits the cap.
    chunks = aggregate_shots([(0, 100), (100, 110)], min_frames=48, max_frames=120)
    assert [c.frame_span for c in chunks] == [[0, 109]]
    # ...but stays separate when merging would overflow the cap.
    chunks = aggregate_shots([(0, 115), (115, 125)], min_frames=48, max_frames=120)
    assert [c.frame_span for c in chunks] == [[0, 114], [115, 124]]
    # Full coverage, no gaps, no overlap (contiguous closed intervals), ids renumbered.
    spans = [c.frame_span for c in aggregate_shots([(0, 48), (48, 96)],
                                                   min_frames=24, max_frames=30)]
    assert spans[0][0] == 0 and spans[-1][1] == 95
    assert all(spans[i][1] + 1 == spans[i + 1][0] for i in range(len(spans) - 1))
    assert shots_from_boundaries([48], 96) == [(0, 48), (48, 96)]  # internal helper stays half-open
    chunks = aggregate_shots([(0, 48), (48, 96)], min_frames=24, max_frames=60)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert layout_hash(chunks, 24.0) != layout_hash(chunks[:-1], 24.0)


def test_forbidden_and_tags() -> None:
    registry = Registry()
    ball = Entity(entity_id="prop_ball", kind="prop", name="Ball", description="a red ball",
                  first_chunk=0, representations=[
                      Representation("prop_ball@c000", 0, "x.jpg", embedding_key="prop_ball@c000"),
                      Representation("prop_ball@c001", 1, "y.jpg", embedding_key="prop_ball@c001")],
                  state_events=[StateEvent("evt", 1, "deflated",
                                           deprecates=["prop_ball@c000", "prop_ball@c001"])])
    registry.entities["prop_ball"] = ball
    registry.embeddings = {"prop_ball@c000": [1.0, 0.0], "prop_ball@c001": [1.0, 0.0]}
    # F_active(t): only events with chunk_id < t (the event chunk itself is not forbidden).
    assert materialize_forbidden(registry, 1) == []
    assert {f.representation_id for f in materialize_forbidden(registry, 2)} == {
        "prop_ball@c000", "prop_ball@c001"}
    # Re-appearance: present at 0,1 then again at 3 (gap at 2).
    tags = scenario_tags_for(3, ["prop_ball"], set(), {"prop_ball": [0, 1]}, registry, False)
    assert "re-appearance" in tags
    tags = scenario_tags_for(1, ["prop_ball"], set(), {"prop_ball": [0]}, registry, True)
    assert tags == ["state-change"]


def test_static_identity_conflict_blocks_name_reuse() -> None:
    registry = Registry()

    def _judge(*_args):
        return True

    first, _, is_new = consolidate_observation(
        registry, chunk_id=0, name="Alex", kind="character",
        description="a red fox", static_attributes={"subcategory": "fox", "primary_color": "red"},
        crop_path="red.jpg", bbox=[0, 0, 100, 100], bbox_source="stub", frame_index=0,
        vector=[1.0, 0.0], judge_same_entity=_judge,
        high_threshold=0.8, low_threshold=0.4, static_overlap_threshold=0.75)
    assert is_new and first.entity_id == "char_alex"

    second, _, is_new = consolidate_observation(
        registry, chunk_id=1, name="Alex", kind="character",
        description="a blue bird", static_attributes={"subcategory": "bird", "primary_color": "blue"},
        crop_path="blue.jpg", bbox=[0, 0, 100, 100], bbox_source="stub", frame_index=1,
        vector=[1.0, 0.0], judge_same_entity=_judge,
        high_threshold=0.8, low_threshold=0.4, static_overlap_threshold=0.75)
    assert is_new and second.entity_id == "char_alex_02"
    assert len(registry.entities) == 2


def test_crop_match_failure_prunes_bad_representation() -> None:
    registry = Registry()
    entity = Entity(entity_id="char_bunny", kind="character", name="Bunny",
                    description="gray rabbit", first_chunk=0,
                    representations=[
                        Representation("char_bunny@c000", 0, "ok.jpg", embedding_key="char_bunny@c000"),
                        Representation("char_bunny@c001", 1, "bad.jpg", embedding_key="char_bunny@c001"),
                    ])
    registry.entities[entity.entity_id] = entity
    registry.embeddings = {"char_bunny@c000": [1.0, 0.0], "char_bunny@c001": [0.0, 1.0]}
    present = [{"entity_id": "char_bunny", "name": "Bunny",
                "representation_id": "char_bunny@c001"}]
    checks = [{"check": "crop_match", "passed": False,
               "detail": "Bunny: crop does not match description"}]

    assert _only_crop_failures(checks)
    kept = _prune_failed_crops(registry, present, checks)
    assert kept == present
    assert [r.representation_id for r in registry.entities["char_bunny"].representations] == [
        "char_bunny@c000"]
    assert "char_bunny@c001" not in registry.embeddings

    new_entity = Entity(entity_id="char_noise", kind="character", name="Noise",
                        description="bad crop", first_chunk=1,
                        representations=[
                            Representation("char_noise@c001", 1, "bad.jpg", embedding_key="char_noise@c001")])
    registry.entities[new_entity.entity_id] = new_entity
    registry.embeddings["char_noise@c001"] = [0.0, 1.0]
    dropped = _prune_failed_crops(
        registry, [{"entity_id": "char_noise", "name": "Noise",
                    "representation_id": "char_noise@c001"}],
        [{"check": "crop_match", "passed": False,
          "detail": "Noise: crop does not match description"}])
    assert dropped == []
    assert "char_noise" not in registry.entities


def test_endpoint_pool_does_not_imply_per_chunk_ensemble() -> None:
    annotators = ["a0", "a1", "a2"]
    verifiers = ["v0", "v1"]

    assert _branch_role_pairs(
        chunk_id=0, attempt=1, branches_per_chunk=1,
        annotators=annotators, verifiers=verifiers) == [(0, "a0", "v0")]
    assert _branch_role_pairs(
        chunk_id=1, attempt=1, branches_per_chunk=1,
        annotators=annotators, verifiers=verifiers) == [(0, "a1", "v1")]
    assert _branch_role_pairs(
        chunk_id=2, attempt=1, branches_per_chunk=1,
        annotators=annotators, verifiers=verifiers) == [(0, "a2", "v0")]

    assert _branch_role_pairs(
        chunk_id=0, attempt=1, branches_per_chunk=3,
        annotators=annotators, verifiers=verifiers) == [
            (0, "a0", "v0"), (1, "a1", "v1"), (2, "a2", "v0")]

    try:
        _branch_role_pairs(
            chunk_id=0, attempt=1, branches_per_chunk=0,
            annotators=annotators, verifiers=verifiers)
        raise AssertionError("branches_per_chunk=0 should fail")
    except ValueError as exc:
        assert "branches_per_chunk" in str(exc)


def test_verifier_skips_location_full_frame_crop_audit() -> None:
    class FakeJudger:
        def __init__(self) -> None:
            self.judge_calls = 0

        def _call_api(self, *_args, **_kwargs):
            return {"checks": [
                {"check": "presence_recall", "passed": True, "detail": ""},
                {"check": "presence_precision", "passed": True, "detail": ""},
                {"check": "prompt_completeness", "passed": True, "detail": ""},
                {"check": "prompt_faithful", "passed": True, "detail": ""},
            ]}

        def judge_same_entity(self, *_args):
            self.judge_calls += 1
            return True

    fake = FakeJudger()
    checks = VerifierRole(fake).verify_chunk([], {"prompt": "A meadow.",
        "state_events": [], "present": [
            {"name": "Meadow", "kind": "location", "description": "a meadow",
             "crop_path": "missing.jpg", "bbox_source": "full_frame"}]})
    assert all(c["passed"] for c in checks)
    assert fake.judge_calls == 0


def test_verifier_skips_high_confidence_grounding_crop_audit() -> None:
    class FakeJudger:
        def __init__(self) -> None:
            self.judge_calls = 0

        def _call_api(self, *_args, **_kwargs):
            return {"checks": [
                {"check": "presence_recall", "passed": True, "detail": ""},
                {"check": "presence_precision", "passed": True, "detail": ""},
                {"check": "prompt_completeness", "passed": True, "detail": ""},
                {"check": "prompt_faithful", "passed": True, "detail": ""},
            ]}

        def judge_same_entity(self, *_args):
            self.judge_calls += 1
            return True

    fake = FakeJudger()
    checks = VerifierRole(fake, crop_audit_score_threshold=0.60).verify_chunk([], {
        "prompt": "Bunny hops.", "state_events": [], "present": [
            {"name": "Bunny", "kind": "character", "description": "gray rabbit",
             "crop_path": "missing.jpg", "bbox_source": "grounding_dino",
             "grounding_score": 0.91}]})
    assert all(c["passed"] for c in checks)
    assert fake.judge_calls == 0


def test_vlm_http_4xx_fails_fast() -> None:
    err = urllib.error.HTTPError("http://127.0.0.1:8000/v1/chat/completions", 404,
                                 "Not Found", hdrs=None, fp=None)
    with mock.patch("urllib.request.urlopen", side_effect=err) as urlopen:
        try:
            VlmJudger(base_url="http://127.0.0.1:8000/v1", model="wrong-model")._call_api(
                [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                {"type": "object", "properties": {}, "additionalProperties": False})
            raise AssertionError("HTTP 404 should fail fast")
        except RuntimeError as exc:
            assert "non-retryable HTTP 404" in str(exc)
            assert "wrong-model" in str(exc)
    assert urlopen.call_count == 1


# ---------------------------------------------------------------------- E2E with stubs

RED_CHUNKS = {0, 1}
CAST = {  # chunk_id -> [(name, kind, description)]
    0: [("Red Ball", "prop", "a shiny red rubber ball"), ("Meadow", "location", "a sunny meadow")],
    1: [("Red Ball", "prop", "a shiny red rubber ball"), ("Meadow", "location", "a sunny meadow")],
    2: [("Green Cube", "prop", "a matte green cube"), ("Meadow", "location", "a sunny meadow")],
    3: [("Red Ball", "prop", "a shiny red rubber ball"), ("Green Cube", "prop", "a matte green cube"),
        ("Meadow", "location", "a sunny meadow")],
}


def _chunk_of(frames: list[Path]) -> int:
    index = int(re.search(r"f(\d+)", frames[0].stem).group(1))
    return index // 24


class StubAnnotator:
    def discover_entities(self, frames, known_entities, feedback, *, temperature=0.0):
        return [{"name": n, "kind": k, "description": d} for n, k, d in CAST[_chunk_of(frames)]]

    def judge_same_entity(self, crop, description, kind):
        return True

    def draft_chunk(self, frames, present, prev_prompt, feedback, *, temperature=0.0):
        chunk = _chunk_of(frames)
        names = ", ".join(p["name"] for p in present)
        events = ([{"entity_id": next(p["entity_id"] for p in present if p["name"] == "Red Ball"),
                    "description": "the red ball deflates"}] if chunk == 1 else [])
        return {"prompt": f"In the meadow, {names} appear; scene {chunk}.", "state_events": events}


class StubVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify_chunk(self, frames, annotation, *, temperature=0.0):
        self.calls += 1
        failed = self.calls == 1  # fail the very first attempt to exercise the retry loop
        return [{"check": "presence_recall", "passed": not failed,
                 "detail": "missing: Red Ball" if failed else ""},
                {"check": "prompt_completeness", "passed": True, "detail": ""}]


class StubGrounder:
    def ground(self, image, phrase):
        if "green cube" in phrase.lower():
            return [250, 50, 750, 450], 0.9
        return [250, 250, 750, 750], 0.9


class StubEmbedder:
    def embed_image(self, image):
        from PIL import Image
        pixel = Image.open(image).convert("RGB").resize((1, 1)).getpixel((0, 0))
        total = sum(pixel) or 1
        return [c / total for c in pixel]


def _make_video(path: Path) -> None:
    from PIL import Image
    frames_dir = path.parent / "src_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(96):
        color = (200, 30, 30) if i < 48 else (30, 200, 30)
        Image.new("RGB", (128, 96), color).save(frames_dir / f"{i:03d}.png")
    from vmem_bench.common.media import ffmpeg_bin
    subprocess.run([ffmpeg_bin(), "-y", "-framerate", "24", "-i", str(frames_dir / "%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True)


@pytest.mark.xfail(reason="upstream: freeze requires human state-event decisions; fails in internal tree too", strict=False)
def test_e2e_stub() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vmem_bench_e2e_"))
    try:
        video = tmp / "movie.mp4"
        _make_video(video)
        out = tmp / "out"
        config = AnnotationConfig(video=video, out_dir=out, movie_id="stub_movie",
                                  min_frames_per_chunk=12, max_frames_per_chunk=24,
                                  max_sampled_frames=3, qa_max_rounds=2)
        summary = annotate_movie(config, annotator=StubAnnotator(), verifier=StubVerifier(),
                                 grounder=StubGrounder(), embedder=StubEmbedder(),
                                 shots=[(0, 48), (48, 96)])
        assert summary["n_chunks"] == 4 and summary["n_flagged_chunks"] == 0

        gold = out / "gold"
        registry = EntityRegistry.from_dict(
            json.loads((gold / "entity_registry.json").read_text()))
        chunks = ChunkAnnotations.from_dict(
            json.loads((gold / "chunk_annotations.json").read_text()))
        assert not registry.human_reviewed and not chunks.human_reviewed
        by_name = {e.name: e for e in registry.entities}
        assert set(by_name) == {"Red Ball", "Green Cube", "Meadow"}, set(by_name)

        # Naming-prior consolidation: one entity per name, one representation per chunk seen.
        assert len(by_name["Red Ball"].representations) == 3   # chunks 0, 1, 3
        assert len(by_name["Meadow"].representations) == 4
        assert by_name["Meadow"].representations[0].bbox_source == "full_frame"
        assert by_name["Red Ball"].representations[0].bbox_source == "grounding_dino"

        # State event on chunk 1 -> forbidden from chunk 2 onward (not chunk 1 itself).
        ball = by_name["Red Ball"]
        assert len(ball.state_events) == 1 and ball.state_events[0].chunk_id == 1
        ann = {c.chunk_id: c for c in chunks.chunks}
        assert ann[1].forbidden == []
        dep = {f.representation_id for f in ann[2].forbidden}
        assert dep == set(ball.state_events[0].deprecates) and len(dep) == 2
        assert {f.representation_id for f in ann[3].forbidden} == dep

        # Presence / first appearances / instructions / tags.
        assert ann[0].first_appearances == sorted(ann[0].present)
        assert ball.entity_id in ann[3].present and ball.entity_id not in ann[3].first_appearances
        reqs = {g.entity_id: g.requirement for g in ann[3].gold_instructions}
        assert reqs[ball.entity_id] == "continuity"
        assert "re-appearance" in ann[3].scenario_tags
        assert "state-change" in ann[1].scenario_tags
        for c in chunks.chunks:
            assert set(c.first_appearances) <= set(c.present)
            assert c.prompt  # never empty

        # QA loop: first verifier call failed -> chunk 0 needed 2 rounds, others 1.
        from vmem_bench.common.paths import MovieDirs, entity_asset_dir, is_entity_asset_path
        qa = json.loads(MovieDirs(out).qa_report.read_text())
        assert qa[0]["rounds"] == 2 and all(q["rounds"] == 1 for q in qa[1:])
        assert all(not q["flagged"] for q in qa)

        # Embedding sidecar (principle #10): all representation keys, none inline in JSON.
        from safetensors.numpy import load_file
        vectors = load_file(str(gold / "embeddings.safetensors"))
        n_reps = sum(len(e.representations) for e in registry.entities)
        assert len(vectors) == n_reps
        assert "\"embedding\"" not in (gold / "entity_registry.json").read_text()

        # Layout hash recorded in provenance and gold/chunk_index.json (legacy: layout/).
        index = json.loads(MovieDirs(out).chunk_index.read_text())
        assert registry.annotation_provenance["layout_hash"] == index["layout_hash"]
        assert (out / "review.html").is_file()

        # Asset library: committed crops are per-entity, relative-pathed, with a cover;
        # no attempt/branch tags leak into the gold registry, and clips are never shipped.
        for e in registry.entities:
            for r in e.representations:
                assert is_entity_asset_path(r.crop_path, e.entity_id, e.kind), r.crop_path
                assert (out / r.crop_path).is_file()
            assert (entity_asset_dir(out / "assets", e.entity_id, e.kind) / "cover.jpg").is_file()
        assert not (out / "chunks").exists() and not (out / "layout" / "clips").exists()
        # manifest ships the download recipe (source sha + how to obtain), not the video.
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["source"]["sha256"] and manifest["source"]["download"]["dataset"]
        assert manifest["layout"]["n_chunks"] == summary["n_chunks"]

        # Review patch: rename + drop + prompt edit, then freeze.
        cube_id = by_name["Green Cube"].entity_id
        patch = {"schema_version": "2.0.0", "merges": [], "splits": [],
                 "renames": {ball.entity_id: "Scarlet Ball"}, "drops": [cube_id],
                 "field_edits": [{"path": "chunks[0].prompt", "value": "Edited prompt."}]}
        patch_path = tmp / "review_patch.json"
        patch_path.write_text(json.dumps(patch))
        apply_patch(gold, patch_path)
        freeze(gold)
        registry2 = EntityRegistry.from_dict(json.loads((gold / "entity_registry.json").read_text()))
        chunks2 = ChunkAnnotations.from_dict(json.loads((gold / "chunk_annotations.json").read_text()))
        assert registry2.human_reviewed and chunks2.human_reviewed
        names2 = {e.name for e in registry2.entities}
        assert "Scarlet Ball" in names2 and "Green Cube" not in names2
        ann2 = {c.chunk_id: c for c in chunks2.chunks}
        assert cube_id not in ann2[3].present
        assert ann2[0].prompt == "Edited prompt."

        # publish: ships manifest + gold/ (+ layout files under gold/) + assets/; never derived/ or tmp/.
        from vmem_bench.publish import publish
        release = publish(out, tmp / "release")
        shipped = {str(p.relative_to(release)) for p in release.rglob("*") if p.is_file()}
        required = {
            "manifest.json", "gold/chunk_index.json", "gold/shot_boundaries.csv",
            "gold/entity_registry.json", "gold/chunk_annotations.json",
            "gold/embeddings.safetensors", "SHA256SUMS"}
        assert required <= shipped, shipped
        assert any(p.startswith("assets/") for p in shipped), shipped
        assert not (release / "derived").exists() and not (release / "build").exists()
        assert not (release / "tmp").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_publish_refuses_unfrozen() -> None:
    import json as _json

    from vmem_bench.publish import publish
    tmp = Path(tempfile.mkdtemp(prefix="vmem_bench_pub_"))
    try:
        (tmp / "layout").mkdir(parents=True)
        (tmp / "gold").mkdir(parents=True)
        (tmp / "manifest.json").write_text("{}")
        (tmp / "layout" / "chunk_index.json").write_text(_json.dumps(
            {"schema_version": "2.0.0", "layout_hash": "x", "chunks": []}))
        (tmp / "layout" / "boundaries.csv").write_text("shot_idx,start_frame,last_frame\n")
        (tmp / "gold" / "entity_registry.json").write_text(_json.dumps(
            {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": False, "entities": []}))
        (tmp / "gold" / "chunk_annotations.json").write_text(_json.dumps(
            {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": True, "chunks": []}))
        try:
            publish(tmp, tmp / "rel")
            raise AssertionError("publish should refuse unfrozen gold")
        except SystemExit:
            pass
        publish(tmp, tmp / "rel_forced", force=True)  # --force bypasses the freeze gate
        assert (tmp / "rel_forced" / "SHA256SUMS").is_file()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_aggregate_shots()
    test_forbidden_and_tags()
    test_static_identity_conflict_blocks_name_reuse()
    test_crop_match_failure_prunes_bad_representation()
    test_endpoint_pool_does_not_imply_per_chunk_ensemble()
    test_verifier_skips_location_full_frame_crop_audit()
    test_verifier_skips_high_confidence_grounding_crop_audit()
    test_vlm_http_4xx_fails_fast()
    test_publish_refuses_unfrozen()
    print("unit checks OK")
    test_e2e_stub()
    print("stub E2E OK")
    sys.exit(0)
