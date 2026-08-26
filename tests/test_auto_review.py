"""Unit tests for machine-assisted auto_review (fakes only; no GPU/model calls)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from vmem_bench.annotation.pipeline_track_first.auto_review import (
    entity_dispersion, name_agreement, run_auto_review, split_merge_tiers, suspicion_score,
)
from vmem_bench.common.schemas import (
    ChunkAnnotation, ChunkAnnotations, Entity, EntityRegistry, Representation,
)


def test_entity_dispersion_orthogonal_and_singleton() -> None:
    ent = Entity(
        entity_id="char_a", kind="character", name="A", description="x", first_chunk=0,
        representations=[
            Representation("char_a@c000", 0, "a.jpg", embedding_key="char_a@c000"),
            Representation("char_a@c001", 1, "b.jpg", embedding_key="char_a@c001"),
        ])
    emb = {"char_a@c000": [1.0, 0.0], "char_a@c001": [0.0, 1.0]}
    assert entity_dispersion(ent, emb) == 0.0  # orthogonal -> min cos 0

    single = Entity(
        entity_id="char_b", kind="character", name="B", description="x", first_chunk=0,
        representations=[Representation("char_b@c000", 0, "c.jpg", embedding_key="char_b@c000")])
    assert entity_dispersion(single, {"char_b@c000": [1.0, 0.0]}) is None


def test_suspicion_score_weights() -> None:
    # dispersion below floor -> +2; singleton -> +1; qa_flagged*0.5; short screen -> +0.5
    assert suspicion_score({"dispersion": 0.1, "singleton": False, "qa_flagged": 0,
                            "screen_time_seconds": 5.0}, dispersion_floor=0.35) == 2.0
    assert suspicion_score({"dispersion": None, "singleton": True, "qa_flagged": 0,
                            "screen_time_seconds": None}) == 1.0
    assert suspicion_score({"dispersion": None, "singleton": False, "qa_flagged": 2,
                            "screen_time_seconds": 0.5}) == 1.5  # 0.5*2 + 0.5
    assert suspicion_score({"dispersion": 0.9, "singleton": False, "qa_flagged": 0,
                            "screen_time_seconds": 2.0}) == 0.0


def test_split_merge_tiers_body_none_is_gray() -> None:
    props = [
        {"keep": "a", "merge": "b", "text_cos": 0.95, "body_cos": 0.80},
        {"keep": "c", "merge": "d", "text_cos": 0.95, "body_cos": None},
        {"keep": "e", "merge": "f", "text_cos": 0.80, "body_cos": 0.90},
    ]
    auto, gray = split_merge_tiers(props, auto_text=0.92, auto_body=0.75)
    assert auto == [props[0]]
    assert gray == [props[1], props[2]]


def _minimal_movie(tmp: Path, *, body_a=None, body_b=None) -> Path:
    """Two near-duplicate character entities with high text+body similarity."""
    gold = tmp / "gold"
    build = tmp / "build"
    gold.mkdir()
    build.mkdir()
    body_a = body_a or [1.0, 0.0]
    body_b = body_b or [0.99, 0.01]
    a = Entity(
        entity_id="char_bunny", kind="character", name="Bunny",
        description="grey rabbit", first_chunk=0,
        representations=[Representation(
            "char_bunny@c000", 0, "assets/char_bunny/c000.jpg",
            embedding_key="char_bunny@c000")],
        screen_time_seconds=5.0)
    b = Entity(
        entity_id="char_bunny_02", kind="character", name="Bunny",
        description="grey rabbit", first_chunk=1,
        representations=[Representation(
            "char_bunny_02@c001", 1, "assets/char_bunny_02/c001.jpg",
            embedding_key="char_bunny_02@c001")],
        screen_time_seconds=4.0)
    er = EntityRegistry(movie_id="m", entities=[a, b], human_reviewed=False)
    chunks = ChunkAnnotations(movie_id="m", human_reviewed=False, chunks=[
        ChunkAnnotation(chunk_id=0, shot_span=[0, 0], frame_span=[0, 99],
                        prompt="bunny hops", present=["char_bunny"],
                        first_appearances=["char_bunny"]),
        ChunkAnnotation(chunk_id=1, shot_span=[1, 1], frame_span=[100, 199],
                        prompt="bunny again", present=["char_bunny_02"],
                        first_appearances=["char_bunny_02"]),
    ])
    (gold / "entity_registry.json").write_text(json.dumps(er.to_dict()), encoding="utf-8")
    (gold / "chunk_annotations.json").write_text(json.dumps(chunks.to_dict()), encoding="utf-8")
    save_file({
        "char_bunny@c000": np.array(body_a, dtype=np.float32),
        "char_bunny_02@c001": np.array(body_b, dtype=np.float32),
    }, str(gold / "embeddings.safetensors"))
    return tmp


def test_run_auto_review_applies_safe_merge(tmp_path: Path | None = None) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="auto_rev_")) if tmp_path is None else tmp_path
    out = _minimal_movie(tmp)

    def fake_text_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    report = run_auto_review(
        out, apply_safe=True, text_embed_fn=fake_text_embed,
        auto_text=0.92, auto_body=0.75,
        merge_text_threshold=0.85, merge_body_threshold=0.5)

    assert report["stats"]["n_applied_merges"] == 1
    assert (out / "tmp" / "auto_review.json").is_file()
    assert (out / "tmp" / "auto_review_patch.json").is_file()
    gold = json.loads((out / "gold" / "entity_registry.json").read_text(encoding="utf-8"))
    ids = {e["entity_id"] for e in gold["entities"]}
    assert "char_bunny" in ids
    assert "char_bunny_02" not in ids
    html = (out / "review.html").read_text(encoding="utf-8")
    assert "机器审核已排序" in html
    assert 'class="machine"' in html or "score=" in html
    # stats count SURVIVING entities: merged-away ids must not inflate the review surface.
    assert report["stats"]["n_entities"] == 1


def test_run_auto_review_apply_safe_false_leaves_gold() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="auto_rev_no_"))
    out = _minimal_movie(tmp)
    before = (out / "gold" / "entity_registry.json").read_text(encoding="utf-8")

    def fake_text_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    report = run_auto_review(
        out, apply_safe=False, text_embed_fn=fake_text_embed,
        auto_text=0.92, auto_body=0.75)
    assert report["stats"]["n_applied_merges"] == 0
    assert (out / "gold" / "entity_registry.json").read_text(encoding="utf-8") == before
    assert not (out / "tmp" / "auto_review_patch.json").exists()
    assert not (out / "build" / "auto_review_patch.json").exists()


def test_name_agreement_returns_none_on_raise() -> None:
    class Boom:
        def classify(self, crop, labels):  # noqa: ANN001
            raise RuntimeError("no model")

    assert name_agreement(Boom(), Path("/tmp/x.jpg"), "Bunny", "grey") is None


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()


def test_vlm_three_vote_confirmed_merges() -> None:
    from vmem_bench.annotation.pipeline_track_first.auto_review import vlm_confirmed_merges
    from vmem_bench.annotation.pipeline_track_first.consolidation import Registry

    tmp = Path(tempfile.mkdtemp(prefix="vlm_vote_"))
    def _ent(eid: str, species: str, n_reps: int) -> Entity:
        reps = []
        for i in range(n_reps):
            crop = tmp / f"{eid}_{i}.jpg"
            crop.write_bytes(b"fake")
            reps.append(Representation(f"{eid}@c{i:03d}", i, str(crop),
                                       embedding_key=f"{eid}@c{i:03d}"))
        return Entity(entity_id=eid, kind="character", name=eid, description="", first_chunk=0,
                      representations=reps, static_attributes={"species": species})

    reg = Registry()
    reg.entities = {
        "char_a": _ent("char_a", "rabbit", 5),
        "char_b": _ent("char_b", "rabbit", 2),
        "char_c": _ent("char_c", "bird", 1),
    }

    class FakeRole:
        def judge_same_individual_pair(self, a_crops, b_crops, a_label, b_label):
            return {"same": True, "reason": "same fur"}

    candidates = [
        {"keep": "char_a", "merge": "char_b", "text_cos": 0.86, "body_cos": 0.6},
        # species guard must veto even though the fake VLM would say yes:
        {"keep": "char_a", "merge": "char_c", "text_cos": 0.86, "body_cos": 0.6},
    ]
    merges, votes = vlm_confirmed_merges(reg, tmp, candidates, FakeRole())
    assert merges == [["char_a", "char_b"]]  # survivor = most representations
    verdicts = {(v["keep"], v["merge"]): v["verdict"] for v in votes}
    assert verdicts[("char_a", "char_b")] == "auto_merge"
    assert verdicts[("char_a", "char_c")] == "guard_reject"


def test_vlm_vote_rejection_leaves_pair_for_humans() -> None:
    from vmem_bench.annotation.pipeline_track_first.auto_review import vlm_confirmed_merges
    from vmem_bench.annotation.pipeline_track_first.consolidation import Registry

    tmp = Path(tempfile.mkdtemp(prefix="vlm_vote_rej_"))
    crop = tmp / "c.jpg"; crop.write_bytes(b"fake")
    def _ent(eid: str) -> Entity:
        return Entity(entity_id=eid, kind="character", name=eid, description="", first_chunk=0,
                      representations=[Representation(f"{eid}@c0", 0, str(crop),
                                                      embedding_key=f"{eid}@c0")],
                      static_attributes={"species": "rabbit"})
    reg = Registry(); reg.entities = {"char_a": _ent("char_a"), "char_b": _ent("char_b")}

    class SayNo:
        def judge_same_individual_pair(self, *a, **k):
            return {"same": False, "reason": "different markings"}

    merges, votes = vlm_confirmed_merges(
        reg, tmp, [{"keep": "char_a", "merge": "char_b", "text_cos": 0.9, "body_cos": 0.7}],
        SayNo())
    assert merges == [] and votes[0]["verdict"] == "vlm_reject"
