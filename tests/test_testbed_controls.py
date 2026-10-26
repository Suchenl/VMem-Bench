"""Testbed controls: last-seen plan format/parsing and own-history selection, no-memory budget."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vmem_bench.testbed.controls import (
    LastSeenReferenceSelector,
    build_last_seen_plan,
    load_plan,
)
from vmem_bench.testbed.retrieval.retrievers import RetrieverConfig
from vmem_bench.testbed.runner import RunConfig, write_json
from vmem_bench.testbed.store import TrackBGeneratedStore, resolve_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def _gold() -> dict:
    return {
        "entities": {"E1": {"kind": "character"}, "E2": {"kind": "character"}, "E3": {"kind": "prop"},
                     "E4": {"kind": "location"}},
        "segments": [
            {"segment_id": "seg_001", "cast": [{"eid": "E1", "op": "introduce"}, {"eid": "E4", "op": "introduce"}]},
            {"segment_id": "seg_002", "cast": [{"eid": "E2", "op": "introduce"}, {"eid": "E3", "op": "introduce"}]},
            {"segment_id": "seg_003", "cast": [
                {"eid": "E3", "op": "persist", "gap": 0, "last_seen": "seg_002"},
                {"eid": "E1", "op": "recall", "gap": 1, "last_seen": "seg_001"},
                {"eid": "E2", "op": "recall", "gap": 0, "last_seen": "seg_002"},
                {"eid": "E4", "op": "recall", "gap": 1, "last_seen": "seg_001"},
            ]},
        ],
    }


def _clip(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([resolve_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc2=size=64x48:rate=8:duration={seconds}", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True, timeout=60)
    return path


def test_plan_file_round_trip(tmp_path: Path) -> None:
    plan, audit = build_last_seen_plan(_gold(), ["seg_001", "seg_002", "seg_003"])
    assert plan["plan"] == {"0": [], "1": [], "2": [0, 1, 1]}
    assert [r["eid"] for r in audit["entities"]["2"]] == ["E1", "E2", "E3"]
    path = write_json(tmp_path / "last_seen_plan.json", plan)
    assert path.read_text(encoding="utf-8") == json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    assert load_plan(path) == {0: [], 1: [], 2: [0, 1, 1]}


def test_real_story_plan_covers_every_shot() -> None:
    stream = json.loads((ROOT / "assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json")
                        .read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "assets/trackB/en/gt/0001_lighthouse_keeper.json").read_text(encoding="utf-8"))
    plan, _ = build_last_seen_plan(gold, [s["segment_id"] for s in stream["segments"]])
    assert set(plan) == {"rule", "plan"} and len(plan["plan"]) == len(stream["segments"])


def test_selector_top_k_dedup_then_preceding_filler(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = TrackBGeneratedStore(RetrieverConfig(), work_dir=run_dir / "retrieval_read")
    for i in range(4):
        store.observe_generated_segment(segment_id=i, prompt_text="p",
                                        segment_video=str(_clip(run_dir / "review" / "segments" / f"seg_{i:03d}.mp4")))
    sel = LastSeenReferenceSelector({4: [1, 1, 0, 2], 3: [0], 2: [2, 5]}, store=store, run_dir=run_dir, budget=3)
    assert sel.sources(segment_id=4) == [1, 0, 3]   # top-3 of the plan [1, 1, 0] deduped, then shot 3
    assert sel.sources(segment_id=3) == [0, 2]
    assert sel.sources(segment_id=2) == [1]          # 2 and 5 are not strictly earlier
    assert sel.sources(segment_id=0) == []
    refs = sel.select(segment_id=4)
    assert [r.representation_id for r in refs] == ["c00001@mid", "c00000@mid", "c00003@mid"]
    assert all(Path(r.image_ref).is_file() and run_dir in Path(r.image_ref).parents for r in refs)


def test_selector_rejects_clips_outside_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = TrackBGeneratedStore(RetrieverConfig(), work_dir=run_dir / "retrieval_read")
    store.observe_generated_segment(segment_id=0, prompt_text="p", segment_video=str(_clip(tmp_path / "other.mp4")))
    sel = LastSeenReferenceSelector({1: [0]}, store=store, run_dir=run_dir, budget=3)
    assert sel.sources(segment_id=1) == []


def test_selector_reads_own_progress_on_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    clip = _clip(run_dir / "review" / "segments" / "seg_000.mp4")
    write_json(run_dir / "progress.json", {"segments": [{"segment_id": 0, "video": str(clip)}]})
    store = TrackBGeneratedStore(RetrieverConfig(), work_dir=run_dir / "retrieval_read")
    sel = LastSeenReferenceSelector({1: [0]}, store=store, run_dir=run_dir, budget=3)
    assert sel.sources(segment_id=1) == [0]


def test_variant_names() -> None:
    assert RunConfig(variant="none", budget=3).budget == 0
    assert RunConfig(variant="oracle").variant == "last_seen"
    assert RunConfig(variant="frame_text", budget=0).budget == 1
    with pytest.raises(ValueError):
        RunConfig(variant="name_match")
