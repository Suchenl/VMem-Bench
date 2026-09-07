"""Focused guards for internal-only Track A/B timing provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRACKA_DIR = ROOT / "scripts/evaluate_baselines/trackA/baseline_adapters/causal"
TRACKB_COMMON = ROOT / "scripts/evaluate_baselines/trackB/baseline_runners/common.py"


def _load(path: Path, name: str, import_dir: Path):
    sys.path.insert(0, str(import_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(import_dir))


tracka = _load(TRACKA_DIR / "runner.py", "_tracka_timing_runner", TRACKA_DIR)
trackb = _load(TRACKB_COMMON, "_trackb_timing_common", TRACKB_COMMON.parent)


class _Adapter:
    name = "fake"

    def reset(self, movie):
        self.movie = movie

    def compose(self, request):
        return sys.modules["contract"].RetrievedMemory(chunk_id=request.chunk_id)

    def observe_segment(self, observation):
        self.last_observation = observation

    def finalize(self):
        return {"method": "fake"}


def test_tracka_writes_separated_internal_timing(tmp_path, monkeypatch):
    movie_dir = tmp_path / "assets" / "trackA" / "Dataset" / "Movie"
    movie_dir.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "outputs"

    monkeypatch.setattr(tracka, "_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        tracka,
        "_load_layout",
        lambda _: ({1: (0.0, 1.0), 2: (1.0, 2.0)}, {1: "first", 2: "selected"}),
    )
    monkeypatch.setattr(tracka, "_resolve_source_video", lambda _: source)
    monkeypatch.setattr(tracka, "gpu_snapshot", lambda: {"available": False, "reason": "test"})

    def fake_cut(_ffmpeg, _src, out, _s0, _s1):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"segment")
        return out

    monkeypatch.setattr(tracka, "_cut_segment", fake_cut)
    monkeypatch.setattr(tracka, "materialize_record_checkpoint", lambda **_: {})
    monkeypatch.setattr(
        tracka,
        "materialize_system",
        lambda **_: {"system": "fake__B16", "out": "selection.json", "chunks": 1},
    )

    adapter = _Adapter()
    summary = tracka.run_movie(
        adapter,
        movie_dir,
        ffmpeg="ffmpeg",
        fps=16.0,
        limit=None,
        budget=16,
        chunk_ids=[2],
    )
    timing_path = Path(summary["internal_timing"])
    timing = json.loads(timing_path.read_text(encoding="utf-8"))

    assert timing["visibility"] == "internal_only"
    assert timing["phases_ms"]["reset_cold_start"] >= 0
    assert timing["phases_ms"]["final_materialization"] >= 0
    assert timing["phases_ms"]["finalize"] >= 0
    assert timing["total_movie_wall_ms"] >= 0
    assert timing["aggregates_ms"]["method"] >= 0
    assert timing["aggregates_ms"]["scorer"] is None
    assert timing["segments"][0]["chunk_id"] == 2
    assert adapter.last_observation.chunk_id == 2
    assert timing["segments"][0].keys() >= {
        "segment_cut_ms",
        "compose_ms",
        "observe_ms",
        "checkpoint_materialization_ms",
    }


def test_trackb_command_and_manifest_reference_internal_timing(tmp_path, monkeypatch):
    out_dir = tmp_path / "run"
    log_path = out_dir / "logs" / "run.log"
    monkeypatch.setattr(trackb, "gpu_snapshot", lambda: {"available": False, "reason": "test"})

    rc = trackb.run_command(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        log_path=log_path,
    )
    stream = trackb.PromptStream(
        path=tmp_path / "prompts.json",
        story_id="story",
        title="Story",
        register="name_anchored",
        segments=[trackb.PromptSegment("seg-1", "prompt", 1.0, "cut")],
        raw={},
    )
    manifest_path = trackb.write_manifest(
        out_dir,
        system="fake",
        stream=stream,
        command=[sys.executable, "-c", "print('ok')"],
        status="done",
        exit_code=rc,
    )
    timing = json.loads((out_dir / "logs" / "timing.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert timing["visibility"] == "internal_only"
    assert timing["exit_code"] == 0
    assert timing["runtime"].keys() >= {
        "host",
        "python_version",
        "cuda_visible_devices",
    }
    assert timing["subprocess_max_rss"]["status"] == "unavailable"
    assert manifest["internal_artifacts"]["timing"] == "logs/timing.json"
    assert "runtime" not in manifest


def test_trackb_deferred_manifest_still_gets_timing(tmp_path, monkeypatch):
    out_dir = tmp_path / "deferred"
    monkeypatch.setattr(trackb, "gpu_snapshot", lambda: {"available": False, "reason": "test"})
    stream = trackb.PromptStream(
        path=tmp_path / "prompts.json",
        story_id="story",
        title="Story",
        register="name_anchored",
        segments=[trackb.PromptSegment("seg-1", "prompt", 1.0, "cut")],
        raw={},
    )
    trackb.write_manifest(
        out_dir,
        system="fake",
        stream=stream,
        command=["fake"],
        status="deferred",
        exit_code=75,
    )
    timing = json.loads((out_dir / "logs" / "timing.json").read_text(encoding="utf-8"))
    assert timing["execution"] == "not_started"
    assert timing["exit_code"] == 75


def test_trackb_launch_failure_is_recorded_without_swallowing_error(tmp_path, monkeypatch):
    log_path = tmp_path / "run" / "logs" / "run.log"
    monkeypatch.setattr(trackb, "gpu_snapshot", lambda: {"available": False, "reason": "test"})
    with pytest.raises(FileNotFoundError):
        trackb.run_command(
            [str(tmp_path / "missing-executable")],
            cwd=tmp_path,
            log_path=log_path,
        )
    timing = json.loads((log_path.parent / "timing.json").read_text(encoding="utf-8"))
    assert timing["exit_code"] is None
    assert timing["launch_error"]["type"] == "FileNotFoundError"
