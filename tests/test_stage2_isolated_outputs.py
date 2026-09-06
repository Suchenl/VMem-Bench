"""Stage-2 must score the isolated canary tree, never stale paper outputs."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from vmem_bench.scoring import visual_coverage
from vmem_bench.scoring.judge_service import DEFAULT_MODEL


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_baselines"
    / "trackA"
    / "stage2_service.py"
)


def _load_service():
    spec = importlib.util.spec_from_file_location("_stage2_service_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_visual_coverage_uses_configured_tracka_root(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "isolated"
    movie = tmp_path / "assets" / "Dataset" / "Movie"
    monkeypatch.setattr(visual_coverage, "_TRACKA_OUTPUT_ROOT", output_root)
    assert visual_coverage._tracka_run_dir(movie, "memstrata__B16") == (
        output_root / "memstrata__B16" / "Dataset" / "Movie"
    )


def test_stage2_preserves_visual_coverage_22_qwen3_contract() -> None:
    assert DEFAULT_MODEL == "qwen3-vl-32b"
    parameters = inspect.signature(visual_coverage._load_selection).parameters
    assert {"video", "ffmpeg"} <= set(parameters)
    assert '"metric_version": "visual-coverage-2.2"' in inspect.getsource(
        visual_coverage.run
    )


def test_stage2_can_inline_video_outside_judge_mount(tmp_path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01video")
    monkeypatch.setenv("VMEM_JUDGE_INLINE_VIDEO", "1")
    part = visual_coverage._vid(clip)
    assert part["type"] == "video_url"
    assert part["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert "file://" not in part["video_url"]["url"]


def test_stage2_discovers_custom_movie_in_isolated_tree(tmp_path, monkeypatch) -> None:
    service = _load_service()
    output_root = tmp_path / "isolated"
    movie = tmp_path / "assets" / "Dataset" / "Movie"
    movie.mkdir(parents=True)
    source_video = tmp_path / "external" / "renamed_movie_720p.mp4"
    source_video.parent.mkdir()
    source_video.write_bytes(b"video")
    selection = (
        output_root
        / "memstrata__B16"
        / "Dataset"
        / "Movie"
        / "visual_selections"
        / "memstrata__B16.json"
    )
    selection.parent.mkdir(parents=True)
    selection.write_text('{"chunks": []}\n', encoding="utf-8")
    monkeypatch.setattr(service, "TRACKA_OUTPUT_ROOT", output_root)
    tasks, skipped = service.discover_tasks(
        systems=["memstrata__B16"],
        movies=["Movie"],
        modes=["name_anchored"],
        movie_dirs={"Movie": movie},
        movie_videos={"Movie": source_video},
    )
    assert not skipped
    assert len(tasks) == 1
    assert tasks[0].video == str(source_video)
    assert Path(tasks[0].out_dir).is_relative_to(output_root)
