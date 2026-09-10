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


def test_stage2_defaults_to_v3_and_preserves_explicit_v22() -> None:
    assert DEFAULT_MODEL == "qwen3-vl-32b"
    parameters = inspect.signature(visual_coverage._load_selection).parameters
    assert {"video", "ffmpeg"} <= set(parameters)
    run_parameters = inspect.signature(visual_coverage.run).parameters
    assert run_parameters["metric_version"].default == "visual-coverage-3.0"
    score_parameters = inspect.signature(
        visual_coverage.score_segment
    ).parameters
    assert score_parameters["metric_version"].default == "visual-coverage-2.2"
    assert visual_coverage.PER_REF_METRIC_VERSION == "visual-coverage-3.0"
    assert set(visual_coverage.SUPPORTED_METRIC_VERSIONS) == {
        "visual-coverage-2.2",
        "visual-coverage-3.0",
    }


def test_stage2_service_accepts_v3_execution_controls(tmp_path) -> None:
    service = _load_service()
    args = service.parse_args(
        [
            "--metric-version",
            "visual-coverage-3.0",
            "--ref-workers",
            "3",
            "--judge-cache",
            str(tmp_path / "cache"),
        ]
    )
    assert args.metric_version == "visual-coverage-3.0"
    assert args.ref_workers == 3
    assert args.judge_cache == tmp_path / "cache"


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
