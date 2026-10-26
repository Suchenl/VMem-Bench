"""CI guard for the bring-your-own-method template.

Runs the Track A example driver on bundled gold (CPU, placeholder frames) and
validates the emitted ``visual_selections/<system>.json`` through the REAL scorer
reader ``vmem_bench.scoring.visual_coverage._load_selection`` — so the documented
output contract cannot silently drift from what the scorer ingests.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TEMPLATE = _REPO / "scripts" / "evaluate_baselines" / "your_method"
_MOVIE = _REPO / "assets" / "trackA" / "BlenderOpenMovies" / "charge"


def _load_driver():
    if str(_TEMPLATE) not in sys.path:
        sys.path.insert(0, str(_TEMPLATE))
    spec = importlib.util.spec_from_file_location(
        "your_method_run_tracka_example", _TEMPLATE / "run_tracka_example.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _MOVIE.exists(), reason="bundled Track A gold not present")
def test_template_emits_scorer_ingestible_selection(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    driver = _load_driver()

    movie = tmp_path / "charge"
    (movie / "gold").mkdir(parents=True)
    for name in ("chunk_annotations.json", "entity_registry.json", "chunk_index.json"):
        shutil.copy(_MOVIE / "gold" / name, movie / "gold" / name)

    system = "your_method-recency"
    out_path = movie / "benchmark_run" / "visual_selections" / f"{system}.json"
    driver.run(movie_dir=movie, system=system, out_path=out_path,
               mem_dir=tmp_path / "_mem", budget=3, limit=3, ffmpeg="ffmpeg")
    assert out_path.is_file()

    # Validate with the actual scorer reader (schema cannot drift from the scorer).
    from vmem_bench.scoring.visual_coverage import _load_selection

    selections, _timing = _load_selection(movie, system)
    assert set(selections) == {0, 1, 2}
    assert selections[0] == []  # first segment: nothing observed yet -> honest empty
    recalled = [p for cid in (1, 2) for p in selections[cid]]
    assert recalled, "later segments should recall stored frames"
    for p in recalled:
        assert Path(p).is_file(), f"selection points at a missing file: {p}"


def test_timestamp_reference_is_materialized_for_scoring(tmp_path: Path) -> None:
    from vmem_bench.scoring.visual_coverage import _load_selection

    movie = tmp_path / "movie"
    selection_dir = movie / "benchmark_run" / "visual_selections"
    selection_dir.mkdir(parents=True)
    system = "timestamp-method"
    (selection_dir / f"{system}.json").write_text(
        json.dumps({
            "system": system,
            "chunks": [{"chunk_id": 0, "selected": [
                {"representations": [{"source_seconds": 1.25}]}
            ]}],
        }),
        encoding="utf-8",
    )
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"placeholder")
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\nout=\nfor arg do out=$arg; done\nprintf 'frame' > \"$out\"\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)

    selections, _ = _load_selection(
        movie, system, video=source_video, ffmpeg=str(fake_ffmpeg)
    )
    assert selections[0] and Path(selections[0][0]).is_file()
