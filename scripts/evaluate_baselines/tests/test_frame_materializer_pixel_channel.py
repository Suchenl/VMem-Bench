from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CAUSAL_DIR = Path(__file__).resolve().parents[1] / "trackA" / "baseline_adapters" / "causal"
if str(CAUSAL_DIR) not in sys.path:
    sys.path.insert(0, str(CAUSAL_DIR))

import frame_materializer
from contract import MovieContext, RetrievedItem, RetrievedMemory


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _layout(tmp_path: Path) -> tuple[MovieContext, Path, Path, Path]:
    run_dir = tmp_path / "run"
    work_dir = run_dir / "_adapter_work" / "memstrata"
    frames_dir = run_dir / "_ref_frames" / "memstrata"
    out_dir = run_dir / "visual_selections"
    work_dir.mkdir(parents=True)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"not a real video")
    movie = MovieContext(
        movie_id="movie",
        source_video=str(source_video),
        fps=16.0,
        seconds_span_by_chunk={0: (0.0, 1.0), 1: (1.0, 2.0)},
        work_dir=str(work_dir),
    )
    return movie, work_dir, frames_dir, out_dir


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "memstrata.json").read_text(encoding="utf-8"))


def test_direct_image_path_is_used_without_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    movie, work_dir, frames_dir, out_dir = _layout(tmp_path)
    crop = work_dir / "segment_00000" / "crop.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(PNG_BYTES)

    def fail_cut(*args, **kwargs) -> bool:
        raise AssertionError("_cut_frame should not be called for a valid SUT image")

    monkeypatch.setattr(frame_materializer, "_cut_frame", fail_cut)

    result = frame_materializer.materialize_system(
        system="memstrata",
        movie=movie,
        records=[
            RetrievedMemory(
                chunk_id=1,
                items=[
                    RetrievedItem(
                        evidence_kind="reference_image",
                        source_seconds=0.5,
                        image_path=str(crop.resolve()),
                    )
                ],
            )
        ],
        out_dir=out_dir,
        frames_dir=frames_dir,
        ffmpeg="ffmpeg",
    )

    manifest = _manifest(out_dir)
    copied = Path(manifest["chunks"][0]["selected"][0]["representations"][0]["crop_abspath"])
    assert result["reference_frames"] == 1
    assert result["pixel_direct"] == 1
    assert manifest["extras"]["n_pixel_direct"] == 1
    assert manifest["extras"]["n_provenance_failed"] == 0
    assert copied == (frames_dir / "c00001_r00.png").resolve()
    assert copied.read_bytes() == PNG_BYTES


def test_missing_image_path_falls_back_to_cut_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    movie, _work_dir, frames_dir, out_dir = _layout(tmp_path)
    calls: list[tuple[Path, float]] = []

    def fake_cut(ffmpeg: str, src: Path, out: Path, seconds: float, *, movie: MovieContext, run_dir: Path) -> bool:
        calls.append((out, seconds))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(PNG_BYTES)
        return True

    monkeypatch.setattr(frame_materializer, "_cut_frame", fake_cut)

    result = frame_materializer.materialize_system(
        system="memstrata",
        movie=movie,
        records=[
            RetrievedMemory(
                chunk_id=1,
                items=[RetrievedItem(evidence_kind="latent", source_seconds=0.25)],
            )
        ],
        out_dir=out_dir,
        frames_dir=frames_dir,
        ffmpeg="ffmpeg",
    )

    manifest = _manifest(out_dir)
    assert len(calls) == 1
    assert calls[0][1] == 0.25
    assert result["reference_frames"] == 1
    assert manifest["extras"]["n_pixel_direct"] == 0
    assert manifest["extras"]["n_cut_failed"] == 0


def test_gold_segment_image_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    movie, _work_dir, frames_dir, out_dir = _layout(tmp_path)
    gold_crop = frames_dir.parents[1] / "gold" / "crop.png"
    gold_crop.parent.mkdir(parents=True)
    gold_crop.write_bytes(PNG_BYTES)

    def fail_cut(*args, **kwargs) -> bool:
        raise AssertionError("rejected SUT images must not fall back to source-frame cutting")

    monkeypatch.setattr(frame_materializer, "_cut_frame", fail_cut)

    frame_materializer.materialize_system(
        system="memstrata",
        movie=movie,
        records=[
            RetrievedMemory(
                chunk_id=1,
                items=[
                    RetrievedItem(
                        evidence_kind="reference_image",
                        source_seconds=0.5,
                        image_path=str(gold_crop.resolve()),
                    )
                ],
            )
        ],
        out_dir=out_dir,
        frames_dir=frames_dir,
        ffmpeg="ffmpeg",
    )

    manifest = _manifest(out_dir)
    assert manifest["chunks"][0]["selected"] == []
    assert manifest["extras"]["n_provenance_failed"] == 1
    assert manifest["extras"]["n_pixel_direct"] == 0


def test_direct_image_path_still_obeys_causal_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    movie, work_dir, frames_dir, out_dir = _layout(tmp_path)
    crop = work_dir / "segment_00001" / "crop.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(PNG_BYTES)

    def fail_cut(*args, **kwargs) -> bool:
        raise AssertionError("future SUT images must be dropped before materialization")

    monkeypatch.setattr(frame_materializer, "_cut_frame", fail_cut)

    frame_materializer.materialize_system(
        system="memstrata",
        movie=movie,
        records=[
            RetrievedMemory(
                chunk_id=1,
                items=[
                    RetrievedItem(
                        evidence_kind="reference_image",
                        source_seconds=1.0,
                        image_path=str(crop.resolve()),
                    )
                ],
            )
        ],
        out_dir=out_dir,
        frames_dir=frames_dir,
        ffmpeg="ffmpeg",
    )

    manifest = _manifest(out_dir)
    assert manifest["chunks"][0]["selected"] == []
    assert manifest["extras"]["n_future_dropped"] == 1
    assert manifest["extras"]["n_provenance_failed"] == 0
