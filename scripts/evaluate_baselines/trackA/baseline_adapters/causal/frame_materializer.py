"""Render a SUT's retrieved memory into real frames + a scorer-ready manifest.

Given, per segment, the :class:`RetrievedMemory` the SUT produced (native memory
items carrying a temporal identity), this either uses a guarded image-native SUT
reference directly or cuts the corresponding frame out of the **real source
video** at the item's absolute source seconds. It writes the
``visual_selections/<system>.json`` manifest consumed by
``vmem_bench.scoring.visual_coverage``.

This is the method-neutral "map retrieved memory to a frame via temporal
consistency" step: it does not inject gold and does not label frames; it only
renders the SUT's own retrieval decision into pixels of the very video the SUT
observed.

Causal guard: an item whose source time is not strictly earlier than the current
segment's start is dropped (a causal SUT can only retrieve the past). Violations
are counted in the manifest ``extras`` for auditing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from contract import MovieContext, RetrievedItem, RetrievedMemory
from _video_io import closest_cached_frame

_BENCH_ROOT = Path(__file__).resolve().parents[5]


def _ffmpeg_threads() -> str:
    return os.environ.get("VMEM_FFMPEG_THREADS", "1")


def _link_or_copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(src, dst)
        return True
    except OSError:
        try:
            shutil.copy2(src, dst)
            return True
        except OSError:
            return False


def _has_gold_segment(path: Path) -> bool:
    return any(part.lower() == "gold" for part in path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validated_sut_image_path(image_path: str, *, run_workspace: Path) -> Path | None:
    """Cheap provenance guard for SUT-provided reference pixels."""
    if not image_path:
        return None
    raw = Path(image_path).expanduser()
    if not raw.is_absolute():
        return None
    if _has_gold_segment(raw):
        return None
    try:
        resolved = raw.resolve(strict=True)
        workspace = run_workspace.resolve(strict=False)
    except OSError:
        return None
    if not resolved.is_file():
        return None
    if _has_gold_segment(resolved):
        return None
    if not _is_relative_to(resolved, workspace):
        return None
    return resolved


def _shared_segment_frame(ffmpeg: str, movie: MovieContext, run_dir: Path, seconds: float) -> Path | None:
    dataset = run_dir.parent.name
    sample = run_dir.name
    for cid, span in movie.seconds_span_by_chunk.items():
        s0, s1 = float(span[0]), float(span[1])
        if s0 - 1e-6 <= float(seconds) <= s1 + 1e-6:
            seg = _BENCH_ROOT / "outputs/evaluation/trackA/_shared_segments" / dataset / sample / f"chunk_{int(cid):05d}.mp4"
            if not seg.is_file():
                return None
            return closest_cached_frame(
                ffmpeg=ffmpeg,
                segment_video=str(seg),
                local_seconds=max(0.0, float(seconds) - s0),
                fps=float(movie.fps),
            )
    return None


def _resolve_seconds(
    item: RetrievedItem,
    movie: MovieContext,
) -> float | None:
    """Absolute source seconds for an item: explicit, else its chunk's mid-span."""
    if item.source_seconds is not None:
        return float(item.source_seconds)
    if item.source_chunk_id is not None:
        span = movie.seconds_span_by_chunk.get(int(item.source_chunk_id))
        if span is not None:
            t0, t1 = float(span[0]), float(span[1])
            return t0 + max(0.0, (t1 - t0)) / 2.0
    return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _cut_frame(ffmpeg: str, src: Path, out: Path, seconds: float, *, movie: MovieContext, run_dir: Path) -> bool:
    """Extract a single frame at ``seconds``. Returns True on success."""
    if out.is_file() and out.stat().st_size > 0:
        return True
    shared = _shared_segment_frame(ffmpeg, movie, run_dir, seconds)
    if shared is not None and shared.is_file() and _link_or_copy(shared, out):
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg, "-y",
            "-ss", f"{max(0.0, float(seconds)):.3f}",
            "-i", str(src),
            "-threads", _ffmpeg_threads(),
            "-frames:v", "1",
            # Reference frames at the SUT's native 832x480 (see _video_io.WAN_*);
            # the scorer downscales further to 384px at send time.
            "-vf", "scale=832:480",
            "-q:v", "2",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0 and out.is_file() and out.stat().st_size > 0


def _materialize_record(
    *,
    system: str,
    movie: MovieContext,
    rec: RetrievedMemory,
    frames_dir: Path,
    ffmpeg: str,
    prompts: dict[int, str],
    src: Path,
    run_dir: Path,
) -> tuple[dict, dict[str, int]]:
    cid = int(rec.chunk_id)
    cur_span = movie.seconds_span_by_chunk.get(cid)
    cur_t0 = float(cur_span[0]) if cur_span else None
    selected: list[dict] = []
    stats = {
        "n_reference_frames": 0,
        "n_future_dropped": 0,
        "n_cut_failed": 0,
        "n_provenance_failed": 0,
        "n_pixel_direct": 0,
    }
    for k, item in enumerate(rec.items):
        frame_path = frames_dir / f"c{cid:05d}_r{k:02d}.png"

        if item.image_path:
            # SUT pixels still need an explicit timestamp so the causal guard
            # remains identical to timestamp-only systems.
            if item.source_seconds is None:
                stats["n_future_dropped"] += 1
                continue
            seconds = float(item.source_seconds)
            if cur_t0 is not None and seconds >= cur_t0:
                stats["n_future_dropped"] += 1
                continue
            sut_image = _validated_sut_image_path(item.image_path, run_workspace=run_dir)
            if sut_image is None:
                stats["n_provenance_failed"] += 1
                continue
            if not _link_or_copy(sut_image, frame_path):
                stats["n_cut_failed"] += 1
                continue
            stats["n_pixel_direct"] += 1
        else:
            seconds = _resolve_seconds(item, movie)
            if seconds is None:
                continue
            # Causal guard: retrieved evidence must be strictly in the past.
            if cur_t0 is not None and seconds >= cur_t0:
                stats["n_future_dropped"] += 1
                continue
            if not _cut_frame(ffmpeg, src, frame_path, seconds, movie=movie, run_dir=run_dir):
                stats["n_cut_failed"] += 1
                continue
        stats["n_reference_frames"] += 1
        selected.append(
            {
                # Baselines have no entity concept; use a synthetic, unlabeled
                # reference id. The VLM scorer treats reference images as
                # UNLABELED, so this id is for bookkeeping only.
                "asset_id": f"{system}:mem@c{cid:05d}_r{k:02d}",
                "function": "reference",
                "representations": [
                    {
                        "crop_abspath": str(frame_path.resolve()),
                        "source_seconds": round(float(seconds), 3),
                        "source_chunk_id": item.source_chunk_id,
                        "evidence_kind": item.evidence_kind,
                        "score": item.score,
                    }
                ],
            }
        )
    # Carry per-segment retrieval timing (compose = retrieval latency,
    # observe = memory-write latency) recorded by the runner. These are
    # time-efficiency metrics; the scorer passes them through and aggregates.
    timing = {
        "compose_ms": rec.extras.get("compose_ms"),
        "observe_ms": rec.extras.get("observe_ms"),
    }
    chunk = {
        "chunk_id": cid,
        "prompt": prompts.get(cid, ""),
        "retrieval_timing": timing,
        "selected": selected,
    }
    return chunk, stats


def materialize_record_checkpoint(
    *,
    system: str,
    movie: MovieContext,
    rec: RetrievedMemory,
    out_dir: Path,
    frames_dir: Path,
    ffmpeg: str,
    prompts: dict[int, str] | None = None,
    expected_chunks: int | None = None,
) -> dict:
    """Incrementally update ``visual_selections/<system>.json`` after one segment.

    The final full-movie ``materialize_system`` pass still rewrites an exact manifest.
    This checkpoint is meant to preserve completed segments if a long runner dies
    near the end.
    """
    prompts = prompts or {}
    src = Path(movie.source_video)
    run_dir = frames_dir.parents[1]
    out_path = out_dir / f"{system}.json"
    chunk, stats = _materialize_record(
        system=system,
        movie=movie,
        rec=rec,
        frames_dir=frames_dir,
        ffmpeg=ffmpeg,
        prompts=prompts,
        src=src,
        run_dir=run_dir,
    )

    previous: dict = {}
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    chunks_by_id = {
        int(row.get("chunk_id")): row
        for row in previous.get("chunks", [])
        if isinstance(row, dict) and row.get("chunk_id") is not None
    }
    chunks_by_id[int(chunk["chunk_id"])] = chunk
    chunks_out = [chunks_by_id[cid] for cid in sorted(chunks_by_id)]

    prev_extras = previous.get("extras", {}) if isinstance(previous.get("extras"), dict) else {}
    extras = {
        "n_reference_frames": int(prev_extras.get("n_reference_frames") or 0) + stats["n_reference_frames"],
        "n_future_dropped": int(prev_extras.get("n_future_dropped") or 0) + stats["n_future_dropped"],
        "n_cut_failed": int(prev_extras.get("n_cut_failed") or 0) + stats["n_cut_failed"],
        "n_provenance_failed": int(prev_extras.get("n_provenance_failed") or 0) + stats["n_provenance_failed"],
        "n_pixel_direct": int(prev_extras.get("n_pixel_direct") or 0) + stats["n_pixel_direct"],
        "source_video": str(src),
        "checkpoint": True,
        "complete": expected_chunks is not None and len(chunks_out) >= int(expected_chunks),
        "expected_chunks": expected_chunks,
    }
    manifest = {
        "movie": movie.movie_id,
        "system": system,
        "chunks": chunks_out,
        "extras": extras,
    }
    _atomic_write_json(out_path, manifest)
    return {"system": system, "out": str(out_path), "chunks": len(chunks_out), **stats}


def materialize_system(
    *,
    system: str,
    movie: MovieContext,
    records: Iterable[RetrievedMemory],
    out_dir: Path,
    frames_dir: Path,
    ffmpeg: str,
    prompts: dict[int, str] | None = None,
) -> dict:
    """Write ``visual_selections/<system>.json`` from retrieved-memory records.

    ``out_dir`` is the run-local ``visual_selections`` directory under
    ``outputs/evaluation/trackA/<system>/<dataset>/<movie>/``; ``frames_dir`` is
    where the cut reference frames live (kept per-system to avoid cross-contamination).
    """
    prompts = prompts or {}
    src = Path(movie.source_video)
    run_dir = frames_dir.parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_out: list[dict] = []
    n_items = 0
    n_future_dropped = 0
    n_cut_failed = 0
    n_provenance_failed = 0
    n_pixel_direct = 0

    for rec in sorted(records, key=lambda r: r.chunk_id):
        chunk, stats = _materialize_record(
            system=system, movie=movie, rec=rec, frames_dir=frames_dir,
            ffmpeg=ffmpeg, prompts=prompts, src=src, run_dir=run_dir,
        )
        chunks_out.append(chunk)
        n_items += stats["n_reference_frames"]
        n_future_dropped += stats["n_future_dropped"]
        n_cut_failed += stats["n_cut_failed"]
        n_provenance_failed += stats["n_provenance_failed"]
        n_pixel_direct += stats["n_pixel_direct"]

    manifest = {
        "movie": movie.movie_id,
        "system": system,
        "chunks": chunks_out,
        "extras": {
            "n_reference_frames": n_items,
            "n_future_dropped": n_future_dropped,
            "n_cut_failed": n_cut_failed,
            "n_provenance_failed": n_provenance_failed,
            "n_pixel_direct": n_pixel_direct,
            "source_video": str(src),
        },
    }
    out_path = out_dir / f"{system}.json"
    _atomic_write_json(out_path, manifest)
    return {
        "system": system,
        "out": str(out_path),
        "chunks": len(chunks_out),
        "reference_frames": n_items,
        "future_dropped": n_future_dropped,
        "cut_failed": n_cut_failed,
        "provenance_failed": n_provenance_failed,
        "pixel_direct": n_pixel_direct,
    }
