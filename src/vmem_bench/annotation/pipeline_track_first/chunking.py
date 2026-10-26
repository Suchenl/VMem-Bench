"""Chunking (workflow step 2): cut shots first, then concatenate into chunks whose frame
count falls within [min_frames, max_frames] (D2).

Boundaries always fall on shot boundaries; a single shot longer than the cap is split
evenly inside the shot. Undersized fragments are merged into a contiguous neighbor when
the merge stays under the cap (a rare isolated fragment may remain below min_frames when
merging would overflow max_frames). The resulting layout is frozen with the gold
annotation via a layout hash (schemas_and_contracts.md §5.3).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from vmem_bench.common.media import probe_media, slice_by_frames  # noqa: F401 (re-export)
from vmem_bench.common.paths import MovieDirs


@dataclass(slots=True)
class ChunkSpec:
    chunk_id: int
    shot_span: list[int]   # [first_shot, last_shot], inclusive
    frame_span: list[int]  # [first_frame, last_frame], inclusive (both endpoints belong to
                           # this chunk; adjacent chunks never share a frame number)


def aggregate_shots(shots: list[tuple[int, int]], min_frames: int,
                    max_frames: int) -> list[ChunkSpec]:
    """Concatenate half-open ``[start_frame, end_frame)`` shots into chunks within [min, max]
    frames. Internal math stays half-open; the emitted ``ChunkSpec.frame_span`` is **closed
    inclusive** ``[first, last]`` so the published layout shows no boundary overlap."""
    if not (0 < min_frames <= max_frames):
        raise ValueError("need 0 < min_frames <= max_frames")

    raw: list[tuple[int, int, int, int]] = []  # (shot_lo, shot_hi, frame_lo, frame_hi)
    pending: tuple[int, int, int, int] | None = None

    def flush() -> None:
        nonlocal pending
        if pending:
            raw.append(pending)
            pending = None

    for idx, (start, end) in enumerate(shots):
        n = end - start
        if n <= 0:
            continue
        if n > max_frames:
            # Oversized shot: flush pending, split the shot evenly into in-range parts.
            flush()
            parts = math.ceil(n / max_frames)
            size = math.ceil(n / parts)
            raw += [(idx, idx, start + p * size, min(start + (p + 1) * size, end))
                    for p in range(parts) if min(start + (p + 1) * size, end) > start + p * size]
            continue
        if pending is None:
            pending = (idx, idx, start, end)
        elif start == pending[3] and end - pending[2] <= max_frames:
            pending = (pending[0], idx, pending[2], end)
        else:
            flush()
            pending = (idx, idx, start, end)
        if pending and pending[3] - pending[2] >= min_frames:
            flush()
    flush()

    # Merge undersized fragments into a contiguous neighbor while staying under the cap.
    merged: list[tuple[int, int, int, int]] = []
    for item in raw:
        if merged:
            prev = merged[-1]
            contiguous = item[2] == prev[3]
            undersized = (prev[3] - prev[2] < min_frames) or (item[3] - item[2] < min_frames)
            if contiguous and undersized and item[3] - prev[2] <= max_frames:
                merged[-1] = (prev[0], item[1], prev[2], item[3])
                continue
        merged.append(item)

    # Half-open [f0, f1) -> closed inclusive [f0, f1 - 1] for the emitted layout.
    return [ChunkSpec(i, [lo, hi], [f0, f1 - 1]) for i, (lo, hi, f0, f1) in enumerate(merged)]


def layout_hash(chunks: list[ChunkSpec], fps: float) -> str:
    payload = json.dumps({"fps": round(fps, 3),
                          "spans": [c.frame_span for c in chunks]}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shots_from_boundaries(boundary_frames: list[int], total_frames: int) -> list[tuple[int, int]]:
    """Interior boundary frame indices -> [(start, end)) shot spans covering the video."""
    cuts = [0] + sorted({b for b in boundary_frames if 0 < b < total_frames}) + [total_frames]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def run_chunking(video: Path, out_dir: Path, *, min_frames: int, max_frames: int,
                 shots: list[tuple[int, int]] | None = None,
                 sbd_method: str = "transnetv2", min_scene_len_sec: float = 1.0,
                 excluded_segments: list[dict] | None = None) -> dict:
    """Detect shots (unless provided) and aggregate into chunks.

    Only the *chunking logic* output is written (``gold/chunk_index.json`` +
    ``gold/shot_boundaries.csv``; legacy readers still accept ``layout/chunk_index.json`` +
    ``layout/boundaries.csv``). Chunk video clips are **not** materialized here — they are
    derived on demand from the source video (``materialize_clip``), so the benchmark ships
    the layout + annotations + download recipe, never the video bytes.
    """
    video = Path(video).resolve()
    out_dir = Path(out_dir)
    dirs = MovieDirs(out_dir, write=True)
    dirs.mkdirs()
    info = probe_media(video)
    fps = info.fps or 24.0
    total_frames = int(round(info.duration_sec * fps))

    if shots is None:
        from vmem_bench.skills.shot_boundary_detection import detect_shot_boundaries
        result = detect_shot_boundaries(video, method=sbd_method,
                                        min_scene_len_sec=min_scene_len_sec)
        boundary_frames = [int(round(b.timestamp_sec * fps)) for b in result.boundaries]
        shots = shots_from_boundaries(boundary_frames, total_frames)
    # Written closed inclusive [start_frame, last_frame] to match frame_span (no overlap).
    with dirs.shot_boundaries.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot_idx", "start_frame", "last_frame"])
        writer.writerows((i, s, e - 1) for i, (s, e) in enumerate(shots))

    chunks = aggregate_shots(shots, min_frames, max_frames)
    index = {
        "schema_version": "2.0.0",
        "source_video": str(video),
        "fps": fps,
        "total_frames": total_frames,
        "min_frames_per_chunk": min_frames,
        "max_frames_per_chunk": max_frames,
        "layout_hash": layout_hash(chunks, fps),
        # Non-diegetic spans (credits/titles) removed BEFORE chunking; recorded so evaluation can
        # prove the SUT never received these frames. Empty when nothing was excluded.
        "excluded_segments": list(excluded_segments or []),
        "chunks": [{"chunk_id": c.chunk_id, "shot_span": c.shot_span,
                    "frame_span": c.frame_span} for c in chunks],
    }
    dirs.chunk_index.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def materialize_clip(video: Path, out_path: Path, *, frame_span: list[int], fps: float) -> Path:
    """Derive one chunk's video clip on demand (eval-time reference); not shipped with the bench.

    ``frame_span`` is closed inclusive ``[first, last]``; slicing needs the half-open end.
    """
    return slice_by_frames(video, out_path, start_frame=frame_span[0],
                           end_frame=frame_span[1] + 1, fps=fps)
