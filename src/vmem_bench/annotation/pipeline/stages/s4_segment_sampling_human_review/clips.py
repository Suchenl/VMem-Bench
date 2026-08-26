"""On-demand segment clips for S4 human review."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vmem_bench.common.media import ffmpeg_bin


def _safe_segment_id(segment_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in segment_id)


def ensure_segment_clip(
    *,
    source_video: Path,
    cache_dir: Path,
    segment_id: str,
    start_seconds: float,
    end_seconds: float,
    max_duration: float = 12.0,
) -> Path:
    """Cut a short review clip once and reuse it from ``cache_dir``."""
    if end_seconds <= start_seconds:
        raise ValueError(f"invalid segment range [{start_seconds}, {end_seconds}]")
    source_video = Path(source_video)
    if not source_video.is_file():
        raise FileNotFoundError(f"source video missing: {source_video}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_segment_id(segment_id)
    out = cache_dir / f"{safe_id}.mp4"
    if out.is_file() and out.stat().st_size > 0:
        return out
    duration = min(float(end_seconds - start_seconds), max_duration)
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale='min(720,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(out),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg clip failed: {(completed.stderr or '')[-500:]}")
    return out


def ensure_segment_poster(
    *,
    source_video: Path,
    cache_dir: Path,
    segment_id: str,
    start_seconds: float,
) -> Path:
    """Extract one JPEG cover frame at segment start for S4 review cards."""
    source_video = Path(source_video)
    if not source_video.is_file():
        raise FileNotFoundError(f"source video missing: {source_video}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_segment_id(segment_id)
    out = cache_dir / f"{safe_id}.jpg"
    if out.is_file() and out.stat().st_size > 0:
        return out
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-ss",
        f"{float(start_seconds):.3f}",
        "-i",
        str(source_video),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(720,iw)':-2",
        "-q:v",
        "3",
        str(out),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not out.is_file() or out.stat().st_size <= 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg poster failed: {(completed.stderr or '')[-500:]}")
    return out
