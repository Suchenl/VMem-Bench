"""Media probing and frame/segment extraction (self-contained, ffmpeg/ffprobe based)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _resolve_media_tool(name: str, env_key: str) -> str:
    """Resolve ffmpeg/ffprobe from env, PATH, or bins near the active interpreter."""
    override = os.environ.get(env_key)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled:
                return bundled
        except Exception:
            pass
    here = Path(sys.executable).resolve().parent
    search_roots = [here, *list(here.parents)[:4]]
    for root in search_roots:
        for candidate in (root / name, root / "bin" / name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return name


def ffmpeg_bin() -> str:
    """Resolve ffmpeg via FFMPEG_BIN, PATH, or bins near sys.executable."""
    return _resolve_media_tool("ffmpeg", "FFMPEG_BIN")


def ffprobe_bin() -> str:
    return _resolve_media_tool("ffprobe", "FFPROBE_BIN")



@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_sec: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    format_name: str | None


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def probe_media(path: Path | str) -> MediaInfo:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file does not exist: {source}")
    command = [ffprobe_bin(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
    except FileNotFoundError:
        # imageio-ffmpeg bundles ffmpeg but not ffprobe; use its reader metadata when a
        # system ffprobe is unavailable so the lightweight annotation smoke remains portable.
        try:
            import imageio_ffmpeg

            reader = imageio_ffmpeg.read_frames(str(source))
            try:
                meta = next(reader)
            finally:
                reader.close()
        except Exception as exc:
            raise RuntimeError(
                "Cannot probe media: install ffprobe or set FFPROBE_BIN"
            ) from exc
        width, height = (meta.get("source_size") or meta.get("size") or (None, None))[:2]
        return MediaInfo(
            duration_sec=float(meta["duration"]),
            width=int(width) if width else None,
            height=int(height) if height else None,
            fps=float(meta["fps"]) if meta.get("fps") else None,
            has_audio=False,
            format_name=None,
        )
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found: {source}")
    duration = payload.get("format", {}).get("duration") or video.get("duration")
    if duration is None:
        raise RuntimeError(f"Cannot determine media duration: {source}")
    return MediaInfo(
        duration_sec=float(duration),
        width=video.get("width"),
        height=video.get("height"),
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
        format_name=payload.get("format", {}).get("format_name"),
    )


def slice_by_frames(video: Path | str, out_path: Path | str, *, start_frame: int,
                    end_frame: int, fps: float) -> Path:
    """Cut ``[start_frame, end_frame)`` into ``out_path`` (re-encode, frame-accurate)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_frame / fps
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        raise ValueError(f"empty frame span [{start_frame}, {end_frame})")
    cmd = [ffmpeg_bin(), "-y", "-ss", f"{start_sec:.6f}", "-i", str(video),
           "-frames:v", str(n_frames), "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-an", str(out_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg slice failed for {out_path.name}: {completed.stderr[-500:]}")
    return out_path


def extract_frame(video: Path | str, out_path: Path | str, *, frame_index: int, fps: float) -> Path:
    """Extract a single frame (by absolute frame index) as an image."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-y", "-ss", f"{frame_index / fps:.6f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg frame extraction failed at frame {frame_index}: {completed.stderr[-500:]}")
    return out_path


def sample_frame_indices(start_frame: int, end_frame: int, *, max_samples: int) -> list[int]:
    """Evenly sample up to ``max_samples`` absolute frame indices from ``[start, end)``."""
    n = end_frame - start_frame
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(start_frame, end_frame))
    step = n / max_samples
    return sorted({start_frame + int(i * step + step / 2) for i in range(max_samples)})
