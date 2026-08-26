"""Method-side video IO shared by causal adapters.

This is a pure decode helper (ffmpeg -> frames -> tensor), NOT perception or
memory: every baseline still runs its own VAE / encoder / memory on the frames it
gets here. All three vendored causal baselines share the Wan2.1-T2V-1.3B backbone
(480x832 pixels, VAE temporal stride 4, native 16 fps), so the preprocessing is
identical; the distinctiveness lives entirely in each adapter's memory/retrieval.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Wan2.1-T2V-1.3B backbone constants (shared by LongLive-RAG / MemFlow / IAMFlow).
WAN_H = 480
WAN_W = 832
WAN_FPS = 16.0
VAE_TEMPORAL_STRIDE = 4  # pixel frames per latent frame after the first
_BENCH_ROOT = Path(__file__).resolve().parents[5]
_CACHE_VERSION = "wan_pixels_v1"


def _ffmpeg_threads() -> str:
    return os.environ.get("MAVE_FFMPEG_THREADS", "1")


def _decode_cache_root() -> Path:
    override = os.environ.get("MAVE_SHARED_DECODE_CACHE_DIR")
    if override:
        return Path(override)
    return _BENCH_ROOT / "outputs/evaluation/trackA/_shared_decoded_frames"


def _cache_profile(height: int, width: int, fps: float) -> str:
    fps_s = f"{float(fps):.6g}".replace(".", "p")
    return f"h{height}_w{width}_fps{fps_s}"


def _cache_key(segment_video: str, *, height: int, width: int, fps: float) -> tuple[str, dict]:
    src = Path(segment_video).resolve()
    stat = src.stat()
    meta = {
        "version": _CACHE_VERSION,
        "source": str(src),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
    }
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    return key, meta


def _manifest_ok(cache_dir: Path, expected: dict) -> list[Path] | None:
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key, value in expected.items():
        if data.get(key) != value:
            return None
    frame_count = int(data.get("frame_count") or 0)
    if frame_count <= 0:
        return None
    frames = sorted(cache_dir.glob("f_*.png"))
    if len(frames) != frame_count:
        return None
    return frames


def _decode_to_dir(
    *,
    ffmpeg: str,
    segment_video: str,
    out_dir: Path,
    height: int,
    width: int,
    fps: float,
    manifest_meta: dict | None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "f_%05d.png"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(segment_video),
         "-threads", _ffmpeg_threads(),
         "-vf", f"fps={fps},scale={width}:{height}",
         "-pix_fmt", "rgb24", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames = sorted(out_dir.glob("f_*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {segment_video}")
    if manifest_meta is not None:
        manifest = dict(manifest_meta)
        manifest["frame_count"] = len(frames)
        tmp = out_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(out_dir / "manifest.json")
    return frames


def _cached_frame_paths(
    *,
    ffmpeg: str,
    segment_video: str,
    height: int,
    width: int,
    fps: float,
) -> list[Path] | None:
    if os.environ.get("MAVE_DISABLE_SHARED_DECODE_CACHE", "0") in {"1", "true", "yes"}:
        return None

    key, meta = _cache_key(segment_video, height=height, width=width, fps=fps)
    profile_dir = _decode_cache_root() / _cache_profile(height, width, fps)
    cache_dir = profile_dir / key
    lock_path = profile_dir / f"{key}.lock"
    profile_dir.mkdir(parents=True, exist_ok=True)

    frames = _manifest_ok(cache_dir, meta)
    if frames is not None:
        return frames

    wait_sec = float(os.environ.get("MAVE_DECODE_CACHE_WAIT_SEC", "600"))
    stale_sec = float(os.environ.get("MAVE_DECODE_CACHE_STALE_SEC", "1800"))
    deadline = time.time() + wait_sec
    got_lock = False

    while time.time() < deadline:
        frames = _manifest_ok(cache_dir, meta)
        if frames is not None:
            return frames
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()} time={time.time()} source={segment_video}\n")
            got_lock = True
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_sec:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.5)

    if not got_lock:
        return None

    tmp_dir = profile_dir / f"{key}.tmp.{os.getpid()}"
    try:
        if cache_dir.exists():
            # Own cache directory for this exact key; safe to clear only under lock.
            shutil.rmtree(cache_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        _decode_to_dir(
            ffmpeg=ffmpeg,
            segment_video=segment_video,
            out_dir=tmp_dir,
            height=height,
            width=width,
            fps=fps,
            manifest_meta=meta,
        )
        tmp_dir.rename(cache_dir)
        return _manifest_ok(cache_dir, meta)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def closest_cached_frame(
    *,
    ffmpeg: str,
    segment_video: str,
    local_seconds: float,
    height: int = WAN_H,
    width: int = WAN_W,
    fps: float = WAN_FPS,
) -> Path | None:
    """Return the nearest shared decoded frame for ``local_seconds`` in a segment.

    This lets materializers reuse ``_shared_decoded_frames`` instead of issuing a
    fresh ``ffmpeg -ss ... -frames:v 1`` for every retrieved reference frame.
    """
    frames = _cached_frame_paths(
        ffmpeg=ffmpeg,
        segment_video=segment_video,
        height=height,
        width=width,
        fps=fps,
    )
    if not frames:
        return None
    idx = int(round(max(0.0, float(local_seconds)) * float(fps)))
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]


def read_segment_pixels(
    segment_video: str,
    *,
    ffmpeg: str,
    height: int = WAN_H,
    width: int = WAN_W,
    fps: float = WAN_FPS,
):
    """Decode a clip into a Wan-ready pixel tensor ``[1, 3, T, H, W]`` in [-1, 1].

    Returns ``(pixel, n_frames)``. Requires torch + numpy + PIL (present in every
    baseline env) and ffmpeg (bench dependency). Decoded PNGs are shared through a
    lock-protected cache so adapters do not repeatedly FFmpeg-decode the same clip.
    """
    import numpy as np
    import torch
    from PIL import Image

    frames = _cached_frame_paths(
        ffmpeg=ffmpeg,
        segment_video=segment_video,
        height=height,
        width=width,
        fps=fps,
    )
    if frames is None:
        with tempfile.TemporaryDirectory() as td:
            frames = _decode_to_dir(
                ffmpeg=ffmpeg,
                segment_video=segment_video,
                out_dir=Path(td),
                height=height,
                width=width,
                fps=fps,
                manifest_meta=None,
            )
            arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frames])
    else:
        arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frames])
    t = arr.shape[0]
    ten = torch.from_numpy(arr).float() / 255.0          # [T,H,W,3] in [0,1]
    ten = ten.permute(3, 0, 1, 2).contiguous()           # [3,T,H,W]
    ten = ten * 2.0 - 1.0                                 # [-1,1]
    return ten.unsqueeze(0), t                            # [1,3,T,H,W]


def latent_local_seconds(latent_index: int, fps: float = WAN_FPS) -> float:
    """Local seconds (within a clip) of a Wan latent frame.

    Latent 0 covers pixel frame 0; latent L>=1 covers pixels [1+4(L-1), 1+4L).
    We take the first pixel of the latent's window as its timestamp.
    """
    if latent_index <= 0:
        pixel_start = 0
    else:
        pixel_start = 1 + VAE_TEMPORAL_STRIDE * (latent_index - 1)
    return float(pixel_start) / float(fps)


def n_latents_for_pixels(n_pixel_frames: int) -> int:
    """Wan VAE latent count for a pixel clip of length ``n_pixel_frames``."""
    if n_pixel_frames <= 0:
        return 0
    return 1 + (n_pixel_frames - 1) // VAE_TEMPORAL_STRIDE
