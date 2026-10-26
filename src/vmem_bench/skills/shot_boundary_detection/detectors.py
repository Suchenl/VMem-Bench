from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from vmem_bench.common.media import MediaInfo, probe_media
from .schemas import BoundaryReason, BoundarySource, ChunkBoundary, SbdResult
from .post_processing import DinoV3Refiner

logger = logging.getLogger(__name__)
PTS_PATTERN = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def detect_shot_boundaries(
    video_path: Path | str,
    *,
    method: str = "auto",
    scenedetect_threshold: float = 27.0,
    ffmpeg_threshold: float = 0.35,
    min_scene_len_sec: float = 1.0,
    use_dino_refinement: bool = True,
    dino_window_size: int = 5,
) -> SbdResult:
    """Detect interior shot boundaries without loading the whole video into memory."""
    source = Path(video_path).expanduser().resolve()
    media_info = probe_media(source)
    requested = method
    notes: list[str] = []

    candidates: list[tuple[str, Callable[[], list[ChunkBoundary]]]]
    if method == "auto":
        candidates = [
            (
                "transnetv2", 
                lambda: _detect_with_transnetv2(
                    source, 
                    media_info=media_info,
                    min_scene_len_sec=min_scene_len_sec,
                    use_dino_refinement=use_dino_refinement,
                    dino_window_size=dino_window_size,
                )
            ),
            (
                "pyscenedetect",
                lambda: _detect_with_pyscenedetect(
                    source,
                    threshold=scenedetect_threshold,
                    min_scene_len_sec=min_scene_len_sec,
                ),
            ),
            ("ffmpeg_scene", lambda: _detect_with_ffmpeg(source, threshold=ffmpeg_threshold)),
        ]
    elif method == "transnetv2":
        candidates = [
            (
                "transnetv2", 
                lambda: _detect_with_transnetv2(
                    source, 
                    media_info=media_info,
                    min_scene_len_sec=min_scene_len_sec,
                    use_dino_refinement=use_dino_refinement,
                    dino_window_size=dino_window_size,
                )
            )
        ]
    elif method == "pyscenedetect":
        candidates = [
            (
                "pyscenedetect",
                lambda: _detect_with_pyscenedetect(
                    source,
                    threshold=scenedetect_threshold,
                    min_scene_len_sec=min_scene_len_sec,
                ),
            )
        ]
    elif method == "ffmpeg_scene":
        candidates = [("ffmpeg_scene", lambda: _detect_with_ffmpeg(source, threshold=ffmpeg_threshold))]
    else:
        raise ValueError("method must be one of: auto, transnetv2, pyscenedetect, ffmpeg_scene")

    last_error: Exception | None = None
    for name, detector in candidates:
        try:
            boundaries = _dedupe_boundaries(detector(), media_info.duration_sec, min_scene_len_sec)
        except Exception as exc:
            last_error = exc
            notes.append(f"{name} unavailable: {exc}")
            continue
            
        notes.append(f"{name} detected {len(boundaries)} boundaries")
        return SbdResult(
            boundaries=boundaries,
            media_info=media_info,
            requested_method=requested,
            used_method=name,
            notes=notes,
        )

    raise RuntimeError(f"No shot detector succeeded for {source}: {last_error}")


def _detect_with_transnetv2(
    video_path: Path, 
    *, 
    media_info: MediaInfo,
    min_scene_len_sec: float,
    use_dino_refinement: bool = True,
    dino_window_size: int = 5,
) -> list[ChunkBoundary]:
    """TransNetV2 adapter with optional DINOv3 temporal refinement."""
    try:
        from vmem_bench.skills.shot_boundary_detection.TransNetV2 import TransNetV2
    except ImportError as exc:
        raise RuntimeError("TransNetV2 implementation is missing or corrupted") from exc

    model = TransNetV2()
    res = model.predict_video(str(video_path))
    if isinstance(res, tuple):
        _predictions, scenes = res
    else:
        scenes = getattr(res, "scenes", None) or res.get("scenes")

    if scenes is None:
        raise RuntimeError("TransNetV2 output does not contain scenes")

    fps = media_info.fps or 30.0
    total_frames = int(round(media_info.duration_sec * fps))

    refiner = None
    if use_dino_refinement:
        try:
            refiner = DinoV3Refiner()
        except Exception as e:
            logger.warning(f"Failed to initialize DINOv3 refiner: {e}. Skipping refinement.")

    boundaries: list[ChunkBoundary] = []
    for scene in scenes[1:]:
        start_sec = float(scene[0]) if isinstance(scene, (list, tuple)) else float(scene["start"])
        
        if start_sec >= min_scene_len_sec:
            candidate_frame_idx = round(start_sec * fps) + 1
            start_sec = candidate_frame_idx / fps

            if refiner is not None:
                refined_frame_idx = refiner.refine_boundary(
                    video_path,
                    candidate_frame_idx=candidate_frame_idx,
                    fps=fps,
                    total_frames=total_frames,
                    window_size=dino_window_size,
                    mode="cls",
                )
                start_sec = refined_frame_idx / fps

            boundaries.append(
                ChunkBoundary(
                    timestamp_sec=start_sec,
                    reason=BoundaryReason.HARD_CUT,
                    source=BoundarySource.OBSERVED,
                    confidence=1.0,
                )
            )
    return boundaries


def _detect_with_pyscenedetect(
    video_path: Path,
    *,
    threshold: float,
    min_scene_len_sec: float,
) -> list[ChunkBoundary]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if min_scene_len_sec <= 0:
        raise ValueError("min_scene_len_sec must be positive")
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError("PySceneDetect is not installed") from exc

    video = open_video(str(video_path))
    fps = float(video.frame_rate) or 24.0
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=max(1, round(min_scene_len_sec * fps)))
    )
    manager.detect_scenes(video)

    def _seconds(tc: object) -> float:
        return float(tc.seconds if hasattr(tc, "seconds") else tc.get_seconds())

    return [
        ChunkBoundary(
            timestamp_sec=_seconds(start),
            reason=BoundaryReason.HARD_CUT,
            source=BoundarySource.OBSERVED,
        )
        for start, _end in manager.get_scene_list()[1:]
    ]


def _detect_with_ffmpeg(video_path: Path, *, threshold: float) -> list[ChunkBoundary]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffmpeg shot detection failed") from exc

    return [
        ChunkBoundary(
            timestamp_sec=float(match.group(1)),
            reason=BoundaryReason.HARD_CUT,
            source=BoundarySource.OBSERVED,
            confidence=threshold,
        )
        for match in PTS_PATTERN.finditer(completed.stderr)
    ]


def _dedupe_boundaries(
    boundaries: list[ChunkBoundary],
    duration_sec: float,
    min_scene_len_sec: float,
) -> list[ChunkBoundary]:
    clean: list[ChunkBoundary] = []
    last = 0.0
    for boundary in sorted(boundaries, key=lambda item: item.timestamp_sec):
        timestamp = round(boundary.timestamp_sec, 6)
        if timestamp <= 0.0 or timestamp >= duration_sec:
            continue
        if timestamp - last < min_scene_len_sec:
            continue
        clean.append(
            ChunkBoundary(
                timestamp_sec=timestamp,
                reason=boundary.reason,
                source=boundary.source,
                confidence=boundary.confidence,
                camera_relation=boundary.camera_relation,
            )
        )
        last = timestamp
    return clean
