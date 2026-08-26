"""Deterministic candidate-frame extraction for one segment crop task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vmem_bench.common.media import extract_frame, probe_media


@dataclass(slots=True)
class FrameCandidate:
    frame_index: int
    seconds: float
    path: str
    sharpness: float
    luminance_std: float

    def to_dict(self) -> dict:
        return asdict(self)


def _quality(path: Path) -> tuple[float, float]:
    """Return cheap sharpness and brightness variation features."""
    import numpy as np
    from PIL import Image

    gray = np.asarray(Image.open(path).convert("L"), dtype="float32")
    if gray.size == 0:
        return 0.0, 0.0
    # Finite-difference Laplacian variance; avoids an OpenCV dependency.
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0, float(gray.std())


def _times(start: float, end: float, count: int) -> list[float]:
    if count <= 0 or end <= start:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    span = end - start
    # Keep away from exact cuts, which often are fades or transition frames.
    return [start + span * (0.1 + 0.8 * i / (count - 1)) for i in range(count)]


def extract_candidates(
    *,
    source_video: Path,
    start_seconds: float,
    end_seconds: float,
    out_dir: Path,
    candidate_count: int = 5,
    keep_count: int = 3,
) -> list[FrameCandidate]:
    """Extract, score and retain the clearest temporally spread candidate frames."""
    info = probe_media(source_video)
    if info.fps is None or info.fps <= 0:
        raise RuntimeError(f"source video lacks a usable fps: {source_video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[FrameCandidate] = []
    for ordinal, seconds in enumerate(_times(start_seconds, end_seconds, candidate_count)):
        frame_index = max(0, int(round(seconds * info.fps)))
        path = out_dir / f"candidate_{ordinal:02d}_f{frame_index:08d}.jpg"
        extract_frame(source_video, path, frame_index=frame_index, fps=info.fps)
        sharpness, luminance_std = _quality(path)
        candidates.append(
            FrameCandidate(
                frame_index=frame_index,
                seconds=round(frame_index / info.fps, 6),
                path=str(path),
                sharpness=sharpness,
                luminance_std=luminance_std,
            )
        )

    # Retain high-quality frames but preserve deterministic time ordering.
    ranked = sorted(candidates, key=lambda item: (item.sharpness, item.luminance_std), reverse=True)
    selected = sorted(ranked[: max(1, min(keep_count, len(ranked)))], key=lambda item: item.frame_index)
    (out_dir / "candidates.json").write_text(
        json.dumps([item.to_dict() for item in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected
