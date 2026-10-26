"""Track-B generated-history store: the retrieval memories' frame pool over this run's own clips.

Track-B generation has no source video — the memory is the film the run is producing, one generated
segment at a time. Each generated segment keeps its OWN clip, and frames are extracted from that clip
on a synthetic monotonic clock so the retrievers' ``as_of_seconds`` causal gate still holds (segment
*i* occupies ``[clock_i, clock_i + duration_i)``; only strictly earlier segments are eligible when the
next one is composed). Only frame materialization differs from the Track-A store; ranking, sampling
and diversity logic are inherited unchanged from :class:`MemoryRetrievalStore`.

Behaviour is that of the frozen Track-B paper runs (duration from ``ffmpeg -i`` stderr, 2.0 s when
unprobeable; stills via ``ffmpeg -ss <local> -frames:v 1 -vf scale=512:-1 -q:v 2``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from vmem_bench.testbed.retrieval.retrievers import (
    MemoryRetrievalStore,
    RetrieverConfig,
    SegmentRecord,
)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def resolve_ffmpeg(preferred: str = "ffmpeg") -> str:
    """An explicit path, then PATH, then the ``imageio-ffmpeg`` bundled binary."""
    if preferred and preferred != "ffmpeg" and Path(preferred).is_file():
        return preferred
    found = shutil.which(preferred or "ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return preferred or "ffmpeg"


def probe_duration(video: str, ffmpeg: str = "ffmpeg") -> float:
    """Clip duration parsed from ``ffmpeg -i`` stderr (centisecond precision); 2.0 s if unavailable."""
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", video],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return 2.0
    match = _DURATION_RE.search(proc.stderr.decode("utf-8", "ignore"))
    if not match:
        return 2.0
    hours, minutes, seconds = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total if total > 0.0 else 2.0


def extract_still(ffmpeg: str, video: str, seconds: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-ss", f"{seconds:.3f}", "-i", video, "-frames:v", "1",
         "-vf", "scale=512:-1", "-q:v", "2", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0 and out.is_file() and out.stat().st_size > 0


class TrackBGeneratedStore(MemoryRetrievalStore):
    """Per-generated-segment frame memory for the Track-B retrieval read path."""

    def __init__(self, config: RetrieverConfig | None = None, *, work_dir: str | Path | None = None) -> None:
        super().__init__(config)
        self.config.ffmpeg = resolve_ffmpeg(self.config.ffmpeg)
        if work_dir is not None:
            self.work_dir = Path(work_dir)
        self._videos: dict[int, str] = {}
        self._clock: float = 0.0

    @property
    def clock(self) -> float:
        return self._clock

    def video_for(self, segment_id: int) -> str | None:
        return self._videos.get(int(segment_id)) or None

    def observe_generated_segment(
        self,
        *,
        segment_id: int,
        prompt_text: str,
        segment_video: str,
        description: str = "",
    ) -> None:
        """Append a freshly generated segment to the causal memory timeline."""
        dur = probe_duration(segment_video, self.config.ffmpeg) if segment_video else 2.0
        span = (self._clock, self._clock + dur)
        self._clock += dur
        self._videos[int(segment_id)] = str(segment_video or "")
        self.segments.append(
            SegmentRecord(
                segment_id=int(segment_id),
                prompt_text=str(prompt_text or ""),
                seconds_span=(float(span[0]), float(span[1])),
                description=str(description or prompt_text or ""),
            )
        )

    def _maybe_extract_frame(self, seg: SegmentRecord, seconds: float) -> str:
        out = self._frame_path(seg, seconds)
        if out.is_file() and out.stat().st_size > 0:
            return str(out)
        video = self._videos.get(int(seg.segment_id))
        if not self.config.extract_frames or not video or not Path(video).is_file():
            return f"gen:c{seg.segment_id}:{seconds:.3f}"
        # Global synthetic seconds -> local offset inside this segment's own clip.
        local = max(0.0, float(seconds) - float(seg.seconds_span[0]))
        if extract_still(self.config.ffmpeg, video, local, out):
            return str(out)
        return f"gen:c{seg.segment_id}:{seconds:.3f}"
