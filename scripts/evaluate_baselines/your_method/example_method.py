"""A trivial, GPU-free reference implementation of the :class:`Method` contract.

``RecentFramesMethod`` is a weak "recency memory" baseline: on ``observe`` it stores
one frame per segment (cut from the real clip if you have source videos, otherwise a
labeled placeholder so the plumbing still runs on CPU); on ``compose`` it recalls the
most recent ``budget`` stored frames. It is NOT competitive — it exists so you can run
the full Track A loop end-to-end before wiring in your real perception + memory.

Replace ``observe`` (your detect/crop/embed/store) and ``compose`` (your recall) with
your method. Keep the fairness rules: compose only from memory built by PRIOR
observations; never read ``present``/roster.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from sut_interface import Method, Reference, Segment


class RecentFramesMethod:
    """Stores one frame per observed segment; recalls the most recent ``budget``."""

    def __init__(self, mem_dir: Path, budget: int = 3, ffmpeg: str = "ffmpeg") -> None:
        self.mem_dir = Path(mem_dir)
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget)
        self.ffmpeg = ffmpeg
        self._memory: list[Path] = []  # chronological frame paths

    # -- recall (before observe) --
    def compose(self, seg: Segment) -> list[Reference]:
        if not self._memory:
            return []  # first segment: nothing seen yet -> empty is honest
        recent = self._memory[-self.budget:]
        return [Reference(crop_abspath=str(p.resolve())) for p in recent]

    # -- write (after compose) --
    def observe(self, seg: Segment) -> None:
        frame = self.mem_dir / f"chunk_{seg.chunk_id:05d}.png"
        if seg.video_path is not None and self._has_ffmpeg():
            self._cut_midpoint_frame(seg, frame)
        if not frame.is_file():
            self._placeholder(seg, frame)
        self._memory.append(frame)

    # -- helpers --
    def _has_ffmpeg(self) -> bool:
        return shutil.which(self.ffmpeg) is not None

    def _cut_midpoint_frame(self, seg: Segment, out: Path) -> None:
        s0, s1 = seg.seconds_span or (0.0, 1.0)
        mid = (float(s0) + float(s1)) / 2.0
        try:
            subprocess.run(
                [self.ffmpeg, "-y", "-ss", f"{mid:.3f}", "-i", str(seg.video_path),
                 "-frames:v", "1", "-q:v", "3", str(out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass  # fall back to placeholder

    def _placeholder(self, seg: Segment, out: Path) -> None:
        # Tiny deterministic PNG so the artifact is schema-valid and scorer-ingestible
        # on CPU. With real source videos, _cut_midpoint_frame supplies real pixels.
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (256, 144), (32, 32, 48))
        ImageDraw.Draw(img).text((8, 8), f"chunk {seg.chunk_id}", fill=(220, 220, 220))
        img.save(out)


def build(mem_dir: Path, budget: int = 3, ffmpeg: str = "ffmpeg") -> Method:
    return RecentFramesMethod(mem_dir=mem_dir, budget=budget, ffmpeg=ffmpeg)
