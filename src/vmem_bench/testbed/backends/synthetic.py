"""CPU smoke backend: a deterministic solid-colour clip per segment (no model, needs only ffmpeg).

The colour hashes ``(prompt, seed, references' bytes)``, so a smoke run exercises the whole loop —
reference selection, causal store updates, layout — and two runs with the same seed are identical.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from vmem_bench.testbed.backends.base import RealizationRequest, RealizationResult
from vmem_bench.testbed.store import resolve_ffmpeg


class SyntheticBackend:
    name = "synthetic"

    def __init__(self, *, seconds: float = 2.0, size: str = "64x48", fps: int = 8) -> None:
        self.seconds, self.size, self.fps = float(seconds), size, int(fps)
        self.ffmpeg = resolve_ffmpeg()

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def realize(self, request: RealizationRequest) -> RealizationResult:
        h = hashlib.sha256(f"{request.prompt}\0{request.seed}".encode("utf-8"))
        for ref in request.references:
            h.update(Path(ref).read_bytes())
        colour = h.hexdigest()[:6]
        out_dir = Path(request.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = out_dir / "clip.mp4"
        subprocess.run(
            [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c=0x{colour}:s={self.size}:d={self.seconds:g}:r={self.fps}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bitexact", str(clip)],
            check=True, capture_output=True, timeout=60,
        )
        return RealizationResult(clip=clip, keyframe=None,
                                 meta={"colour": colour, "n_references": len(request.references),
                                       "seed": int(request.seed)})
