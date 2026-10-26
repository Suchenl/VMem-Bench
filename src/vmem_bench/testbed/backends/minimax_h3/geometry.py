"""Clip geometry of the MiniMax-H3-Turbo pipeline, aligned with the paper's Wan2.2 pipeline.

The paper's shared generator (MemStrata ``configs/video_gen/wan22_ti2v5b_turbo.toml``) renders
121 frames at 24 fps (5.04 s, one Track-B segment) and delivers 480x832. H3 also runs at 24 fps but
snaps frame counts up to 17n+5, so 121 requested frames render as 124 and the clip is trimmed back
to the first 121. Every geometry delivers 480x832 so scorers see the same resolution for both
pipelines; only the render size differs:

- ``480x832`` (default): render at the delivery size, no resampling.
- ``704x1280``: the Wan2.2 render size (within H3's trained range, native short side 768), then a
  Lanczos downscale to 480x832.
- ``544x960``: the Ref2V Turbo 4-step v0.1 training resolution (544p, lightx2v/ModelTC release notes;
  the official ComfyUI graph uses 960x544), then a Lanczos downscale to 480x832.

Measured on one B200 (2026-09-27 smoke): 704x1280 costs ~61 s (0 refs) to ~80 s (3 refs) per segment,
480x832 ~20-32 s. ``REF_IMAGE_SHORT_EDGE`` is 768 rather than DiffSynth's 2048 default: 2048 made a
2-reference segment 2.3x slower (170 s) without a visible fidelity gain.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "minimax_h3_ref2v_turbo"
FL2V_MODEL_ID = "minimax_h3_fl2v_turbo"   # first-frame (FL2VA) server, used by the flux2_h3 backend
NUM_FRAMES = 121
FPS = 24
OUT_HEIGHT = 480
OUT_WIDTH = 832
STEPS = 4
REF_IMAGE_SHORT_EDGE = 768
SEGMENT_SECONDS = 5.0


@dataclass(frozen=True)
class Geometry:
    name: str
    height: int
    width: int
    out_height: int = OUT_HEIGHT
    out_width: int = OUT_WIDTH


GEOMETRIES: dict[str, Geometry] = {
    g.name: g for g in (Geometry("480x832", 480, 832), Geometry("544x960", 544, 960),
                        Geometry("704x1280", 704, 1280))
}
DEFAULT_GEOMETRY = "480x832"
RENDER_HEIGHT = GEOMETRIES[DEFAULT_GEOMETRY].height
RENDER_WIDTH = GEOMETRIES[DEFAULT_GEOMETRY].width


def geometry(name: str) -> Geometry:
    try:
        return GEOMETRIES[name]
    except KeyError:
        raise ValueError(f"unknown H3 geometry {name!r}; choices: {sorted(GEOMETRIES)}") from None


def aligned_frame_count(n: int) -> int:
    """Smallest 17k+5 >= n (H3's temporal VAE layout)."""
    current = max(int(n), 5)
    while current % 17 != 5:
        current += 1
    return current
