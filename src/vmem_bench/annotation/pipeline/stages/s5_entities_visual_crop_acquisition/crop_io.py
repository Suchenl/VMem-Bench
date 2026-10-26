"""Crop I/O: store masked crops as RGBA PNG; composite white only for model feed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Default composite background when a crop is consumed by VLM / encoders / generators.
MODEL_FEED_BACKGROUND: tuple[int, int, int] = (255, 255, 255)


def _pixel_box(bbox_norm: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = bbox_norm
    left = max(0, min(width - 1, round(x0 / 1000 * width)))
    top = max(0, min(height - 1, round(y0 / 1000 * height)))
    right = max(left + 1, min(width, round(x1 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
    return left, top, right, bottom


def ensure_png_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")


def unmasked_companion_path(crop_path: Path) -> Path:
    """Return the opaque same-bbox companion path for a masked crop."""
    crop_path = ensure_png_path(Path(crop_path))
    return crop_path.with_name(f"{crop_path.stem}_unmasked.png")


def materialize_crop(
    *,
    frame: Path,
    bbox_norm: list[int],
    out_path: Path,
    mask: object | None = None,
    mask_fill: tuple[int, int, int] | None = None,
) -> Path:
    """Crop one image and write it to disk.

    With a mask: store **RGBA PNG** (transparent outside the mask; no fill).
    Without a mask: store RGB PNG (opaque bbox crop).

    ``mask_fill`` is accepted for backward compatibility and ignored — fill happens
    at model-feed time via :func:`load_crop_rgb_for_model`.
    """
    del mask_fill
    from PIL import Image
    import numpy as np

    image = Image.open(frame).convert("RGB")
    left, top, right, bottom = _pixel_box(bbox_norm, *image.size)
    crop = image.crop((left, top, right, bottom))
    destination = ensure_png_path(Path(out_path))
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        array = np.asarray(mask, dtype=bool)[top:bottom, left:right]
        if array.shape[:2] != (crop.size[1], crop.size[0]):
            raise ValueError(
                f"mask crop shape {array.shape[:2]} != image crop {(crop.size[1], crop.size[0])}"
            )
        rgb = np.asarray(crop, dtype=np.uint8)
        alpha = (array.astype(np.uint8) * 255)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba, mode="RGBA").save(destination)
    else:
        crop.save(destination)

    return destination


def materialize_unmasked_companion(
    *,
    frame: Path,
    bbox_norm: list[int],
    crop_path: Path,
) -> Path:
    """Write the opaque same-bbox companion beside a masked crop."""
    return materialize_crop(
        frame=frame,
        bbox_norm=bbox_norm,
        out_path=unmasked_companion_path(crop_path),
    )


def load_crop_rgb_for_model(
    path: Path | str,
    *,
    background: tuple[int, int, int] = MODEL_FEED_BACKGROUND,
) -> Any:
    """Load a stored crop as RGB, compositing RGBA onto ``background`` (default white)."""
    from PIL import Image

    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def crop_to_model_data_url(
    path: Path | str,
    *,
    background: tuple[int, int, int] = MODEL_FEED_BACKGROUND,
    mime: str = "image/jpeg",
    quality: int = 95,
) -> str:
    """Encode a crop for multimodal APIs after white-background composite."""
    import base64
    import io

    rgb = load_crop_rgb_for_model(path, background=background)
    buffer = io.BytesIO()
    fmt = "JPEG" if mime.endswith("jpeg") or mime.endswith("jpg") else "PNG"
    if fmt == "JPEG":
        rgb.save(buffer, format=fmt, quality=quality)
    else:
        rgb.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
