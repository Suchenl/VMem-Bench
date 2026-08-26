"""VLM bbox+point grounding for one entity crop task."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from vmem_bench.annotation.pipeline.servers.direct_http import (
    ensure_no_proxy_env,
    ensure_no_proxy_host,
    urlopen_direct,
)

ensure_no_proxy_env()

DEFAULT_MODEL = "Qwen3VL-32B-Instruct"


@dataclass(slots=True)
class GroundingResult:
    frame_index: int
    usable: bool
    bbox_norm: list[int]
    point_norm: list[int]
    visibility: str
    confidence: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _area(box: list[int]) -> float:
    y0, x0, y1, x1 = box
    return max(0, y1 - y0) * max(0, x1 - x0) / 1_000_000


def _valid_yxyx(box: list[int]) -> bool:
    y0, x0, y1, x1 = box
    return 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000


def canonicalize_box_point(
    bbox: list[int], point: list[int]
) -> tuple[list[int], list[int]] | None:
    """Return canonical [ymin,xmin,ymax,xmax] + [y,x], or None if unusable.

    Accepts either yxyx or xyxy model output (Qwen often emits xyxy).
    """
    def xyxy_to_yxyx(values: list[int]) -> list[int]:
        x0, y0, x1, y1 = values
        return [y0, x0, y1, x1]

    def point_inside(box: list[int], pt: list[int]) -> bool:
        if len(pt) != 2:
            return False
        y, x = pt
        y0, x0, y1, x1 = box
        return y0 <= y <= y1 and x0 <= x <= x1

    candidates = [list(bbox), xyxy_to_yxyx(bbox)]
    points = [list(point), [point[1], point[0]]]
    best: tuple[list[int], list[int]] | None = None
    best_key = (-1.0, False, -1.0)
    for box in candidates:
        if not _valid_yxyx(box):
            continue
        for pt in points:
            inside = point_inside(box, pt)
            key = (1.0 if inside else 0.0, _area(box) >= 0.02, _area(box))
            if key > best_key:
                best_key = key
                center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
                best = (box, pt if inside else center)
    if best is None or _area(best[0]) < 0.01:
        return None
    return best


class Grounder(Protocol):
    def ground(
        self,
        *,
        image: Path,
        frame_index: int,
        entity_id: str,
        name: str,
        description: str,
        action: str,
    ) -> GroundingResult: ...


class FullFrameGrounder:
    """Dependency-free dry-run grounder; never suitable for a frozen gold."""

    def ground(
        self, *, image: Path, frame_index: int, entity_id: str, name: str, description: str, action: str
    ) -> GroundingResult:
        del image, entity_id, name, description, action
        return GroundingResult(
            frame_index=frame_index,
            usable=True,
            bbox_norm=[0, 0, 1000, 1000],
            point_norm=[500, 500],
            visibility="dry_run_full_frame",
            confidence="low",
            reason="dry-run full-frame grounding",
        )


class QwenImageGrounder:
    """OpenAI-compatible image reviewer returning normalized bbox and positive point."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = 300,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        ensure_no_proxy_host(self.base_url)

    @staticmethod
    def _data_url(image: Path) -> str:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def ground(
        self, *, image: Path, frame_index: int, entity_id: str, name: str, description: str, action: str
    ) -> GroundingResult:
        # Ask for xyxy — Qwen-VL is more reliable with that convention — then
        # canonicalize to the pipeline's yxyx storage format.
        prompt = (
            "Locate the named entity in THIS image only. Return a TIGHT axis-aligned "
            "box covering the entity body (not the whole frame).\n"
            "Coordinate system: integers 0-1000, bbox_norm=[xmin,ymin,xmax,ymax], "
            "point_norm=[x,y] must lie INSIDE the box.\n"
            "If the entity is absent/too occluded: usable=false and empty arrays.\n"
            f"entity_id={entity_id}\nname={name}\ndescription={description}\n"
            f"segment_action={action}"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["usable", "bbox_norm", "point_norm", "visibility", "confidence", "reason"],
            "properties": {
                "usable": {"type": "boolean"},
                "bbox_norm": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 0,
                    "maxItems": 4,
                },
                "point_norm": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 0,
                    "maxItems": 2,
                },
                "visibility": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
        }
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "crop_grounding", "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen_direct(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"crop grounder HTTP {exc.code}: {exc.read().decode('utf-8')[:500]}") from exc
        raw = json.loads(body["choices"][0]["message"]["content"])
        bbox = [int(value) for value in raw["bbox_norm"]]
        point = [int(value) for value in raw["point_norm"]]
        usable = bool(raw["usable"]) and len(bbox) == 4 and len(point) == 2
        canon = canonicalize_box_point(bbox, point) if usable else None
        if canon is None:
            usable, bbox, point = False, [], []
        else:
            bbox, point = canon
        return GroundingResult(
            frame_index=frame_index,
            usable=usable,
            bbox_norm=bbox if usable else [],
            point_norm=point if usable else [],
            visibility=str(raw["visibility"]),
            confidence=str(raw["confidence"]),
            reason=str(raw["reason"]),
        )
