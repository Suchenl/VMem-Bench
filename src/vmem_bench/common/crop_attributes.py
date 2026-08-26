"""Crop attribute pack for MemStrata-Bench S5 (mirror of ``memstrata.mllm.crop_attributes``).

Packages must not import each other. Keep enums / JSON schema / prompt in sync with
``memstrata.mllm.crop_attributes`` and ``docs/benchmark/crop_contract.md``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class SpatialAngle(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    TOP = "top"
    UNKNOWN = "unknown"


class StateAngle(str, Enum):
    DEFAULT = "default"
    CHANGED = "changed"
    DAMAGED = "damaged"
    UNKNOWN = "unknown"


class ShotSize(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    INSERT = "insert"
    UNKNOWN = "unknown"


class Lighting(str, Enum):
    DAY = "day"
    NIGHT = "night"
    INDOOR = "indoor"
    OUTDOOR_OVERCAST = "outdoor_overcast"
    ARTIFICIAL = "artificial"
    BACKLIGHT = "backlight"
    UNKNOWN = "unknown"


class Occlusion(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


def _parse_enum(enum_cls: type[Enum], raw: Any) -> Any:
    try:
        return enum_cls(str(raw))
    except ValueError:
        return enum_cls["UNKNOWN"]


CROP_ATTRIBUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spatial_angle": {
            "type": "string",
            "enum": [e.value for e in SpatialAngle],
        },
        "state_angle": {
            "type": "string",
            "enum": [e.value for e in StateAngle],
        },
        "shot_size": {
            "type": "string",
            "enum": [e.value for e in ShotSize],
        },
        "lighting": {
            "type": "string",
            "enum": [e.value for e in Lighting],
        },
        "occlusion": {
            "type": "string",
            "enum": [e.value for e in Occlusion],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "spatial_angle",
        "state_angle",
        "shot_size",
        "lighting",
        "occlusion",
        "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}

CLASSIFY_PROMPT = (
    "You classify one entity crop for a stratified visual memory bank.\n"
    "Pick exactly one value from each closed enum. Do not invent labels.\n\n"
    "spatial_angle: front | side | back | top | unknown\n"
    "state_angle: default | changed | damaged | unknown\n"
    "shot_size: wide | medium | close_up | extreme_close_up | insert | unknown\n"
    "lighting: day | night | indoor | outdoor_overcast | artificial | backlight | unknown\n"
    "occlusion: none | partial | heavy | unknown\n\n"
    "Entity kind: {kind}\n"
    "Entity name: {name}\n"
    "Return JSON only."
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3.5-9B-Instruct"


@dataclass(slots=True)
class CropAttributePack:
    """Full attribute pack attached to one crop / representation."""

    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    shot_size: ShotSize = ShotSize.UNKNOWN
    lighting: Lighting = Lighting.UNKNOWN
    occlusion: Occlusion = Occlusion.UNKNOWN
    chunk_id: int | None = None
    frame_index: int | None = None
    seconds: float | None = None
    confidence: float = 0.0
    reasoning: str = ""
    source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def diversity_bucket(self) -> tuple[str, str, str, str]:
        """Bucket used by attribute-diverse dedup (occlusion excluded)."""
        return (
            self.spatial_angle.value,
            self.state_angle.value,
            self.shot_size.value,
            self.lighting.value,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "spatial_angle": self.spatial_angle.value,
            "state_angle": self.state_angle.value,
            "shot_size": self.shot_size.value,
            "lighting": self.lighting.value,
            "occlusion": self.occlusion.value,
            "chunk_id": self.chunk_id,
            "frame_index": self.frame_index,
            "seconds": self.seconds,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "source": self.source,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def to_annotations(self) -> dict[str, Any]:
        return {
            "crop_attributes": self.to_dict(),
            "angle_source": self.source,
            "angle_confidence": float(self.confidence),
            "angle_reasoning": self.reasoning,
            "shot_size": self.shot_size.value,
            "lighting": self.lighting.value,
            "occlusion": self.occlusion.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropAttributePack:
        return cls(
            spatial_angle=_parse_enum(SpatialAngle, data.get("spatial_angle")),
            state_angle=_parse_enum(StateAngle, data.get("state_angle")),
            shot_size=_parse_enum(ShotSize, data.get("shot_size")),
            lighting=_parse_enum(Lighting, data.get("lighting")),
            occlusion=_parse_enum(Occlusion, data.get("occlusion")),
            chunk_id=(int(data["chunk_id"]) if data.get("chunk_id") is not None else None),
            frame_index=(
                int(data["frame_index"]) if data.get("frame_index") is not None else None
            ),
            seconds=(float(data["seconds"]) if data.get("seconds") is not None else None),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reasoning=str(data.get("reasoning", "")),
            source=str(data.get("source", "unknown")),
            extra=dict(data.get("extra") or {}),
        )


class CropAttributeClassifier(Protocol):
    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        chunk_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack: ...


def _image_data_url(image_path: str) -> str:
    import io

    from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
        load_crop_rgb_for_model,
    )

    rgb = load_crop_rgb_for_model(image_path)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class NullCropAttributeClassifier:
    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        chunk_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        _ = image_path, kind, name
        return CropAttributePack(
            chunk_id=chunk_id,
            frame_index=frame_index,
            seconds=seconds,
            source="null",
        )


class HeuristicCropAttributeClassifier:
    """Filename-stem hints for tests (e.g. ``hero_front_default_close_up_day``)."""

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        chunk_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        _ = kind, name
        stem = Path(image_path).stem.lower().replace("-", "_")
        tokens = set(stem.split("_"))
        stem_compact = stem.replace("_", "")

        def _pick(enum_cls: type[Enum]) -> Any:
            for item in enum_cls:
                if item.value == "unknown":
                    continue
                value = item.value
                if (
                    value in tokens
                    or value in stem
                    or value.replace("_", "") in stem_compact
                ):
                    return item
            return enum_cls["UNKNOWN"]

        return CropAttributePack(
            spatial_angle=_pick(SpatialAngle),
            state_angle=_pick(StateAngle),
            shot_size=_pick(ShotSize),
            lighting=_pick(Lighting),
            occlusion=_pick(Occlusion),
            chunk_id=chunk_id,
            frame_index=frame_index,
            seconds=seconds,
            confidence=1.0,
            reasoning="heuristic_filename",
            source="heuristic",
        )


class VlmCropAttributeClassifier:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout_sec: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MEMSTRATA_BENCH_CROP_ATTR_BASE_URL")
            or os.environ.get("MEMSTRATA_CROP_ATTR_BASE_URL")
            or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.model = (
            model
            or os.environ.get("MEMSTRATA_BENCH_CROP_ATTR_MODEL")
            or os.environ.get("MEMSTRATA_CROP_ATTR_MODEL")
            or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER_MODEL")
            or DEFAULT_MODEL
        )
        self.timeout_sec = timeout_sec

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        chunk_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        try:
            data_url = _image_data_url(image_path)
        except OSError:
            return CropAttributePack(
                chunk_id=chunk_id,
                frame_index=frame_index,
                seconds=seconds,
                source="vlm_error",
                reasoning="unreadable_image",
            )

        prompt = CLASSIFY_PROMPT.format(kind=kind or "unknown", name=name or "unknown")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        try:
            result = self._call_api(messages)
        except Exception as exc:  # noqa: BLE001
            return CropAttributePack(
                chunk_id=chunk_id,
                frame_index=frame_index,
                seconds=seconds,
                source="vlm_error",
                reasoning=str(exc)[:200],
            )

        return CropAttributePack(
            spatial_angle=_parse_enum(SpatialAngle, result.get("spatial_angle")),
            state_angle=_parse_enum(StateAngle, result.get("state_angle")),
            shot_size=_parse_enum(ShotSize, result.get("shot_size")),
            lighting=_parse_enum(Lighting, result.get("lighting")),
            occlusion=_parse_enum(Occlusion, result.get("occlusion")),
            chunk_id=chunk_id,
            frame_index=frame_index,
            seconds=seconds,
            confidence=float(result.get("confidence", 0.0) or 0.0),
            reasoning=str(result.get("reasoning", "")),
            source="vlm",
        )

    def _call_api(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": int(
                os.environ.get("MEMSTRATA_BENCH_CROP_ATTR_MAX_TOKENS")
                or os.environ.get("MEMSTRATA_CROP_ATTR_MAX_TOKENS")
                or "2048"
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "crop_attribute_pack",
                    "schema": CROP_ATTRIBUTE_SCHEMA,
                    "strict": True,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            return json.loads(content)


def build_crop_attribute_classifier(
    *,
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> CropAttributeClassifier:
    chosen = (
        mode
        or os.environ.get("MEMSTRATA_BENCH_CROP_ATTR_CLASSIFIER")
        or os.environ.get("MEMSTRATA_CROP_ATTR_CLASSIFIER")
        or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER")
        or "null"
    ).strip().lower()
    if chosen in {"vlm", "mllm", "api"}:
        return VlmCropAttributeClassifier(base_url=base_url, model=model)
    if chosen in {"heuristic", "stub", "test"}:
        return HeuristicCropAttributeClassifier()
    return NullCropAttributeClassifier()


__all__ = [
    "CLASSIFY_PROMPT",
    "CROP_ATTRIBUTE_SCHEMA",
    "CropAttributeClassifier",
    "CropAttributePack",
    "HeuristicCropAttributeClassifier",
    "Lighting",
    "NullCropAttributeClassifier",
    "Occlusion",
    "ShotSize",
    "SpatialAngle",
    "StateAngle",
    "VlmCropAttributeClassifier",
    "build_crop_attribute_classifier",
]
