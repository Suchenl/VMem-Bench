"""VLM multiple-choice crop picker for route B (no bbox regression)."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from vmem_bench.annotation.pipeline.servers.direct_http import (
    ensure_no_proxy_env,
    ensure_no_proxy_host,
    urlopen_direct,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
    ReviewerEndpointPool,
    parse_endpoint_urls,
)

ensure_no_proxy_env()


class CropPicker(Protocol):
    def pick(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        action: str,
        candidate_crops: list[Path],
    ) -> int:
        """Return 0-based candidate index, or -1 when none match."""


class FirstCandidatePicker:
    """Deterministic dry-run picker: always take the first candidate when present."""

    def __init__(self) -> None:
        self.last_result: dict[str, object] = {}

    def pick(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        action: str,
        candidate_crops: list[Path],
    ) -> int:
        del name, description, kind, action
        index = 0 if candidate_crops else -1
        self.last_result = {
            "index": index,
            "confidence": "low",
            "reason": "deterministic dry-run first candidate",
            "picker": "first_candidate",
        }
        return index


class QwenCropPicker:
    """OpenAI-compatible image picker; supports a single URL or a comma-separated pool."""

    def __init__(
        self,
        *,
        base_url: str | list[str],
        model: str,
        timeout_seconds: int = 300,
        max_tokens: int = 2048,
        max_retry_tokens: int = 8192,
        max_retries: int = 3,
    ) -> None:
        self._pool = ReviewerEndpointPool(parse_endpoint_urls(base_url))
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_retry_tokens = max_retry_tokens
        self.max_retries = max_retries
        self.last_result: dict[str, object] = {}
        for url in self._pool.base_urls:
            ensure_no_proxy_host(url)

    @property
    def base_url(self) -> str:
        """First endpoint (compat for callers that only log one URL)."""
        return self._pool.base_urls[0]

    @staticmethod
    def _data_url(image: Path) -> str:
        from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
            load_crop_rgb_for_model,
        )

        rgb = load_crop_rgb_for_model(image)
        rgb.thumbnail((512, 512))
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def pick(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        action: str,
        candidate_crops: list[Path],
    ) -> int:
        if not candidate_crops:
            return -1
        del action
        content: list[dict] = []
        for index, crop in enumerate(candidate_crops):
            content.append({"type": "text", "text": f"candidate_index={index}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": self._data_url(crop)},
            })
        content.append({
            "type": "text",
            "text": (
                "上面是检测器提出的候选 crop（按编号）。请为指定实体选择最匹配的一张。"
                "这是 closed-set identity 选择：只能选择确实属于目标 name/description 的 crop，"
                "不要因为另一只共现角色更显眼就选它。若身份特征不足、crop 混入多只实体、"
                "或都不匹配，index=-1。只做选择，不要输出 bbox。\n"
                f"kind={kind}\nname={name}\ndescription={description}\n"
                '返回 JSON {"index": int, "confidence": "high|medium|low", "reason": str}'
            ),
        })
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["index", "confidence", "reason"],
            "properties": {
                "index": {"type": "integer"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "crop_pick", "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err: Exception | None = None
        prefer_not: str | None = None
        for attempt in range(self.max_retries):
            with self._pool.lease(
                prefer_not=prefer_not,
                workload={"stage": "s5_crop_pick", "attempt": attempt + 1},
            ) as lease:
                prefer_not = lease.base_url
                request = urllib.request.Request(
                    f"{lease.base_url}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen_direct(request, timeout=self.timeout_seconds) as response:
                        body = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    last_err = RuntimeError(
                        f"crop picker HTTP {exc.code} on {lease.base_url}: "
                        f"{exc.read().decode('utf-8')[:500]}"
                    )
                    if attempt < self.max_retries - 1:
                        continue
                    raise last_err from exc
                except Exception as exc:  # endpoint disconnect/timeout: rotate and retry
                    last_err = exc
                    if attempt < self.max_retries - 1:
                        continue
                    raise RuntimeError(
                        f"crop picker request failed after {self.max_retries} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                choice = body["choices"][0]
                content_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
                try:
                    raw = json.loads(content_text)
                except json.JSONDecodeError as exc:
                    last_err = exc
                    # Truncated JSON (often Unterminated string near "reason") → raise budget.
                    if finish_reason == "length" and payload["max_tokens"] < self.max_retry_tokens:
                        payload["max_tokens"] = min(
                            self.max_retry_tokens, int(payload["max_tokens"]) * 2
                        )
                        continue
                    if attempt < self.max_retries - 1:
                        payload["max_tokens"] = min(
                            self.max_retry_tokens,
                            max(int(payload["max_tokens"]) * 2, self.max_tokens),
                        )
                        continue
                    raise RuntimeError(
                        f"crop picker returned non-JSON after {self.max_retries} attempts "
                        f"(finish_reason={finish_reason!r}, max_tokens={payload['max_tokens']}): "
                        f"{exc}\n--content--\n{str(content_text)[:500]}"
                    ) from exc
                index = int(raw["index"])
                self.last_result = {
                    "index": index,
                    "confidence": str(raw.get("confidence") or "low"),
                    "reason": str(raw.get("reason") or ""),
                    "picker": "qwen_closed_set",
                    "endpoint": lease.base_url,
                }
                if index < 0 or index >= len(candidate_crops):
                    return -1
                return index
        raise RuntimeError(f"crop picker exhausted retries: {last_err}")
