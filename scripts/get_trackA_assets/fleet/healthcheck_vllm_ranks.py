#!/usr/bin/env python3
"""Run an overwrite-only multimodal task health check for every local VLM rank."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


RED_SQUARE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAX0lEQVR4nO3PQQ0AIBDAMMC/50ME"
    "j4ZkVbDtWX87OuBVA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oD"
    "WgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA9oFUoUBf3Xr7AgAAAAASUVORK5CYII="
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _request(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["color"],
        "properties": {"color": {"type": "string", "enum": ["red"]}},
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{RED_SQUARE_PNG}"},
                    },
                    {"type": "text", "text": 'Return JSON {"color":"red"} for this image.'},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rank_health", "schema": schema, "strict": True},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return json.loads(body["choices"][0]["message"]["content"])


def _check_rank(*, rank: int, base_port: int, model: str, timeout: float, output: Path) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{base_port + rank}/v1"
    started = time.time()
    result: dict[str, Any] = {
        "rank": rank,
        "base_url": base_url,
        "checked_at_epoch": started,
        "task": "multimodal structured color=red",
    }
    try:
        response = _request(base_url, model, timeout)
        result.update(
            {
                "ok": response == {"color": "red"},
                "response": response,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001 - a health log must capture every failure
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    # The persistent monitor owns rank<N>.log. Keep the latest real-task probe
    # beside it without racing or appending to the liveness record.
    _atomic_write(output / f"rank{rank}.task.log", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--base-port", type=int, default=8110)
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    ranks = [int(value.strip()) for value in args.ranks.split(",") if value.strip()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranks) or 1) as executor:
        futures = [
            executor.submit(
                _check_rank,
                rank=rank,
                base_port=args.base_port,
                model=args.model,
                timeout=args.timeout,
                output=args.output,
            )
            for rank in ranks
        ]
        results = [future.result() for future in futures]
    summary = {
        "checked_at_epoch": time.time(),
        "model": args.model,
        "ranks": results,
        "healthy_count": sum(bool(result["ok"]) for result in results),
        "total_count": len(results),
    }
    _atomic_write(args.output / "summary.task.log", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["healthy_count"] == summary["total_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
