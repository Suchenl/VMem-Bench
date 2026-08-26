"""Shared OpenAI-compatible VLM judge service helpers for MaVE scoring.

Track A and Track B use different scoring prompts and task enumerators, but the
model-service layer is the same: normalize endpoint URLs, optionally resolve the
annotation fleet, lease one endpoint per request, and report workload sidecars.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_API = "http://127.0.0.1:8110/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-32b"
DEFAULT_MM_PROCESSOR_KWARGS = {"fps": 2.0}

# A single dead/restarting replica used to abort the whole movie task (one
# ConnectionRefused killed all remaining segments). These control transparent
# retry-with-rotation: on a transient endpoint failure the same request is
# re-issued against a DIFFERENT healthy endpoint. This changes only which
# replica serves the call, never the payload/sampling, so scores are unchanged.
JUDGE_MAX_ATTEMPTS = int(os.environ.get("VMEM_JUDGE_MAX_ATTEMPTS", "6"))
JUDGE_RETRY_BACKOFF_SEC = float(os.environ.get("VMEM_JUDGE_RETRY_BACKOFF_SEC", "5.0"))


def _is_transient_judge_error(exc: Exception) -> bool:
    """True for connection/timeout/5xx failures worth retrying on another replica.

    Deterministic client errors (HTTP 4xx, e.g. a malformed prompt) are NOT
    transient: retrying them just wastes calls and hides a real bug.
    """
    if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)):
        # urllib.error.HTTPError subclasses URLError; treat only 5xx as transient.
        if isinstance(exc, urllib.error.HTTPError):
            return 500 <= int(exc.code) < 600
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if "VLM call failed HTTP 5" in msg:
            return True
    return False


def chat_endpoint(api: str) -> str:
    """Accept either a full chat endpoint or an OpenAI-compatible /v1 base URL."""
    url = str(api or "").rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def call_judge_http(
    api: str,
    model: str,
    content: list,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "mm_processor_kwargs": mm_processor_kwargs or DEFAULT_MM_PROCESSOR_KWARGS,
        }
    ).encode()
    req = urllib.request.Request(chat_endpoint(api), data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:  # noqa: S310 - operator-controlled judge URLs
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode()[:800]
        except Exception:
            detail = "<no body>"
        raise RuntimeError(f"VLM call failed HTTP {exc.code}: {detail}") from None


class PooledJudgeCaller:
    """Endpoint pool for VLM judge calls.

    The pool size controls throughput only: every individual chat/completions
    request leases one slot. Workload markers reuse the annotation fleet
    registry so the existing console can show active Track A/B scoring work.

    ``endpoint_slots`` is the number of concurrent requests allowed per endpoint
    and defaults to 1 (one in-flight request per replica, the historical
    behaviour). Raise it ONLY when the replicas run with a matching vLLM
    ``MAX_NUM_SEQS``; it changes scheduling/throughput only, never the request
    payloads, so the judged content is unaffected.
    """

    def __init__(
        self,
        base_urls: list[str] | str,
        *,
        model: str,
        stage: str,
        fleet_root: Path | str | None = None,
        report_workload: bool | None = None,
        endpoint_slots: int = 1,
    ) -> None:
        from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
            ReviewerEndpointPool,
            parse_endpoint_urls,
        )

        self.model = model
        self.stage = stage
        self.pool = ReviewerEndpointPool(
            parse_endpoint_urls(base_urls),
            fleet_root=fleet_root,
            report_workload=report_workload,
            default_workload={"stage": stage},
            slots_per_url=endpoint_slots,
        )
        self._local = threading.local()

    @property
    def size(self) -> int:
        """Total concurrent lease capacity (endpoints x endpoint_slots)."""
        return self.pool.size

    @property
    def endpoint_slots(self) -> int:
        return self.pool.slots_per_url

    @property
    def base_urls(self) -> list[str]:
        return self.pool.base_urls

    @contextmanager
    def workload(self, **fields: Any) -> Iterator[None]:
        prev = getattr(self._local, "workload", None)
        merged = {**(prev or {}), **{k: v for k, v in fields.items() if v not in (None, "")}}
        self._local.workload = merged
        try:
            yield
        finally:
            self._local.workload = prev

    def current_workload(self) -> dict[str, Any] | None:
        """Snapshot of this thread's workload tags.

        Workload context is thread-local, so a helper thread spawned by the
        scorer would otherwise report an unlabelled busy endpoint to the fleet
        console. Callers re-enter ``workload(**snapshot)`` inside the new thread
        to keep console attribution identical to the single-threaded path.
        """
        current = getattr(self._local, "workload", None)
        return dict(current) if current else None

    def __call__(
        self,
        content: list,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> str:
        workload = getattr(self._local, "workload", None)
        attempts = max(1, JUDGE_MAX_ATTEMPTS)
        last_url: str | None = None
        last_exc: Exception | None = None
        for attempt in range(attempts):
            # After a transient failure, prefer a DIFFERENT replica so a single
            # dead/restarting endpoint can't keep failing the same request.
            with self.pool.lease(workload=workload, prefer_not=last_url) as lease:
                last_url = lease.base_url
                try:
                    return call_judge_http(
                        lease.base_url,
                        self.model,
                        content,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 - decide retry vs re-raise below
                    last_exc = exc
                    if attempt + 1 >= attempts or not _is_transient_judge_error(exc):
                        raise
            # Sleep OUTSIDE the lease so the endpoint is freed for other workers
            # (and the failed replica has a moment to recover) before retrying.
            time.sleep(JUDGE_RETRY_BACKOFF_SEC * (attempt + 1))
        assert last_exc is not None
        raise last_exc


def build_judge_api(
    *,
    api: str = DEFAULT_API,
    model: str = DEFAULT_MODEL,
    api_list: str | list[str] | None = None,
    use_fleet: bool = False,
    fleet_root: Path | str | None = None,
    fleet_role: str | None = "reviewer",
    stage: str = "mave_scoring",
    endpoint_slots: int = 1,
) -> str | PooledJudgeCaller:
    """Build a judge caller. ``endpoint_slots`` > 1 needs replicas with matching MAX_NUM_SEQS."""
    # NB: not ``endpoint_slots or 1`` — that would silently turn an explicit 0 into 1.
    endpoint_slots = 1 if endpoint_slots is None else int(endpoint_slots)
    if endpoint_slots < 1:
        raise ValueError(f"endpoint_slots must be >= 1, got {endpoint_slots}")
    urls: list[str] = []
    if use_fleet:
        from vmem_bench.annotation.pipeline.servers.fleet.registry import resolve_dispatch_urls

        urls = resolve_dispatch_urls(
            fleet_root=Path(fleet_root).resolve() if fleet_root else None,
            role=fleet_role,
            probe=True,
        )
        if not urls:
            raise RuntimeError("no online VLM endpoints found in fleet registry")
    elif api_list:
        from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
            parse_endpoint_urls,
        )

        urls = parse_endpoint_urls(api_list)
    elif "," in str(api) or "\n" in str(api) or ";" in str(api):
        from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
            parse_endpoint_urls,
        )

        urls = parse_endpoint_urls(api)
    if not urls and endpoint_slots > 1:
        # A bare single --api normally returns a plain URL string with no pool, so
        # slots would silently do nothing. Opting into slots is explicit, so build a
        # one-endpoint pool to make the setting take effect. Default (slots == 1)
        # still returns the plain string, unchanged.
        urls = [str(api)]
    if urls:
        return PooledJudgeCaller(
            urls, model=model, stage=stage, fleet_root=fleet_root, endpoint_slots=endpoint_slots
        )
    return chat_endpoint(api)


def call_judge(
    api: str | PooledJudgeCaller,
    model: str,
    content: list,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> str:
    if callable(api) and not isinstance(api, str):
        return api(
            content,
            temperature=temperature,
            max_tokens=max_tokens,
            mm_processor_kwargs=mm_processor_kwargs,
        )
    return call_judge_http(
        str(api),
        model,
        content,
        temperature=temperature,
        max_tokens=max_tokens,
        mm_processor_kwargs=mm_processor_kwargs,
    )
