"""Bounded pool of OpenAI-compatible VLM reviewer endpoints.

Endpoint count controls throughput only: each in-flight segment holds one
endpoint slot.  Per-segment revise rounds stay bounded separately.

When ``report_workload`` is enabled, leases write busy/idle markers under
``runtime/services/vlm_fleet/workloads/`` so the console can show which
cluster/node/rank is actively working.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


def parse_endpoint_urls(raw: str | list[str]) -> list[str]:
    """Parse comma/whitespace-separated URLs into a de-duplicated list."""
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw]
    else:
        text = str(raw or "").replace("\n", ",").replace(";", ",")
        parts = []
        for chunk in text.split(","):
            parts.extend(chunk.split())
    urls: list[str] = []
    seen: set[str] = set()
    for part in parts:
        url = part.rstrip("/")
        if not url or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid reviewer endpoint URL: {part!r}")
        seen.add(url)
        urls.append(url)
    if not urls:
        raise ValueError("at least one reviewer endpoint URL is required")
    return urls


@dataclass(slots=True)
class EndpointLease:
    """One exclusive lease on a reviewer HTTP endpoint."""

    base_url: str
    endpoint_index: int


class ReviewerEndpointPool:
    """Queue of reviewer base URLs with ``slots_per_url`` concurrent leases per URL.

    ``MAX_NUM_SEQS=1`` vLLM replicas should expose one URL each.  Workers block
    on ``acquire`` until a free endpoint is available, so pool size equals the
    maximum number of concurrent segment reviews.

    ``slots_per_url`` defaults to 1, which is the historical one-at-a-time
    behaviour and the ONLY correct setting for ``MAX_NUM_SEQS=1`` replicas.
    Raise it above 1 only when the backing vLLM replicas are launched with a
    matching ``MAX_NUM_SEQS`` (and enough KV cache), otherwise the extra leases
    just queue inside the server and add no throughput.
    """

    def __init__(
        self,
        base_urls: list[str],
        *,
        report_workload: bool | None = None,
        fleet_root: Path | str | None = None,
        default_workload: dict[str, Any] | None = None,
        slots_per_url: int = 1,
    ) -> None:
        slots_per_url = int(slots_per_url)
        if slots_per_url < 1:
            raise ValueError("slots_per_url must be >= 1")
        unique_urls = parse_endpoint_urls(base_urls)
        # Slot-major interleave, NOT url-major. With [u0,u0,u1,u1,...] the rotation
        # in ``acquire`` hands out both of u0's slots before it ever reaches u1, so
        # K concurrent workers pile onto K/slots replicas and leave the rest idle.
        # [u0,u1,...,uN,u0,u1,...,uN] spreads the first pass across every replica
        # before any replica takes a second request.
        self._urls = [url for _ in range(slots_per_url) for url in unique_urls]
        self._unique_urls = unique_urls
        self._slots_per_url = slots_per_url
        self._condition = threading.Condition()
        self._free = list(range(len(self._urls)))
        self._round_robin = 0
        self._busy_slots_by_url = {url: 0 for url in unique_urls}
        if report_workload is None:
            report_workload = os.environ.get("MEMSTRATA_FLEET_REPORT_WORKLOAD", "1") != "0"
        self._report_workload = bool(report_workload)
        self._fleet_root = Path(fleet_root).resolve() if fleet_root else None
        self._default_workload = dict(default_workload or {})

    @property
    def size(self) -> int:
        """Total lease capacity = ``len(base_urls) * slots_per_url``.

        Callers use this as the default worker count, so it must be the number of
        simultaneously servable requests, not the number of distinct endpoints.
        """
        return len(self._urls)

    @property
    def slots_per_url(self) -> int:
        return self._slots_per_url

    @property
    def base_urls(self) -> list[str]:
        """The DISTINCT endpoints, one entry each (not one entry per slot)."""
        return list(self._unique_urls)

    def acquire(self, *, prefer_not: str | None = None, timeout: float | None = None) -> EndpointLease:
        """Block until an endpoint is free. Prefer a different URL when possible."""
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._condition:
            while True:
                chosen: int | None = None
                if self._free:
                    # Always select in round-robin rotation order (not ascending
                    # index order). Ascending order biased every lease — and
                    # especially every ``prefer_not`` revise re-lease — toward the
                    # lowest indices. With the URL list ordered [a800×8, h800×16],
                    # that starved the h800 replicas even though there are twice
                    # as many of them. Fair rotation gives each replica an equal
                    # share, so h800 naturally receives ~2x the a800 traffic.
                    ordered = sorted(
                        self._free,
                        key=lambda idx: (idx - self._round_robin) % len(self._urls),
                    )
                    if prefer_not and len(ordered) > 1:
                        target = prefer_not.rstrip("/")
                        chosen = next(
                            (idx for idx in ordered if self._urls[idx] != target),
                            ordered[0],
                        )
                    else:
                        chosen = ordered[0]
                if chosen is not None:
                    self._free.remove(chosen)
                    self._round_robin = (chosen + 1) % len(self._urls)
                    return EndpointLease(base_url=self._urls[chosen], endpoint_index=chosen)
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for a free reviewer endpoint")
                self._condition.wait(timeout=remaining)

    def release(self, lease: EndpointLease) -> None:
        with self._condition:
            if lease.endpoint_index not in self._free:
                self._free.append(lease.endpoint_index)
                self._free.sort()
            self._condition.notify()

    def _mark_busy(self, base_url: str, workload: dict[str, Any] | None) -> None:
        if not self._report_workload:
            return
        try:
            from vmem_bench.annotation.pipeline.servers.fleet.registry import mark_endpoint_busy

            meta = {**self._default_workload, **(workload or {})}
            mark_endpoint_busy(
                base_url,
                fleet_root=self._fleet_root,
                job_id=str(meta.pop("job_id", "") or ""),
                movie_id=str(meta.pop("movie_id", "") or ""),
                dataset=str(meta.pop("dataset", "") or ""),
                segment_id=str(meta.pop("segment_id", "") or ""),
                stage=str(meta.pop("stage", "") or ""),
                extra=meta or None,
            )
        except Exception:  # noqa: BLE001 — workload reporting must not break reviews
            return

    def _mark_idle(self, base_url: str) -> None:
        if not self._report_workload:
            return
        try:
            from vmem_bench.annotation.pipeline.servers.fleet.registry import mark_endpoint_idle

            mark_endpoint_idle(base_url, fleet_root=self._fleet_root)
        except Exception:  # noqa: BLE001
            return

    def _begin_workload(self, base_url: str, workload: dict[str, Any] | None) -> None:
        with self._condition:
            self._busy_slots_by_url[base_url] = self._busy_slots_by_url.get(base_url, 0) + 1
        self._mark_busy(base_url, workload)

    def _finish_workload(self, base_url: str) -> bool:
        with self._condition:
            remaining = max(0, self._busy_slots_by_url.get(base_url, 0) - 1)
            self._busy_slots_by_url[base_url] = remaining
            return remaining == 0

    @contextmanager
    def lease(
        self,
        *,
        prefer_not: str | None = None,
        timeout: float | None = None,
        workload: dict[str, Any] | None = None,
    ) -> Iterator[EndpointLease]:
        held = self.acquire(prefer_not=prefer_not, timeout=timeout)
        self._begin_workload(held.base_url, workload)
        try:
            yield held
        finally:
            if self._finish_workload(held.base_url):
                self._mark_idle(held.base_url)
            self.release(held)
