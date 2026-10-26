"""Shared-filesystem queue for durable, reusable S3 segment clips.

Queue state is represented by JSON task files under ``pending/``, ``claimed/``,
``done/``, and ``failed/``.  Moving a file between those directories uses an
atomic rename, so independently started workers on the same filesystem can
claim a task without a coordinator service.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from .segment_media import worker_clip

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_STATES = ("pending", "claimed", "done", "failed")


def _now() -> float:
    return time.time()


def _is_ready_clip(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON only after its complete contents reach the local filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task record {path} must contain a JSON object")
    return payload


def _worker_file_name(worker_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_name).strip("._")
    if not slug:
        raise ValueError("worker_name must contain at least one path-safe character")
    return f"{slug}.json"


@dataclass(frozen=True, slots=True)
class ClipTask:
    """One output-addressed ffmpeg segment job."""

    task_id: str
    source_video: Path
    output_path: Path
    start_seconds: float
    end_seconds: float
    max_attempts: int = 3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id must be 1-128 path-safe characters")
        if self.end_seconds <= self.start_seconds:
            raise ValueError(f"invalid segment range [{self.start_seconds}, {self.end_seconds}]")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_video": str(self.source_video),
            "output_path": str(self.output_path),
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "max_attempts": self.max_attempts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ClipTask":
        try:
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be an object")
            return cls(
                task_id=str(record["task_id"]),
                source_video=Path(str(record["source_video"])),
                output_path=Path(str(record["output_path"])),
                start_seconds=float(record["start_seconds"]),
                end_seconds=float(record["end_seconds"]),
                max_attempts=int(record.get("max_attempts", 3)),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid clip task record: {error}") from error


@dataclass(frozen=True, slots=True)
class ClipClaim:
    """A task exclusively moved into ``claimed/`` by one worker."""

    task: ClipTask
    worker_name: str
    path: Path
    attempts: int


@dataclass(frozen=True, slots=True)
class WorkerRunResult:
    """Counters from one invocation of a polling clip worker."""

    claimed: int = 0
    completed: int = 0
    retried: int = 0
    failed: int = 0


class SharedClipQueue:
    """Atomic shared-filesystem lifecycle for S3 clip materialization."""

    def __init__(self, root: Path, *, stale_after_seconds: float = 15 * 60) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self.root = Path(root)
        self.stale_after_seconds = float(stale_after_seconds)
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for name in (*_STATES, "workers"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, state: str, task_id: str) -> Path:
        if state not in _STATES:
            raise ValueError(f"unknown queue state {state!r}")
        return self.root / state / f"{task_id}.json"

    def _record_path(self, task_id: str) -> tuple[str, Path] | None:
        for state in _STATES:
            path = self._path(state, task_id)
            if path.is_file():
                return state, path
        return None

    def _move(self, source: Path, state: str, record: Mapping[str, Any]) -> Path:
        destination = self._path(state, str(record["task_id"]))
        record = dict(record)
        record["state"] = state
        _atomic_write_json(source, record)
        os.replace(source, destination)
        return destination

    def enqueue(self, task: ClipTask) -> str:
        """Idempotently enqueue ``task`` or skip it when its output already exists.

        The returned state is one of ``pending``, ``claimed``, ``done``,
        ``failed``, or ``skipped``.  A non-empty output is authoritative: a
        missing/stale ledger entry is recorded as ``done`` without re-encoding.
        """
        self._ensure_layout()
        existing = self._record_path(task.task_id)
        if _is_ready_clip(task.output_path):
            if existing is None:
                record = task.to_record()
                record.update(
                    {
                        "state": "done",
                        "attempts": 0,
                        "created_at": _now(),
                        "completed_at": _now(),
                        "skipped_existing_output": True,
                    }
                )
                _atomic_write_json(self._path("done", task.task_id), record)
            elif existing[0] == "pending":
                state, path = existing
                _ = state
                record = _read_json(path)
                record.update(
                    {
                        "completed_at": _now(),
                        "skipped_existing_output": True,
                    }
                )
                self._move(path, "done", record)
            return "skipped"

        if existing is not None:
            state, path = existing
            if state == "done":
                # A stale done record must not suppress regeneration forever.
                record = _read_json(path)
                record.pop("completed_at", None)
                self._move(path, "pending", record)
                return "pending"
            return state

        record = task.to_record()
        record.update({"state": "pending", "attempts": 0, "created_at": _now()})
        pending = self._path("pending", task.task_id)
        try:
            with pending.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return self.enqueue(task)
        return "pending"

    def heartbeat(self, worker_name: str, *, task_id: str | None = None) -> Path:
        """Record liveness for a named worker without touching task ownership."""
        payload: dict[str, Any] = {
            "worker_name": worker_name,
            "heartbeat_at": _now(),
        }
        if task_id is not None:
            payload["task_id"] = task_id
        path = self.root / "workers" / _worker_file_name(worker_name)
        _atomic_write_json(path, payload)
        return path

    def _worker_is_stale(self, worker_name: str, now: float) -> bool:
        path = self.root / "workers" / _worker_file_name(worker_name)
        try:
            heartbeat_at = float(_read_json(path).get("heartbeat_at", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return True
        return now - heartbeat_at > self.stale_after_seconds

    def requeue_stale_claims(self) -> int:
        """Return abandoned claims to ``pending`` when their worker stops heartbeating."""
        now = _now()
        restored = 0
        for path in sorted((self.root / "claimed").glob("*.json")):
            try:
                record = _read_json(path)
                worker_name = str(record["claimed_by"])
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
            if not self._worker_is_stale(worker_name, now):
                continue
            record.pop("claimed_by", None)
            record.pop("claimed_at", None)
            try:
                self._move(path, "pending", record)
            except FileNotFoundError:
                continue
            restored += 1
        return restored

    def claim(self, worker_name: str) -> ClipClaim | None:
        """Atomically move one pending task to ``claimed`` for ``worker_name``."""
        self._ensure_layout()
        self.heartbeat(worker_name)
        self.requeue_stale_claims()
        for pending in sorted((self.root / "pending").glob("*.json")):
            try:
                record = _read_json(pending)
                task = ClipTask.from_record(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _is_ready_clip(task.output_path):
                record["completed_at"] = _now()
                record["skipped_existing_output"] = True
                try:
                    self._move(pending, "done", record)
                except FileNotFoundError:
                    pass
                continue
            claimed = self._path("claimed", task.task_id)
            try:
                os.replace(pending, claimed)
            except FileNotFoundError:
                continue
            record["state"] = "claimed"
            record["claimed_by"] = worker_name
            record["claimed_at"] = _now()
            _atomic_write_json(claimed, record)
            self.heartbeat(worker_name, task_id=task.task_id)
            return ClipClaim(
                task=task,
                worker_name=worker_name,
                path=claimed,
                attempts=int(record.get("attempts", 0)),
            )
        return None

    def complete(self, claim: ClipClaim) -> Path:
        """Commit a completed claim after verifying its requested output path."""
        if not _is_ready_clip(claim.task.output_path):
            raise ValueError(f"task {claim.task.task_id} has no valid output clip")
        record = _read_json(claim.path)
        if str(record.get("claimed_by")) != claim.worker_name:
            raise RuntimeError(f"claim for {claim.task.task_id} is not owned by {claim.worker_name}")
        record["completed_at"] = _now()
        record.pop("claimed_by", None)
        record.pop("claimed_at", None)
        return self._move(claim.path, "done", record)

    def fail(self, claim: ClipClaim, error: BaseException | str) -> str:
        """Release a failed claim or terminally mark it failed after max attempts."""
        record = _read_json(claim.path)
        if str(record.get("claimed_by")) != claim.worker_name:
            raise RuntimeError(f"claim for {claim.task.task_id} is not owned by {claim.worker_name}")
        attempts = int(record.get("attempts", 0)) + 1
        record["attempts"] = attempts
        record["last_error"] = str(error)[-4_000:]
        record["failed_at"] = _now()
        record.pop("claimed_by", None)
        record.pop("claimed_at", None)
        if attempts >= claim.task.max_attempts:
            self._move(claim.path, "failed", record)
            return "failed"
        self._move(claim.path, "pending", record)
        return "pending"

    def state(self, task_id: str) -> str | None:
        """Return the durable ledger state for ``task_id``, if it exists."""
        found = self._record_path(task_id)
        return found[0] if found is not None else None

    def wait_for_ready(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.25,
    ) -> Path:
        """Wait until a task's output becomes valid or a terminal failure occurs."""
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        started = time.monotonic()
        while True:
            found = self._record_path(task_id)
            if found is None:
                raise KeyError(f"unknown task {task_id!r}")
            state, path = found
            record = _read_json(path)
            task = ClipTask.from_record(record)
            if _is_ready_clip(task.output_path):
                return task.output_path
            if state == "failed":
                raise RuntimeError(
                    f"clip task {task_id} failed after {record.get('attempts', 0)} attempts: "
                    f"{record.get('last_error', 'unknown error')}"
                )
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"timed out waiting for clip task {task_id}")
            time.sleep(poll_interval_seconds)


class _ClaimHeartbeat(AbstractContextManager["_ClaimHeartbeat"]):
    """Keep a long ffmpeg invocation from looking abandoned."""

    def __init__(self, queue: SharedClipQueue, claim: ClipClaim, interval_seconds: float) -> None:
        self.queue = queue
        self.claim = claim
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.queue.heartbeat(self.claim.worker_name, task_id=self.claim.task.task_id)

    def __enter__(self) -> "_ClaimHeartbeat":
        self.queue.heartbeat(self.claim.worker_name, task_id=self.claim.task.task_id)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 1)
        self.queue.heartbeat(self.claim.worker_name)


def run_clip_worker(
    queue: SharedClipQueue,
    *,
    worker_name: str,
    poll_interval_seconds: float = 2.0,
    heartbeat_interval_seconds: float = 15.0,
    max_tasks: int | None = None,
    once: bool = False,
) -> WorkerRunResult:
    """Poll ``queue`` and materialize claimed clips until the requested stop condition."""
    if poll_interval_seconds <= 0 or heartbeat_interval_seconds <= 0:
        raise ValueError("poll and heartbeat intervals must be positive")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be >= 1")

    result = WorkerRunResult()
    while max_tasks is None or result.claimed < max_tasks:
        claim = queue.claim(worker_name)
        if claim is None:
            if once:
                return result
            queue.heartbeat(worker_name)
            time.sleep(poll_interval_seconds)
            continue

        result = replace(result, claimed=result.claimed + 1)
        try:
            with _ClaimHeartbeat(queue, claim, heartbeat_interval_seconds):
                with worker_clip(
                    source_video=claim.task.source_video,
                    cache_root=queue.root / "worker_cache",
                    worker_id=worker_name,
                    start_seconds=claim.task.start_seconds,
                    end_seconds=claim.task.end_seconds,
                    max_attempts=1,
                    output_path=claim.task.output_path,
                    remove_on_exit=False,
                ):
                    pass
            queue.complete(claim)
            result = replace(result, completed=result.completed + 1)
        except Exception as error:  # noqa: BLE001 - persist task failures for another worker.
            next_state = queue.fail(claim, error)
            if next_state == "failed":
                result = replace(result, failed=result.failed + 1)
            else:
                result = replace(result, retried=result.retried + 1)
    return result

