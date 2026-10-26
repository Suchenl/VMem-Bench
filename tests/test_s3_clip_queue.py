"""Focused lifecycle tests for the shared S3 clip queue."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise import clip_queue


def _task(tmp_path: Path, *, task_id: str = "seg_001") -> clip_queue.ClipTask:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return clip_queue.ClipTask(
        task_id=task_id,
        source_video=source,
        output_path=tmp_path / "clips" / task_id / "segment.mp4",
        start_seconds=0,
        end_seconds=1,
        max_attempts=2,
    )


def test_enqueue_skips_existing_valid_output_and_marks_done(tmp_path: Path) -> None:
    queue = clip_queue.SharedClipQueue(tmp_path / "queue")
    task = _task(tmp_path)

    assert queue.enqueue(task) == "pending"
    task.output_path.parent.mkdir(parents=True)
    task.output_path.write_bytes(b"clip")

    assert queue.enqueue(task) == "skipped"
    assert queue.state(task.task_id) == "done"
    assert not list((queue.root / "pending").glob("*.json"))


def test_claim_failure_retry_then_completion(tmp_path: Path) -> None:
    queue = clip_queue.SharedClipQueue(tmp_path / "queue")
    task = _task(tmp_path)
    queue.enqueue(task)

    first = queue.claim("node-a")
    assert first is not None
    assert queue.fail(first, "transient encoder error") == "pending"

    second = queue.claim("node-b")
    assert second is not None
    assert second.attempts == 1
    task.output_path.parent.mkdir(parents=True)
    task.output_path.write_bytes(b"clip")
    queue.complete(second)

    assert queue.state(task.task_id) == "done"
    assert queue.wait_for_ready(task.task_id, timeout_seconds=0) == task.output_path


def test_only_one_worker_can_claim_one_pending_task(tmp_path: Path) -> None:
    queue = clip_queue.SharedClipQueue(tmp_path / "queue")
    task = _task(tmp_path)
    queue.enqueue(task)

    claims: list[clip_queue.ClipClaim | None] = []
    lock = threading.Lock()

    def claim(worker_name: str) -> None:
        acquired = queue.claim(worker_name)
        with lock:
            claims.append(acquired)

    threads = [threading.Thread(target=claim, args=(f"node-{index}",)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(item is not None for item in claims) == 1
    assert queue.state(task.task_id) == "claimed"


def test_worker_materializes_requested_output_without_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    queue = clip_queue.SharedClipQueue(tmp_path / "queue")
    task = _task(tmp_path)
    queue.enqueue(task)
    observed: dict[str, object] = {}

    @contextmanager
    def fake_worker_clip(**kwargs: object):
        observed.update(kwargs)
        output = Path(str(kwargs["output_path"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")
        yield output

    monkeypatch.setattr(clip_queue, "worker_clip", fake_worker_clip)
    result = clip_queue.run_clip_worker(
        queue,
        worker_name="node-a",
        max_tasks=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
    )

    assert result.completed == 1
    assert queue.state(task.task_id) == "done"
    assert observed["output_path"] == task.output_path
    assert observed["remove_on_exit"] is False

