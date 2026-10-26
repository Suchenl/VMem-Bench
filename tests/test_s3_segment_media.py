"""S3 segment-media retry and concurrency tests."""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise import (
    segment_media,
)


def test_worker_clip_retries_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 75, stdout="first stdout", stderr="first stderr")
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(segment_media.subprocess, "run", fake_run)
    with segment_media.worker_clip(
        source_video=source,
        cache_root=tmp_path / "cache",
        worker_id="worker",
        start_seconds=0,
        end_seconds=1,
        retry_delay_seconds=0,
    ) as clip:
        assert clip.read_bytes() == b"clip"
    assert calls == 2
    assert not clip.exists()


def test_worker_clip_failure_preserves_actionable_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 137, stdout="out", stderr="fatal encoder failure")

    monkeypatch.setattr(segment_media.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"after 2 attempts.*returncode=137.*fatal encoder failure"):
        with segment_media.worker_clip(
            source_video=source,
            cache_root=tmp_path / "cache",
            worker_id="worker",
            start_seconds=0,
            end_seconds=1,
            max_attempts=2,
            retry_delay_seconds=0,
        ):
            pass


def test_worker_clip_semaphore_limits_only_parallel_encodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    state_lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        Path(command[-1]).write_bytes(b"clip")
        with state_lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(segment_media.subprocess, "run", fake_run)
    semaphore = threading.BoundedSemaphore(1)

    def cut(worker_id: str) -> None:
        with segment_media.worker_clip(
            source_video=source,
            cache_root=tmp_path / "cache",
            worker_id=worker_id,
            start_seconds=0,
            end_seconds=1,
            cut_semaphore=semaphore,
        ):
            pass

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(cut, ("a", "b", "c")))
    assert maximum == 1
