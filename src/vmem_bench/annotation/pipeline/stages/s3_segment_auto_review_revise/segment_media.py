"""Ephemeral per-worker segment video media for S3 review."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from vmem_bench.common.media import ffmpeg_bin


def _ffmpeg_failure_detail(
    *,
    command: list[str],
    attempt: int,
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    """Keep the useful start and end of ffmpeg diagnostics."""
    def excerpt(value: str) -> str:
        text = str(value or "").strip()
        if len(text) <= 1_200:
            return text
        return f"{text[:400]}\n... <truncated> ...\n{text[-800:]}"

    parts = [
        f"attempt={attempt}",
        f"returncode={returncode}",
        f"command={command!r}",
    ]
    if stdout.strip():
        parts.append(f"stdout={excerpt(stdout)}")
    if stderr.strip():
        parts.append(f"stderr={excerpt(stderr)}")
    return "; ".join(parts)


@contextmanager
def worker_clip(
    *,
    source_video: Path,
    cache_root: Path,
    worker_id: str,
    start_seconds: float,
    end_seconds: float,
    cut_semaphore: threading.Semaphore | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.25,
    timing: dict[str, float] | None = None,
    output_path: Path | None = None,
    remove_on_exit: bool = True,
) -> Iterator[Path]:
    """Materialize one segment at a stable worker-local path, then remove it.

    The caller may reuse the same ``worker_id`` for every segment.  This avoids
    retaining a full duplicate video corpus under ``tmp/`` while satisfying
    video-url VLM endpoints that require an independent file.

    ``cut_semaphore`` bounds only the local transcode—not downstream VLM
    inference—so a large endpoint pool cannot stampede a shared source video.
    When ``timing`` is provided, ``queue_seconds`` records waits for that
    semaphore and ``encode_seconds`` records the actual ffmpeg invocation.
    """
    if end_seconds <= start_seconds:
        raise ValueError(f"invalid segment range [{start_seconds}, {end_seconds}]")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    clip = Path(output_path) if output_path is not None else cache_root / worker_id / "segment.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{end_seconds - start_seconds:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        str(clip),
    ]
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        clip.unlink(missing_ok=True)
        queue_started = time.perf_counter()
        if cut_semaphore is None:
            encode_started = queue_started
            completed = subprocess.run(cmd, capture_output=True, text=True)
        else:
            with cut_semaphore:
                encode_started = time.perf_counter()
                completed = subprocess.run(cmd, capture_output=True, text=True)
        if timing is not None:
            timing["queue_seconds"] = timing.get("queue_seconds", 0.0) + (
                encode_started - queue_started
            )
            timing["encode_seconds"] = timing.get("encode_seconds", 0.0) + (
                time.perf_counter() - encode_started
            )
        if completed.returncode == 0 and clip.is_file():
            break
        failures.append(
            _ffmpeg_failure_detail(
                command=cmd,
                attempt=attempt,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
        if attempt < max_attempts:
            time.sleep(retry_delay_seconds * attempt)
    else:
        raise RuntimeError(
            f"ffmpeg segment cut failed after {max_attempts} attempts: "
            + "\n---\n".join(failures)
        )
    try:
        yield clip
    finally:
        if remove_on_exit:
            clip.unlink(missing_ok=True)


def cleanup_cache(cache_root: Path) -> None:
    """Delete S3's ephemeral clip cache after normal or failed completion."""
    shutil.rmtree(cache_root, ignore_errors=True)
