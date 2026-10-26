"""Run one named worker against a shared S3 clip-materialization queue."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.clip_queue import (
    SharedClipQueue,
    run_clip_worker,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=15.0)
    parser.add_argument("--stale-after-seconds", type=float, default=15 * 60)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit immediately when no task is currently pending.",
    )
    args = parser.parse_args()
    if not args.worker_name.strip():
        parser.error("--worker-name must not be empty")

    queue = SharedClipQueue(
        args.queue_root,
        stale_after_seconds=args.stale_after_seconds,
    )
    result = run_clip_worker(
        queue,
        worker_name=args.worker_name,
        poll_interval_seconds=args.poll_interval_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        max_tasks=args.max_tasks,
        once=args.once,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
