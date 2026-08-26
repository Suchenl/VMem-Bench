#!/usr/bin/env python3
"""Overwrite per-GPU health logs for an S5 worker tmux session."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _tmux_windows(session: str) -> set[str]:
    completed = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(completed.stdout.splitlines()) if completed.returncode == 0 else set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU ranks")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    windows = _tmux_windows(args.session)
    logs = args.run_root / "logs"
    results: list[dict[str, Any]] = []
    for shard, gpu in enumerate(gpus):
        log_path = logs / f"worker_shard{shard}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        tail = text[-4000:]
        window = f"s{shard}_g{gpu}"
        status = "running" if window in windows else ("exited" if "EXIT=" in tail else "not_started")
        payload: dict[str, Any] = {
            "service": "s5_crop_worker",
            "gpu_rank": gpu,
            "shard": shard,
            "checked_at_epoch": time.time(),
            "tmux_session": args.session,
            "tmux_window": window,
            "status": status,
            "source_log": str(log_path),
            "source_log_tail": tail,
        }
        _atomic_write(args.output / f"rank{gpu}.log", payload)
        results.append(payload)
    summary = {
        "service": "s5_crop_workers",
        "checked_at_epoch": time.time(),
        "session": args.session,
        "workers": results,
        "running_count": sum(item["status"] == "running" for item in results),
    }
    _atomic_write(args.output / "summary.log", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
