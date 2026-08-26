"""Continuously refresh one liveness log per registered VLM service."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.servers.fleet.registry import (
    default_fleet_root,
    list_fleet,
)


def _safe_component(value: object, fallback: str) -> str:
    text = str(value if value not in (None, "") else fallback)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _log_path(root: Path, row: dict[str, Any]) -> Path:
    cluster = _safe_component(row.get("cluster"), "unmapped")
    node = _safe_component(row.get("node_id"), "unknown")
    rank = _safe_component(row.get("gpu_rank"), "unknown")
    return root / cluster / f"node{node}" / f"rank{rank}.log"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _prune_lost_registry_record(fleet_root: Path, row: dict[str, Any]) -> bool:
    """Remove terminal/stale records so the console has no phantom history."""
    terminal = row.get("status") == "terminated"
    stale = bool(row.get("stale") or row.get("starting_stale"))
    if not (terminal or stale):
        return False
    instance_id = str(row.get("instance_id") or "")
    if not instance_id:
        return False
    for directory in ("intents", "instances", "workloads", "breaks"):
        (fleet_root / directory / f"{instance_id}.json").unlink(missing_ok=True)
    return True


def refresh_once(*, fleet_root: Path, output_root: Path, probe_timeout: float) -> dict[str, int]:
    """Refresh lifecycle logs, deleting a rank log as soon as its service is lost."""
    snapshot = list_fleet(
        fleet_root=fleet_root,
        # Every supervisor already probes its local vLLM endpoint and writes a
        # heartbeat. Re-probing a mixed 40+ endpoint fleet serially here would
        # turn a 20-second monitor into multi-minute stale snapshots.
        probe=False,
        probe_timeout=probe_timeout,
    )
    updated = 0
    removed = 0
    pruned = 0
    expected_paths: set[Path] = set()
    for row in snapshot["instances"]:
        path = _log_path(output_root, row)
        if not row.get("online"):
            if path.exists():
                path.unlink()
                removed += 1
            if _prune_lost_registry_record(fleet_root, row):
                pruned += 1
            continue
        expected_paths.add(path)
        payload: dict[str, Any] = {
            "service": "vlm_reviewer",
            "checked_at_epoch": time.time(),
            "ok": bool(row.get("online")),
            "console_status": row.get("console_status"),
            "display_name": row.get("display_name"),
            "base_url": row.get("base_url"),
            "cluster": row.get("cluster"),
            "node_id": row.get("node_id"),
            "gpu_rank": row.get("gpu_rank"),
            "model": row.get("model"),
            "role": row.get("role"),
            "heartbeat_at": row.get("heartbeat_at"),
            "workload": row.get("workload"),
        }
        _atomic_write(path, payload)
        updated += 1
    # A rank can disappear entirely from the registry (e.g. legacy intent
    # cleanup), in which case no row above would visit its old log path.
    for orphan in output_root.glob("*/node*/rank*.log"):
        if orphan not in expected_paths:
            orphan.unlink(missing_ok=True)
            removed += 1
    return {
        "updated": updated,
        "removed": removed,
        "pruned": pruned,
        "online": snapshot["online_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-root", type=Path, default=default_fleet_root())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    fleet_root = args.fleet_root.resolve()
    output_root = (args.output_root or (fleet_root / "health")).resolve()
    while True:
        summary = refresh_once(
            fleet_root=fleet_root,
            output_root=output_root,
            probe_timeout=args.probe_timeout,
        )
        print(json.dumps(summary), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
