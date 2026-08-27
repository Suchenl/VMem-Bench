"""Overwrite/delete semantics for per-service fleet health logs."""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline.servers.fleet import health_monitor


def test_refresh_overwrites_online_log_and_deletes_lost_service(tmp_path: Path) -> None:
    row = {
        "cluster": "gpu-h800",
        "node_id": "0",
        "gpu_rank": 3,
        "online": True,
        "reachable": True,
        "console_status": "idle",
        "display_name": "idle · gpu-h800/node0/rank3 · reviewer/qwen3-vl-8b",
        "base_url": "http://10.83.1.79:8113/v1",
        "model": "qwen3-vl-8b",
        "role": "reviewer",
        "heartbeat_at": "2026-07-20 12:00:00",
        "workload": None,
    }
    original = health_monitor.list_fleet
    try:
        health_monitor.list_fleet = lambda **_kwargs: {"online_count": 1, "instances": [row]}  # type: ignore[assignment]
        health_monitor.refresh_once(
            fleet_root=tmp_path / "fleet",
            output_root=tmp_path / "health",
            probe_timeout=1.0,
        )
        path = tmp_path / "health" / "gpu-h800" / "node0" / "rank3.log"
        assert path.is_file()
        orphan = tmp_path / "health" / "gpu-h800" / "node0" / "rank7.log"
        orphan.write_text("old", encoding="utf-8")

        offline = {**row, "online": False, "console_status": "broke"}
        health_monitor.list_fleet = lambda **_kwargs: {"online_count": 0, "instances": [offline]}  # type: ignore[assignment]
        health_monitor.refresh_once(
            fleet_root=tmp_path / "fleet",
            output_root=tmp_path / "health",
            probe_timeout=1.0,
        )
        assert not path.exists()
        assert not orphan.exists()
    finally:
        health_monitor.list_fleet = original  # type: ignore[assignment]


def test_terminal_record_is_pruned_with_its_health_log(tmp_path: Path) -> None:
    fleet_root = tmp_path / "fleet"
    instance_id = "host-8110"
    for directory in ("intents", "instances"):
        path = fleet_root / directory / f"{instance_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    row = {
        "instance_id": instance_id,
        "cluster": "bdy-a800",
        "node_id": "0",
        "gpu_rank": 0,
        "online": False,
        "status": "terminated",
        "stale": False,
        "starting_stale": False,
    }
    original = health_monitor.list_fleet
    try:
        health_monitor.list_fleet = lambda **_kwargs: {"online_count": 0, "instances": [row]}  # type: ignore[assignment]
        health_monitor.refresh_once(
            fleet_root=fleet_root,
            output_root=tmp_path / "health",
            probe_timeout=1.0,
        )
        assert not (fleet_root / "intents" / f"{instance_id}.json").exists()
        assert not (fleet_root / "instances" / f"{instance_id}.json").exists()
    finally:
        health_monitor.list_fleet = original  # type: ignore[assignment]
