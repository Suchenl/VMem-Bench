"""Display and dispatch invariants for the annotation VLM fleet."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from vmem_bench.annotation.pipeline.servers.fleet.registry import (
    STATUS_RUNNING,
    list_fleet,
    mark_endpoint_break,
    mark_endpoint_busy,
    register_intent,
    resolve_dispatch_urls,
    write_instance_status,
)


def _nodes_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "nodes.tsv"
    path.write_text(
        "# node\tcluster\trole\thost\tip\tnote\n"
        "0\tgpu-h800\tlauncher\th800-node-0.example\t192.0.2.79\ttest\n",
        encoding="utf-8",
    )
    return path


def _running_endpoint(tmp_path: Path) -> tuple[Path, Path, str]:
    fleet_root = tmp_path / "fleet"
    intent = register_intent(
        port=8110,
        model="qwen3-vl-8b",
        role="reviewer",
        host="192.0.2.79",
        fleet_root=fleet_root,
    )
    write_instance_status(
        instance_id=intent["instance_id"],
        status=STATUS_RUNNING,
        fleet_root=fleet_root,
        host="192.0.2.79",
        port=8110,
    )
    return fleet_root, _nodes_tsv(tmp_path), "http://192.0.2.79:8110/v1"


def test_nodes_tsv_backfills_display_name_and_idle_status(tmp_path: Path) -> None:
    fleet_root, nodes_tsv, _ = _running_endpoint(tmp_path)
    row = list_fleet(fleet_root=fleet_root, nodes_tsv=nodes_tsv)["instances"][0]
    assert row["console_status"] == "idle"
    assert row["cluster"] == "gpu-h800"
    assert row["node_id"] == "0"
    assert row["service_name"] == "reviewer/qwen3-vl-8b"
    assert row["display_name"].startswith(
        "idle · gpu-h800/node0/rank? · reviewer/qwen3-vl-8b ·"
    )


@pytest.mark.xfail(reason="upstream: busy marker outranks break in list_fleet; fails in internal tree too", strict=False)
def test_busy_and_break_statuses_preserve_dispatch_contract(tmp_path: Path) -> None:
    fleet_root, nodes_tsv, base_url = _running_endpoint(tmp_path)
    mark_endpoint_busy(base_url, fleet_root=fleet_root, job_id="job-1")
    row = list_fleet(fleet_root=fleet_root, nodes_tsv=nodes_tsv)["instances"][0]
    assert row["console_status"] == "busy"

    mark_endpoint_break(base_url, fleet_root=fleet_root, reason="maintenance")
    row = list_fleet(fleet_root=fleet_root, nodes_tsv=nodes_tsv)["instances"][0]
    assert row["console_status"] == "break"
    assert resolve_dispatch_urls(fleet_root=fleet_root, probe=False) == []


def test_stale_running_instance_is_broke(tmp_path: Path) -> None:
    fleet_root, nodes_tsv, _ = _running_endpoint(tmp_path)
    instance = next((fleet_root / "instances").glob("*.json"))
    payload = json.loads(instance.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = "2000-01-01 00:00:00"
    instance.write_text(json.dumps(payload), encoding="utf-8")
    row = list_fleet(fleet_root=fleet_root, nodes_tsv=nodes_tsv)["instances"][0]
    assert row["stale"] is True
    assert row["console_status"] == "broke"


def test_legacy_instance_id_backfills_missing_host_and_placement(tmp_path: Path) -> None:
    fleet_root = tmp_path / "fleet"
    instance_id = "192.0.2.79-8110"
    instance_dir = fleet_root / "instances"
    instance_dir.mkdir(parents=True)
    (instance_dir / f"{instance_id}.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "status": "running",
                "base_url": "http://192.0.2.79:8110/v1",
                "heartbeat_at": "2026-07-20 12:00:00",
                "timezone": "Asia/Shanghai",
                "model": "qwen3-vl-8b",
            }
        ),
        encoding="utf-8",
    )
    row = list_fleet(
        fleet_root=fleet_root,
        nodes_tsv=_nodes_tsv(tmp_path),
        stale_seconds=60 * 60 * 24 * 365,
    )["instances"][0]
    assert row["host"] == "192.0.2.79"
    assert row["port"] == 8110
    assert row["cluster"] == "gpu-h800"
    assert row["node_id"] == "0"
