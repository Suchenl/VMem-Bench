"""Supervise a VLM process and write fleet instance status to the shared path.

Launcher writes intent; this supervisor writes ``starting`` / ``running`` /
``terminated`` (+ heartbeats) so a dead process cannot look healthy forever.

Usage::

    python -m vmem_bench.annotation.pipeline.servers.fleet.supervise \\
      --gpu 0 --port 8110 --model qwen3-vl-32b -- \\
      bash path/to/start_qwen32_vllm.sh 0 8110
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from vmem_bench.annotation.pipeline.servers.fleet.registry import (
    STATUS_RUNNING,
    STATUS_STARTING,
    STATUS_TERMINATED,
    advertise_host,
    base_url_for,
    default_fleet_root,
    make_instance_id,
    probe_models,
    register_intent,
    write_instance_status,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES value for the intent record")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "qwen3-vl-32b"))
    parser.add_argument("--role", default="reviewer", help="reviewer | grounder | any")
    parser.add_argument("--host", default="", help="Advertise host (default: FLEET_ADVERTISE_HOST / fqdn)")
    parser.add_argument("--cluster", default=os.environ.get("FLEET_CLUSTER", ""), help="e.g. gpu-h800")
    parser.add_argument("--node", default=os.environ.get("FLEET_NODE_ID", ""), help="tgpu node id, e.g. 0")
    parser.add_argument(
        "--gpu-rank",
        default=os.environ.get("FLEET_GPU_RANK", ""),
        help="Physical GPU rank on the node (0..N-1)",
    )
    parser.add_argument(
        "--fleet-root",
        type=Path,
        default=None,
        help="Override shared fleet root (default: MemStrata/runtime/services/vlm_fleet)",
    )
    parser.add_argument("--probe-interval", type=float, default=5.0)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after -- ; e.g. -- bash start_qwen32_vllm.sh 0 8110",
    )
    args = parser.parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing supervised command after --")
    args.command = cmd
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fleet_root = Path(args.fleet_root).resolve() if args.fleet_root else default_fleet_root()
    host = str(args.host or advertise_host())
    port = int(args.port)
    instance_id = make_instance_id(host=host, port=port)
    advertise_url = base_url_for(host=host, port=port)
    local_probe_url = base_url_for(host="127.0.0.1", port=port)

    intent = register_intent(
        port=port,
        model=str(args.model),
        gpu=str(args.gpu),
        role=str(args.role),
        host=host,
        command=" ".join(args.command),
        fleet_root=fleet_root,
        cluster=str(args.cluster or ""),
        node_id=str(args.node or ""),
        gpu_rank=args.gpu_rank if str(args.gpu_rank).strip() != "" else None,
    )
    placement = {
        "cluster": intent.get("cluster") or "",
        "node_id": intent.get("node_id") if intent.get("node_id") not in (None, "") else "",
        "gpu_rank": intent.get("gpu_rank") if intent.get("gpu_rank") not in (None, "") else "",
    }
    write_instance_status(
        instance_id=instance_id,
        status=STATUS_STARTING,
        fleet_root=fleet_root,
        host=host,
        port=port,
        base_url=advertise_url,
        model=str(args.model),
        gpu=str(args.gpu),
        role=str(args.role),
        intent_id=intent["instance_id"],
        supervisor_pid=os.getpid(),
        **placement,
    )

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(args.command, env=env)  # noqa: S603 — operator-supplied launcher
    write_instance_status(
        instance_id=instance_id,
        status=STATUS_STARTING,
        fleet_root=fleet_root,
        pid=proc.pid,
        base_url=advertise_url,
    )

    stopping = {"value": False}

    def _stop(signum: int, _frame: object) -> None:
        stopping["value"] = True
        if proc.poll() is None:
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    last_heartbeat = 0.0
    became_running = False
    exit_code = 1
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                exit_code = int(rc)
                break
            if stopping["value"]:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                exit_code = int(proc.returncode or 1)
                break

            now = time.monotonic()
            healthy = probe_models(local_probe_url) or probe_models(advertise_url)
            if healthy:
                due = (not became_running) or ((now - last_heartbeat) >= float(args.heartbeat_interval))
                became_running = True
                if due:
                    write_instance_status(
                        instance_id=instance_id,
                        status=STATUS_RUNNING,
                        fleet_root=fleet_root,
                        pid=proc.pid,
                        base_url=advertise_url,
                        model=str(args.model),
                        gpu=str(args.gpu),
                        role=str(args.role),
                        probe_ok=True,
                        **placement,
                    )
                    last_heartbeat = now
            elif became_running and (now - last_heartbeat) >= float(args.heartbeat_interval):
                # Process still up but /models failed — keep running stamp; stale TTL covers death.
                write_instance_status(
                    instance_id=instance_id,
                    status=STATUS_RUNNING,
                    fleet_root=fleet_root,
                    pid=proc.pid,
                    base_url=advertise_url,
                    probe_ok=False,
                    **placement,
                )
                last_heartbeat = now
            time.sleep(float(args.probe_interval))
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            exit_code = int(proc.returncode if proc.returncode is not None else exit_code)
        write_instance_status(
            instance_id=instance_id,
            status=STATUS_TERMINATED,
            fleet_root=fleet_root,
            pid=proc.pid if proc.pid else None,
            base_url=advertise_url,
            exit_code=exit_code,
            reason="process_exit" if not stopping["value"] else "signal_stop",
            **placement,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
