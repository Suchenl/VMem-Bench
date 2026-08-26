# VMem-Bench operational scripts

This directory is organized by responsibility. Use only the paths below; the
previous flat script layout has been removed.

| Area | Canonical entry points | Use |
|---|---|---|
| `core/` | `run_annotation.sh`, `run_s5_crops_skip_s3.py` | Stable annotation and S5 worker drivers |
| `servers/` | `start_annotation_vllm.sh` | One-endpoint VLM (Qwen3-VL) supervisor-backed launcher |
| `fleet/h800/` | `submit_8b_fleet.sh`, `launch_8b_fleet.sh` | Production 2-node × 8-GPU Qwen3-VL-8B fleet |
| `fleet/` | `healthcheck_vllm_ranks.py`, `healthcheck_s5_workers.py` | Overwrite-only real-task rank health checks |
| `fleet/lib/` | `stage_qwen3vl8b_local.sh` | Reusable Qwen3-VL staging helper for training nodes only |
| `ops/bdy-a800/s3/` | `submit_tmp_fleets.sh`, `supervise_fleets_then_s3.sh`, `continue_fleet.sh` | BDY S3 campaign fleet and recovery runbooks |
| `ops/bdy-a800/s5/` | `launch_8b_fleet.sh`, `launch_workers.sh`, `restart_node.sh` | BDY S5 skip-S3 campaign |
| `ops/kml-a800/s5/` | `launch_workers.sh`, `run_american_beauty.sh` | KML A800 S5 campaign |
| `maintenance/` | `build_gold_from_vlm_output.py`, `migrate_flat_layout.py`, `regen_crops_from_checkpoints.py`, `report_gold_readiness.py` | Gold, readiness reporting and data-layout maintenance |
| `_archive/` | historical diagnostics only | Never use for production or paper-facing gold |

## H800 production fleet

From the development machine:

```bash
bash benchmarks/MemStrata/scripts/vmem_bench/fleet/h800/submit_8b_fleet.sh
```

The launcher preserves one Qwen3-VL-8B endpoint per H800 GPU, registers each
endpoint with its cluster/node/rank, and keeps processes under node-local tmux
sessions. Do not replace this path with an ad-hoc vLLM command.

## Canonical service startup paths

| Service | Script | Scope |
|---|---|---|
| One VLM rank | `servers/start_annotation_vllm.sh <gpu> <port>` | Reusable supervisor-backed Qwen3-VL endpoint |
| H800 16-rank Fleet | `fleet/h800/submit_8b_fleet.sh` | Login-side two-node submission |
| One H800/KML node Fleet | `fleet/h800/launch_8b_fleet.sh` | Node-side 8-rank launcher; set `FLEET_*`, `SESSION`, and `LOG_DIR` |
| BDY 32-rank restart | `ops/bdy-a800/s3/submit_uniform_restart.sh` | Canonical task-queue submitter; uses routable nodes.tsv host names |
| BDY 8-rank node restart | `ops/bdy-a800/s3/restart_all_ranks.sh` | Node-side worker; clears stale MemStrata VLLM children before starting all ranks |
| KML S5 workers | `ops/kml-a800/s5/bootstrap_workers.sh` | Wait for VLM pool, then launch configured worker GPUs |

## BDY uniform restart

To rebuild all BDY nodes consistently, use
`ops/bdy-a800/s3/submit_uniform_restart.sh`. It sends the node-side
`restart_all_ranks.sh` through `tgpu_fs` with routable nodes.tsv host names,
not non-routable control IPs. The node script clears VLLM API, supervisor, and
EngineCore children before scheduling all ranks, preventing residual port,
engine, and routing conflicts.

## Safety boundaries

- `fleet/h800/` is the annotation-console fleet. Do not mix it with BDY
  campaign scripts or stop its tmux sessions from a BDY recovery command.
- Never stop `gpu.py`, `occupy.sh`, `dev_occupy.sh`, or another platform
  keepalive process. The BDY restart only targets MemStrata VLLM API,
  supervisor, and EngineCore processes.
- `ops/` scripts are campaign-specific. Their run directories, GPU maps and
  staging choices are part of their contract.
- Training-node `/dev/shm` staging is allowed only through the relevant
  training-node runbook; never copy model weights into development-machine
  `/dev/shm`.
- Gold maintenance scripts do not make a candidate paper-ready. The S7 strict
  lint and human-review gates remain mandatory.
