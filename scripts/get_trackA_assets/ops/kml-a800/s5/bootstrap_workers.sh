#!/usr/bin/env bash
# Wait for the remote VLM pool, then start the node-local S5 worker shards.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
POOL="${POOL:-http://10.82.120.211:8110/v1}"
WORKER_SCRIPT="$REPO/benchmarks/MemStrata/scripts/vmem_bench/ops/kml-a800/s5/launch_workers.sh"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
deadline=$((SECONDS + READY_TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
  if curl --noproxy "*" -fsS -m 5 "${POOL%/}/models" >/dev/null; then
    exec env \
      RUN="${RUN:-$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/kml_a800_node1_6shard}" \
      POOL="$POOL" \
      MODEL="${MODEL:-qwen3-vl-8b}" \
      WORKER_GPUS="${WORKER_GPUS:-0,1,2,3,4,7}" \
      SESSION="${SESSION:-memstrata_s5_kml_node1_6workers}" \
      STOP_EXISTING_WORKERS="${STOP_EXISTING_WORKERS:-0}" \
      bash "$WORKER_SCRIPT"
  fi
  sleep 5
done

echo "VLM_NOT_READY pool=${POOL} timeout=${READY_TIMEOUT_SECONDS}s" >&2
exit 2
