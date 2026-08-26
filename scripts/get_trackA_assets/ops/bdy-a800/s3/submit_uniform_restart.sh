#!/usr/bin/env bash
# Submit uniform 8-rank BDY restarts through the shared-filesystem worker queue.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
TGPU_FS="$REPO/scripts/tgpu_fs.py"
FS_ROOT="${FS_ROOT:-${UNSET_INTERNAL_PATH}"
RESTART="$REPO/benchmarks/MemStrata/scripts/vmem_bench/ops/bdy-a800/s3/restart_all_ranks.sh"

# Use the routable host names from nodes.tsv, not the 10.252.* control IPs.
declare -A NODE_HOST=(
  [0]=a800bcctest0033-bd.bce.example.org
  [1]=a800bcctest0075-bd.bce.example.org
  [2]=a800bcctest0039-bd.bce.example.org
  [3]=a800bcctest0104-bd.bce.example.org
)

for node in 0 1 2 3; do
  host="${NODE_HOST[$node]}"
  job_id="bdy-uniform-host-restart-n${node}-$(date +%s)"
  cmd="export FLEET_CLUSTER=bdy-a800 FLEET_NODE_ID=${node} FLEET_ADVERTISE_HOST=${host}; bash '${RESTART}'"
  python3 "$TGPU_FS" --root "$FS_ROOT" run \
    --cluster bdy-a800 --node "$node" --cwd "$REPO" \
    --timeout 180 --wait 0 --job-id "$job_id" --cmd "$cmd"
done
