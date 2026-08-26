#!/usr/bin/env bash
# From the IDE / login machine: start Qwen3-VL-8B on all 16 H800 GPUs (2 nodes × 8).
#
# Usage:
#   bash benchmarks/MemStrata/scripts/vmem_bench/fleet/h800/submit_8b_fleet.sh
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
TGPU="${TGPU:-${UNSET_INTERNAL_PATH}"
CLUSTER="${CLUSTER:-kml-h800}"
LAUNCH="$REPO/benchmarks/MemStrata/scripts/vmem_bench/fleet/h800/launch_8b_fleet.sh"

# node_id -> advertise IP (Console must reach these; from tgpu nodes.tsv)
declare -A NODE_IP=(
  [0]=10.83.1.79
  [1]=10.83.3.86
)

NODES="${NODES:-0,1}"
IFS=',' read -r -a NODE_ARR <<< "$NODES"

for node in "${NODE_ARR[@]}"; do
  node="${node// /}"
  ip="${NODE_IP[$node]:-}"
  [[ -n "$ip" ]] || { echo "unknown node $node (add to NODE_IP map)" >&2; exit 2; }
  echo "=== submit cluster=$CLUSTER node=$node advertise=$ip ==="
  "$TGPU" -c "$CLUSTER" -node "$node" bash -lc "
    set -euo pipefail
    export FLEET_CLUSTER='$CLUSTER' FLEET_NODE_ID='$node' FLEET_ADVERTISE_HOST='$ip'
    export REPO='$REPO' PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}
    bash '$LAUNCH'
  "
done

echo
echo "Submitted. Watch:"
echo "  python -m vmem_bench.annotation.pipeline.servers.fleet list"
echo "  curl -s http://127.0.0.1:7864/api/fleet | python3 -m json.tool | head"
echo "Hard-refresh the annotation console to see cluster/node/rank."
