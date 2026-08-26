#!/usr/bin/env bash
# Restart one BDY node as a uniform eight-rank Qwen3-VL-8B Fleet.
#
# Required environment: FLEET_CLUSTER, FLEET_NODE_ID, FLEET_ADVERTISE_HOST.
# This script starts all rank windows together after local weight staging.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
STOP="$MS/scripts/vmem_bench/ops/bdy-a800/s3/stop_node_vllm.sh"
STAGE="$MS/scripts/vmem_bench/fleet/lib/stage_qwen3vl8b_local.sh"
START="$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh"

: "${FLEET_CLUSTER:?set FLEET_CLUSTER}"
: "${FLEET_NODE_ID:?set FLEET_NODE_ID}"
: "${FLEET_ADVERTISE_HOST:?set FLEET_ADVERTISE_HOST}"
case "$FLEET_ADVERTISE_HOST" in
  10.252.*)
    echo "BDY FLEET_ADVERTISE_HOST must use the routable nodes.tsv host, not control IP $FLEET_ADVERTISE_HOST" >&2
    exit 2
    ;;
esac

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8110}"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
SESSION="${SESSION:-memstrata_bdy_8b_n${FLEET_NODE_ID}}"
LOCAL_PUBLIC_MODELS_ROOT="${LOCAL_PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
mkdir -p "$LOG_DIR"

export FLEET_CLUSTER FLEET_NODE_ID FLEET_ADVERTISE_HOST
export PUBLIC_MODELS_ROOT="$LOCAL_PUBLIC_MODELS_ROOT"
export MODEL_SIZE=8B MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,0.0.0.0"

bash "$STOP"
bash "$STAGE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" -n boot "sleep infinity"

IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
idx=0
for raw_gpu in "${GPU_ARR[@]}"; do
  gpu="${raw_gpu// /}"
  [[ -n "$gpu" ]] || continue
  port=$((BASE_PORT + idx))
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  : > "$log"
  tmux new-window -t "$SESSION" -n "g${gpu}_p${port}" \
    "export FLEET_CLUSTER='$FLEET_CLUSTER' FLEET_NODE_ID='$FLEET_NODE_ID' FLEET_ADVERTISE_HOST='$FLEET_ADVERTISE_HOST' FLEET_GPU_RANK='$gpu' PUBLIC_MODELS_ROOT='$LOCAL_PUBLIC_MODELS_ROOT' MODEL_SIZE=8B MAX_MODEL_LEN='$MAX_MODEL_LEN' GPU_MEM_UTIL='$GPU_MEM_UTIL' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' ALLOWED_LOCAL_MEDIA_PATH='$ALLOWED_LOCAL_MEDIA_PATH' NO_PROXY='$NO_PROXY' no_proxy='$no_proxy'; bash '$START' '$gpu' '$port' >>'$log' 2>&1; echo EXIT:\$? >>'$log'; sleep 5"
  echo "scheduled node=$FLEET_NODE_ID rank=$gpu port=$port log=$log"
  idx=$((idx + 1))
done

echo "session=$SESSION host=$HOST ranks=${#GPU_ARR[@]}"
