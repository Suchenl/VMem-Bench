#!/usr/bin/env bash
# Launch Qwen3-VL-8B on all GPUs of one H800 node, registering into the shared fleet.
#
# Required env (set by submit_8b_fleet.sh):
#   FLEET_CLUSTER     e.g. kml-h800
#   FLEET_NODE_ID     e.g. 0
#   FLEET_ADVERTISE_HOST  reachable IP/hostname for Console (e.g. 10.83.1.79)
#
# Optional:
#   GPUS=0,1,2,3,4,5,6,7   BASE_PORT=8110
#
# Run ON the GPU node (via tgpu), not on the IDE machine.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
START_VLLM="$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh"
# shellcheck source=/dev/null
source "$MS/src/vmem_bench/annotation/pipeline/servers/env_no_proxy.sh" || true

: "${FLEET_CLUSTER:?set FLEET_CLUSTER (e.g. kml-h800)}"
: "${FLEET_NODE_ID:?set FLEET_NODE_ID (e.g. 0)}"
: "${FLEET_ADVERTISE_HOST:?set FLEET_ADVERTISE_HOST to a Console-reachable IP}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8110}"
HOST="$(hostname -s)"
LOG_DIR="${LOG_DIR:-$MS/runtime/services/vlm_fleet/logs/${FLEET_CLUSTER}/node${FLEET_NODE_ID}}"
SESSION="${SESSION:-memstrata_h800_8b_n${FLEET_NODE_ID}}"
mkdir -p "$LOG_DIR"

export MODEL_SIZE=8B
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b}"
export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"
# vLLM accepts a single directory; never pass comma lists.
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export FLEET_CLUSTER FLEET_NODE_ID FLEET_ADVERTISE_HOST
export PYTHONPATH="$MS/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"

# Stop previous 8B fleet on these ports only (do not touch unrelated gpu.py).
for i in "${!GPU_ARR[@]}"; do
  port=$((BASE_PORT + i))
  pkill -f "vllm serve .* --port ${port}" 2>/dev/null || true
  pkill -f "fleet.supervise .* --port ${port}" 2>/dev/null || true
done
sleep 2

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" -n "boot" "echo fleet ${FLEET_CLUSTER} node${FLEET_NODE_ID}; sleep infinity"

idx=0
URLS=()
for gpu in "${GPU_ARR[@]}"; do
  gpu="${gpu// /}"
  [[ -n "$gpu" ]] || continue
  port=$((BASE_PORT + idx))
  win="g${gpu}_p${port}"
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  : >"$log"
  tmux new-window -t "$SESSION" -n "$win" \
    "export MODEL_SIZE=8B MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL \
     SERVED_MODEL_NAME=$SERVED_MODEL_NAME PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT \
     ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH \
     FLEET_CLUSTER=$FLEET_CLUSTER FLEET_NODE_ID=$FLEET_NODE_ID \
     FLEET_ADVERTISE_HOST=$FLEET_ADVERTISE_HOST FLEET_GPU_RANK=$gpu \
     PYTHONPATH='$MS/src'; \
     bash '$START_VLLM' $gpu $port >>'$log' 2>&1; echo EXIT:\$? >>'$log'; sleep 5"
  echo "scheduled cluster=$FLEET_CLUSTER node=$FLEET_NODE_ID rank=$gpu port=$port log=$log"
  URLS+=("http://${FLEET_ADVERTISE_HOST}:${port}/v1")
  idx=$((idx + 1))
done

printf '%s\n' "${URLS[@]}" > "$LOG_DIR/endpoints.txt"
echo "$FLEET_ADVERTISE_HOST" > "$LOG_DIR/advertise_host.txt"
echo "session=$SESSION host=$HOST cluster=$FLEET_CLUSTER node=$FLEET_NODE_ID endpoints=${#URLS[@]}"
echo "fleet root: $MS/runtime/services/vlm_fleet"
