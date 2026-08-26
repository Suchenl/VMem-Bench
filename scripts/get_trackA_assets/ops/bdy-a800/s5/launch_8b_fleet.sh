#!/usr/bin/env bash
# Launch a local Qwen3-VL-8B vLLM fleet on one BDY A800 node (8 GPUs).
# GPUs 0..5 → 6× 8B picker endpoints :8110-8115
# GPUs 6..7 reserved for SAM3 crop workers.
#
# Usage (on the BDY node, via tgpu_fs):
#   bash .../ops/bdy-a800/s5/launch_8b_fleet.sh
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
RUN_ROOT="${RUN_ROOT:-$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/bdy}"
START_VLLM="$REPO/benchmarks/MemStrata/scripts/vmem_bench/servers/start_annotation_vllm.sh"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
mkdir -p "$LOG_DIR"

export MODEL_SIZE=8B
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b}"
export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,0.0.0.0"

# Stop previous fleet on these ports (do NOT touch gpu.py).
for port in 8110 8111 8112 8113 8114 8115; do
  pkill -f "vllm serve .* --port ${port}" 2>/dev/null || true
done
sleep 2

SESSION="memstrata_s5_8b_fleet"
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" -n "boot" "echo fleet; sleep infinity"

idx=0
for gpu in 0 1 2 3 4 5; do
  port=$((8110 + idx))
  win="g${gpu}_p${port}"
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  pidf="$LOG_DIR/vllm_g${gpu}_p${port}.pid"
  tmux new-window -t "$SESSION" -n "$win" \
    "export MODEL_SIZE=8B MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL SERVED_MODEL_NAME=$SERVED_MODEL_NAME PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT; \
     bash '$START_VLLM' $gpu $port >'$log' 2>&1 & echo \$! >'$pidf'; wait"
  echo "scheduled $win -> $log"
  idx=$((idx + 1))
done

echo "fleet session=$SESSION host=$HOST ports=8110-8115 (GPUs 0-5); GPUs 6-7 free for SAM3"
echo "$HOST" > "$LOG_DIR/fleet_host.txt"
printf '%s\n' \
  "http://127.0.0.1:8110/v1" \
  "http://127.0.0.1:8111/v1" \
  "http://127.0.0.1:8112/v1" \
  "http://127.0.0.1:8113/v1" \
  "http://127.0.0.1:8114/v1" \
  "http://127.0.0.1:8115/v1" > "$LOG_DIR/endpoints.txt"
