#!/usr/bin/env bash
# On one BDY node: stage 8B to /tmp, then start a single vLLM smoke (GPU0:8110).
# Env: LOCAL_PUBLIC_MODELS_ROOT (default /tmp/memstrata_public_models)
#      SMOKE_GPU (default 0) SMOKE_PORT (default 8110)
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
mkdir -p "$LOG_DIR"

export LOCAL_PUBLIC_MODELS_ROOT="${LOCAL_PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
SMOKE_GPU="${SMOKE_GPU:-0}"
SMOKE_PORT="${SMOKE_PORT:-8110}"

exec > >(tee -a "$LOG_DIR/tmp_stage_smoke.log") 2>&1
echo "=== $(date -Is) tmp-stage+smoke host=$HOST gpu=$SMOKE_GPU port=$SMOKE_PORT ==="

# Kill leftover S3 / our vLLM ports only (do not touch gpu.py).
pkill -f 's3_segment_auto_review_revise.vlm_auto_review' 2>/dev/null || true
for p in 8110 8111 8112 8113 8114 8115 8116 8117; do
  pkill -f "vllm serve .* --port ${p}" 2>/dev/null || true
done
sleep 2
for p in 8110 8111 8112 8113 8114 8115 8116 8117; do
  pkill -9 -f "vllm serve .* --port ${p}" 2>/dev/null || true
done
tmux kill-session -t memstrata_s3_8b_fleet 2>/dev/null || true

echo "staging -> $LOCAL_PUBLIC_MODELS_ROOT"
bash "$MS/scripts/vmem_bench/fleet/lib/stage_qwen3vl8b_local.sh"
export PUBLIC_MODELS_ROOT="$LOCAL_PUBLIC_MODELS_ROOT"

export MODEL_SIZE=8B
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=$NO_PROXY

LOG="$LOG_DIR/vllm_smoke_g${SMOKE_GPU}_p${SMOKE_PORT}.log"
: > "$LOG"
echo "MODEL_PATH will be $PUBLIC_MODELS_ROOT/Qwen/Qwen3-VL-8B-Instruct"
nohup bash "$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh" "$SMOKE_GPU" "$SMOKE_PORT" >>"$LOG" 2>&1 &
echo $! > "$LOG_DIR/vllm_smoke_g${SMOKE_GPU}_p${SMOKE_PORT}.pid"
echo "vllm_pid=$(cat "$LOG_DIR/vllm_smoke_g${SMOKE_GPU}_p${SMOKE_PORT}.pid")"

echo "waiting for http://127.0.0.1:${SMOKE_PORT}/v1/models (900s)..."
deadline=$((SECONDS + 900))
ready=0
while (( SECONDS < deadline )); do
  if curl -sf -m 3 "http://127.0.0.1:${SMOKE_PORT}/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if (( ready == 1 )); then
  echo "SMOKE_OK port=$SMOKE_PORT"
  curl -s "http://127.0.0.1:${SMOKE_PORT}/v1/models" | head -c 300; echo
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
else
  echo "SMOKE_FAIL timeout"
  tail -n 80 "$LOG" || true
  exit 1
fi
echo "=== $(date -Is) smoke done ==="
