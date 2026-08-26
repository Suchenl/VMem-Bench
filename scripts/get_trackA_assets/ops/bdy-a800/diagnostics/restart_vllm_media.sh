#!/usr/bin/env bash
# Restart one 8B vLLM on BDY with --allowed-local-media-path (safe for tgpu_fs; no pkill pattern in job cmd).
set -euo pipefail
REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
HOST="$(hostname -s)"
LOG_DIR="$MS/data/_runs/s3_bdy_8b/logs/$HOST"
mkdir -p "$LOG_DIR"
GPU="${1:-1}"
PORT="${2:-8110}"
LOG="$LOG_DIR/vllm_media_g${GPU}_p${PORT}.log"

export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
export MODEL_SIZE=8B
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=$NO_PROXY

echo "=== $(date -Is) restart media vLLM host=$HOST gpu=$GPU port=$PORT root=$PUBLIC_MODELS_ROOT ==="

# Kill only matching python vllm children by reading /proc (avoid pkill matching this script's argv).
while read -r pid; do
  [[ -n "$pid" ]] || continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"/envs/vllm/bin/vllm serve"*"--port ${PORT}"*)
      echo "kill $pid"
      kill "$pid" 2>/dev/null || true
      ;;
  esac
done < <(ls /proc | grep -E '^[0-9]+$' || true)
sleep 2
while read -r pid; do
  [[ -n "$pid" ]] || continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"/envs/vllm/bin/vllm serve"*"--port ${PORT}"*)
      kill -9 "$pid" 2>/dev/null || true
      ;;
  esac
done < <(ls /proc | grep -E '^[0-9]+$' || true)
sleep 1

: > "$LOG"
nohup bash "$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh" "$GPU" "$PORT" >>"$LOG" 2>&1 &
echo $! > "$LOG_DIR/vllm_media_g${GPU}_p${PORT}.pid"
echo "STARTED_PID=$(cat "$LOG_DIR/vllm_media_g${GPU}_p${PORT}.pid")"
sleep 3
ps -p "$(cat "$LOG_DIR/vllm_media_g${GPU}_p${PORT}.pid")" -o pid,etime,stat,cmd || true
head -c 400 "$LOG" || true
echo "=== $(date -Is) launch submitted; wait for /v1/models separately ==="
