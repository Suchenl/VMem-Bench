#!/usr/bin/env bash
# After main fleet is up, add any missing GPUs (e.g. node0 GPU0 -> :8117).
# Waits for boot_bdy_tmp_fleet to finish, then starts extras one-by-one.
set -euo pipefail
REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/append_missing_gpus.log") 2>&1

export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
export MODEL_SIZE=8B MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export PYTHONUNBUFFERED=1 NO_PROXY=localhost,127.0.0.1,0.0.0.0 no_proxy=localhost,127.0.0.1,0.0.0.0

# Extra GPU/port pairs not covered by main fleet (node0 skipped GPU0).
# Format: "gpu:port"
EXTRAS="${EXTRAS:-0:8117}"

echo "=== $(date -Is) append missing GPUs host=$HOST extras=[$EXTRAS] ==="
echo "waiting for main boot to finish (no boot_bdy_tmp_fleet)..."
deadline=$((SECONDS + 28800))
while (( SECONDS < deadline )); do
  if ! pgrep -f "boot_bdy_tmp_fleet.sh" >/dev/null 2>&1; then
    break
  fi
  sleep 30
done
if pgrep -f "boot_bdy_tmp_fleet.sh" >/dev/null 2>&1; then
  echo "WARN: main boot still running after timeout; continue append anyway"
fi

for pair in $EXTRAS; do
  gpu=${pair%%:*}
  port=${pair##*:}
  url="http://127.0.0.1:${port}/v1"
  if curl -sf -m 2 "${url}/models" >/dev/null; then
    echo "already READY $url"
    grep -qxF "$url" "$LOG_DIR/endpoints.txt" 2>/dev/null || echo "$url" >> "$LOG_DIR/endpoints.txt"
    continue
  fi
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  pidf="$LOG_DIR/vllm_g${gpu}_p${port}.pid"
  : > "$log"
  echo "starting extra g${gpu}_p${port}"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t memstrata_s3_8b_fleet 2>/dev/null; then
    tmux new-window -t memstrata_s3_8b_fleet -n "g${gpu}_p${port}" \
      "export MODEL_SIZE=8B MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL SERVED_MODEL_NAME=$SERVED_MODEL_NAME PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH PYTHONUNBUFFERED=1; \
       bash '$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh' $gpu $port >'$log' 2>&1 & echo \$! >'$pidf'; wait"
  else
    nohup bash "$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh" "$gpu" "$port" >>"$log" 2>&1 &
    echo $! > "$pidf"
  fi
  deadline2=$((SECONDS + 2700))
  ready=0
  while (( SECONDS < deadline2 )); do
    if curl -sf -m 3 "${url}/models" >/dev/null; then ready=1; break; fi
    sleep 8
  done
  if (( ready == 1 )); then
    echo "READY $url"
    echo "$url" >> "$LOG_DIR/endpoints.txt"
  else
    echo "ERROR timeout $url"
    tail -n 40 "$log" || true
    exit 1
  fi
done
echo "=== $(date -Is) append done; endpoints=$(wc -l <"$LOG_DIR/endpoints.txt") ==="
