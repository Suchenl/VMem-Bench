#!/usr/bin/env bash
# Continue fleet: start only missing GPUs; never kill healthy ready endpoints.
# Writes pidfile so supervisor can detect real boot (not pgrep false positive).
set -euo pipefail
REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
mkdir -p "$LOG_DIR"
PIDFILE="$LOG_DIR/continue_fleet.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
export MODEL_SIZE=8B MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export PYTHONUNBUFFERED=1 NO_PROXY=localhost,127.0.0.1,0.0.0.0 no_proxy=localhost,127.0.0.1,0.0.0.0

# pairs "gpu:port" — node0 uses 1:8110..7:8116 + 0:8117; others 0:8110..7:8117
GPU_PORT_PAIRS="${GPU_PORT_PAIRS:-}"
if [[ -z "$GPU_PORT_PAIRS" ]]; then
  GPU_PORT_PAIRS="0:8110 1:8111 2:8112 3:8113 4:8114 5:8115 6:8116 7:8117"
fi

READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-2700}"
exec > >(tee -a "$LOG_DIR/continue_fleet.log") 2>&1
echo "=== $(date -Is) continue fleet host=$HOST pairs=[$GPU_PORT_PAIRS] ==="

# ensure weights
bash "$MS/scripts/vmem_bench/fleet/lib/stage_qwen3vl8b_local.sh" || true
export PUBLIC_MODELS_ROOT=/tmp/memstrata_public_models

: > "$LOG_DIR/endpoints.txt"
for pair in $GPU_PORT_PAIRS; do
  gpu=${pair%%:*}
  port=${pair##*:}
  url="http://127.0.0.1:${port}/v1"
  if curl -sf -m 2 "${url}/models" >/dev/null; then
    echo "KEEP READY $url"
    echo "$url" >> "$LOG_DIR/endpoints.txt"
    continue
  fi
  # if something dead on this port, stop only that port via /proc
  for pidpath in /proc/[0-9]*; do
    pid=${pidpath#/proc/}
    cmd=$(tr '\0' ' ' < "$pidpath/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/envs/vllm/bin/vllm serve"*"--port ${port}"*)
        echo "stop stale pid=$pid port=$port"
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        ;;
    esac
  done
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  pidf="$LOG_DIR/vllm_g${gpu}_p${port}.pid"
  : > "$log"
  echo "START g${gpu}_p${port}"
  nohup bash "$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh" "$gpu" "$port" >>"$log" 2>&1 &
  echo $! > "$pidf"
  deadline=$((SECONDS + READY_TIMEOUT_SEC))
  ready=0
  while (( SECONDS < deadline )); do
    if curl -sf -m 3 "${url}/models" >/dev/null; then ready=1; break; fi
    pid=$(cat "$pidf" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR pid exited; tail:"; tail -n 50 "$log" || true
      exit 1
    fi
    sleep 8
  done
  if (( ready != 1 )); then
    echo "ERROR timeout $url"; tail -n 60 "$log" || true
    exit 1
  fi
  echo "READY $url"
  echo "$url" >> "$LOG_DIR/endpoints.txt"
done
echo "=== $(date -Is) continue fleet DONE n=$(wc -l <"$LOG_DIR/endpoints.txt") ==="
