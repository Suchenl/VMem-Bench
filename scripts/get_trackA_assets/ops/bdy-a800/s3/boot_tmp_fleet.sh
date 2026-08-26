#!/usr/bin/env bash
# On one BDY node: stage 8B to /tmp, then start a staggered vLLM fleet in background.
# Designed to survive tgpu_fs job exit (caller should: nohup this script &).
# Does NOT use pkill -f patterns that can match the parent job cmdline.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
mkdir -p "$LOG_DIR"

export LOCAL_PUBLIC_MODELS_ROOT="${LOCAL_PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
export PUBLIC_MODELS_ROOT="$LOCAL_PUBLIC_MODELS_ROOT"
export MODEL_SIZE=8B
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export SERVED_MODEL_NAME=qwen3-vl-8b
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
export PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=$NO_PROXY

# node0 often has GPU0 occupied → default skip 0
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"
PORT_BASE="${PORT_BASE:-8110}"
FIRST_READY_TIMEOUT_SEC="${FIRST_READY_TIMEOUT_SEC:-2700}"
NEXT_READY_TIMEOUT_SEC="${NEXT_READY_TIMEOUT_SEC:-1200}"
STAGE_BEFORE_SEC="${STAGE_BEFORE_SEC:-0}"

exec > >(tee -a "$LOG_DIR/tmp_fleet_boot.log") 2>&1
echo "=== $(date -Is) tmp-fleet boot host=$HOST gpus=[$GPU_LIST] stage_delay=${STAGE_BEFORE_SEC}s ==="

if (( STAGE_BEFORE_SEC > 0 )); then
  echo "cross-node stage sleep ${STAGE_BEFORE_SEC}s"
  sleep "$STAGE_BEFORE_SEC"
fi

echo "staging weights -> $LOCAL_PUBLIC_MODELS_ROOT"
bash "$MS/scripts/vmem_bench/fleet/lib/stage_qwen3vl8b_local.sh"
export PUBLIC_MODELS_ROOT="$LOCAL_PUBLIC_MODELS_ROOT"

kill_port() {
  local port="$1"
  local pid cmd
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/envs/vllm/bin/vllm serve"*"--port ${port}"*)
        echo "stop pid=$pid port=$port"
        kill "$pid" 2>/dev/null || true
        ;;
    esac
  done
  sleep 1
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/envs/vllm/bin/vllm serve"*"--port ${port}"*)
        kill -9 "$pid" 2>/dev/null || true
        ;;
    esac
  done
}

idx=0
for gpu in $GPU_LIST; do
  port=$((PORT_BASE + idx))
  kill_port "$port"
  idx=$((idx + 1))
done
sleep 1

SESSION="memstrata_s3_8b_fleet"
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
  tmux new-session -d -s "$SESSION" -n boot "sleep infinity"
fi

idx=0
ENDPOINTS=()
: > "$LOG_DIR/endpoints.txt"
for gpu in $GPU_LIST; do
  port=$((PORT_BASE + idx))
  log="$LOG_DIR/vllm_g${gpu}_p${port}.log"
  pidf="$LOG_DIR/vllm_g${gpu}_p${port}.pid"
  : > "$log"
  if command -v tmux >/dev/null 2>&1; then
    tmux new-window -t "$SESSION" -n "g${gpu}_p${port}" \
      "export MODEL_SIZE=8B MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL SERVED_MODEL_NAME=$SERVED_MODEL_NAME PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH PYTHONUNBUFFERED=1 NO_PROXY=$NO_PROXY no_proxy=$no_proxy; \
       export FLEET_CLUSTER='${FLEET_CLUSTER:-}' FLEET_NODE_ID='${FLEET_NODE_ID:-}' FLEET_ADVERTISE_HOST='${FLEET_ADVERTISE_HOST:-}' FLEET_GPU_RANK=$gpu; \
       stdbuf -oL -eL bash '$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh' $gpu $port >'$log' 2>&1 & echo \$! >'$pidf'; wait"
  else
    nohup bash "$MS/scripts/vmem_bench/servers/start_annotation_vllm.sh" "$gpu" "$port" >>"$log" 2>&1 &
    echo $! > "$pidf"
  fi
  url="http://127.0.0.1:${port}/v1"
  echo "started g${gpu}_p${port} pid=$(cat "$pidf" 2>/dev/null || echo ?) -> $log"
  if (( idx == 0 )); then
    timeout_sec=$FIRST_READY_TIMEOUT_SEC
  else
    timeout_sec=$NEXT_READY_TIMEOUT_SEC
  fi
  echo "waiting $url/models (timeout ${timeout_sec}s)..."
  deadline=$((SECONDS + timeout_sec))
  ready=0
  while (( SECONDS < deadline )); do
    if curl -sf -m 3 "${url}/models" >/dev/null; then
      ready=1
      break
    fi
    if [[ -f "$pidf" ]]; then
      pid=$(cat "$pidf" 2>/dev/null || true)
      if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: pid=$pid exited early; tail log:"
        tail -n 40 "$log" || true
        exit 1
      fi
    fi
    sleep 8
  done
  if (( ready != 1 )); then
    echo "ERROR: timeout waiting for $url"
    tail -n 60 "$log" || true
    exit 1
  fi
  echo "READY $url"
  echo "$url" >> "$LOG_DIR/endpoints.txt"
  ENDPOINTS+=("$url")
  idx=$((idx + 1))
done

echo "$HOST" > "$LOG_DIR/fleet_host.txt"
echo "fleet host=$HOST n=${#ENDPOINTS[@]} gpus=[$GPU_LIST]"
echo "=== $(date -Is) fleet all ready ==="
