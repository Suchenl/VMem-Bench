#!/usr/bin/env bash
set -euo pipefail

ROOT="${MONTAGE_ROOT:-.}"
OUT_ROOT="$ROOT/benchmarks/VMem-Bench/outputs/evaluation/trackB/_services/iamflow_vllm"
PY="${IAMFLOW_VLLM_PY:-python3}"
SERVICE_GPU="${IAMFLOW_SERVICE_GPU:-6}"
LLM_SERVICE_GPU="${IAMFLOW_LLM_SERVICE_GPU:-$SERVICE_GPU}"
VLM_SERVICE_GPU="${IAMFLOW_VLM_SERVICE_GPU:-7}"
LLM_PORT="${IAMFLOW_LLM_PORT:-8100}"
VLM_PORT="${IAMFLOW_VLM_PORT:-8101}"
HOST="${IAMFLOW_SERVICE_HOST:-0.0.0.0}"

LLM_MODEL_PATH="${IAMFLOW_LLM_MODEL_PATH:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-4B-Instruct-2507}"
VLM_MODEL_PATH="${IAMFLOW_VLM_MODEL_PATH:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-2B-Instruct}"
LLM_MODEL="${IAMFLOW_LLM_MODEL:-$(basename "$LLM_MODEL_PATH")}"
VLM_MODEL="${IAMFLOW_VLM_MODEL:-$(basename "$VLM_MODEL_PATH")}"

LLM_GPU_UTIL="${IAMFLOW_LLM_GPU_UTIL:-0.25}"
VLM_GPU_UTIL="${IAMFLOW_VLM_GPU_UTIL:-0.35}"
LLM_MAX_MODEL_LEN="${IAMFLOW_LLM_MAX_MODEL_LEN:-4096}"
VLM_MAX_MODEL_LEN="${IAMFLOW_VLM_MAX_MODEL_LEN:-4096}"
LOG_ROOT="${IAMFLOW_SERVICE_LOGDIR:-$OUT_ROOT/$(date +%Y%m%d_%H%M%S)}"
LLM_SESSION="${IAMFLOW_LLM_SESSION:-trackb_iamflow_llm_${LLM_PORT}}"
VLM_SESSION="${IAMFLOW_VLM_SESSION:-trackb_iamflow_vlm_${VLM_PORT}}"

mkdir -p "$LOG_ROOT"
rm -f "$OUT_ROOT/latest"
ln -s "$LOG_ROOT" "$OUT_ROOT/latest"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$LLM_SERVICE_GPU" == "$VLM_SERVICE_GPU" && "${IAMFLOW_ALLOW_SHARED_SERVICE_GPU:-0}" != "1" ]]; then
  echo "refusing to launch IAMFlow LLM and VLM services on the same GPU: $LLM_SERVICE_GPU" >&2
  echo "set IAMFLOW_LLM_SERVICE_GPU and IAMFLOW_VLM_SERVICE_GPU to different GPUs, or set IAMFLOW_ALLOW_SHARED_SERVICE_GPU=1 for a smoke-only override" >&2
  exit 88
fi

if [[ -n "${GPU_KEEPALIVE_STATUS_DIR:-}" ]]; then
  host="$(hostname)"
  status_file="${GPU_KEEPALIVE_STATUS_DIR}/${host}.status"
  if [[ ! -f "$status_file" ]]; then
    echo "missing keepalive status: $status_file" >&2
    exit 86
  fi
  status_text="$(<"$status_file")"
  if [[ "$status_text" != *"alive_gpu_processes=8/8"* ]]; then
    echo "unhealthy keepalive status: $status_file" >&2
    exit 87
  fi
fi

start_tmux_service() {
  local session="$1"
  local log_file="$2"
  shift 2
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session"
    return
  fi
  tmux new-session -d -s "$session" "$* 2>&1 | tee -a '$log_file'"
  echo "started tmux session: $session"
}

start_tmux_service "$LLM_SESSION" "$LOG_ROOT/llm_${LLM_PORT}.log" \
  "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
   CUDA_VISIBLE_DEVICES=$LLM_SERVICE_GPU '$PY' -m vllm.entrypoints.openai.api_server \
    --host '$HOST' \
    --port '$LLM_PORT' \
    --model '$LLM_MODEL_PATH' \
    --served-model-name '$LLM_MODEL' \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization '$LLM_GPU_UTIL' \
    --max-model-len '$LLM_MAX_MODEL_LEN' \
    --max-num-seqs 1"

start_tmux_service "$VLM_SESSION" "$LOG_ROOT/vlm_${VLM_PORT}.log" \
  "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
   CUDA_VISIBLE_DEVICES=$VLM_SERVICE_GPU '$PY' -m vllm.entrypoints.openai.api_server \
    --host '$HOST' \
    --port '$VLM_PORT' \
    --model '$VLM_MODEL_PATH' \
    --served-model-name '$VLM_MODEL' \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization '$VLM_GPU_UTIL' \
    --max-model-len '$VLM_MAX_MODEL_LEN' \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{\"image\": 3}'"

cat >"$LOG_ROOT/iamflow_service.env" <<EOF
export IAMFLOW_LLM_ENDPOINT=http://127.0.0.1:${LLM_PORT}/v1
export IAMFLOW_VLM_ENDPOINT=http://127.0.0.1:${VLM_PORT}/v1
export IAMFLOW_LLM_MODEL=${LLM_MODEL}
export IAMFLOW_VLM_MODEL=${VLM_MODEL}
export IAMFLOW_HTTP_TIMEOUT=${IAMFLOW_HTTP_TIMEOUT:-900}
EOF

cat <<EOF
IAMFlow Track B vLLM services launched or already running.
log_dir=$LOG_ROOT
llm_gpu=$LLM_SERVICE_GPU
vlm_gpu=$VLM_SERVICE_GPU
llm_session=$LLM_SESSION
vlm_session=$VLM_SESSION

Before launching Track B IAMFlow workers on this node:
  source $LOG_ROOT/iamflow_service.env
EOF
