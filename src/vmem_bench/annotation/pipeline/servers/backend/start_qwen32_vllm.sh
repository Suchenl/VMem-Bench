#!/usr/bin/env bash
# Start Qwen3-VL-32B-Instruct for pipeline S3 review and S5 grounding.
# Canonical VLM launcher — do not copy this script into stages/ or data/.
#
# Usage (on a GPU node under tmux):
#   bash start_qwen32_vllm.sh <gpu_id_or_csv> <port>
# H800 single-card example:
#   TENSOR_PARALLEL_SIZE=1 bash start_qwen32_vllm.sh 6 8110
# A800 two-card example:
#   TENSOR_PARALLEL_SIZE=2 bash start_qwen32_vllm.sh 0,1 8110
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env_no_proxy.sh
source "${SCRIPT_DIR}/../env_no_proxy.sh"

GPU="${1:?usage: start_qwen32_vllm.sh <gpu_id> <port>}"
PORT="${2:?usage: start_qwen32_vllm.sh <gpu_id> <port>}"
VLLM_ENV="${VLLM_ENV:-}"
MODEL_PATH="${MODEL_PATH:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:--1}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${ALLOWED_LOCAL_MEDIA_PATH:-.}}"
# vLLM accepts a SINGLE directory (comma lists break file:// video loads).
if [[ "${ALLOWED_LOCAL_MEDIA_PATH}" == *","* ]]; then
  echo "WARNING: ALLOWED_LOCAL_MEDIA_PATH must be one directory; got '${ALLOWED_LOCAL_MEDIA_PATH}'. Using first entry." >&2
  ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH%%,*}"
fi
if [[ ! -d "${ALLOWED_LOCAL_MEDIA_PATH}" ]]; then
  echo "ALLOWED_LOCAL_MEDIA_PATH is not a directory: ${ALLOWED_LOCAL_MEDIA_PATH}" >&2
  exit 1
fi

# vLLM's video loader defaults to 32 frames. Use -1 so request-time
# mm_processor_kwargs.fps is applied by Qwen3VLVideoProcessor over the full clip.
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
if [[ -z "${MM_PROCESSOR_KWARGS}" ]]; then
  MM_PROCESSOR_KWARGS='{"size":{"longest_edge":25165824,"shortest_edge":4096},"max_pixels":8388608,"min_pixels":524288}'
fi

[[ -d "${MODEL_PATH}" ]] || { echo "model path missing: ${MODEL_PATH}" >&2; exit 1; }
export PATH="${VLLM_ENV}/bin:${PATH}"
for d in "${VLLM_ENV}"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
  [[ -d "${d}" ]] && { export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; break; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"

EAGER_ARGS=()
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && EAGER_ARGS+=(--enforce-eager)

# Optional serving-throughput knobs, used by the Track A/B scoring pools. Both are
# unset by default, so the annotation pipeline's launch line stays byte-identical.
# They change scheduling/parallelism only - never the prompt, sampling or output.
#   API_SERVER_COUNT   : N API server processes. Multimodal preprocessing (video
#                        decode/resize) runs in the API server's asyncio loop and
#                        serialises there; N>1 parallelises it. Costs host RAM:
#                        mm_processor_cache_gb x (API_SERVER_COUNT + dp_size).
#   MM_ENCODER_TP_MODE : "data" shards vision-encoder work across TP ranks instead
#                        of replicating it (helps video-heavy prefill).
PERF_ARGS=()
[[ -n "${API_SERVER_COUNT:-}" ]] && PERF_ARGS+=(--api-server-count "${API_SERVER_COUNT}")
[[ -n "${MM_ENCODER_TP_MODE:-}" ]] && PERF_ARGS+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
MEDIA_IO_ARGS=()
if [[ -n "${VIDEO_NUM_FRAMES}" ]]; then
  MEDIA_IO_ARGS+=(--media-io-kwargs "{\"video\":{\"num_frames\":${VIDEO_NUM_FRAMES}}}")
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}" \
  "${MEDIA_IO_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 24, "video": 1}' \
  "${EAGER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  --trust-remote-code
