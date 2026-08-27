#!/usr/bin/env bash
# Launch one Qwen3-VL vLLM endpoint for the MemStrata annotation pipeline.
# Registers into runtime/services/vlm_fleet via supervise.
#
# Usage:
#   MODEL_SIZE=8B  FLEET_ADVERTISE_HOST=$(hostname -s) start_annotation_vllm.sh <gpu_id> <port>
#   MODEL_SIZE=32B start_annotation_vllm.sh <gpu_id> <port>
set -euo pipefail

GPU="${1:?usage: start_annotation_vllm.sh <gpu_id> <port>}"
PORT="${2:?usage: start_annotation_vllm.sh <gpu_id> <port>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# servers/start_annotation_vllm.sh -> servers -> vmem_bench -> scripts -> MemStrata
MEMSTRATA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_ROOT="${MEMSTRATA_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "${VLLM_ENV:-}" ]]; then
  export PATH="${VLLM_ENV}/bin:${PATH}"
  if [[ "${PYTHON_BIN}" == "python3" && -x "${VLLM_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VLLM_ENV}/bin/python"
  fi
fi
MODEL_SIZE="${MODEL_SIZE:-8B}"
PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:?export PUBLIC_MODELS_ROOT=/path/to/hf-style-models}"
case "${MODEL_SIZE}" in
  8B|8b)
    MODEL_PATH="${MODEL_PATH:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-8B-Instruct}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
    ;;
  32B|32b)
    MODEL_PATH="${MODEL_PATH:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-32B-Instruct}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
    ;;
  *)
    echo "MODEL_SIZE must be 8B or 32B, got: ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

[[ -d "${MODEL_PATH}" ]] || { echo "model path missing: ${MODEL_PATH}" >&2; exit 1; }

_nvjitlink_lib="$(
  "${PYTHON_BIN}" - <<'PY'
import pathlib, sys
site = pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "nvjitlink" / "lib"
print(site if site.is_dir() else "")
PY
)"
if [[ -n "${_nvjitlink_lib}" ]]; then
  export LD_LIBRARY_PATH="${_nvjitlink_lib}:${LD_LIBRARY_PATH:-}"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,0.0.0.0"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SERVED_MODEL_NAME

ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-.}"
# vLLM accepts a SINGLE directory. Comma-separated values are treated as one
# nonexistent path (e.g. "/data,/tmp") and break all file:// video reviews.
if [[ "${ALLOWED_LOCAL_MEDIA_PATH}" == *","* ]]; then
  echo "WARNING: ALLOWED_LOCAL_MEDIA_PATH must be one directory; got '${ALLOWED_LOCAL_MEDIA_PATH}'. Using first entry." >&2
  ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH%%,*}"
fi
if [[ ! -d "${ALLOWED_LOCAL_MEDIA_PATH}" ]]; then
  echo "ALLOWED_LOCAL_MEDIA_PATH is not a directory: ${ALLOWED_LOCAL_MEDIA_PATH}" >&2
  exit 1
fi

VLLM_CMD=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0 --port "${PORT}"
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --limit-mm-per-prompt '{"image": 24, "video": 1}'
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --trust-remote-code
)

exec "${PYTHON_BIN}" -m vmem_bench.annotation.pipeline.servers.fleet.supervise \
  --gpu "${GPU}" \
  --port "${PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --role reviewer \
  --cluster "${FLEET_CLUSTER:-}" \
  --node "${FLEET_NODE_ID:-}" \
  --gpu-rank "${FLEET_GPU_RANK:-${GPU}}" \
  -- \
  "${VLLM_CMD[@]}"
