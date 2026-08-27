#!/usr/bin/env bash
# Launch one or more Qwen3-VL-32B reviewer replicas with fleet status registration.
#
# H800: one 32B service per GPU.
#   FLEET_ADVERTISE_HOST=$(hostname -s) bash start_reviewer_pool.sh 6:8110,7:8111
# A800: one 32B service per TWO GPUs (tensor parallel).
#   FLEET_CLUSTER=gpu-a800 FLEET_ADVERTISE_HOST=$(hostname -s) \
#     bash start_reviewer_pool.sh 0+1:8110,2+3:8111
#
# H800 tensor-parallel (SCORING/CONCURRENCY EXPERIMENT ONLY, opt-in, non-default):
# a 1-GPU H800 replica has only ~4 GiB of KV cache left after the 32B weights, so it
# is stuck at MAX_NUM_SEQS=1. Sharding one service across 2 GPUs frees ~36 GiB of KV
# per rank, so the replica can batch many judge requests. Trade fewer replicas for
# much higher per-replica concurrency. Raising MAX_NUM_SEQS is REQUIRED to get any
# benefit - ALLOW_H800_TP alone only changes the sharding.
#   ALLOW_H800_TP=1 MAX_NUM_SEQS=8 FLEET_ADVERTISE_HOST=$(hostname -s) \
#     bash start_reviewer_pool.sh 0+1:8110,2+3:8111,4+5:8112,6+7:8113
#
# Instance intent + status land under:
#   runtime/services/vlm_fleet/{intents,instances}/
# Console reads that path and dispatches jobs — no manual Base URL needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env_no_proxy.sh
source "${SCRIPT_DIR}/../env_no_proxy.sh"
START_ONE="${SCRIPT_DIR}/start_qwen32_vllm.sh"
[[ -f "${START_ONE}" ]] || { echo "missing ${START_ONE}" >&2; exit 1; }

SPEC="${1:?usage: $0 <gpu[:+gpu...]:port>[,gpu[:+gpu...]:port...] }"
MEMSTRATA_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
SRC_ROOT="${MEMSTRATA_ROOT}/src"
LOG_ROOT="${LOG_ROOT:-${MEMSTRATA_ROOT}/runtime/services/vlm_fleet/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "${LOG_ROOT}"

# 30s@fps=2 (~12k multimodal tokens). H800 runs TP=1, A800 runs TP=2.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b}"
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

detect_gpu_family() {
  local first_gpu="$1"
  local cluster="${FLEET_CLUSTER:-${GPU_CLUSTER:-}}"
  case "${cluster,,}" in
    *a800*) echo "a800"; return ;;
    *h800*) echo "h800"; return ;;
  esac
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${first_gpu}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
  case "${name}" in
    *a800*) echo "a800" ;;
    *h800*) echo "h800" ;;
    *) echo "unknown" ;;
  esac
}

gpu_count_for_spec() {
  local spec="$1"
  awk -F'+' '{print NF}' <<< "${spec}"
}

gpu_csv_for_spec() {
  local spec="$1"
  tr '+' ',' <<< "${spec}"
}

first_gpu_for_spec() {
  local spec="$1"
  awk -F'+' '{print $1}' <<< "${spec}"
}

safe_gpu_tag() {
  local spec="$1"
  tr '+,' '__' <<< "${spec}"
}

validate_topology() {
  local family="$1"
  local gpu_count="$2"
  if [[ "${family}" == "a800" && "${gpu_count}" -ne 2 ]]; then
    echo "ABORT: A800 32B reviewer must use exactly two GPUs per service; use e.g. 0+1:8110" >&2
    exit 3
  fi
  if [[ "${family}" == "h800" && "${gpu_count}" -ne 1 ]]; then
    # SCORING/CONCURRENCY EXPERIMENT MODE (opt-in, non-default).
    # One H800 fits the 32B weights but leaves only ~4 GiB of KV cache, which caps
    # the replica at MAX_NUM_SEQS=1. Sharding one service over 2 GPUs frees ~36 GiB
    # of KV per rank so the replica can serve many concurrent judge requests.
    # Fewer replicas, each far more concurrent. Default H800 stays 1 GPU/service.
    if [[ "${ALLOW_H800_TP:-0}" != "1" ]]; then
      echo "ABORT: H800 32B reviewer defaults to exactly one GPU per service; use e.g. 6:8110." >&2
      echo "       For the scoring/concurrency experiment set ALLOW_H800_TP=1 to allow" >&2
      echo "       tensor-parallel H800 services (e.g. ALLOW_H800_TP=1 ... 6+7:8110)." >&2
      exit 3
    fi
    echo "NOTE: ALLOW_H800_TP=1 -> H800 tensor-parallel service over ${gpu_count} GPUs" >&2
    echo "      (scoring/concurrency experiment mode; not the annotation default)." >&2
  fi
  if [[ "${family}" == "unknown" ]]; then
    echo "ABORT: cannot detect GPU family; set FLEET_CLUSTER=gpu-h800 or gpu-a800" >&2
    exit 3
  fi
}

check_free_memory() {
  local family="$1"
  local gpu_spec="$2"
  local min_free
  if [[ "${family}" == "a800" ]]; then
    min_free="${A800_MIN_FREE_MIB:-65000}"
  else
    min_free="${H800_MIN_FREE_MIB:-100000}"
  fi
  IFS='+' read -r -a GPUS <<< "${gpu_spec}"
  for gpu in "${GPUS[@]}"; do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    echo "GPU${gpu} free_MiB=${free} min_required=${min_free}"
    if [[ "${free}" -lt "${min_free}" ]]; then
      echo "ABORT: GPU${gpu} free ${free} MiB < ${min_free}; not enough headroom for ${family} 32B reviewer" >&2
      exit 4
    fi
  done
}

IFS=',' read -r -a ITEMS <<< "${SPEC}"
URLS=()
for item in "${ITEMS[@]}"; do
  item="${item// /}"
  [[ -n "${item}" ]] || continue
  GPU_SPEC="${item%%:*}"
  PORT="${item##*:}"
  [[ "${GPU_SPEC}" =~ ^[0-9]+(\+[0-9]+)*$ && "${PORT}" =~ ^[0-9]+$ ]] || {
    echo "bad item ${item}; expected gpu[:+gpu...]:port, e.g. 6:8110 or 0+1:8110" >&2
    exit 2
  }
  FIRST_GPU="$(first_gpu_for_spec "${GPU_SPEC}")"
  GPU_FAMILY="$(detect_gpu_family "${FIRST_GPU}")"
  GPU_COUNT="$(gpu_count_for_spec "${GPU_SPEC}")"
  GPU_CSV="$(gpu_csv_for_spec "${GPU_SPEC}")"
  GPU_TAG="$(safe_gpu_tag "${GPU_SPEC}")"
  validate_topology "${GPU_FAMILY}" "${GPU_COUNT}"
  if [[ -n "${TENSOR_PARALLEL_SIZE:-}" && ! "${TENSOR_PARALLEL_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "ABORT: TENSOR_PARALLEL_SIZE must be numeric, got '${TENSOR_PARALLEL_SIZE}'" >&2
    exit 3
  fi
  if [[ -n "${TENSOR_PARALLEL_SIZE:-}" && "${TENSOR_PARALLEL_SIZE}" -ne "${GPU_COUNT}" ]]; then
    echo "ABORT: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} conflicts with ${GPU_COUNT} GPU(s) in ${GPU_SPEC}" >&2
    exit 3
  fi
  check_free_memory "${GPU_FAMILY}" "${GPU_SPEC}"
  echo "${GPU_FAMILY^^} service gpu=${GPU_CSV} tp=${GPU_COUNT} -> :${PORT}"
  LOG="${LOG_ROOT}/gpu${GPU_TAG}_p${PORT}.log"
  PIDF="${LOG_ROOT}/gpu${GPU_TAG}_p${PORT}.pid"
  if [[ -s "${PIDF}" ]] && kill -0 "$(cat "${PIDF}")" 2>/dev/null; then
    echo "already running gpu=${GPU_CSV} port=${PORT} pid=$(cat "${PIDF}")"
  else
    nohup env TENSOR_PARALLEL_SIZE="${GPU_COUNT}" "${PYTHON_BIN}" -m vmem_bench.annotation.pipeline.servers.fleet.supervise \
      --gpu "${GPU_CSV}" \
      --port "${PORT}" \
      --model "${SERVED_MODEL_NAME}" \
      --role reviewer \
      --cluster "${FLEET_CLUSTER:-}" \
      --node "${FLEET_NODE_ID:-}" \
      --gpu-rank "${FIRST_GPU}" \
      -- \
      bash "${START_ONE}" "${GPU_CSV}" "${PORT}" \
      >"${LOG}" 2>&1 &
    echo $! >"${PIDF}"
    echo "started supervised gpu=${GPU_CSV} port=${PORT} supervisor_pid=$(cat "${PIDF}") log=${LOG}"
  fi
  HOST="${FLEET_ADVERTISE_HOST:-$(hostname -s)}"
  URLS+=("http://${HOST}:${PORT}/v1")
done

echo
echo "Fleet advertise URLs (console auto-discovers via runtime/services/vlm_fleet):"
IFS=','; echo "${URLS[*]}"
echo "Fleet root: ${MEMSTRATA_ROOT}/runtime/services/vlm_fleet"
