#!/usr/bin/env bash
# One-click launcher for the annotation pipeline console.
# Prefer: bash ensure_console.sh --watch   (idempotent + auto-restart)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=console_lib.sh
source "${SCRIPT_DIR}/console_lib.sh"

if [[ "${START_VLM:-0}" == "1" ]]; then
  GPU="${GPU:-0}"
  VLLM_PORT="${VLLM_PORT:-8110}"
  console_start_bg qwen32_vllm "" bash "${SCRIPT_DIR}/backend/start_qwen32_vllm.sh" "${GPU}" "${VLLM_PORT}"
  export VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
fi

BACKEND_ARGS=(
  "${PYTHON_BIN}" -m vmem_bench.annotation.pipeline.servers.backend.server
  --host "${BACKEND_HOST}"
  --port "${BACKEND_PORT}"
  --data-root "${DATA_ROOT}"
  --jobs-root "${JOBS_ROOT}"
  --python "${PYTHON_BIN}"
)
# Prefer local download catalogs so Blender source resolves under Videos/<id>/.
if [[ -z "${BLENDER_INDEX:-}" ]]; then
  for cand in \
    "${DATA_ROOT}/_services/blender_index.json" \
    "${VMEM_DATASETS_ROOT}/BlenderOpenMovies/download_status.json" \
    "/data/public_datasets/BlenderOpenMovies/download_status.json"
  do
    if [[ -f "${cand}" ]]; then BLENDER_INDEX="${cand}"; break; fi
  done
fi
if [[ -z "${LSMDC_INDEX:-}" ]]; then
  for cand in \
    "${DATA_ROOT}/_services/lsmdc_index.json" \
    "${VMEM_DATASETS_ROOT}/LSMDC/complete_movies.json" \
    "/data/public_datasets/LSMDC/complete_movies.json"
  do
    if [[ -f "${cand}" ]]; then LSMDC_INDEX="${cand}"; break; fi
  done
fi
[[ -n "${BLENDER_INDEX:-}" ]] && BACKEND_ARGS+=(--blender-index "${BLENDER_INDEX}")
[[ -n "${LSMDC_INDEX:-}" ]] && BACKEND_ARGS+=(--lsmdc-index "${LSMDC_INDEX}")

console_start_bg annotation_backend "${BACKEND_HEALTH_URL}" "${BACKEND_ARGS[@]}"
console_start_bg annotation_frontend "${FRONTEND_URL_LOCAL}/" \
  "${PYTHON_BIN}" -m vmem_bench.annotation.pipeline.servers.frontend.server \
  --host "${FRONTEND_HOST}" \
  --port "${FRONTEND_PORT}" \
  --backend-url "${BACKEND_URL}" \
  --vlm-base-url "${VLM_BASE_URL:-}"

console_wait_healthy "annotation_backend" "${BACKEND_HEALTH_URL}" 40 || true
console_wait_healthy "annotation_frontend" "${FRONTEND_URL_LOCAL}/" 40 || true
console_print_access
echo "Tip: keep it up with: bash ${SCRIPT_DIR}/ensure_console.sh"
echo "If source videos are not auto-resolved, set BLENDER_INDEX and LSMDC_INDEX before launching."
