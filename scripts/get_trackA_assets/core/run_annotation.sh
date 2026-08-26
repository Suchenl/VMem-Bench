#!/usr/bin/env bash
# Run the MemStrata offline annotation pipeline against one or more vLLM endpoints.
# Waits for all servers to become ready, then runs the pipeline on CLIENT_GPU
# (GroundingDINO + DINOv3 + SBD).
#
# Usage (on a GPU node, under tmux):
#   CLIENT_GPU=2 bash benchmarks/MemStrata/scripts/vmem_bench/core/run_annotation.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${PY:-python3}"

VIDEO="${VIDEO:-${VMEM_DATASETS_ROOT}/BlenderOpenMovies/big_buck_bunny_720p/big_buck_bunny_720p_h264.mp4}"
MOVIE_ID="${MOVIE_ID:-big_buck_bunny}"
OUT="${OUT:-${REPO}/benchmarks/MemStrata/data/blender_open_movies/${MOVIE_ID}}"
# Comma-separated port lists are endpoint pools for dataset-level throughput.
# A single chunk uses BRANCHES_PER_CHUNK branches (default 1), rotated over the pool.
# For 16 terminals, use e.g. ANNOTATOR_PORTS=8001,8003,8005,8007,8009,8011,8013,8015
# and VERIFIER_PORTS=8002,8004,8006,8008,8010,8012,8014,8016.
ANNOTATOR_PORTS="${ANNOTATOR_PORTS:-8001}"
VERIFIER_PORTS="${VERIFIER_PORTS:-8002}"
# Text-embedding endpoint (Qwen3-Embedding). Without it the auto-merge tier of auto_review is
# disabled (it requires text+body double agreement) and every duplicate lands on the human queue.
TEXT_EMBED_PORT="${TEXT_EMBED_PORT:-8003}"
TEXT_EMBED_MODEL="${TEXT_EMBED_MODEL:-qwen3-embedding-4b}"
# Hybrid serving: drafting on the fast model (ANNOTATOR_PORTS/VLM_MODEL), judgment roles on the
# large judge model when its endpoint is up (roster/naming/adjudication/auto-review).
JUDGE_PORT="${JUDGE_PORT:-8101}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen3-vl-32b}"
# Perception route: gdino_track (A, language-grounded) | sam3_track (B, exemplar-grounded).
PERCEPTION_BACKEND="${PERCEPTION_BACKEND:-gdino_track}"
VLM_MODEL="${VLM_MODEL:-qwen3-vl-8b}"
MIN_FRAMES="${MIN_FRAMES:-120}"
MAX_FRAMES="${MAX_FRAMES:-480}"
QA_ROUNDS="${QA_ROUNDS:-2}"
BRANCHES_PER_CHUNK="${BRANCHES_PER_CHUNK:-1}"
GROUNDING_SCORE_THRESHOLD="${GROUNDING_SCORE_THRESHOLD:-0.30}"
CROP_AUDIT_SCORE_THRESHOLD="${CROP_AUDIT_SCORE_THRESHOLD:-0.60}"
STATIC_OVERLAP_THRESHOLD="${STATIC_OVERLAP_THRESHOLD:-0.75}"
ROSTER_SEED="${ROSTER_SEED:-}"
PROPOSAL_ONLY="${PROPOSAL_ONLY:-0}"

ROSTER_ARGS=()
if [[ -n "${ROSTER_SEED}" ]]; then
  [[ -f "${ROSTER_SEED}" ]] || { echo "[run_annotation] roster seed missing: ${ROSTER_SEED}" >&2; exit 2; }
  ROSTER_ARGS=(--roster-seed "${ROSTER_SEED}")
elif [[ "${PROPOSAL_ONLY}" == "1" ]]; then
  ROSTER_ARGS=(--proposal-only)
  echo "[run_annotation] proposal-only: automatic roster output cannot be frozen"
else
  echo "[run_annotation] production runs require ROSTER_SEED=<human-confirmed.json>; set PROPOSAL_ONLY=1 only for diagnostics" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CLIENT_GPU:-2}"
export PYTHONPATH="${REPO}/benchmarks/MemStrata/src"
# Route B needs transformers>=5.9 (Sam3*); the vendored copy must lead the WHOLE process
# (mixing transformers versions mid-process corrupts model init).
if [[ "${PERCEPTION_BACKEND:-gdino_track}" == "sam3_track" || "${PERCEPTION_BACKEND:-}" == "fusion_track" ]]; then
  SAM3_DEPS="${MEMSTRATA_SAM3_DEPS:-${REPO}/models/vendor/sam3_transformers59}"
  [[ -d "${SAM3_DEPS}" ]] && export PYTHONPATH="${SAM3_DEPS}:${PYTHONPATH}" \
    && echo "[run_annotation] sam3_track: vendored transformers at ${SAM3_DEPS}"
fi
# torch>=2.10 (cu129) in the vllm env ships a newer nvjitlink than the system's; force it onto the
# loader path so `import torch` (GroundingDINO/DINOv3/face) does not die with an undefined-symbol
# ImportError (libcusparse.so.12 -> __nvJitLinkGetErrorLogSize_12_9).
for _d in "$(dirname "$(dirname "${PY}")")"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
  [[ -d "${_d}" ]] && { export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; break; }
done
export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"
export MONTAGE_WEIGHTS_ROOT="${MONTAGE_WEIGHTS_ROOT:-${REPO}/models/model_weights}"
export HF_HUB_OFFLINE=1
export FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
export FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1"

JUDGE_ARGS=()
if [[ -n "${JUDGE_PORT}" ]] && curl -sf "http://127.0.0.1:${JUDGE_PORT}/v1/models" 2>/dev/null | grep -q "${JUDGE_MODEL}"; then
  JUDGE_ARGS=(--judge-base-url "http://127.0.0.1:${JUDGE_PORT}/v1" --judge-model "${JUDGE_MODEL}")
  echo "[run_annotation] judge model ${JUDGE_MODEL} wired on :${JUDGE_PORT}"
else
  echo "[run_annotation] judge model not serving on :${JUDGE_PORT}; all roles use ${VLM_MODEL}"
fi

# Only wire the text-embed endpoint when it is actually serving; a dead URL must not stall the run.
TEXT_EMBED_ARGS=()
if [[ -n "${TEXT_EMBED_PORT}" ]] && curl -sf "http://127.0.0.1:${TEXT_EMBED_PORT}/v1/models" >/dev/null 2>&1; then
  TEXT_EMBED_ARGS=(--text-embed-base-url "http://127.0.0.1:${TEXT_EMBED_PORT}/v1"
                   --text-embed-model "${TEXT_EMBED_MODEL}")
  echo "[run_annotation] text-embed endpoint :${TEXT_EMBED_PORT} wired"
else
  echo "[run_annotation] WARNING: text-embed endpoint :${TEXT_EMBED_PORT} not serving; auto-merge tier disabled"
fi

ALL_PORTS="${ANNOTATOR_PORTS},${VERIFIER_PORTS}"
echo "[run_annotation] waiting for vLLM endpoints :${ALL_PORTS} ..."
for port in ${ALL_PORTS//,/ }; do
  until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 20; done
  echo "[run_annotation] endpoint :${port} ready"
done

to_urls() { local out=""; for p in ${1//,/ }; do out="${out:+${out},}http://127.0.0.1:${p}/v1"; done; echo "$out"; }
FIRST_ANNOTATOR_PORT="${ANNOTATOR_PORTS%%,*}"
VLM_BASE_URL="http://127.0.0.1:${FIRST_ANNOTATOR_PORT}/v1"

mkdir -p "${OUT}"
rc=0
"${PY}" -m vmem_bench.annotation.pipeline_track_first.run \
  --video "${VIDEO}" \
  --out "${OUT}" \
  --movie-id "${MOVIE_ID}" \
  --vlm-base-url "${VLM_BASE_URL}" \
  --annotator-urls "$(to_urls "${ANNOTATOR_PORTS}")" \
  --verifier-urls "$(to_urls "${VERIFIER_PORTS}")" \
  --vlm-model "${VLM_MODEL}" \
  --min-frames "${MIN_FRAMES}" \
  --max-frames "${MAX_FRAMES}" \
  --qa-rounds "${QA_ROUNDS}" \
  --branches-per-chunk "${BRANCHES_PER_CHUNK}" \
  --grounding-score-threshold "${GROUNDING_SCORE_THRESHOLD}" \
  --crop-audit-score-threshold "${CROP_AUDIT_SCORE_THRESHOLD}" \
  --static-overlap-threshold "${STATIC_OVERLAP_THRESHOLD}" \
  --perception-backend "${PERCEPTION_BACKEND}" \
  "${ROSTER_ARGS[@]}" \
  "${TEXT_EMBED_ARGS[@]}" \
  "${JUDGE_ARGS[@]}" \
  "$@" || rc=$?
echo "EXIT:${rc}"
exit "${rc}"
