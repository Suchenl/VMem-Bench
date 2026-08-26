#!/bin/bash
# Stage-2 (VLM visual-coverage scoring) for a completed Stage-1 fan-out.
# This wrapper now delegates to stage2_service.py: one long-lived scorer process
# keeps the DINOv3 embedder warm and uses segment-level pooled endpoint concurrency.
#
# Usage: bash tgpu_score_stage2.sh [JUDGE_API]
#   JUDGE_API default http://127.0.0.1:8110/v1/chat/completions
#     (qwen3-vl-32b server; launch separately). Both a full /chat/completions endpoint
#     and a bare OpenAI-compatible /v1 base URL are accepted.
#
# High-concurrency options:
#   JUDGE_APIS="http://host1:8110/v1,http://host2:8110/v1" bash .../tgpu_score_stage2.sh
#   STAGE2_FLEET=1 STAGE2_WORKERS=0 bash .../tgpu_score_stage2.sh
#
# Terminology note: outputs use SEGMENT wording. The legacy input field remains
# chunk_id for backward compatibility with existing gold/manifests.
set -u

BENCH=.
TRACKA=$BENCH/scripts/evaluate_baselines/trackA
AGG=$BENCH/scripts/evaluate_baselines/trackA/aggregate_two_movie_run.py
VACE=python3
PUB=${PUBLIC_MODELS_ROOT}
JUDGE_API=${1:-http://127.0.0.1:8110/v1/chat/completions}
JUDGE_MODEL=${JUDGE_MODEL:-qwen3-vl-32b}
SCORE_GPU=${SCORE_GPU:-0}
STAGE2_WORKERS=${STAGE2_WORKERS:-0}
STAGE2_FLEET=${STAGE2_FLEET:-0}
STAGE2_FLEET_ROLE=${STAGE2_FLEET_ROLE:-reviewer}
STAGE2_FLEET_ROOT=${STAGE2_FLEET_ROOT:-}
JUDGE_APIS=${JUDGE_APIS:-}

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=$BENCH/_tgpu_run/score_$STAMP
mkdir -p "$LOGDIR"
echo "[score] judge=$JUDGE_API model=$JUDGE_MODEL gpu=$SCORE_GPU workers=$STAGE2_WORKERS logdir=$LOGDIR"

cmd=( "$VACE" "$TRACKA/stage2_service.py"
  --api "$JUDGE_API"
  --model "$JUDGE_MODEL"
  --workers "$STAGE2_WORKERS"
  --score-gpu "$SCORE_GPU"
  --log-dir "$LOGDIR" )
[ -n "$JUDGE_APIS" ] && cmd+=( --api-list "$JUDGE_APIS" )
[ "$STAGE2_FLEET" = "1" ] && cmd+=( --fleet --fleet-role "$STAGE2_FLEET_ROLE" )
[ -n "$STAGE2_FLEET_ROOT" ] && cmd+=( --fleet-root "$STAGE2_FLEET_ROOT" )

( cd "$BENCH" && NO_PROXY=localhost,127.0.0.1 CUDA_VISIBLE_DEVICES=$SCORE_GPU \
  PUBLIC_MODELS_ROOT=$PUB PYTHONPATH=src "${cmd[@]}" )
rc=$?
echo "[score] service exit=$rc progress=$LOGDIR/progress.json"
echo "[score] aggregate: $VACE $AGG --out $LOGDIR"
exit $rc
