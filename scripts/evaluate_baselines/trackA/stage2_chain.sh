#!/bin/bash
# After Stage-1 runner jobs finish, score (Stage-2) and aggregate.
# Point STAGE1_LOG_DIR at the directory that holds per-job logs (default _eval_run/latest).
set -u
BENCH=.
TRACKA=$BENCH/scripts/evaluate_baselines/trackA
PYTHON=${PYTHON:-python3}
JUDGE=${JUDGE:-http://127.0.0.1:8110/v1}
cd "$BENCH" || exit 97
LATEST=${STAGE1_LOG_DIR:-$(readlink -f _eval_run/latest 2>/dev/null || true)}
if [ -z "$LATEST" ] || [ ! -d "$LATEST" ]; then
  echo "set STAGE1_LOG_DIR to your Stage-1 log directory" >&2
  exit 2
fi
echo "[chain] Stage-2 scoring (judge $JUDGE) via stage2_service.py"
"$PYTHON" "$TRACKA/stage2_service.py" --log-dir "$LATEST" || true
echo "[chain] aggregating"
mkdir -p "$LATEST/_agg"
"$PYTHON" "$TRACKA/aggregate_two_movie_run.py" --out "$LATEST/_agg"
echo "[chain] done: $LATEST/_agg"
