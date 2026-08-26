#!/bin/bash
# Wait for the formal Track A Stage-1 EXIT sentinels, then score (Stage-2) + aggregate.
# Launch DETACHED with setsid so it survives the launching shell:
#   setsid bash scripts/evaluate_baselines/trackA/stage2_chain.sh > _tgpu_run/stage2_chain.out 2>&1 </dev/null &
set -u
BENCH=.
TRACKA=$BENCH/scripts/evaluate_baselines/trackA
VACE=python3
JUDGE=${JUDGE:-http://127.0.0.1:8110/v1}
EXPECTED=${STAGE1_EXPECTED:-18}
cd "$BENCH" || exit 97
LATEST=$(readlink -f _tgpu_run/latest)

echo "[chain] waiting for $EXPECTED Stage-1 EXIT sentinels in $LATEST"
for i in $(seq 1 300); do
  n=0
  for f in "$LATEST"/*.log; do
    [ "$(basename "$f")" = "master.log" ] && continue
    [ -f "$f" ] && grep -q "^EXIT:" "$f" && n=$((n+1))
  done
  echo "[chain] $(date '+%H:%M:%S') sentinels=$n/$EXPECTED"
  [ "$n" -ge "$EXPECTED" ] && break
  sleep 60
done

echo "[chain] === Stage-2 scoring (judge $JUDGE) ==="
bash "$TRACKA/tgpu_score_stage2.sh" "$JUDGE"
echo "[chain] === aggregating ==="
mkdir -p "$LATEST/_agg"
"$VACE" "$TRACKA/aggregate_two_movie_run.py" --out "$LATEST/_agg"
echo "[chain] === STAGE-2 + AGGREGATE COMPLETE: $LATEST/_agg/results.md ==="
