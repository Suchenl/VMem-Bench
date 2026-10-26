#!/usr/bin/env bash
# Track A Stage-1 smoke on bundled BBB gold, first N segments.
# Copies the internal causal runner invocation; only the paths are public-repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${PY:-python3}"
LIMIT="${LIMIT:-2}"
export VMEM_DATASETS_ROOT="${VMEM_DATASETS_ROOT:-$HERE/data}"

if ! "$PY" "$HERE/scripts/doctor.py"; then
  echo "[run_tracka_smoke] doctor failed; not starting a GPU/perception run." >&2
  exit 2
fi

CAUSAL="$HERE/scripts/evaluate_baselines/trackA/baseline_adapters/causal"
MOVIE="$HERE/assets/trackA/BlenderOpenMovies/big_buck_bunny"
echo "[run_tracka_smoke] adapter=memstrata movie=$MOVIE limit=$LIMIT"
exec "$PY" "$CAUSAL/runner.py" --adapter memstrata --movie-dir "$MOVIE" --limit "$LIMIT"
