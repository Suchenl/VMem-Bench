#!/usr/bin/env bash
# Wait until first S5 movie writes crop_proposals.json + progress line.
# Only touches known movie paths (no recursive data/ scan).
set -eu
REPO=${MONTAGE_ROOT}
RUN=$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/kml_a800
LOG=$RUN/logs/first_movie_ready.log
OUT=$RUN/logs/first_movie_ready.out
: > "$LOG"
MOVIES=(
  BlenderOpenMovies/big_buck_bunny_720p
  BlenderOpenMovies/caminandes_1_llama_drama
  LSMDC/0001_American_Beauty
  LSMDC/0002_As_Good_As_It_Gets
)
notify() {
  local title=$1 body=$2
  python3 "$REPO/.agents/tools/scripts/send_notification.py" \
    --title "$title" --body "$body" --group MontageAgent || true
}
for i in $(seq 1 180); do
  props=()
  for m in "${MOVIES[@]}"; do
    f="$REPO/benchmarks/MemStrata/data/$m/tmp/pipeline/s5_entities_visual_crop_acquisition/crop_proposals.json"
    if [[ -f "$f" ]]; then props+=("$f"); fi
  done
  prog=$(cat "$RUN"/shards/shard*_progress.jsonl 2>/dev/null | wc -l || echo 0)
  cand=0
  bbb_c="$REPO/benchmarks/MemStrata/data/BlenderOpenMovies/big_buck_bunny_720p/tmp/pipeline/s5_entities_visual_crop_acquisition/candidates"
  if [[ -d "$bbb_c" ]]; then
    cand=$(ls -1 "$bbb_c"/character/*/*.jpg "$bbb_c"/location/*/*.jpg "$bbb_c"/prop/*/*.jpg 2>/dev/null | wc -l || echo 0)
  fi
  echo "[$(date -Is)] i=$i progress_lines=$prog bbb_cands=$cand n_props=${#props[@]}" | tee -a "$LOG"
  if (( ${#props[@]} > 0 && prog > 0 )); then
    echo READY "${props[@]}" | tee -a "$LOG"
    notify "[Done] S5 first movie crops" "progress=$prog proposals=${#props[@]} bbb_cands=$cand"
    exit 0
  fi
  if ! tgpu -c kml-a800 -node 1 pgrep -f run_s5_crops_skip_s3.py >/dev/null 2>&1; then
    echo WORKERS_DEAD | tee -a "$LOG"
    notify "[Blocked] S5 workers died" "see $LOG"
    exit 1
  fi
  sleep 30
done
echo TIMEOUT | tee -a "$LOG"
notify "[Blocked] S5 first movie timeout" "60min no proposals"
exit 1
