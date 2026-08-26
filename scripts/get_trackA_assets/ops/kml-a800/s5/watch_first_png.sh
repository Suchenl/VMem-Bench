#!/usr/bin/env bash
# Wait for first RGBA PNG crop under known first-movie S5 dirs.
set -eu
REPO=${MONTAGE_ROOT}
RUN=$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/kml_a800_png
LOG=$RUN/logs/first_png_ready.log
mkdir -p "$RUN/logs"
: > "$LOG"
MOVIES=(
  BlenderOpenMovies/big_buck_bunny_720p
  BlenderOpenMovies/caminandes_1_llama_drama
  LSMDC/0001_American_Beauty
  LSMDC/0002_As_Good_As_It_Gets
)
notify() {
  python3 "$REPO/.agents/tools/scripts/send_notification.py" \
    --title "$1" --body "$2" --group MontageAgent || true
}
for i in $(seq 1 120); do
  found=""
  for m in "${MOVIES[@]}"; do
    c="$REPO/benchmarks/MemStrata/data/$m/tmp/pipeline/s5_entities_visual_crop_acquisition/candidates"
    if ls "$c"/*/*/*.png >/dev/null 2>&1; then
      found=$(ls "$c"/*/*/*.png 2>/dev/null | head -1)
      break
    fi
  done
  prog=$(cat "$RUN"/shards/shard*_progress.jsonl 2>/dev/null | wc -l || echo 0)
  echo "[$(date -Is)] i=$i prog=$prog found=${found:-none}" | tee -a "$LOG"
  if [[ -n "$found" ]]; then
    notify "[Done] S5 PNG crops running" "first=$found prog=$prog"
    exit 0
  fi
  if ! tgpu -c kml-a800 -node 1 pgrep -f run_s5_crops_skip_s3.py >/dev/null 2>&1; then
    notify "[Blocked] S5 PNG relaunch died" "see $RUN/logs"
    exit 1
  fi
  sleep 30
done
notify "[Blocked] S5 PNG timeout" "no png in 60min"
exit 1
