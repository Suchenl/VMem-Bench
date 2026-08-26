#!/usr/bin/env bash
# On one BDY node: start S3 shard worker against local endpoints.txt (fleet already up).
# Args: <shard_index> <shard_count>
set -euo pipefail
SHARD_INDEX="${1:?need shard_index}"
SHARD_COUNT="${2:?need shard_count}"
REPO="${REPO:-${MONTAGE_ROOT}}"
MS="$REPO/benchmarks/MemStrata"
RUN_ROOT="${RUN_ROOT:-$MS/data/_runs/s3_bdy_8b}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
STAGE_ROOT="$MS/data/LSMDC/0001_American_Beauty/tmp/pipeline/s3_segment_auto_review_revise"
STAGE="$STAGE_ROOT/shard_${SHARD_INDEX}"
mkdir -p "$LOG_DIR" "$STAGE"

exec > >(tee -a "$LOG_DIR/s3_shard${SHARD_INDEX}_boot.log") 2>&1
echo "=== $(date -Is) S3 shard boot shard=$SHARD_INDEX/$SHARD_COUNT host=$HOST ==="

EP_FILE="$LOG_DIR/endpoints.txt"
if [[ ! -s "$EP_FILE" ]]; then
  # rebuild from live ports
  : > "$EP_FILE"
  for p in 8110 8111 8112 8113 8114 8115 8116 8117; do
    if curl -sf -m 2 "http://127.0.0.1:${p}/v1/models" >/dev/null; then
      echo "http://127.0.0.1:${p}/v1" >> "$EP_FILE"
    fi
  done
fi
mapfile -t ENDPOINTS < "$EP_FILE"
need=${#ENDPOINTS[@]}
if (( need < 1 )); then
  echo "ERROR: no ready endpoints"
  exit 1
fi
ok=0
for url in "${ENDPOINTS[@]}"; do
  curl -sf -m 3 "${url}/models" >/dev/null && ok=$((ok + 1)) || true
done
echo "endpoints ready=$ok/$need"
if (( ok < need )); then
  echo "WARN: not all listed endpoints healthy; using healthy subset"
  : > "$EP_FILE"
  for url in "${ENDPOINTS[@]}"; do
    curl -sf -m 2 "${url}/models" >/dev/null && echo "$url" >> "$EP_FILE"
  done
  mapfile -t ENDPOINTS < "$EP_FILE"
  need=${#ENDPOINTS[@]}
fi
(( need >= 1 )) || { echo "ERROR: zero healthy endpoints"; exit 1; }

# stop previous S3 client only (do not touch vLLM)
while read -r pid; do
  [[ -n "$pid" ]] || continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"s3_segment_auto_review_revise.vlm_auto_review"*)
      kill "$pid" 2>/dev/null || true
      ;;
  esac
done < <(ls /proc | grep -E '^[0-9]+$' || true)
sleep 1

POOL=$(IFS=,; echo "${ENDPOINTS[*]}")
N_WORKERS=$need
source "$MS/src/vmem_bench/annotation/pipeline/services/env_no_proxy.sh" || true
export PYTHONPATH="$MS/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY=${CONDA_ENVS_ROOT}/vllm/bin/python
LOG="$LOG_DIR/s3_shard${SHARD_INDEX}.log"
: > "$LOG"

cd "$MS"
nohup "$PY" -u -m vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review \
  --annotation data/LSMDC/0001_American_Beauty/tmp/pipeline/s2_annotation_postprocess/normalized_annotation.json \
  --source-video ${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched/0001_American_Beauty.mp4 \
  --stage-dir "$STAGE" \
  --reviewer qwen \
  --base-url "$POOL" \
  --model qwen3-vl-8b \
  --max-tokens 4096 \
  --max-review-rounds 1 \
  --fps 2.0 \
  --max-workers "$N_WORKERS" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  >> "$LOG" 2>&1 &
echo $! > "$LOG_DIR/s3_shard${SHARD_INDEX}.pid"
echo "S3_PID=$(cat "$LOG_DIR/s3_shard${SHARD_INDEX}.pid") stage=$STAGE workers=$N_WORKERS pool=$POOL"
sleep 3
ps -p "$(cat "$LOG_DIR/s3_shard${SHARD_INDEX}.pid")" -o pid,etime,stat,cmd || echo 'S3 missing'
echo "=== $(date -Is) S3 shard boot done ==="
