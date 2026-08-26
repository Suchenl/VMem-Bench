#!/usr/bin/env bash
# Dedicated American_Beauty S5 crop on kml-a800 training node (NOT the DT/dev machine).
# GPU6: local qwen3-vl-8b :8113 (avoids shared 32B :8110 contention)
# GPU7: SAM3 propose_and_pick — uses existing training-node SAM3 under /dev/shm if present,
#        else PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} (never stage on DT).
set -eu
REPO=${MONTAGE_ROOT}
RUN=$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/american_beauty_kml
PY=python3
PP=$REPO/models/vendor/sam3_transformers59:$REPO/benchmarks/MemStrata/src
VLLM_SH=$REPO/benchmarks/MemStrata/scripts/vmem_bench/servers/start_annotation_vllm.sh
mkdir -p "$RUN/logs"

if [ -f /dev/shm/memstrata_public_models/facebook/sam3/model.safetensors ]; then
  export PUBLIC_MODELS_ROOT=/dev/shm/memstrata_public_models
else
  export PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}
fi

export PATH=${CONDA_ENVS_ROOT}/vace/bin:$PATH
export PYTHONPATH=$PP
export MEMSTRATA_SAM3_DEPS=$REPO/models/vendor/sam3_transformers59
export PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export no_proxy=$NO_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Start dedicated 8B picker on GPU6 if missing
c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8113/v1/models || true)
if [ "$c" != 200 ]; then
  echo "starting_vlm_8b_g6_p8113 $(date -Is)" | tee -a "$RUN/logs/boot.log"
  export MODEL_SIZE=8B MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.88 SERVED_MODEL_NAME=qwen3-vl-8b
  # Read weights from shared Ceph on the training node — do NOT copy into DT shm.
  export MODEL_PATH=${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-8B-Instruct
  nohup bash "$VLLM_SH" 6 8113 >"$RUN/logs/vllm_8b_g6_p8113.log" 2>&1 &
  echo $! > "$RUN/logs/vllm_8b_g6_p8113.pid"
fi

echo "waiting_vlm $(date -Is)" | tee -a "$RUN/logs/american_beauty_s5.log"
ready=0
for i in $(seq 1 180); do
  c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8113/v1/models || true)
  if [ "$c" = 200 ]; then ready=1; break; fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  echo "VLM_NOT_READY $(date -Is)" | tee -a "$RUN/logs/american_beauty_s5.log"
  tail -50 "$RUN/logs/vllm_8b_g6_p8113.log" | tee -a "$RUN/logs/american_beauty_s5.log" || true
  exit 2
fi

echo "vlm_ready PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT $(date -Is)" | tee -a "$RUN/logs/american_beauty_s5.log"
export CUDA_VISIBLE_DEVICES=7
# Default coverage library + ≤t slot bind (not per-segment Cartesian).
export MEMSTRATA_BENCH_CROP_ATTR_CLASSIFIER="${MEMSTRATA_BENCH_CROP_ATTR_CLASSIFIER:-vlm}"
export MEMSTRATA_BENCH_CROP_ATTR_BASE_URL="${MEMSTRATA_BENCH_CROP_ATTR_BASE_URL:-http://127.0.0.1:8113/v1}"
export MEMSTRATA_BENCH_CROP_ATTR_MODEL="${MEMSTRATA_BENCH_CROP_ATTR_MODEL:-qwen3-vl-8b}"
echo "start_s5 task_mode=coverage $(date -Is)" | tee -a "$RUN/logs/american_beauty_s5.log"
set +e
"$PY" -u "$REPO/benchmarks/MemStrata/scripts/vmem_bench/core/run_s5_crops_skip_s3.py" \
  --grounder-base-url http://127.0.0.1:8113/v1 \
  --grounder-model qwen3-vl-8b \
  --crop-route propose_and_pick \
  --proposer sam3 \
  --task-mode coverage \
  --movie-id 0001_American_Beauty \
  --out "$RUN/american_beauty_results.json" \
  --progress "$RUN/american_beauty_progress.jsonl" \
  >>"$RUN/logs/american_beauty_s5.log" 2>&1
code=$?
set -e
echo "EXIT=$code $(date -Is)" | tee -a "$RUN/logs/american_beauty_s5.log"
S5=$REPO/benchmarks/MemStrata/data/LSMDC/0001_American_Beauty/tmp/pipeline/s5_entities_visual_crop_acquisition
n_png=$(find "$S5" -name '*.png' 2>/dev/null | wc -l || echo 0)
prop=0
if [ -f "$S5/crop_proposals.json" ]; then
  prop=$("$PY" -c "import json;print(len(json.load(open('$S5/crop_proposals.json'))))" 2>/dev/null || echo 0)
fi
if [ "$code" -eq 0 ]; then
  python3 "$REPO/.agents/tools/scripts/send_notification.py" \
    --title "[Done] American_Beauty S5 crop" \
    --body "kml exit=0 proposals=$prop png=$n_png" --group MontageAgent || true
else
  python3 "$REPO/.agents/tools/scripts/send_notification.py" \
    --title "[Blocked] American_Beauty S5 crop" \
    --body "kml exit=$code png=$n_png see $RUN/logs/american_beauty_s5.log" --group MontageAgent || true
fi
exit "$code"
