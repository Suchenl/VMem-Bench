#!/usr/bin/env bash
# Launch S5 crop workers on selected kml-a800 node1 GPUs.
# Stages facebook/sam3 onto /dev/shm first so Ceph does not D-state all workers.
set -eu
REPO=${MONTAGE_ROOT}
# Fresh run dir after crop_io RGBA-PNG storage change (old jpg outputs under kml_a800/).
RUN=${RUN:-$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/kml_a800_png}
LOG=$RUN/logs
mkdir -p "$LOG" "$RUN/shards"
PY=python3
export PATH=${CONDA_ENVS_ROOT}/vace/bin:$PATH
PP=$REPO/models/vendor/sam3_transformers59:$REPO/benchmarks/MemStrata/src
export PYTHONPATH=$PP
export MEMSTRATA_SAM3_DEPS=$REPO/models/vendor/sam3_transformers59
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export no_proxy=$NO_PROXY
POOL=${POOL:-http://127.0.0.1:8110/v1}
MODEL=${MODEL:-qwen3-vl-32b}
WORKER_GPUS=${WORKER_GPUS:-2,3,4,5}
IFS=',' read -r -a GPU_ARR <<< "$WORKER_GPUS"
(( ${#GPU_ARR[@]} > 0 )) || { echo "WORKER_GPUS must not be empty" >&2; exit 2; }

# Local public-models root (shm) — only facebook/sam3 is required for proposer.
LOCAL_PUBLIC=${LOCAL_PUBLIC_MODELS_ROOT:-/dev/shm/memstrata_public_models}
SRC_SAM3=${SRC_SAM3:-${PUBLIC_MODELS_ROOT}/facebook/sam3}
DST_SAM3=$LOCAL_PUBLIC/facebook/sam3

c=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${POOL%/}/models" || true)
echo "vlm_ready_code=$c" | tee "$LOG/workers_launch.log"
[ "$c" = "200" ] || exit 2

$PY -c "import vmem_bench; print('import_ok', vmem_bench.__file__)" | tee -a "$LOG/workers_launch.log"

echo "staging sam3 -> $DST_SAM3" | tee -a "$LOG/workers_launch.log"
mkdir -p "$LOCAL_PUBLIC/facebook"
if [ ! -f "$DST_SAM3/model.safetensors" ]; then
  # Single-stream copy; avoid parallel Ceph stampede.
  rm -rf "$DST_SAM3"
  mkdir -p "$DST_SAM3"
  # Prefer model.safetensors + tokenizer/config; skip duplicate sam3.pt (same size).
  for f in config.json configuration.json processor_config.json \
           special_tokens_map.json tokenizer_config.json tokenizer.json \
           merges.txt vocab.json LICENSE README.md model.safetensors; do
    if [ -f "$SRC_SAM3/$f" ]; then
      echo "  copy $f" | tee -a "$LOG/workers_launch.log"
      cp -f "$SRC_SAM3/$f" "$DST_SAM3/$f"
    fi
  done
fi
ls -lah "$DST_SAM3" | tee -a "$LOG/workers_launch.log"
export PUBLIC_MODELS_ROOT=$LOCAL_PUBLIC

# Kill previous session / stuck D-state workers.
SESSION=${SESSION:-memstrata_s5_kml_workers}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION" || true
fi
if [[ "${STOP_EXISTING_WORKERS:-0}" == "1" ]]; then
  pkill -f 'run_s5_crops_skip_s3.py' 2>/dev/null || true
  sleep 2
  # Force-kill any remaining D-stuck children only when explicitly requested.
  pkill -9 -f 'run_s5_crops_skip_s3.py' 2>/dev/null || true
fi

tmux new-session -d -s "$SESSION" -n boot "sleep infinity"

# Stagger starts so only one process hits from_pretrained at a time.
STAGGER_SEC=${STAGGER_SEC:-90}
for local_w in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$local_w]// /}
  shard=$local_w
  log=$LOG/worker_shard${shard}.log
  : > "$log"
  delay=$((local_w * STAGGER_SEC))
  tmux new-window -t "$SESSION" -n "s${shard}_g${gpu}" \
    "export CUDA_VISIBLE_DEVICES=${gpu} PATH=${PATH} PYTHONPATH=${PP} \
       MEMSTRATA_SAM3_DEPS=${MEMSTRATA_SAM3_DEPS} \
       PUBLIC_MODELS_ROOT=${LOCAL_PUBLIC} \
       PYTHONUNBUFFERED=1 \
       NO_PROXY=${NO_PROXY} no_proxy=${no_proxy}; \
     echo waiting_stagger_${delay}s; sleep ${delay}; \
     echo start_shard_${shard} \$(date -Is); \
     ${PY} -u ${REPO}/benchmarks/MemStrata/scripts/vmem_bench/core/run_s5_crops_skip_s3.py \
       --grounder-base-url ${POOL} \
       --grounder-model ${MODEL} \
       --crop-route propose_and_pick \
       --proposer sam3 \
       --shard-index ${shard} \
       --num-shards ${#GPU_ARR[@]} \
       --out ${RUN}/shards/shard${shard}_results.json \
       --progress ${RUN}/shards/shard${shard}_progress.jsonl \
       >>${log} 2>&1; echo EXIT=\$? >>${log}"
  echo "queued shard=$shard gpu=$gpu stagger=${delay}s"
done
tmux list-windows -t "$SESSION"
echo DONE $(date -Is) | tee -a "$LOG/workers_launch.log"
