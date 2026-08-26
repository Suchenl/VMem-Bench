#!/usr/bin/env bash
# On one BDY node: wait for local 8B fleet, then run two SAM3 crop workers on GPUs 6/7.
set -euo pipefail

REPO="${REPO:-${MONTAGE_ROOT}}"
RUN_ROOT="${RUN_ROOT:-$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/bdy}"
HOST="$(hostname -s)"
LOG_DIR="$RUN_ROOT/logs/$HOST"
NODE_ID="${BDY_NODE_ID:?set BDY_NODE_ID to 0 or 1}"
NUM_NODES="${NUM_NODES:-2}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-2}"
PY=python3

mkdir -p "$LOG_DIR" "$RUN_ROOT/shards"
export PATH=${CONDA_ENVS_ROOT}/vace/bin:$PATH
export PYTHONPATH="$REPO/models/vendor/sam3_transformers59:$REPO/benchmarks/MemStrata/src"
export MEMSTRATA_SAM3_DEPS="$REPO/models/vendor/sam3_transformers59"
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,10.252.0.0/16
export no_proxy=$NO_PROXY

POOL="http://127.0.0.1:8110/v1,http://127.0.0.1:8111/v1,http://127.0.0.1:8112/v1,http://127.0.0.1:8113/v1,http://127.0.0.1:8114/v1,http://127.0.0.1:8115/v1"

echo "[$(date -Is)] node=$NODE_ID host=$HOST waiting for 8B fleet" | tee -a "$LOG_DIR/workers.out"
for i in $(seq 1 180); do
  ok=0
  for p in 8110 8111 8112 8113 8114 8115; do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${p}/v1/models" || true)
    [ "$code" = "200" ] && ok=$((ok + 1))
  done
  echo "[$(date -Is)] ready_endpoints=$ok/6" | tee -a "$LOG_DIR/workers.out"
  if [ "$ok" -ge 4 ]; then
    break
  fi
  if [ "$i" = "180" ]; then
    echo "fleet not ready" | tee -a "$LOG_DIR/workers.out"
    exit 2
  fi
  sleep 10
done

# Two SAM3 workers on GPUs 6 and 7; global shard = node*workers + local_worker
SESSION="memstrata_s5_crop_workers"
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" -n boot "sleep infinity"

for local_w in 0 1; do
  gpu=$((6 + local_w))
  shard=$((NODE_ID * WORKERS_PER_NODE + local_w))
  total_shards=$((NUM_NODES * WORKERS_PER_NODE))
  win="sam3_g${gpu}_shard${shard}"
  log="$LOG_DIR/worker_shard${shard}.log"
  tmux new-window -t "$SESSION" -n "$win" \
    "export CUDA_VISIBLE_DEVICES=$gpu; \
     $PY '$REPO/benchmarks/MemStrata/scripts/vmem_bench/core/run_s5_crops_skip_s3.py' \
       --grounder-base-url '$POOL' \
       --grounder-model qwen3-vl-8b \
       --crop-route propose_and_pick \
       --proposer sam3 \
       --shard-index $shard \
       --num-shards $total_shards \
       --out '$RUN_ROOT/shards/shard${shard}_results.json' \
       --progress '$RUN_ROOT/shards/shard${shard}_progress.jsonl' \
       >'$log' 2>&1"
  echo "started $win shard=$shard/$total_shards gpu=$gpu"
done

echo "[$(date -Is)] workers launched on $HOST" | tee -a "$LOG_DIR/workers.out"
