#!/usr/bin/env bash
# From login: nohup stage+/tmp fleet boot on all 4 BDY nodes (staggered copies).
# Short tgpu jobs only launch background boots — boots keep running after job ends.
set -euo pipefail
ROOT=${MONTAGE_ROOT}
MS=$ROOT/benchmarks/MemStrata
TGPU_FS=$ROOT/scripts/tgpu_fs.py
FS_ROOT=${UNSET_INTERNAL_PATH}
BOOT=$MS/scripts/vmem_bench/ops/bdy-a800/s3/boot_tmp_fleet.sh
LOGDIR=$MS/data/_runs/s3_bdy_8b/logs
declare -A NODE_HOST=(
  [0]=a800bcctest0033-bd.bce.example.org
  [1]=a800bcctest0075-bd.bce.example.org
  [2]=a800bcctest0039-bd.bce.example.org
  [3]=a800bcctest0104-bd.bce.example.org
)
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/submit_tmp_fleets.log") 2>&1
echo "=== $(date -Is) submit tmp fleets ==="
python3 "$TGPU_FS" --root "$FS_ROOT" status --cluster bdy-a800

for node in 0 1 2 3; do
  host="${NODE_HOST[$node]}"
  if [[ "$node" == "0" ]]; then
    gpu_list="1 2 3 4 5 6 7"
  else
    gpu_list="0 1 2 3 4 5 6 7"
  fi
  # 3 min between nodes for Ceph copy; node0 already staged → delay still ok (skip copy)
  stage_delay=$((node * 180))
  job_id="tmp-fleet-n${node}-$(date +%s)"
  # IMPORTANT: job cmdline must NOT contain 'vllm serve' / broad pkill patterns.
  cmd="export FLEET_CLUSTER=bdy-a800 FLEET_NODE_ID='$node' FLEET_ADVERTISE_HOST='$host' GPU_LIST='$gpu_list' STAGE_BEFORE_SEC=$stage_delay LOCAL_PUBLIC_MODELS_ROOT=/tmp/memstrata_public_models ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-.}; nohup bash '$BOOT' >'$LOGDIR/node${node}_tmp_fleet_launcher.out' 2>&1 & echo NODE${node}_BOOT_PID=\$!; sleep 2; tail -n 8 '$LOGDIR/node${node}_tmp_fleet_launcher.out' 2>/dev/null || true"
  python3 "$TGPU_FS" --root "$FS_ROOT" run \
    --cluster bdy-a800 --node "$node" --cwd /tmp --timeout 45 --wait 40 \
    --job-id "$job_id" --cmd "$cmd" || echo "WARN submit node $node"
done
echo "submitted 4 background fleet boots (nodeN stage delay=N*180s)"
echo "=== $(date -Is) done ==="
