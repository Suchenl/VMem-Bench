#!/usr/bin/env bash
# Login-side supervisor: wait until all 4 BDY nodes have 8 ready vLLMs, then start S3 shards.
# Runs under nohup; safe to leave overnight. Does not kill fleets.
set -euo pipefail
ROOT=${MONTAGE_ROOT}
MS=$ROOT/benchmarks/MemStrata
TGPU=$ROOT/scripts/tgpu_fs.py
FS=${UNSET_INTERNAL_PATH}
LOGDIR=$MS/data/_runs/s3_bdy_8b/logs
CONTINUE_FLEET=$MS/scripts/vmem_bench/ops/bdy-a800/s3/continue_fleet.sh
BOOT_S3=$MS/scripts/vmem_bench/ops/bdy-a800/s3/boot_shard.sh
MARKER=$LOGDIR/s3_american_beauty_launched.ok
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/supervise_fleets_then_s3.log") 2>&1
echo "=== $(date -Is) supervisor start pid=$$ ==="

if [[ -f "$MARKER" ]]; then
  echo "S3 already launched ($(cat "$MARKER")); exit"
  exit 0
fi

NEED_PER_NODE=8
POLL_SEC=120
MAX_WAIT_SEC=$((20 * 3600))
start=$SECONDS

# Host short names for continue_fleet.pid (must match hostname -s on each node)
NODE_HOST=(
  [0]=a800bcctest0033-bd
  [1]=a800bcctest0075-bd
  [2]=a800bcctest0039-bd
  [3]=a800bcctest0104-bd
)

ready_on_node() {
  local node=$1
  local host="${NODE_HOST[$node]}"
  local out
  # Detect boot via pidfile only (never pgrep script name — tgpu cmdline false-positives).
  out=$(python3 "$TGPU" --root "$FS" run \
    --cluster bdy-a800 --node "$node" --cwd /tmp --timeout 50 --wait 45 \
    --job-id "sup-rdy-n${node}-$(date +%s)" \
    --cmd "ready=0; for p in 8110 8111 8112 8113 8114 8115 8116 8117; do curl -sf -m 1 http://127.0.0.1:\$p/v1/models >/dev/null && ready=\$((ready+1)); done; boot=no; pf='$LOGDIR/$host/continue_fleet.pid'; if [[ -f \"\$pf\" ]]; then pid=\$(cat \"\$pf\" 2>/dev/null || true); if [[ -n \"\${pid:-}\" ]] && kill -0 \"\$pid\" 2>/dev/null; then boot=yes; fi; fi; echo READY=\$ready BOOT=\$boot" \
    2>/dev/null | tail -n 5) || true
  echo "$out"
}

ensure_boot() {
  local node=$1
  local pairs
  if [[ "$node" == "0" ]]; then
    pairs='1:8110 2:8111 3:8112 4:8113 5:8114 6:8115 7:8116 0:8117'
  else
    pairs='0:8110 1:8111 2:8112 3:8113 4:8114 5:8115 6:8116 7:8117'
  fi
  local host="${NODE_HOST[$node]}"
  python3 "$TGPU" --root "$FS" run \
    --cluster bdy-a800 --node "$node" --cwd /tmp --timeout 45 --wait 40 \
    --job-id "sup-cont-n${node}-$(date +%s)" \
    --cmd "pf='$LOGDIR/$host/continue_fleet.pid'; if [[ -f \"\$pf\" ]] && kill -0 \$(cat \"\$pf\") 2>/dev/null; then echo already_continuing; else export GPU_PORT_PAIRS='$pairs' PUBLIC_MODELS_ROOT=/tmp/memstrata_public_models; nohup bash '$CONTINUE_FLEET' >'$LOGDIR/node${node}_continue.out' 2>&1 & echo RESTARTED_PID=\$!; fi" \
    || echo "WARN ensure_boot node $node"
}

echo "waiting for ready>=$NEED_PER_NODE on all nodes..."
while (( SECONDS - start < MAX_WAIT_SEC )); do
  all_ok=1
  total=0
  for n in 0 1 2 3; do
    line=$(ready_on_node "$n")
    ready=$(echo "$line" | grep -oE 'READY=[0-9]+' | tail -1 | cut -d= -f2 || echo 0)
    boot=$(echo "$line" | grep -oE 'BOOT=(yes|no)' | tail -1 | cut -d= -f2 || echo no)
    ready=${ready:-0}
    total=$((total + ready))
    echo "$(date -Is) node=$n ready=$ready/8 boot=$boot"
    if (( ready < NEED_PER_NODE )); then
      all_ok=0
      if [[ "$boot" != "yes" ]]; then
        echo "relaunching continue_fleet on node $n"
        ensure_boot "$n"
      fi
    fi
  done
  echo "$(date -Is) total_ready=$total/32"
  if (( all_ok == 1 )); then
    echo "ALL NODES READY"
    break
  fi
  sleep "$POLL_SEC"
done

if (( all_ok != 1 )); then
  echo "ERROR: timed out waiting for full fleets; total_ready=$total"
  exit 1
fi

# node0: ensure GPU0 append if only 7 from main list — ready_on_node already requires 8 ports
echo "launching S3 shards 0..3"
SHARD_COUNT=4
for node in 0 1 2 3; do
  python3 "$TGPU" --root "$FS" run \
    --cluster bdy-a800 --node "$node" --cwd /tmp --timeout 60 --wait 50 \
    --job-id "sup-s3-n${node}-$(date +%s)" \
    --cmd "nohup bash '$BOOT_S3' $node $SHARD_COUNT >'$LOGDIR/node${node}_s3_launcher.out' 2>&1 & echo S3_LAUNCH_PID=\$!; sleep 3; tail -n 15 '$LOGDIR/node${node}_s3_launcher.out' 2>/dev/null || true" \
    || echo "WARN S3 launch node $node"
done

date -Is > "$MARKER"
echo "S3 launched; marker=$MARKER"
# optional bark
python3 "$ROOT/.agents/tools/scripts/send_notification.py" \
  --title "[Done] BDY全卡就绪并开S3" \
  --body "4x8 vLLM ready; American Beauty S3 shards 0-3 started." \
  --group MontageAgent || true
echo "=== $(date -Is) supervisor done ==="
