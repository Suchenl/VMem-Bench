#!/usr/bin/env bash
# Hard-reset picker ports and start exactly one 8B on GPU0:8110.
# Avoids pkill -f patterns that can match the tgpu_fs wrapper.
set -eu
REPO=${MONTAGE_ROOT}
RUN=$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/bdy
HOST=$(hostname -s)
LOG_DIR=$RUN/logs/$HOST
mkdir -p "$LOG_DIR"
REPORT=$LOG_DIR/single_launch_report.txt
exec > >(tee "$REPORT") 2>&1
echo "START $(date -Is) host=$HOST"

# Collect PIDs of vllm serve (numeric only), kill them.
mapfile -t PIDS < <(pgrep -f '/vllm serve ' || true)
echo "found_vllm_pids=${PIDS[*]:-none}"
for pid in "${PIDS[@]:-}"; do
  [ -n "${pid:-}" ] || continue
  echo "kill $pid"
  kill "$pid" 2>/dev/null || true
done
sleep 3
mapfile -t LEFT < <(pgrep -f '/vllm serve ' || true)
echo "left_after_kill=${LEFT[*]:-none}"
for pid in "${LEFT[@]:-}"; do
  [ -n "${pid:-}" ] || continue
  kill -9 "$pid" 2>/dev/null || true
done
tmux has-session -t memstrata_s5_8b_fleet 2>/dev/null && tmux kill-session -t memstrata_s5_8b_fleet || true
tmux has-session -t memstrata_s5_crop_workers 2>/dev/null && tmux kill-session -t memstrata_s5_crop_workers || true

NVJIT=$(${CONDA_ENVS_ROOT}/vllm/bin/python -c 'import pathlib,sys; print(pathlib.Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"/"nvidia"/"nvjitlink"/"lib")')
export LD_LIBRARY_PATH="${NVJIT}:${LD_LIBRARY_PATH:-}"
export MODEL_SIZE=8B MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.88 SERVED_MODEL_NAME=qwen3-vl-8b
export PYTHONUNBUFFERED=1 PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=$NO_PROXY

LOG=$LOG_DIR/vllm_single_g0_p8110.log
: > "$LOG"
echo "launching single 8B LD=$NVJIT"
nohup bash "$REPO/benchmarks/MemStrata/scripts/vmem_bench/servers/start_annotation_vllm.sh" 0 8110 >>"$LOG" 2>&1 &
echo $! > "$LOG_DIR/vllm_single_g0_p8110.pid"
echo "pid=$(cat $LOG_DIR/vllm_single_g0_p8110.pid)"
sleep 2
ps -o pid,stat,etime,cmd -p "$(cat $LOG_DIR/vllm_single_g0_p8110.pid)" || true
echo "DONE $(date -Is)"
