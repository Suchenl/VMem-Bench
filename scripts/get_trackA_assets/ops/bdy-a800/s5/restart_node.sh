#!/usr/bin/env bash
# Minimal BDY restart; always writes a shared log even on failure.
NODE_ID="${1:?need node id}"
REPO=${MONTAGE_ROOT}
RUN=$REPO/benchmarks/MemStrata/data/_runs/s5_skip_s3/bdy
LOG=$RUN/logs/restart_node${NODE_ID}.log
mkdir -p "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1
set -x
echo "START $(date -Is) host=$(hostname) node=$NODE_ID"

# Stop old vLLMs gently
for port in 8110 8111 8112 8113 8114 8115; do
  pids=$(pgrep -f "vllm serve .*--port ${port}" || true)
  echo "port $port pids=$pids"
  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
  done
done
tmux kill-session -t memstrata_s5_8b_fleet 2>/dev/null || true
tmux kill-session -t memstrata_s5_crop_workers 2>/dev/null || true
sleep 3

NVJIT=$(${CONDA_ENVS_ROOT}/vllm/bin/python -c 'import pathlib,sys; print(pathlib.Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"/"nvidia"/"nvjitlink"/"lib")')
echo "NVJIT=$NVJIT"
export LD_LIBRARY_PATH="${NVJIT}:${LD_LIBRARY_PATH:-}"
${CONDA_ENVS_ROOT}/vllm/bin/python -c 'import torch; print("torch_ok", torch.cuda.is_available(), torch.cuda.device_count())'

bash "$REPO/benchmarks/MemStrata/scripts/vmem_bench/ops/bdy-a800/s5/launch_8b_fleet.sh"
export BDY_NODE_ID="$NODE_ID"
nohup bash "$REPO/benchmarks/MemStrata/scripts/vmem_bench/ops/bdy-a800/s5/launch_workers.sh" \
  >"$RUN/logs/node${NODE_ID}_workers_launcher.out" 2>&1 &
echo "workers_pid=$!"
echo "DONE $(date -Is)"
