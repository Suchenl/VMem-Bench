#!/usr/bin/env bash
# Stop all MemStrata annotation vLLM endpoints on this BDY node.
# Safe for tgpu_fs: no "vllm serve" / broad pkill in the parent job argv.
# Run ON the GPU node (via tgpu_fs), not on the IDE machine.
set -euo pipefail

HOST="$(hostname -s)"
echo "=== $(date -Is) stop BDY vLLM host=$HOST ==="

kill_port() {
  local port="$1"
  local pid cmd
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/envs/vllm/bin/vllm serve"*"--port ${port}"*|*"/bin/vllm serve"*"--port ${port}"*)
        echo "stop pid=$pid port=$port"
        kill "$pid" 2>/dev/null || true
        ;;
      *"fleet.supervise"*"--port ${port}"*)
        echo "stop supervise pid=$pid port=$port"
        kill "$pid" 2>/dev/null || true
        ;;
    esac
  done
  sleep 1
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/envs/vllm/bin/vllm serve"*"--port ${port}"*|*"/bin/vllm serve"*"--port ${port}"*|*"fleet.supervise"*"--port ${port}"*)
        kill -9 "$pid" 2>/dev/null || true
        ;;
    esac
  done
}

for port in 8110 8111 8112 8113 8114 8115 8116 8117; do
  kill_port "$port"
done

# vLLM 0.16 separates EngineCore into children whose command line does not
# include a port. A parent-only stop leaves GPU contexts alive and causes the
# next launch to hit address-in-use / engine-init conflicts. This BDY restart
# owns every MemStrata VLM rank on the node, so clear those children too.
for pidpath in /proc/[0-9]*; do
  pid=${pidpath#/proc/}
  cmd=$(tr '\0' ' ' < "$pidpath/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"VLLM::EngineCore"*|*"/envs/vllm/bin/vllm serve"*|*"fleet.supervise"*)
      echo "stop vllm process pid=$pid"
      kill "$pid" 2>/dev/null || true
      ;;
  esac
done
sleep 2
for pidpath in /proc/[0-9]*; do
  pid=${pidpath#/proc/}
  cmd=$(tr '\0' ' ' < "$pidpath/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"VLLM::EngineCore"*|*"/envs/vllm/bin/vllm serve"*|*"fleet.supervise"*)
      kill -9 "$pid" 2>/dev/null || true
      ;;
  esac
done

# This is intentionally BDY-only. Never target H800/KML sessions from a BDY
# recovery command.
for sess in memstrata_s3_8b_fleet memstrata_s5_8b_fleet memstrata_bdy_8b_fleet; do
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$sess" 2>/dev/null; then
    echo "kill tmux session=$sess"
    tmux kill-session -t "$sess" || true
  fi
done

# Stop continue/boot launchers if still holding a pidfile.
REPO="${REPO:-${MONTAGE_ROOT}}"
LOGDIR="$REPO/benchmarks/MemStrata/data/_runs/s3_bdy_8b/logs/$HOST"
if [[ -f "$LOGDIR/continue_fleet.pid" ]]; then
  pid=$(cat "$LOGDIR/continue_fleet.pid" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "stop continue_fleet pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$LOGDIR/continue_fleet.pid"
fi

for attempt in $(seq 1 30); do
  engines=$(ps -eo args 2>/dev/null | awk '/VLLM::EngineCore|\/envs\/vllm\/bin\/vllm serve|fleet\.supervise/ {count++} END {print count+0}')
  if [[ "$engines" == "0" ]]; then
    break
  fi
  echo "waiting for $engines vllm process(es) to exit (${attempt}/30)"
  sleep 2
done

alive=0
for port in 8110 8111 8112 8113 8114 8115 8116 8117; do
  if curl -sf -m 1 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "STILL_UP port=$port"
    alive=$((alive + 1))
  fi
done
echo "DONE host=$HOST still_up=$alive"
exit 0
