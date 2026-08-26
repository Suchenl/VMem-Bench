#!/usr/bin/env bash
# Idempotent ensure for the annotation console (+ optional watchdog).
#
#   bash ensure_console.sh           # start/repair + keep watchdog running (default)
#   bash ensure_console.sh --once    # start/repair only (no watch)
#   bash ensure_console.sh --status  # print health only
#   bash ensure_console.sh --stop    # stop frontend/backend/watch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=console_lib.sh
source "${SCRIPT_DIR}/console_lib.sh"

MODE="watch"
for arg in "$@"; do
  case "${arg}" in
    --watch) MODE="watch" ;;
    --status) MODE="status" ;;
    --stop) MODE="stop" ;;
    --ensure|--once) MODE="ensure" ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
  esac
done

start_stack() {
  # shellcheck source=start_all.sh
  # Re-enter via start_all so index paths / VLM flags stay in one place.
  bash "${SCRIPT_DIR}/start_all.sh"
  console_wait_healthy "annotation_backend" "${BACKEND_HEALTH_URL}" 40 || true
  console_wait_healthy "annotation_frontend" "${FRONTEND_URL_LOCAL}/" 40 || true
}

# Container PID 1 (tail -f /dev/null) never reaps orphaned descendants, so the
# console stack must run under a child-subreaper parent (console_init.py) to
# avoid zombie accumulation. For long-running modes, hand off to that reaper
# unless we are already running under it.
REAPER_NAME="annotation_console_init"
maybe_bootstrap_reaper() {
  [[ "${MODE}" == "watch" || "${MODE}" == "ensure" ]] || return 0
  [[ "${MEMSTRATA_CONSOLE_UNDER_REAPER:-0}" == "1" ]] && return 0
  local pid
  pid="$(console_read_pid "${REAPER_NAME}" || true)"
  if console_pid_alive "${pid}"; then
    echo "${REAPER_NAME} already running: pid=${pid} (its watchdog keeps the stack healthy)"
    console_print_access
    exit 0
  fi
  rm -f "${LOG_ROOT}/${REAPER_NAME}.pid"
  local log="${LOG_ROOT}/${REAPER_NAME}.log"
  echo "starting console subreaper (anti-zombie): log=${log}"
  MEMSTRATA_CONSOLE_UNDER_REAPER=1 nohup "${PYTHON_BIN}" \
    -m vmem_bench.annotation.pipeline.servers.console_init \
    >>"${log}" 2>&1 &
  echo "$!" >"${LOG_ROOT}/${REAPER_NAME}.pid"
  console_wait_healthy "annotation_backend" "${BACKEND_HEALTH_URL}" 60 || true
  console_wait_healthy "annotation_frontend" "${FRONTEND_URL_LOCAL}/" 40 || true
  console_print_access
  if console_backend_ok && console_frontend_ok; then exit 0; fi
  exit 1
}

ensure_fleet_health_monitor() {
  console_start_bg fleet_health_monitor "" \
    "${PYTHON_BIN}" -m vmem_bench.annotation.pipeline.servers.fleet.health_monitor \
      --interval "${FLEET_HEALTH_INTERVAL_SECONDS:-20}" \
      --probe-timeout "${FLEET_HEALTH_PROBE_TIMEOUT_SECONDS:-3}"
}

ensure_watch() {
  local name="annotation_console_watch"
  local pidfile="${LOG_ROOT}/${name}.pid"
  local log="${LOG_ROOT}/${name}.log"
  local old_pid
  old_pid="$(console_read_pid "${name}" || true)"
  if console_pid_alive "${old_pid}"; then
    echo "${name} already running: pid=${old_pid}"
    return 0
  fi
  rm -f "${pidfile}"
  echo "starting ${name}: log=${log}"
  nohup bash "${SCRIPT_DIR}/watch_console.sh" >>"${log}" 2>&1 &
  echo "$!" >"${pidfile}"
}

case "${MODE}" in
  status)
    if console_backend_ok; then echo "backend:  OK  ${BACKEND_HEALTH_URL}"; else echo "backend:  DOWN ${BACKEND_HEALTH_URL}"; fi
    if console_frontend_ok; then echo "frontend: OK  ${FRONTEND_URL_LOCAL}/"; else echo "frontend: DOWN ${FRONTEND_URL_LOCAL}/"; fi
    if console_pid_alive "$(console_read_pid annotation_console_watch || true)"; then
      echo "watch:    OK  pid=$(console_read_pid annotation_console_watch)"
    else
      echo "watch:    DOWN"
    fi
    if console_pid_alive "$(console_read_pid fleet_health_monitor || true)"; then
      echo "fleet:    OK  pid=$(console_read_pid fleet_health_monitor)"
    else
      echo "fleet:    DOWN"
    fi
    if console_pid_alive "$(console_read_pid "${REAPER_NAME}" || true)"; then
      echo "reaper:   OK  pid=$(console_read_pid "${REAPER_NAME}") (anti-zombie subreaper)"
    else
      echo "reaper:   DOWN (zombies will accumulate under PID 1=tail)"
    fi
    console_print_access
    if console_backend_ok && console_frontend_ok; then exit 0; fi
    exit 1
    ;;
  stop)
    console_stop_pidfile annotation_console_watch
    console_stop_pidfile fleet_health_monitor
    console_stop_pidfile annotation_frontend
    console_stop_pidfile annotation_backend
    # Only the top-level (non-reaper) invocation tears down the subreaper, so
    # console_init's own SIGTERM-triggered --stop does not signal itself.
    if [[ "${MEMSTRATA_CONSOLE_UNDER_REAPER:-0}" != "1" ]]; then
      console_stop_pidfile "${REAPER_NAME}"
    fi
    echo "console stopped"
    ;;
  ensure|watch)
    maybe_bootstrap_reaper
    if console_backend_ok && console_frontend_ok; then
      echo "console already healthy"
    else
      echo "console unhealthy — repairing"
      start_stack
    fi
    ensure_fleet_health_monitor
    if [[ "${MODE}" == "watch" ]]; then
      ensure_watch
    fi
    console_print_access
    if console_backend_ok && console_frontend_ok; then exit 0; fi
    exit 1
    ;;
esac
