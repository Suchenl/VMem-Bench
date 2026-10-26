#!/usr/bin/env bash
# Restart the annotation console when health checks fail.
# Intended to run under ensure_console.sh --watch (single pidfile instance).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=console_lib.sh
source "${SCRIPT_DIR}/console_lib.sh"

INTERVAL_SECONDS="${CONSOLE_WATCH_INTERVAL_SECONDS:-20}"
FAILS_BEFORE_RESTART="${CONSOLE_WATCH_FAILS:-3}"
CONFIRM_DELAY_SECONDS="${CONSOLE_WATCH_CONFIRM_DELAY_SECONDS:-2}"

echo "[$(date '+%F %T')] watch_console started interval=${INTERVAL_SECONDS}s fails=${FAILS_BEFORE_RESTART}"

fail_streak=0
while true; do
  # Restart only when a service is TRULY dead (process gone or port refusing
  # connections). On a heavily loaded machine (load >> cores) a live backend can
  # answer HTTP health slowly; killing it then causes a self-inflicted restart
  # loop that looks like "frontend keeps dropping". Liveness = process + TCP.
  backend_ok=0
  frontend_ok=0
  console_backend_live && backend_ok=1
  console_frontend_live && frontend_ok=1
  if (( backend_ok && frontend_ok )); then
    fail_streak=0
  else
    fail_streak=$((fail_streak + 1))
    echo "[$(date '+%F %T')] not-live streak=${fail_streak} backend=$([[ ${backend_ok} == 1 ]] && echo live || echo dead) frontend=$([[ ${frontend_ok} == 1 ]] && echo live || echo dead)"
    if (( fail_streak >= FAILS_BEFORE_RESTART )); then
      # Confirm once after a short delay before killing anything.
      sleep "${CONFIRM_DELAY_SECONDS}"
      backend_ok=0
      frontend_ok=0
      console_backend_live && backend_ok=1
      console_frontend_live && frontend_ok=1
      if (( backend_ok && frontend_ok )); then
        echo "[$(date '+%F %T')] recovered on confirmation; no restart"
      else
        echo "[$(date '+%F %T')] confirmed dead; repairing failed component(s)"
        (( frontend_ok )) || console_stop_pidfile annotation_frontend || true
        (( backend_ok )) || console_stop_pidfile annotation_backend || true
        bash "${SCRIPT_DIR}/start_all.sh" || true
        console_wait_healthy "annotation_backend" "${BACKEND_HEALTH_URL}" 40 || true
        console_wait_healthy "annotation_frontend" "${FRONTEND_URL_LOCAL}/" 40 || true
      fi
      fail_streak=0
    fi
  fi
  sleep "${INTERVAL_SECONDS}"
done
