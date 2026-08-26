#!/usr/bin/env bash
# Shared helpers for MemStrata annotation console control scripts.
# shellcheck shell=bash

console_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_no_proxy.sh
source "${console_lib_dir}/env_no_proxy.sh"

MEMSTRATA_ROOT="$(cd "${console_lib_dir}/../../../../.." && pwd)"
SRC_ROOT="${MEMSTRATA_ROOT}/src"
DATA_ROOT="${DATA_ROOT:-${MEMSTRATA_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/_services/annotation_console/logs}"
JOBS_ROOT="${JOBS_ROOT:-${DATA_ROOT}/_services/annotation_jobs}"
# Resolve the shared tgpu node map from the checkout layout without embedding a
# user-specific absolute path. Operators may still override this explicitly.
if [[ -z "${MEMSTRATA_NODES_TSV:-}" ]]; then
  _workspace_root="$(cd "${MEMSTRATA_ROOT}/../.." && pwd)"
  _nodes_candidate="${_workspace_root}/../../ssh_tunnel/nodes.tsv"
  if [[ -f "${_nodes_candidate}" ]]; then
    export MEMSTRATA_NODES_TSV="${_nodes_candidate}"
  fi
  unset _workspace_root _nodes_candidate
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Prefer the F3GS env used by this machine's annotation console when unset/default.
if [[ "${PYTHON_BIN}" == "python3" || "${PYTHON_BIN}" == "python" ]]; then
  for _py in \
    "F3GS/bin/python" \
    "${HOME}/miniconda3/envs/F3GS/bin/python"
  do
    if [[ -x "${_py}" ]]; then
      PYTHON_BIN="${_py}"
      break
    fi
  done
  unset _py
fi
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-7864}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
# The user-facing console is served directly by the development machine.
# Keep this distinct from Cursor/SSH port forwarding, which is less stable.
FRONTEND_PORT="${FRONTEND_PORT:-8890}"
BACKEND_URL="${BACKEND_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"
FRONTEND_URL_LOCAL="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_HEALTH_URL="${BACKEND_URL%/}/api/health"

mkdir -p "${LOG_ROOT}" "${JOBS_ROOT}"
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

console_primary_ips() {
  # Prefer non-loopback IPv4 addresses for access hints / no_proxy.
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF && $1 !~ /^127\./ {print $1}'
  fi
}

console_http_ok() {
  local url="$1"
  local timeout="${2:-3}"
  curl -fsS --noproxy '*' --max-time "${timeout}" "${url}" >/dev/null 2>&1
}

# HTTP health with load-tolerant timeouts. On a busy dev machine (load > cores)
# a live backend can take >5s to answer; keep these generous so the watchdog
# does not kill a slow-but-alive service.
CONSOLE_BACKEND_HTTP_TIMEOUT="${CONSOLE_BACKEND_HTTP_TIMEOUT:-25}"
CONSOLE_FRONTEND_HTTP_TIMEOUT="${CONSOLE_FRONTEND_HTTP_TIMEOUT:-10}"

console_backend_ok() {
  console_http_ok "${BACKEND_HEALTH_URL}" "${CONSOLE_BACKEND_HTTP_TIMEOUT}"
}

console_frontend_ok() {
  console_http_ok "${FRONTEND_URL_LOCAL}/" "${CONSOLE_FRONTEND_HTTP_TIMEOUT}"
}

# TCP liveness: is anything accepting connections on the port right now?
# Distinguishes "truly dead / connection refused" from "alive but slow HTTP".
console_tcp_open() {
  local host="$1"
  local port="$2"
  local timeout="${3:-3}"
  timeout "${timeout}" bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null
}

# A service is considered LIVE if its process is alive AND its port accepts a
# TCP connection. HTTP slowness alone must not count as "down".
console_backend_live() {
  local pid
  pid="$(console_read_pid annotation_backend || true)"
  console_pid_alive "${pid}" || return 1
  console_tcp_open "${BACKEND_HOST}" "${BACKEND_PORT}" 3
}

console_frontend_live() {
  local pid
  pid="$(console_read_pid annotation_frontend || true)"
  console_pid_alive "${pid}" || return 1
  console_tcp_open "127.0.0.1" "${FRONTEND_PORT}" 3
}

console_pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local stat
  stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  [[ "${stat}" != *Z* ]]
}

console_read_pid() {
  local name="$1"
  local pidfile="${LOG_ROOT}/${name}.pid"
  [[ -s "${pidfile}" ]] || return 0
  tr -d '[:space:]' <"${pidfile}"
}

console_stop_pidfile() {
  local name="$1"
  local pidfile="${LOG_ROOT}/${name}.pid"
  local pid
  pid="$(console_read_pid "${name}" || true)"
  if console_pid_alive "${pid}"; then
    echo "${name}: stopping pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      console_pid_alive "${pid}" || break
      sleep 0.2
    done
    if console_pid_alive "${pid}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pidfile}"
}

console_start_bg() {
  # Usage: console_start_bg NAME HEALTH_URL cmd...
  local name="$1"
  local health_url="$2"
  shift 2
  local log="${LOG_ROOT}/${name}.log"
  local pidfile="${LOG_ROOT}/${name}.pid"
  local old_pid
  old_pid="$(console_read_pid "${name}" || true)"

  if console_pid_alive "${old_pid}"; then
    if [[ -n "${health_url}" ]] && console_http_ok "${health_url}" 3; then
      echo "${name} already healthy: pid=${old_pid} log=${log}"
      return 0
    fi
    echo "${name}: pid=${old_pid} alive but unhealthy — restarting"
    console_stop_pidfile "${name}"
  else
    rm -f "${pidfile}"
  fi

  # Port may still be held by an orphan outside our pidfile.
  if [[ -n "${health_url}" ]] && console_http_ok "${health_url}" 2; then
    echo "${name}: port already healthy (external/orphan process); skipping launch"
    return 0
  fi

  echo "starting ${name}: log=${log}"
  nohup "$@" >>"${log}" 2>&1 &
  echo "$!" >"${pidfile}"
}

console_wait_healthy() {
  local label="$1"
  local url="$2"
  local tries="${3:-30}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if console_http_ok "${url}" 3; then
      echo "${label}: healthy (${url})"
      return 0
    fi
    sleep 0.4
  done
  echo "${label}: still unhealthy after ${tries} probes — see ${LOG_ROOT}" >&2
  return 1
}

console_print_access() {
  local ip
  echo
  echo "MemStrata annotation console"
  echo "  Local:    ${FRONTEND_URL_LOCAL}"
  echo "  Backend:  ${BACKEND_HEALTH_URL}"
  echo "  Logs:     ${LOG_ROOT}"
  echo "  Primary UI: KML HTTPS AccessProxy → :${FRONTEND_PORT} (e.g. https://…example.com/)."
  echo "  If the UI shows SSO/login HTML, refresh the page to renew accessproxy_session."
  while read -r ip; do
    [[ -n "${ip}" ]] || continue
    echo "  LAN debug: http://${ip}:${FRONTEND_PORT}  (optional; bypass corporate squid)"
  done < <(console_primary_ips)
  echo
}
