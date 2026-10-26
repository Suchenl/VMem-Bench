#!/usr/bin/env bash
# Shared NO_PROXY for MemStrata annotation services.
# Source from reusable launchers under pipeline/servers/ only.
#
# Dev machines export http_proxy -> corporate squid. Traffic to A800/H800
# (10.x) and this machine's LAN IP (11.x) must bypass it, or:
# - VLM calls die with 504 Gateway Time-out
# - browser access to the development-machine console on :8890 hangs
#
# curl <7.86 ignores CIDR in no_proxy — we therefore also inject concrete
# hostname + IPs. Browsers use a separate proxy list; access the development
# machine directly on :8890 and do not route the console through Cursor/SSH.

_MEMSTRATA_NO_PROXY_EXTRA="localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,11.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# Concrete hosts (curl + Python urllib honor these even without CIDR support).
if command -v hostname >/dev/null 2>&1; then
  _hn="$(hostname 2>/dev/null || true)"
  _hnf="$(hostname -f 2>/dev/null || true)"
  [[ -n "${_hn}" ]] && _MEMSTRATA_NO_PROXY_EXTRA="${_MEMSTRATA_NO_PROXY_EXTRA},${_hn}"
  [[ -n "${_hnf}" && "${_hnf}" != "${_hn}" ]] && _MEMSTRATA_NO_PROXY_EXTRA="${_MEMSTRATA_NO_PROXY_EXTRA},${_hnf}"
  for _ip in $(hostname -I 2>/dev/null || true); do
    [[ -n "${_ip}" ]] || continue
    _MEMSTRATA_NO_PROXY_EXTRA="${_MEMSTRATA_NO_PROXY_EXTRA},${_ip}"
  done
  unset _hn _hnf _ip
fi

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_MEMSTRATA_NO_PROXY_EXTRA}"
export no_proxy="${no_proxy:+${no_proxy},}${_MEMSTRATA_NO_PROXY_EXTRA}"
unset _MEMSTRATA_NO_PROXY_EXTRA
