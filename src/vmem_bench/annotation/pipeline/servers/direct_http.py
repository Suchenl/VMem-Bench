"""Direct (proxy-bypassing) HTTP for in-cluster VLM endpoints.

Dev machines often export ``http_proxy`` to a corporate squid. Calls to
A800/H800 reviewer IPs (10.x) then go through the proxy and can fail with
``504 Gateway Time-out`` on long multimodal requests. Pipeline clients must
talk to those endpoints directly.

Notes:
- Python's ``urllib.request.proxy_bypass`` on this stack only matches exact
  hosts / DNS suffixes — not CIDR. So clients use ``urlopen_direct`` (empty
  ``ProxyHandler``) and also register the concrete endpoint host into
  ``NO_PROXY`` for any leftover ``urlopen`` / subprocess tools.
- Shell start scripts still export RFC1918 CIDRs because ``curl`` honors them.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

_DEFAULT_NO_PROXY = (
    "localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,11.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
)
_DIRECT_OPENER = build_opener(ProxyHandler({}))


def ensure_no_proxy_env(extra: str = "") -> None:
    parts: list[str] = []
    parts.extend(p.strip() for p in _DEFAULT_NO_PROXY.split(",") if p.strip())
    if extra:
        parts.extend(p.strip() for p in extra.split(",") if p.strip())
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        if existing:
            parts.extend(p.strip() for p in existing.split(",") if p.strip())
        merged = ",".join(dict.fromkeys(parts))
        os.environ[key] = merged


def ensure_no_proxy_host(host_or_url: str) -> None:
    raw = host_or_url.strip()
    host = urlparse(raw).hostname if "://" in raw else raw
    if host:
        ensure_no_proxy_env(extra=host)
    else:
        ensure_no_proxy_env()


def urlopen_direct(request: Request, timeout: float | None = None) -> Any:
    return _DIRECT_OPENER.open(request, timeout=timeout)
