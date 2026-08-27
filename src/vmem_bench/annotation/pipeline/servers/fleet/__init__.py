"""Shared-path VLM fleet registry for the annotation console.

Runtime root (not under ``data/``)::

    runtime/services/vlm_fleet/
      intents/<instance_id>.json      # written by launcher (intent)
      instances/<instance_id>.json    # written by supervisor (truth)

Statuses: ``starting`` | ``running`` | ``terminated``.
Stale heartbeats are treated as offline for dispatch.
"""

from .registry import (
    STATUS_RUNNING,
    STATUS_STARTING,
    STATUS_TERMINATED,
    default_fleet_root,
    list_fleet,
    mark_endpoint_busy,
    mark_endpoint_idle,
    register_intent,
    resolve_dispatch_urls,
    write_instance_status,
)
from .timeutil import now_beijing, now_beijing_iso

__all__ = [
    "STATUS_RUNNING",
    "STATUS_STARTING",
    "STATUS_TERMINATED",
    "default_fleet_root",
    "list_fleet",
    "mark_endpoint_busy",
    "mark_endpoint_idle",
    "now_beijing",
    "now_beijing_iso",
    "register_intent",
    "resolve_dispatch_urls",
    "write_instance_status",
]
