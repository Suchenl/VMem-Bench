"""Atomic intent / instance status files on the shared fleet root."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from vmem_bench.annotation.pipeline.servers.fleet.nodes import load_nodes_tsv
from vmem_bench.annotation.pipeline.servers.fleet.timeutil import BEIJING, now_beijing

STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_TERMINATED = "terminated"
VALID_STATUSES = frozenset({STATUS_STARTING, STATUS_RUNNING, STATUS_TERMINATED})

# If supervisor stops heartbeating, treat as offline even if status says running.
DEFAULT_STALE_SECONDS = 90
DEFAULT_PROBE_TIMEOUT_SECONDS = 2.0
# Lease heartbeat missing longer than this → treat as idle (crashed worker).
DEFAULT_BUSY_STALE_SECONDS = 1800
DEFAULT_STARTING_STALE_SECONDS = 900

# The console polls /api/health and the fleet panel frequently. Each non-probe
# list_fleet() call stats/reads ~4 KFS files per instance (intents + instances +
# workloads + breaks); with 24 instances that is ~100 Ceph metadata ops. Under
# KFS pressure (e.g. right after a rerun) those calls degraded to 30s+, so the
# the reverse proxy timed out and returned an HTML page that the frontend misread as
# an SSO logout. Cache the read-only (probe=False) snapshot for a few seconds so
# repeated health polls collapse into one KFS scan. probe=True stays live so
# dispatch decisions never use a stale reachability result.
_FLEET_CACHE_LOCK = threading.Lock()
_FLEET_CACHE: dict[str, Any] = {}
try:
    _FLEET_CACHE_TTL_SEC = float(os.environ.get("MEMSTRATA_FLEET_CACHE_TTL_SEC", "4") or 4)
except (TypeError, ValueError):
    _FLEET_CACHE_TTL_SEC = 4.0


def _invalidate_fleet_cache(root: Path) -> None:
    """Drop cached console snapshots after a mutation to this fleet root."""
    prefix = f"{root}|"
    with _FLEET_CACHE_LOCK:
        for key in tuple(_FLEET_CACHE):
            if key.startswith(prefix):
                del _FLEET_CACHE[key]


def memstrata_root_from_here() -> Path:
    # fleet/registry.py -> fleet -> servers -> pipeline -> annotation -> vmem_bench -> src -> MemStrata
    return Path(__file__).resolve().parents[6]


def default_fleet_root() -> Path:
    override = os.environ.get("MEMSTRATA_FLEET_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return memstrata_root_from_here() / "runtime" / "services" / "vlm_fleet"


def advertise_host() -> str:
    return (
        os.environ.get("FLEET_ADVERTISE_HOST", "").strip()
        or socket.getfqdn()
        or socket.gethostname()
        or "127.0.0.1"
    )


def make_instance_id(*, host: str, port: int) -> str:
    safe_host = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in host)
    return f"{safe_host}-{int(port)}"


def base_url_for(*, host: str, port: int) -> str:
    return f"http://{host}:{int(port)}/v1"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt shared files are offline, not fatal
        return None
    return payload if isinstance(payload, dict) else None


def register_intent(
    *,
    port: int,
    model: str,
    gpu: str = "",
    role: str = "reviewer",
    host: str | None = None,
    command: str = "",
    fleet_root: Path | None = None,
    cluster: str = "",
    node_id: str = "",
    gpu_rank: int | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Launcher writes desired config (intent). Does not claim the service is up."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    host_name = host or advertise_host()
    instance_id = make_instance_id(host=host_name, port=port)
    now = now_beijing()
    rank: int | str | None = gpu_rank
    if rank is None or str(rank).strip() == "":
        rank = None
    elif str(rank).isdigit():
        rank = int(rank)
    if rank is None and str(gpu).isdigit():
        rank = int(gpu)
    payload: dict[str, Any] = {
        "instance_id": instance_id,
        "host": host_name,
        "port": int(port),
        "base_url": base_url_for(host=host_name, port=port),
        "model": str(model or ""),
        "gpu": str(gpu or ""),
        "gpu_rank": rank if rank is not None else "",
        "cluster": str(cluster or os.environ.get("FLEET_CLUSTER", "") or ""),
        "node_id": str(node_id if node_id != "" else os.environ.get("FLEET_NODE_ID", "") or ""),
        "role": str(role or "reviewer"),
        "command": str(command or ""),
        "registered_at": now,
        "registered_by_pid": os.getpid(),
        "registered_by_host": socket.gethostname(),
        "timezone": "Asia/Shanghai",
    }
    if extra:
        payload.update(extra)
    _atomic_write_json(root / "intents" / f"{instance_id}.json", payload)
    _invalidate_fleet_cache(root)
    return payload


def write_instance_status(
    *,
    instance_id: str,
    status: str,
    fleet_root: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Supervisor / server process writes truth status (+ heartbeat)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    path = root / "instances" / f"{instance_id}.json"
    current = _read_json(path) or {"instance_id": instance_id}
    now = now_beijing()
    current.update(fields)
    current["instance_id"] = instance_id
    current["status"] = status
    current["updated_at"] = now
    current["timezone"] = "Asia/Shanghai"
    if status == STATUS_RUNNING:
        current["heartbeat_at"] = now
    if status == STATUS_TERMINATED:
        current["terminated_at"] = now
    # A supervisor restart is a new service start, not a continuation of the
    # prior process. Keeping an old timestamp misleads console operators.
    if status == STATUS_STARTING:
        current["started_at"] = now
    _atomic_write_json(path, current)
    _invalidate_fleet_cache(root)
    return current


def instance_id_from_base_url(base_url: str) -> str:
    """Map an OpenAI-compatible base URL to the fleet instance id."""
    parsed = urlparse(str(base_url or "").rstrip("/"))
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"invalid base_url for fleet workload: {base_url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return make_instance_id(host=host, port=int(port))


def mark_endpoint_busy(
    base_url: str,
    *,
    fleet_root: Path | None = None,
    job_id: str = "",
    movie_id: str = "",
    dataset: str = "",
    segment_id: str = "",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Client lease: mark endpoint busy in a sidecar workload file (not overwritten by heartbeat)."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    instance_id = instance_id_from_base_url(base_url)
    payload: dict[str, Any] = {
        "instance_id": instance_id,
        "base_url": str(base_url).rstrip("/"),
        "busy": True,
        "job_id": str(job_id or ""),
        "movie_id": str(movie_id or ""),
        "dataset": str(dataset or ""),
        "segment_id": str(segment_id or ""),
        "stage": str(stage or ""),
        "leased_at": now_beijing(),
        "updated_at": now_beijing(),
        "timezone": "Asia/Shanghai",
        "worker_pid": os.getpid(),
        "worker_host": socket.gethostname(),
    }
    if extra:
        payload.update(extra)
    _atomic_write_json(root / "workloads" / f"{instance_id}.json", payload)
    _invalidate_fleet_cache(root)
    return payload


def mark_endpoint_idle(base_url: str, *, fleet_root: Path | None = None) -> None:
    """Client release: clear busy marker."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    try:
        instance_id = instance_id_from_base_url(base_url)
    except ValueError:
        return
    path = root / "workloads" / f"{instance_id}.json"
    if path.is_file():
        path.unlink(missing_ok=True)
    _invalidate_fleet_cache(root)


def _parse_ts(value: Any, *, timezone_hint: str | None = None) -> float | None:
    """Parse a timestamp to epoch seconds.

    Naive stamps:
    - ``Asia/Shanghai`` hint, or space-separated ``YYYY-MM-DD HH:MM:SS`` → Beijing
    - legacy ``YYYY-MM-DDTHH:MM:SS`` without hint → UTC (old supervise writers)
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            hint = (timezone_hint or "").strip()
            if hint in {"Asia/Shanghai", "CST", "Beijing"} or ("T" not in text and " " in text):
                dt = dt.replace(tzinfo=BEIJING)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
            if fmt.startswith("%Y-%m-%d %H") or (timezone_hint or "").strip() in {
                "Asia/Shanghai",
                "CST",
                "Beijing",
            }:
                dt = naive.replace(tzinfo=BEIJING)
            else:
                dt = naive.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _is_stale(instance: dict[str, Any], *, now: float, stale_seconds: float) -> bool:
    status = str(instance.get("status") or "")
    if status != STATUS_RUNNING:
        return False
    hint = str(instance.get("timezone") or "")
    hb = _parse_ts(
        instance.get("heartbeat_at") or instance.get("updated_at"),
        timezone_hint=hint,
    )
    if hb is None:
        return True
    return (now - hb) > stale_seconds


def _read_workload(
    root: Path,
    instance_id: str,
    *,
    now: float,
    busy_stale_seconds: float,
) -> dict[str, Any] | None:
    payload = _read_json(root / "workloads" / f"{instance_id}.json")
    if not payload or not payload.get("busy"):
        return None
    leased = _parse_ts(payload.get("updated_at") or payload.get("leased_at"))
    if leased is not None and (now - leased) > busy_stale_seconds:
        return None
    return payload


def _read_break(root: Path, instance_id: str, *, now: float) -> dict[str, Any] | None:
    payload = _read_json(root / "breaks" / f"{instance_id}.json")
    if not payload or not payload.get("active", True):
        return None
    until = _parse_ts(payload.get("until"), timezone_hint=str(payload.get("timezone") or ""))
    if until is not None and until <= now:
        return None
    return payload


def mark_endpoint_break(
    base_url: str,
    *,
    fleet_root: Path | None = None,
    reason: str,
    until: str = "",
) -> dict[str, Any]:
    """Pause dispatch to an otherwise healthy endpoint without killing it."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    instance_id = instance_id_from_base_url(base_url)
    payload = {
        "instance_id": instance_id,
        "active": True,
        "reason": str(reason or "operator pause"),
        "until": str(until or ""),
        "set_at": now_beijing(),
        "timezone": "Asia/Shanghai",
    }
    _atomic_write_json(root / "breaks" / f"{instance_id}.json", payload)
    _invalidate_fleet_cache(root)
    return payload


def clear_endpoint_break(base_url: str, *, fleet_root: Path | None = None) -> None:
    """Resume dispatch to an endpoint previously paused with ``mark_endpoint_break``."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    try:
        instance_id = instance_id_from_base_url(base_url)
    except ValueError:
        return
    (root / "breaks" / f"{instance_id}.json").unlink(missing_ok=True)
    _invalidate_fleet_cache(root)


def _resolved_start_at(intent: dict[str, Any], instance: dict[str, Any]) -> str:
    """Prefer the latest intent registration over a stale inherited start time."""
    hint = str(intent.get("timezone") or instance.get("timezone") or "")
    registered = _parse_ts(intent.get("registered_at"), timezone_hint=hint)
    started = _parse_ts(instance.get("started_at"), timezone_hint=hint)
    if registered is not None and (started is None or registered >= started):
        return str(intent.get("registered_at") or "")
    return str(instance.get("started_at") or intent.get("registered_at") or "")


def _start_is_stale(value: str, *, timezone_hint: str, now: float) -> bool:
    epoch = _parse_ts(value, timezone_hint=timezone_hint)
    return epoch is not None and (now - epoch) > DEFAULT_STARTING_STALE_SECONDS


def _format_beijing(value: str, *, timezone_hint: str = "") -> str:
    epoch = _parse_ts(value, timezone_hint=timezone_hint)
    if epoch is None:
        return "unknown time"
    return datetime.fromtimestamp(epoch, tz=BEIJING).strftime("%Y-%m-%d %H:%M:%S")


def _derive_console_status(row: dict[str, Any]) -> str:
    if row.get("break"):
        return "break"
    if row.get("busy"):
        return "busy"
    if row.get("status") == STATUS_STARTING and not row.get("starting_stale"):
        return "starting"
    if row.get("online"):
        return "idle"
    return "broke"


def _display_name(row: dict[str, Any]) -> str:
    cluster = str(row.get("cluster") or "?")
    node = str(row.get("node_id") if row.get("node_id") not in (None, "") else "?")
    rank = str(row.get("gpu_rank") if row.get("gpu_rank") not in (None, "") else "?")
    service = str(row.get("service_name") or "vlm")
    start = str(row.get("started_at_bj") or "unknown time")
    return f"{row['console_status']} · {cluster}/node{node}/rank{rank} · {service} · {start}"


def probe_models(base_url: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if OpenAI-compatible ``/models`` responds."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-controlled fleet URLs
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def list_fleet(
    *,
    fleet_root: Path | None = None,
    nodes_tsv: Path | None = None,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    busy_stale_seconds: float = DEFAULT_BUSY_STALE_SECONDS,
    probe: bool = False,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Merge intents + instance status + workload for the console."""
    root = Path(fleet_root) if fleet_root else default_fleet_root()
    # Serve the read-only snapshot from a short-TTL cache so frequent health
    # polls do not repeatedly hammer KFS metadata. Live probes bypass the cache.
    cache_key = None
    if not probe and _FLEET_CACHE_TTL_SEC > 0:
        cache_key = f"{root}|{stale_seconds}|{busy_stale_seconds}|{nodes_tsv}"
        now_mono = time.monotonic()
        with _FLEET_CACHE_LOCK:
            hit = _FLEET_CACHE.get(cache_key)
            if hit and (now_mono - hit[0]) < _FLEET_CACHE_TTL_SEC:
                return hit[1]
    intents_dir = root / "intents"
    instances_dir = root / "instances"
    intent_ids = {path.stem for path in intents_dir.glob("*.json")} if intents_dir.is_dir() else set()
    instance_ids = (
        {path.stem for path in instances_dir.glob("*.json")} if instances_dir.is_dir() else set()
    )
    nodes_index = load_nodes_tsv(nodes_tsv)
    by_cluster_node = nodes_index["by_cluster_node"]
    by_host = nodes_index["by_host"]
    now = time.time()
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(intent_ids | instance_ids):
        intent = _read_json(intents_dir / f"{instance_id}.json") or {}
        instance = _read_json(instances_dir / f"{instance_id}.json") or {}
        if instance:
            status = str(instance.get("status") or "")
        elif intent:
            status = "registered"
        else:
            status = "unknown"
        stale = _is_stale(instance, now=now, stale_seconds=stale_seconds) if instance else False
        base_url = str(instance.get("base_url") or intent.get("base_url") or "")
        reachable = None
        if probe and base_url and status == STATUS_RUNNING and not stale:
            reachable = probe_models(base_url, timeout=probe_timeout)
        online = bool(status == STATUS_RUNNING and not stale and (reachable is not False))
        workload = _read_workload(
            root, instance_id, now=now, busy_stale_seconds=busy_stale_seconds
        )
        host = str(intent.get("host") or instance.get("host") or "")
        port = intent.get("port") or instance.get("port")
        if (not host or port in (None, "")) and base_url:
            parsed_url = urlparse(base_url)
            host = host or str(parsed_url.hostname or "")
            port = port or parsed_url.port
        if (not host or port in (None, "")) and "-" in instance_id:
            legacy_host, legacy_port = instance_id.rsplit("-", 1)
            host = host or legacy_host
            port = port or (int(legacy_port) if legacy_port.isdigit() else None)
        cluster = str(intent.get("cluster") or instance.get("cluster") or "")
        node_id = str(
            intent.get("node_id")
            if intent.get("node_id") not in (None, "")
            else instance.get("node_id", "")
        )
        node_meta = by_cluster_node.get((cluster, node_id), {})
        if not node_meta:
            for key in {host.lower(), host.split(".", 1)[0].lower()} - {""}:
                node_meta = by_host.get(key, {})
                if node_meta:
                    break
        if not cluster and node_meta:
            cluster = str(node_meta.get("cluster") or "")
        if not node_id and node_meta:
            node_id = str(node_meta.get("node_id") or "")
        busy = bool(workload)
        start_at = _resolved_start_at(intent, instance)
        timezone_hint = str(intent.get("timezone") or instance.get("timezone") or "")
        starting_stale = status == STATUS_STARTING and _start_is_stale(
            start_at, timezone_hint=timezone_hint, now=now
        )
        break_marker = _read_break(root, instance_id, now=now)
        model = str(intent.get("model") or instance.get("model") or "")
        role = str(intent.get("role") or instance.get("role") or "reviewer")
        row = {
            "instance_id": instance_id,
            "host": host,
            "port": port,
            "base_url": base_url,
            "model": model,
            "gpu": intent.get("gpu") or instance.get("gpu") or "",
            "gpu_rank": (
                intent.get("gpu_rank")
                if intent.get("gpu_rank") not in (None, "")
                else instance.get("gpu_rank")
            ),
            "cluster": cluster,
            "node_id": node_id,
            "role": role,
            "node_meta": node_meta,
            "cluster_order": node_meta.get("cluster_order", 10_000),
            "node_order": node_meta.get("node_order", 10_000),
            "status": status or "unknown",
            "stale": stale,
            "starting_stale": starting_stale,
            "online": online,
            "busy": busy,
            "break": break_marker,
            "workload": workload,
            "reachable": reachable,
            "registered_at": intent.get("registered_at"),
            "started_at": instance.get("started_at"),
            "start_at": start_at,
            "started_at_bj": _format_beijing(start_at, timezone_hint=timezone_hint),
            "heartbeat_at": instance.get("heartbeat_at"),
            "updated_at": instance.get("updated_at"),
            "pid": instance.get("pid"),
            "intent": intent,
            "instance": instance,
        }
        row["service_name"] = "/".join(part for part in (role, model) if part)
        row["console_status"] = _derive_console_status(row)
        row["display_name"] = _display_name(row)
        rows.append(row)
    online_urls = [
        row["base_url"]
        for row in rows
        if row["online"] and row["console_status"] != "break" and row["base_url"]
    ]
    busy_count = sum(1 for row in rows if row.get("busy"))
    models = [str(row["model"]) for row in rows if row["online"] and row.get("model")]
    default_model = max(set(models), key=models.count) if models else ""
    summary = {
        "fleet_root": str(root),
        "stale_seconds": stale_seconds,
        "timezone": "Asia/Shanghai",
        "instances": rows,
        "online_count": sum(1 for row in rows if row["online"]),
        "busy_count": busy_count,
        "idle_count": sum(1 for row in rows if row["online"] and not row.get("busy")),
        "break_count": sum(1 for row in rows if row["console_status"] == "break"),
        "broke_count": sum(1 for row in rows if row["console_status"] == "broke"),
        "total_count": len(rows),
        "online_base_urls": online_urls,
        "default_model": default_model,
        "dispatch_url": ",".join(online_urls),
    }
    if cache_key is not None:
        with _FLEET_CACHE_LOCK:
            _FLEET_CACHE[cache_key] = (time.monotonic(), summary)
    return summary


def resolve_dispatch_urls(
    *,
    fleet_root: Path | None = None,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    probe: bool = True,
    role: str | None = None,
    exclude_clusters: set[str] | None = None,
) -> list[str]:
    """URLs safe to hand to ``ReviewerEndpointPool`` (online + optional live probe)."""
    summary = list_fleet(
        fleet_root=fleet_root,
        stale_seconds=stale_seconds,
        probe=probe,
    )
    preferred: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()
    for row in summary["instances"]:
        if not row.get("online") or row.get("console_status") == "break":
            continue
        if exclude_clusters and str(row.get("cluster") or "") in exclude_clusters:
            continue
        url = str(row.get("base_url") or "").rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        row_role = str(row.get("role") or "reviewer")
        if role and row_role not in {role, "any"}:
            fallback.append(url)
        else:
            preferred.append(url)
    return preferred or fallback
