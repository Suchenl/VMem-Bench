"""SSH remote dispatch for annotation pipeline batches.

CPU-heavy batch work (ffmpeg clip cutting, SAM3/GroundingDINO pre/post) is
offloaded onto the shared remote training nodes so the dev machine only serves the
console frontend + Cursor.  The design keeps the console responsive:

- **Launch / stop go over SSH** (via ``MEMSTRATA_TGPU`` if you configure a
  cluster launcher).  A batch is started
  *detached* with ``setsid --fork`` so the SSH call returns in ~0.1s; the remote
  process keeps running after the connection closes.
- **Status never goes over SSH.**  ``/data`` is shared between the dev
  machine and every remote node, so the remote batch writes ``progress.json`` /
  ``return_code.txt`` straight into the shared ``job_dir``.  The backend reads
  those files locally on each refresh — instant, no SSH round-trip, no 504.

Node placement is load-weighted: nodes with more idle CPU (lower loadavg per
core) win, with a small per-cluster penalty so the weaker a800 CPUs get less
than the h800 CPUs, and a penalty for jobs we already dispatched there so a
burst of submissions spreads across the fleet instead of piling on one node.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

TGPU = os.environ.get("MEMSTRATA_TGPU", "").strip()
_NODES_RAW = os.environ.get("TGPU_NODES_FILE", "").strip()
NODES_FILE = Path(_NODES_RAW) if _NODES_RAW else Path()

# Cluster labels in nodes.tsv that mark an SSH-reachable remote GPU node. Set this
# to your own cluster's prefix (e.g. "gpu-", "dgx-", "hpc-"); rows whose cluster
# label does not start with it are ignored by the remote dispatcher.
REMOTE_CLUSTER_PREFIX = os.environ.get("REMOTE_CLUSTER_PREFIX", "gpu-")

# Per-cluster CPU-quality penalty. a800 CPUs are weaker than h800, so the same
# normalized loadavg counts as "more loaded" on a800 → h800 receives more work.
# Override with e.g. MEMSTRATA_REMOTE_CLUSTER_PENALTY="gpu-a800:1.3,gpu-h800:1.0".
_DEFAULT_CLUSTER_PENALTY = {"gpu-a800": 1.25, "gpu-h800": 1.0}

# Extra effective-load added per job we already placed on a node this session.
# loadavg lags real utilization by ~1 min, so without this a burst of submits
# all pick the same "currently idle" node.
_ACTIVE_JOB_PENALTY = float(os.environ.get("MEMSTRATA_REMOTE_ACTIVE_JOB_PENALTY", "0.12"))

_PROBE_TIMEOUT = int(os.environ.get("MEMSTRATA_REMOTE_PROBE_TIMEOUT", "5"))
_LAUNCH_TIMEOUT = int(os.environ.get("MEMSTRATA_REMOTE_LAUNCH_TIMEOUT", "20"))
_STOP_TIMEOUT = int(os.environ.get("MEMSTRATA_REMOTE_STOP_TIMEOUT", "15"))


@dataclass(slots=True)
class RemoteNode:
    cluster: str
    node: str
    ip: str
    host: str = ""
    role: str = ""

    @property
    def key(self) -> str:
        return f"{self.cluster}#{self.node}"


@dataclass(slots=True)
class NodeState:
    node: RemoteNode
    online: bool
    loadavg1: float
    ncpu: int
    active_jobs: int

    @property
    def normalized_load(self) -> float:
        return self.loadavg1 / max(1, self.ncpu)

    def effective_load(self, penalty: float) -> float:
        return self.normalized_load * penalty + self.active_jobs * _ACTIVE_JOB_PENALTY


@dataclass(slots=True)
class Placement:
    node: RemoteNode
    state: NodeState
    reason: str


def _cluster_penalty() -> dict[str, float]:
    raw = os.environ.get("MEMSTRATA_REMOTE_CLUSTER_PENALTY", "").strip()
    if not raw:
        return dict(_DEFAULT_CLUSTER_PENALTY)
    out = dict(_DEFAULT_CLUSTER_PENALTY)
    for chunk in raw.replace(";", ",").split(","):
        if ":" not in chunk:
            continue
        cluster, _, value = chunk.partition(":")
        try:
            out[cluster.strip()] = float(value)
        except ValueError:
            continue
    return out


def load_remote_nodes(nodes_file: Path | None = None) -> list[RemoteNode]:
    """Parse ``nodes.tsv`` and return every remote (SSH-reachable) node."""
    path = Path(nodes_file) if nodes_file else NODES_FILE
    nodes: list[RemoteNode] = []
    if not path.is_file():
        return nodes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        node, cluster, role, host, ip = (f.strip() for f in fields[:5])
        # Skip rows that are not SSH-reachable training clusters.
        if not cluster.startswith(REMOTE_CLUSTER_PREFIX):
            continue
        nodes.append(RemoteNode(cluster=cluster, node=node, ip=ip, host=host, role=role))
    return nodes


def _tgpu_run(node: RemoteNode, argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    if not TGPU:
        raise RuntimeError(
            "Remote GPU launch is not configured. Set MEMSTRATA_TGPU to your "
            "cluster launcher (and TGPU_NODES_FILE to a nodes table) or run locally."
        )
    return subprocess.run(  # noqa: S603
        [TGPU, "-c", node.cluster, "-node", node.node, "--", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _probe_one(node: RemoteNode) -> tuple[float, int] | None:
    """Return ``(loadavg1, ncpu)`` for a node, or ``None`` if unreachable."""
    try:
        proc = _tgpu_run(
            node,
            ["sh", "-c", "cut -d' ' -f1 /proc/loadavg; nproc"],
            timeout=_PROBE_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    parts = [p for p in proc.stdout.split() if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), int(parts[1])
    except ValueError:
        return None


def probe_states(
    nodes: list[RemoteNode],
    active_counts: dict[str, int] | None = None,
) -> list[NodeState]:
    """Probe all nodes in parallel; offline nodes come back with online=False."""
    active_counts = active_counts or {}
    states: dict[str, NodeState] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        results = list(pool.map(_probe_one, nodes))
    for node, result in zip(nodes, results):
        if result is None:
            states[node.key] = NodeState(node, False, 0.0, 1, active_counts.get(node.key, 0))
        else:
            load, ncpu = result
            states[node.key] = NodeState(node, True, load, ncpu, active_counts.get(node.key, 0))
    return [states[node.key] for node in nodes]


def select_node(
    nodes: list[RemoteNode] | None = None,
    active_counts: dict[str, int] | None = None,
) -> Placement | None:
    """Pick the least effectively-loaded online remote node, or ``None`` if all down."""
    nodes = nodes if nodes is not None else load_remote_nodes()
    if not nodes:
        return None
    states = probe_states(nodes, active_counts=active_counts)
    penalty = _cluster_penalty()
    online = [s for s in states if s.online]
    if not online:
        return None
    best = min(online, key=lambda s: s.effective_load(penalty.get(s.node.cluster, 1.0)))
    reason = (
        f"load1={best.loadavg1:.2f}/{best.ncpu}c "
        f"(norm={best.normalized_load:.3f}, active={best.active_jobs}) "
        f"→ eff={best.effective_load(penalty.get(best.node.cluster, 1.0)):.3f}"
    )
    return Placement(node=best.node, state=best, reason=reason)


def build_detached_command(
    *,
    inner_script: str,
    cwd: str,
    log_path: str,
    env_exports: dict[str, str],
) -> list[str]:
    """Build the tgpu argv that launches ``inner_script`` detached on a node.

    ``inner_script`` is expected to already end with the
    ``; rc=$?; echo $rc > <return_code.txt>; exit $rc`` status trailer so the
    shared-FS status file is written on every exit path except SIGKILL.
    """
    exports = "".join(
        f"export {name}={shlex.quote(value)}; " for name, value in env_exports.items()
    )
    # ``exec >>log 2>&1 </dev/null`` first so the SSH channel fds are released
    # immediately (setsid --fork returns, then the child detaches its stdio),
    # letting the tgpu call return in ~0.1s while the batch keeps running.
    body = (
        f"exec >>{shlex.quote(log_path)} 2>&1 </dev/null; "
        f"cd {shlex.quote(cwd)}; {exports}{inner_script}"
    )
    return ["setsid", "--fork", "bash", "-lc", body]


def launch(
    node: RemoteNode,
    *,
    inner_script: str,
    cwd: str,
    log_path: str,
    env_exports: dict[str, str],
) -> None:
    """Fire-and-forget launch of a detached batch on ``node`` via tgpu SSH."""
    argv = build_detached_command(
        inner_script=inner_script,
        cwd=cwd,
        log_path=log_path,
        env_exports=env_exports,
    )
    proc = _tgpu_run(node, argv, timeout=_LAUNCH_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote launch on {node.key} ({node.ip}) failed rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
        )


def stop(node: RemoteNode, *, match: str) -> bool:
    """SIGTERM the remote batch whose command line contains ``match``.

    ``match`` is the unique job_dir path (contains the job_id), so pkill -f only
    hits this job's process tree.  Returns True if pkill reported a kill.
    """
    try:
        proc = _tgpu_run(node, ["pkill", "-TERM", "-f", match], timeout=_STOP_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return False
    # pkill exits 0 if it killed something, 1 if nothing matched (already gone).
    return proc.returncode == 0
