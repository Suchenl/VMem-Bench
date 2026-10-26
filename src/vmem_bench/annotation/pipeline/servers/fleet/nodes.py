"""Read optional cluster placement metadata from a ``tgpu`` nodes.tsv file."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


def configured_nodes_tsv() -> Path | None:
    """Return the operator-configured nodes.tsv path, if it exists."""
    raw = os.environ.get("MEMSTRATA_NODES_TSV", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def _host_keys(value: object) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    return {text, text.split(".", 1)[0]}


def load_nodes_tsv(path: Path | None = None) -> dict[str, dict[Any, dict[str, str | int]]]:
    """Load cluster/node and host lookup tables without requiring a local file."""
    selected = path or configured_nodes_tsv()
    by_cluster_node: dict[tuple[str, str], dict[str, str | int]] = {}
    by_host: dict[str, dict[str, str | int]] = {}
    if selected is None:
        return {"by_cluster_node": by_cluster_node, "by_host": by_host}

    try:
        with selected.open(encoding="utf-8", newline="") as handle:
            lines: list[str] = []
            for raw in handle:
                if raw.lstrip().startswith("#"):
                    candidate = raw.lstrip()[1:].lstrip()
                    if candidate.startswith("node\t"):
                        lines.append(candidate)
                    continue
                lines.append(raw)
            rows = csv.DictReader(
                lines,
                delimiter="\t",
            )
            for order, row in enumerate(rows):
                cluster = str(row.get("cluster") or "").strip()
                node_id = str(row.get("node") or "").strip()
                if not cluster or not node_id:
                    continue
                meta: dict[str, str | int] = {
                    "cluster": cluster,
                    "node_id": node_id,
                    "role": str(row.get("role") or "").strip(),
                    "host": str(row.get("host") or "").strip(),
                    "ip": str(row.get("ip") or "").strip(),
                    "note": str(row.get("note") or "").strip(),
                    "cluster_order": order,
                    "node_order": int(node_id) if node_id.isdigit() else order,
                }
                by_cluster_node[(cluster, node_id)] = meta
                for key in _host_keys(meta["host"]) | _host_keys(meta["ip"]):
                    by_host[key] = meta
    except OSError:
        return {"by_cluster_node": {}, "by_host": {}}
    return {"by_cluster_node": by_cluster_node, "by_host": by_host}
