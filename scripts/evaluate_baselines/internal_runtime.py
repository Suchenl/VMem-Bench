"""Internal-only runtime provenance for benchmark run artifacts.

This module is deliberately outside scoring and publication code.  Its payloads
may contain company-internal machine details and must never be copied into paper
tables or public result exports.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTERNAL_ONLY_NOTICE = (
    "INTERNAL ONLY: hardware/runtime provenance; exclude from paper tables "
    "and publication outputs."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def gpu_snapshot() -> dict[str, Any]:
    """Return a best-effort, non-fatal nvidia-smi snapshot."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    query = (
        "index,uuid,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu"
    )
    try:
        proc = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:1000],
        }
    keys = [
        "index",
        "uuid",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_gpu_percent",
        "temperature_gpu_c",
    ]
    gpus = []
    for line in proc.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(keys):
            gpus.append(dict(zip(keys, values)))
    return {"available": True, "gpus": gpus}


def runtime_context(
    *,
    env: dict[str, str] | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    effective_env = os.environ if env is None else env
    executable = python_executable or sys.executable
    python_version = platform.python_version()
    if python_executable is not None:
        try:
            proc = subprocess.run(
                [python_executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env=effective_env,
            )
            version_output = (proc.stdout or proc.stderr).strip()
            python_version = version_output.removeprefix("Python ").strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired) as exc:
            python_version = f"unavailable: {type(exc).__name__}"
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": python_version,
        "python_executable": executable,
        "launcher_python_executable": sys.executable,
        "cuda_visible_devices": effective_env.get("CUDA_VISIBLE_DEVICES"),
    }


def max_rss_omission() -> dict[str, Any]:
    return {
        "value": None,
        "unit": "KiB",
        "status": "unavailable",
        "reason": (
            "Portable per-subprocess max RSS is not exposed by subprocess.run; "
            "resource.getrusage(RUSAGE_CHILDREN) is cumulative across runs and "
            "would misattribute multi-story runner memory."
        ),
    }


def atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)
