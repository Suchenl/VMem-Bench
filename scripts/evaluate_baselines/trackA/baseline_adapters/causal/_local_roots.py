"""Locate sibling checkouts and dataset roots for the public two-repo layout.

Legacy Montage checkouts are still accepted for migration, but public users
should use sibling repositories or set ``MEMSTRATA_SRC`` explicitly. This file
keeps the compatibility fallback without adding a second public protocol.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../scripts/evaluate_baselines/trackA/baseline_adapters/causal/<this>
_CAUSAL_DIR = Path(__file__).resolve().parent
BENCH_ROOT = _CAUSAL_DIR.parents[4]


def _existing_dir(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return None
    return resolved if resolved.is_dir() else None


def find_memstrata_src() -> Path:
    """Return the directory that contains the ``memstrata`` package (i.e. ``.../src``).

    Order: ``MEMSTRATA_SRC`` → sibling ``../MemStrata/src`` → legacy checkout.
    """
    env = os.environ.get("MEMSTRATA_SRC")
    if env:
        hit = _existing_dir(Path(env))
        if hit is not None:
            if (hit / "memstrata").is_dir():
                return hit
            nested = _existing_dir(hit / "src")
            if nested is not None and (nested / "memstrata").is_dir():
                return nested
        raise FileNotFoundError(
            f"MEMSTRATA_SRC={env!r} is set but does not contain memstrata/. "
            "Point it at MemStrata/src (the folder that contains the memstrata package)."
        )
    candidates = [
        BENCH_ROOT.parent / "MemStrata" / "src",
        BENCH_ROOT.parent / "MemStrata",
        BENCH_ROOT.parents[1] / "methods" / "MemStrata" / "src",
    ]
    parent = BENCH_ROOT.parent
    if parent.is_dir():
        for child in sorted(parent.iterdir()):
            if child.name.startswith("."):
                continue
            candidates.append(child / "src")
            candidates.append(child)
    for cand in candidates:
        hit = _existing_dir(cand)
        if hit is None:
            continue
        if (hit / "memstrata").is_dir():
            return hit
        nested = _existing_dir(hit / "src")
        if nested is not None and (nested / "memstrata").is_dir():
            return nested
    raise FileNotFoundError(
        "Cannot find MemStrata. Clone it next to VMem-Bench:\n"
        "  git clone https://github.com/Suchenl/MemStrata.git\n"
        "  git clone https://github.com/Suchenl/VMem-Bench.git\n"
        "so that ../MemStrata/src/memstrata exists, or export MEMSTRATA_SRC=/path/to/MemStrata/src."
    )


def default_datasets_root() -> Path:
    env = os.environ.get("VMEM_DATASETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (BENCH_ROOT / "data").resolve()


def expand_dataset_root(raw: str) -> Path:
    os.environ.setdefault("VMEM_DATASETS_ROOT", str(default_datasets_root()))
    expanded = os.path.expanduser(os.path.expandvars(raw.strip()))
    if "${" in expanded:
        expanded = expanded.replace(
            "${VMEM_DATASETS_ROOT}", str(default_datasets_root())
        )
    return Path(expanded)
