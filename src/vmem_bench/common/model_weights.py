"""Weight-path resolution for vmem_bench (self-contained copy, principle #7).

Weights live under the repo's ``models/model_weights`` (override with
``MONTAGE_WEIGHTS_ROOT``); shared public checkpoints resolve through
``PUBLIC_MODELS_ROOT``. Adjust on standalone release.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Montage monorepo root (or MemStrata standalone root after split).

    Nested ``benchmarks/MemStrata/AGENTS.md`` must not win: a previous bug also
    mkdir'd ``benchmarks/MemStrata/models/model_weights``, which would otherwise
    keep stealing HF hub resolution away from Montage ``models/model_weights/hub``.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "montage").is_dir() and (parent / "models" / "model_weights").is_dir():
            return parent
    for parent in current.parents:
        if (parent / ".git").exists() and (parent / "models" / "model_weights").is_dir():
            return parent
    for parent in current.parents:
        if (parent / "AGENTS.md").is_file() or (parent / ".git").exists():
            return parent
    # .../benchmarks/MemStrata/src/vmem_bench/common/model_weights.py -> Montage
    return current.parents[5]


def weights_root() -> Path:
    override = os.environ.get("MONTAGE_WEIGHTS_ROOT")
    root = Path(override).expanduser().resolve() if override else repo_root() / "models" / "model_weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> str:
    cache = weights_root() / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(weights_root()))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    return str(cache)


def public_models_root() -> Path:
    override = os.environ.get("PUBLIC_MODELS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    raise RuntimeError("PUBLIC_MODELS_ROOT is not set")
