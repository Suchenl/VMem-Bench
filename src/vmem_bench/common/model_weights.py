"""Weight-path resolution for vmem_bench (self-contained; principle #7).

Local caches live under this checkout's ``models/model_weights`` (override with
``MONTAGE_WEIGHTS_ROOT``). Shared public checkpoints resolve through
``PUBLIC_MODELS_ROOT``; that env var is required, weights are not in git.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """This VMem-Bench checkout (directory that owns ``src/vmem_bench``)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "vmem_bench").is_dir() and (
            (parent / "AGENTS.md").is_file() or (parent / ".git").exists()
        ):
            return parent
    for parent in current.parents:
        if (parent / "AGENTS.md").is_file() or (parent / ".git").exists():
            return parent
    return current.parents[3]


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
