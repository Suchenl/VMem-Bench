"""Regression: weight helpers resolve inside this standalone repo."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vmem_bench.common.model_weights import hf_cache_dir, public_models_root, repo_root


def test_repo_root_is_this_checkout() -> None:
    root = repo_root()
    assert (root / "src" / "vmem_bench").is_dir()
    assert (root / "AGENTS.md").is_file()
    assert not (root / "src" / "montage").is_dir()


def test_hf_cache_dir_points_at_local_hub() -> None:
    cache = Path(hf_cache_dir())
    assert cache == repo_root() / "models" / "model_weights" / "hub"


def test_public_models_root_requires_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PUBLIC_MODELS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="PUBLIC_MODELS_ROOT"):
        public_models_root()
    monkeypatch.setenv("PUBLIC_MODELS_ROOT", str(tmp_path))
    assert public_models_root() == tmp_path.resolve()
