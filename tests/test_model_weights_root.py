"""Regression: HF hub cache must resolve to Montage models/model_weights/hub."""

from __future__ import annotations

from pathlib import Path

from vmem_bench.common.model_weights import hf_cache_dir, public_models_root, repo_root


def test_repo_root_is_montage_not_nested_memstrata_agents() -> None:
    root = repo_root()
    assert (root / "src" / "montage").is_dir()
    assert (root / "models" / "model_weights").is_dir()
    # Nested benchmarks/MemStrata/AGENTS.md must not win.
    assert not str(root).endswith("benchmarks/MemStrata")


def test_hf_cache_dir_points_at_montage_hub() -> None:
    cache = Path(hf_cache_dir())
    assert cache == repo_root() / "models" / "model_weights" / "hub"
    assert "benchmarks/MemStrata/models" not in str(cache)


def test_public_sam3_dir_resolves() -> None:
    sam3 = public_models_root() / "facebook" / "sam3"
    assert sam3.is_dir()
    assert (sam3 / "config.json").is_file()
