"""Checkout-path helper used by the public two-repo layout."""

from __future__ import annotations

import sys
from pathlib import Path

CAUSAL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_baselines"
    / "trackA"
    / "baseline_adapters"
    / "causal"
)
sys.path.insert(0, str(CAUSAL))

from _local_roots import BENCH_ROOT, expand_dataset_root, find_memstrata_src  # noqa: E402


def test_bench_root_is_this_repo() -> None:
    assert (BENCH_ROOT / "src" / "vmem_bench").is_dir()


def test_expand_dataset_root_defaults_under_repo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VMEM_DATASETS_ROOT", raising=False)
    monkeypatch.setenv("VMEM_DATASETS_ROOT", str(tmp_path))
    got = expand_dataset_root("${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos")
    assert got == tmp_path / "BlenderOpenMovies" / "Videos"


def test_find_memstrata_src_from_env(monkeypatch, tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "memstrata"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("MEMSTRATA_SRC", str(tmp_path / "src"))
    assert find_memstrata_src() == (tmp_path / "src").resolve()
