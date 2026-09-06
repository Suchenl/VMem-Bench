"""Prevent Track A from growing a second MemStrata implementation again."""

from __future__ import annotations

import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_baselines"
    / "trackA"
    / "baseline_adapters"
    / "causal"
    / "memstrata.py"
)


def test_memstrata_adapter_only_imports_public_production_entry() -> None:
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    method_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("memstrata")
    }
    assert method_imports == {"memstrata.production.realized"}


def test_memstrata_adapter_does_not_construct_method_internals() -> None:
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    forbidden = {
        "Observation",
        "MemoryUpdater",
        "IntentInterpreter",
        "RoleAwareDecomposer",
        "AssetBank",
    }
    constructed = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (constructed & forbidden)


def test_memstrata_adapter_selects_strict_paper_profile_by_default() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    pipeline_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_realized_segment_pipeline"
    ]
    assert len(pipeline_calls) == 1
    profile = next(
        keyword.value
        for keyword in pipeline_calls[0].keywords
        if keyword.arg == "profile"
    )
    assert isinstance(profile, ast.Attribute)
    assert profile.attr == "production_profile"
    keywords = {keyword.arg for keyword in pipeline_calls[0].keywords}
    assert {"mllm_base_url", "mllm_model"} <= keywords
    assert '"MEMSTRATA_TRACKA_PROFILE", "paper_tracka_202607"' in source
    assert '"MEMSTRATA_TRACKA_NAME_SOURCE", "mllm"' in source
