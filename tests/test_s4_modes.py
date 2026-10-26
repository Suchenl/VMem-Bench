"""S4→S5 mode resolution and resume command wiring."""

import inspect
from pathlib import Path

import pytest

from vmem_bench.annotation.pipeline.orchestration.contracts import resolve_s4_mode
from vmem_bench.annotation.pipeline.orchestration import orchestrator
from vmem_bench.annotation.pipeline.servers.backend.jobs import JobStore, _public_options


def test_auto_mode_defaults_by_human_policy() -> None:
    assert resolve_s4_mode("auto", skip_human=False) == "blocking"
    assert resolve_s4_mode("auto", skip_human=True) == "nonblocking"
    assert resolve_s4_mode("nonblocking", skip_human=False) == "nonblocking"


def test_blocking_cannot_skip_human() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_s4_mode("blocking", skip_human=True)


def test_orchestrator_s4_mode_signatures_are_importable() -> None:
    continue_signature = inspect.signature(orchestrator.continue_after_s4)
    pipeline_signature = inspect.signature(orchestrator.run_pipeline)
    assert list(continue_signature.parameters).count("s4_mode") == 1
    assert continue_signature.parameters["s4_mode"].default == "auto"
    assert pipeline_signature.parameters["s4_mode"].default == "auto"


def test_continue_job_command_preserves_resume_and_mode(tmp_path: Path) -> None:
    store = JobStore(
        jobs_root=tmp_path / "jobs",
        data_root=tmp_path / "data",
        python="/usr/bin/python3",
        fleet_root=tmp_path / "fleet",
    )
    command = store._command(
        {
            "continue_from": "after_s4",
            "s4_mode": "blocking",
            "reviewer": "passthrough",
        },
        tmp_path / "catalog.jsonl",
        tmp_path / "result.json",
        tmp_path / "return_code.txt",
    )
    assert "'--continue-from' 'after_s4'" in command
    assert "'--s4-mode' 'blocking'" in command


def test_job_options_record_effective_auto_mode() -> None:
    assert _public_options({"s4_mode": "auto", "skip_human": False})[
        "s4_mode_effective"
    ] == "blocking"
    assert _public_options({"s4_mode": "auto", "skip_human": True})[
        "s4_mode_effective"
    ] == "nonblocking"

