"""Stage-1 per-movie job lock semantics for the Track-A causal runner.

The lock has to separate two failure modes that look identical on disk:

* a runner SIGKILLed by the pod cgroup never unlinks its lock, and the movie
  would stay unrunnable until somebody deletes the file by hand;
* a runner that is merely slow (one memflow_sma segment can take 165 s, a movie
  8+ hours) must keep its lock, or a stale-lock sweep starts a second runner on
  the same movie and both burn a GPU on byte-identical work.

Locks written before the heartbeat existed carry no ``host=`` field, so they must
always be treated as live -- the live fleet holds such locks for hours.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_baselines"
    / "trackA"
    / "baseline_adapters"
    / "causal"
)


def _load_runner():
    if str(_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(_ADAPTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_mave_trackA_runner_under_test", _ADAPTER_DIR / "runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_acquire_then_second_caller_is_refused(tmp_path):
    lock = tmp_path / ".stage1.lock"
    fd = runner._acquire_job_lock(lock)
    assert fd is not None
    try:
        assert runner._acquire_job_lock(lock) is None
        body = lock.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in body
        assert f"host={runner._HOSTNAME}" in body
    finally:
        os.close(fd)
        lock.unlink()


def test_dead_same_host_owner_lock_is_reclaimed(tmp_path):
    lock = tmp_path / ".stage1.lock"
    # pid 2**22 is above the default pid_max and cannot be running.
    lock.write_text(
        f"pid=4194303 host={runner._HOSTNAME} start={time.time():.3f}\n", encoding="utf-8"
    )
    assert runner._lock_owner_is_alive(lock) is False
    fd = runner._acquire_job_lock(lock)
    assert fd is not None, "a provably dead owner must not block the movie forever"
    try:
        assert f"pid={os.getpid()}" in lock.read_text(encoding="utf-8")
    finally:
        os.close(fd)
        lock.unlink()


def test_live_same_host_owner_is_respected(tmp_path):
    lock = tmp_path / ".stage1.lock"
    lock.write_text(
        f"pid={os.getpid()} host={runner._HOSTNAME} start={time.time():.3f}\n",
        encoding="utf-8",
    )
    assert runner._lock_owner_is_alive(lock) is True
    assert runner._acquire_job_lock(lock) is None


def test_legacy_lock_without_host_is_never_stolen(tmp_path):
    """Pre-heartbeat lock format: conservative, even when very old."""
    lock = tmp_path / ".stage1.lock"
    lock.write_text("pid=4194303 start=1785158888.673\n", encoding="utf-8")
    old = time.time() - 12 * 3600
    os.utime(lock, (old, old))
    assert runner._lock_owner_is_alive(lock) is True
    assert runner._acquire_job_lock(lock) is None


def test_remote_owner_uses_heartbeat_age(tmp_path, monkeypatch):
    lock = tmp_path / ".stage1.lock"
    lock.write_text("pid=123 host=some-other-node start=1.0\n", encoding="utf-8")
    monkeypatch.setenv("VMEM_STAGE1_LOCK_STALE_MINUTES", "45")

    fresh = time.time() - 60
    os.utime(lock, (fresh, fresh))
    assert runner._lock_owner_is_alive(lock) is True, "recent heartbeat means alive"

    cold = time.time() - 3 * 3600
    os.utime(lock, (cold, cold))
    assert runner._lock_owner_is_alive(lock) is False, "no heartbeat for 3 h means dead"


def test_touch_job_lock_refreshes_heartbeat(tmp_path):
    lock = tmp_path / ".stage1.lock"
    lock.write_text(f"pid=1 host={runner._HOSTNAME} start=1.0\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    runner._touch_job_lock(lock)
    assert time.time() - lock.stat().st_mtime < 5

    # Must stay quiet when the lock is already gone (cleanup races).
    lock.unlink()
    runner._touch_job_lock(lock)


def test_run_name_matches_output_layout():
    assert runner._run_name("iamflow", "name_anchored", 16) == "iamflow__B16"
    assert runner._run_name("memflow_sma", "name_anchored", None) == "memflow_sma"
    assert runner._run_name("memflow", "description_provided", 8) == "memflow__descprov__B8"
    assert runner._run_name("memflow", "description_only", None) == "memflow__desconly"


def test_env_float_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("VMEM_TEST_FLOAT", "not-a-number")
    assert runner._env_float("VMEM_TEST_FLOAT", 7.5) == pytest.approx(7.5)
    monkeypatch.setenv("VMEM_TEST_FLOAT", "2.5")
    assert runner._env_float("VMEM_TEST_FLOAT", 7.5) == pytest.approx(2.5)
