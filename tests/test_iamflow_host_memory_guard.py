"""Host-memory guard tests for the Track-A IAMFlow adapter.

IAMFlow archives a ~274 MiB KV slice per archived frame and never evicts it, so
long movies were SIGKILLed by the pod cgroup (exit 137) with no diagnosis. These
tests pin the guard behaviour that replaced that failure mode: the numerics-exact
reclaims, the projection arithmetic, and the deliberate abort. No GPU, no model
and no vendored IAMFlow import is required.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
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


def _load_adapter_module():
    """Import the flat adapter file under a private name.

    The file is normally loaded by ``runner.py`` as top-level ``iamflow``, which
    collides with the vendored ``iamflow`` package; use a distinct name here so
    importing it in a test can never shadow the real package.
    """
    if str(_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(_ADAPTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_mave_iamflow_adapter_under_test", _ADAPTER_DIR / "iamflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


iamflow = _load_adapter_module()


class _Movie:
    """Minimal MovieContext stand-in: only the span map is read."""

    def __init__(self, spans):
        self.movie_id = "unit_movie"
        self.seconds_span_by_chunk = spans


def test_rss_and_cgroup_probes_are_usable():
    assert iamflow._rss_anon_bytes() > 0
    # 0 is a legal answer (no cgroup); anything else must be a sane byte count.
    limit = iamflow._cgroup_mem_limit_bytes()
    assert limit == 0 or limit > (1 << 30)


def test_archived_frames_matches_kv_block_arithmetic():
    # Three 15 s segments: 15 / 0.25 = 60 latents, // 3 = 20 blocks each.
    spans = {i: (i * 15.0, i * 15.0 + 15.0) for i in range(3)}
    assert iamflow._archived_frames_for_movie(_Movie(spans), 3) == 60


def test_kv_bytes_per_frame_fallback_is_274_mib():
    # 30 blocks x {k,v} x 1560 tokens x 1536 dim x 2 B == 274.2 MiB. This constant
    # is what makes long movies impossible; if it moves, the docs must move too.
    mib = iamflow._KV_BYTES_PER_FRAME_FALLBACK / (1024 ** 2)
    assert 274.0 < mib < 274.5
    assert iamflow._kv_bytes_per_archived_frame(None) == iamflow._KV_BYTES_PER_FRAME_FALLBACK


def test_budget_respects_explicit_override(monkeypatch):
    monkeypatch.setenv("MAVE_IAMFLOW_MAX_RSS_GB", "100")
    assert iamflow._host_memory_budget_bytes() == 100 * 1024 ** 3


def _guard(monkeypatch, *, total_segments, budget_gb, warmup=1, floor="0.0"):
    monkeypatch.setenv("MAVE_IAMFLOW_MAX_RSS_GB", str(budget_gb))
    monkeypatch.setenv("MAVE_IAMFLOW_RSS_WARMUP_SEGMENTS", str(warmup))
    monkeypatch.setenv("MAVE_IAMFLOW_PROJECTION_FLOOR_FRACTION", floor)
    monkeypatch.setenv("MAVE_IAMFLOW_RSS_LOG", "0")
    return iamflow._HostMemoryGuard(
        movie_id="unit_movie",
        total_segments=total_segments,
        kv_bytes_per_frame=iamflow._KV_BYTES_PER_FRAME_FALLBACK,
    )


def test_watchdog_aborts_on_projection_from_measured_slope(monkeypatch):
    """A 3 GB/segment slope over 200 segments cannot fit 100 GB: stop early."""
    guard = _guard(monkeypatch, total_segments=200, budget_gb=100)
    rss = iter([10, 13, 16, 19])  # GB after segments 1..4
    monkeypatch.setattr(iamflow, "_rss_anon_bytes", lambda: next(rss) * 1024 ** 3)
    guard.observe(1)  # warmup anchor
    with pytest.raises(iamflow.HostMemoryBudgetExceeded) as excinfo:
        for idx in range(2, 5):
            guard.observe(idx)
    assert "iamflow_host_memory_budget_exceeded" in str(excinfo.value)
    assert guard.abort_reason
    assert guard.last_slope_gb == pytest.approx(3.0, abs=0.01)


def test_watchdog_lets_a_movie_that_fits_run_to_completion(monkeypatch):
    """Same slope, few enough segments: no abort, and telemetry is reported."""
    guard = _guard(monkeypatch, total_segments=10, budget_gb=100)
    box = {"gb": 10.0}

    def _rss():
        box["gb"] += 3.0
        return int(box["gb"] * 1024 ** 3)

    monkeypatch.setattr(iamflow, "_rss_anon_bytes", _rss)
    for idx in range(1, 11):
        guard.observe(idx)
    telemetry = guard.telemetry()
    assert telemetry["aborted"] is None
    assert telemetry["rss_anon_peak_gb"] == pytest.approx(40.0, abs=0.1)
    assert telemetry["kv_bytes_per_archived_frame"] == iamflow._KV_BYTES_PER_FRAME_FALLBACK


def test_projection_floor_protects_short_smoke_runs(monkeypatch):
    """A --limit smoke sees the full segment count; it must not self-abort."""
    guard = _guard(monkeypatch, total_segments=500, budget_gb=800, floor="0.5")
    rss = iter([20, 25, 30])
    monkeypatch.setattr(iamflow, "_rss_anon_bytes", lambda: next(rss) * 1024 ** 3)
    for idx in range(1, 4):
        guard.observe(idx)  # projects far over budget, but RSS is nowhere near it
    assert guard.abort_reason is None


#: Replay of the real Track-A IAMFlow runs, from measured ``RssAnon`` slopes and
#: the pod's own cgroup limit. Each row is
#: ``(movie, segments, GiB/segment, cgroup GiB)``. The oracle for "doomed" is
#: ``peak >= cgroup - _POD_OVERHEAD_GIB``, because the co-located Qwen3-4B and
#: Qwen3-VL-2B vLLM servers hold ~10 GB of host RSS in the same pod, so peaking a
#: few GiB below the hard limit is not survivable.
_REPLAY = [
    ("0053_Rendezvous_mit_Joe_Black", 221, 2.4, 800),
    ("0001_American_Beauty", 300, 2.3, 800),  # really did complete: must not abort
    ("0013_Halloween", 128, 5.5, 800),
    ("0017_Pianist", 235, 3.6, 800),  # really was killed at ~215/235
    ("1005_Signs", 296, 4.7, 900),  # H800 pod, still doomed
    ("1048_Gran_Torino", 732, 1.07, 800),  # longest remaining movie
]
_POD_OVERHEAD_GIB = 20.0
_BASELINE_GIB = 12.0


@pytest.mark.parametrize("movie,segments,rate_gib,cgroup_gib", _REPLAY)
def test_watchdog_calibration_against_measured_runs(
    monkeypatch, movie, segments, rate_gib, cgroup_gib
):
    """Abort exactly the runs that cannot finish, and none of the ones that can.

    This is the calibration that matters: a watchdog that fires on a movie which
    would have completed destroys hours of good GPU work, and one that stays quiet
    on a doomed movie leaves the exit-137 problem unsolved.
    """
    monkeypatch.delenv("MAVE_IAMFLOW_MAX_RSS_GB", raising=False)
    monkeypatch.setenv("MAVE_IAMFLOW_RSS_LOG", "0")
    monkeypatch.setenv("MAVE_IAMFLOW_RSS_WARMUP_SEGMENTS", "3")
    monkeypatch.setattr(
        iamflow, "_cgroup_mem_limit_bytes", lambda: int(cgroup_gib * 1024 ** 3)
    )
    guard = iamflow._HostMemoryGuard(
        movie_id=movie,
        total_segments=segments,
        kv_bytes_per_frame=iamflow._KV_BYTES_PER_FRAME_FALLBACK,
    )

    aborted_at = None
    for index in range(1, segments + 1):
        rss = int((_BASELINE_GIB + rate_gib * index) * 1024 ** 3)
        monkeypatch.setattr(iamflow, "_rss_anon_bytes", lambda rss=rss: rss)
        try:
            guard.observe(index)
        except iamflow.HostMemoryBudgetExceeded:
            aborted_at = index
            break

    peak_gib = _BASELINE_GIB + rate_gib * segments
    doomed = peak_gib >= cgroup_gib - _POD_OVERHEAD_GIB
    if doomed:
        assert aborted_at is not None, f"{movie} peaks at {peak_gib:.0f} GiB and must be stopped"
        assert aborted_at < segments
    else:
        assert aborted_at is None, (
            f"{movie} peaks at {peak_gib:.0f} GiB under a {cgroup_gib:.0f} GiB pod "
            f"and must be allowed to finish"
        )


def test_watchdog_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MAVE_IAMFLOW_RSS_WATCHDOG", "0")
    guard = _guard(monkeypatch, total_segments=200, budget_gb=1)
    monkeypatch.setattr(iamflow, "_rss_anon_bytes", lambda: 500 * 1024 ** 3)
    guard.observe(1)
    guard.observe(2)
    assert guard.abort_reason is None


# ---------------------------------------------------------------------------
# Instance patches on fake pipeline/bank objects
# ---------------------------------------------------------------------------


class _FrameInfo:
    def __init__(self):
        self.pixel_frame = object()
        self.kv_cache = ["kv"]


class _Bank:
    def __init__(self):
        self.frame_archive = {}
        self._frame_kv_store = {}
        self.active_memory = []
        self.id_memory = []
        self._memory_kv_cache = "stale"
        self._memory_kv_cache_key = ("stale",)
        self._memory_kv_cache_device = "cpu"
        self.calls = 0

    @property
    def frame_active_memory(self):
        return list(self.active_memory)

    @staticmethod
    def _frame_sort_key(frame_id):
        match = re.match(r"p(\d+)_c(\d+)_f(\d+)", frame_id)
        if not match:
            return (1 << 30, 1 << 30, 1 << 30)
        return tuple(int(g) for g in match.groups())

    def select_frame_from_chunk(self, frame_id="p1_c1_f0"):
        self.calls += 1
        info = _FrameInfo()
        self.frame_archive[frame_id] = info
        self._frame_kv_store[frame_id] = ["kv"]
        return frame_id, 0.5


def _fake_pipe():
    pipe = types.SimpleNamespace()
    pipe.agent_memory_bank = _Bank()
    pipe._sync_pixel_store = {}
    pipe._sync_pixel_order = []

    def _run_sync_vae_vlm(chunk_key):
        pipe._sync_pixel_store[chunk_key] = f"pixels::{chunk_key}"
        pipe._sync_pixel_order.append(chunk_key)

    pipe._run_sync_vae_vlm = _run_sync_vae_vlm
    return pipe


def test_write_only_pixel_frame_view_is_dropped():
    """FrameInfo.pixel_frame is never read but pins a whole decoded block."""
    pipe = _fake_pipe()
    iamflow._install_host_memory_guards(pipe)
    bank = pipe.agent_memory_bank
    frame_id, score = bank.select_frame_from_chunk("p1_c1_f0")
    assert score == 0.5
    assert bank.frame_archive[frame_id].pixel_frame is None
    # The algorithmically load-bearing KV slice is untouched by default.
    assert bank._frame_kv_store[frame_id] == ["kv"]


def test_sync_pixel_store_is_bounded(monkeypatch):
    monkeypatch.setenv("MAVE_IAMFLOW_PIXEL_STORE_KEEP", "4")
    pipe = _fake_pipe()
    iamflow._install_host_memory_guards(pipe)
    for i in range(12):
        pipe._run_sync_vae_vlm(f"p1_c{i}")
    assert len(pipe._sync_pixel_store) == 4
    assert pipe._sync_pixel_order == ["p1_c8", "p1_c9", "p1_c10", "p1_c11"]
    # The eviction lag is 3 chunks back, so the retained window must cover it.
    assert "p1_c8" in pipe._sync_pixel_store


def test_guards_are_installed_only_once():
    pipe = _fake_pipe()
    iamflow._install_host_memory_guards(pipe)
    first = pipe._run_sync_vae_vlm
    iamflow._install_host_memory_guards(pipe)
    assert pipe._run_sync_vae_vlm is first


def test_kv_archive_cap_is_off_by_default():
    bank = _Bank()
    for i in range(20):
        bank._frame_kv_store[f"p1_c{i}_f0"] = ["kv"]
        bank.frame_archive[f"p1_c{i}_f0"] = _FrameInfo()
    assert iamflow._trim_kv_archive(bank) == 0
    assert len(bank._frame_kv_store) == 20


def test_kv_archive_cap_drops_oldest_and_spares_active(monkeypatch):
    """Opt-in variant: bounded recall pool, active frames never dropped."""
    monkeypatch.setenv("MAVE_IAMFLOW_MAX_ARCHIVE_KV_FRAMES", "5")
    bank = _Bank()
    for i in range(20):
        bank._frame_kv_store[f"p1_c{i}_f0"] = ["kv"]
        bank.frame_archive[f"p1_c{i}_f0"] = _FrameInfo()
    bank.active_memory = ["p1_c0_f0"]  # oldest, but currently recalled
    dropped = iamflow._trim_kv_archive(bank)
    assert dropped == 15
    assert len(bank._frame_kv_store) == 5
    assert "p1_c0_f0" in bank._frame_kv_store
    # Metadata survives so entity-coverage recall still ranks dropped frames.
    assert len(bank.frame_archive) == 20
    assert bank.frame_archive["p1_c1_f0"].kv_cache is None
    assert bank._memory_kv_cache is None
