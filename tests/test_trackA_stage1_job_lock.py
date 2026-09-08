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
import json
import os
import subprocess
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


def test_output_root_can_be_isolated_for_canaries(tmp_path):
    env = os.environ.copy()
    env["VMEM_TRACKA_OUTPUT_ROOT"] = str(tmp_path / "isolated")
    value = subprocess.check_output(
        [sys.executable, "-c", "import runner; print(runner._OUTPUT_ROOT)"],
        cwd=_ADAPTER_DIR,
        env=env,
        text=True,
    ).strip()
    assert value == str((tmp_path / "isolated").resolve())


def test_limited_smoke_completion_uses_requested_chunk_count(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    output = tmp_path / "output"
    movie = bench / "assets" / "trackA" / "Dataset" / "Movie"
    selection = output / "memstrata__B16" / "Dataset" / "Movie" / "visual_selections"
    (movie / "gold").mkdir(parents=True)
    selection.mkdir(parents=True)
    (movie / "gold" / "chunk_annotations.json").write_text(
        json.dumps({"chunks": [{"chunk_id": index} for index in range(10)]}),
        encoding="utf-8",
    )
    (selection / "memstrata__B16.json").write_text(
        json.dumps({"chunks": [{"chunk_id": index} for index in range(3)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_BENCH_ROOT", bench)
    monkeypatch.setattr(runner, "_OUTPUT_ROOT", output)
    assert runner._summary_selection_complete(
        {
            "system": "memstrata__B16",
            "dataset": "Dataset",
            "movie": "Movie",
            "expected_chunks": 3,
        }
    )


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
    monkeypatch.setenv("MAVE_STAGE1_LOCK_STALE_MINUTES", "45")

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
    monkeypatch.setenv("MAVE_TEST_FLOAT", "not-a-number")
    assert runner._env_float("MAVE_TEST_FLOAT", 7.5) == pytest.approx(7.5)
    monkeypatch.setenv("MAVE_TEST_FLOAT", "2.5")
    assert runner._env_float("MAVE_TEST_FLOAT", 7.5) == pytest.approx(2.5)


def test_resume_checkpoint_requires_exact_config_and_prefix(tmp_path):
    path = tmp_path / "stage1_checkpoint.json"
    config = {"chunk_ids": [0, 1, 2], "benchmark_git": {"commit": "bench-a"}}
    payload = {
        "schema_version": 1,
        "status": "committed",
        "config": config,
        "completed_chunk_ids": [0, 1],
        "records": [
            {"chunk_id": 0, "items": [], "extras": {"intent_resolution_source": "recency"}},
            {
                "chunk_id": 1,
                "items": [
                    {
                        "evidence_kind": "reference_image",
                        "source_seconds": 0.0,
                        "source_chunk_id": 0,
                        "latent_index": None,
                        "score": None,
                        "raw_ref": "memstrata:a:r",
                        "image_path": "/tmp/ref.png",
                    }
                ],
                "extras": {},
            },
        ],
        "adapter_state": {"bank_sha256": "abc"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, adapter_state = runner._load_stage1_checkpoint(path, config)
    assert [record.chunk_id for record in records] == [0, 1]
    assert records[1].items[0].source_chunk_id == 0
    assert adapter_state == {"bank_sha256": "abc"}

    with pytest.raises(RuntimeError, match="commit/config mismatch"):
        runner._load_stage1_checkpoint(
            path, {"chunk_ids": [0, 1, 2], "benchmark_git": {"commit": "bench-b"}}
        )

    payload["completed_chunk_ids"] = [0, 2]
    payload["records"][1]["chunk_id"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact prefix"):
        runner._load_stage1_checkpoint(path, config)


def test_recovery_lineage_requires_approved_patch_and_ancestry():
    proof = runner._recovery_lineage_proof(
        root=runner._BENCH_ROOT,
        old_commit="9301039e0741d879c40ff9f454c879b1446a976a",
        current_commit="d8bbc0d8fa4c39a1dd858a53ad9cf9d734ed7477",
        allowed_patch_ids=runner._RECOVERY_PATCH_IDS,
        allowed_paths=runner._RECOVERY_PATHS,
    )
    assert proof["verdict"] == "PASS"
    assert proof["patch_ids"] == ["f8e036c3684e8d76d92667242db81ecef2ac3b4a"]

    with pytest.raises(RuntimeError, match="not an ancestor"):
        runner._recovery_lineage_proof(
            root=runner._BENCH_ROOT,
            old_commit="996e975",
            current_commit="d8bbc0d8",
            allowed_patch_ids=runner._RECOVERY_PATCH_IDS,
            allowed_paths=runner._RECOVERY_PATHS,
        )


def test_partial_output_requires_explicit_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert runner._has_partial_output(run_dir) is False
    (run_dir / ".stage1.lock").write_text("pid=1", encoding="utf-8")
    assert runner._has_partial_output(run_dir) is False
    (run_dir / "_adapter_work").mkdir()
    assert runner._has_partial_output(run_dir) is True


def test_legacy_adoption_validates_provenance_and_freezes_bank(tmp_path, monkeypatch):
    run_dir = tmp_path / "output" / "memstrata__B16" / "Dataset" / "Movie"
    work_dir = run_dir / "_adapter_work" / "memstrata__B16"
    selection = run_dir / "visual_selections" / "memstrata__B16.json"
    selection.parent.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": 0,
                        "retrieval_timing": {"compose_ms": 1, "observe_ms": 2},
                        "selected": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bank = work_dir / "bank.json"
    bank.write_text(json.dumps({"assets": {}, "version": 1}), encoding="utf-8")
    provenance = tmp_path / "results.json"
    provenance.write_text(
        json.dumps(
            {
                "method_git_sha": "method-base",
                "benchmark_git_sha": "benchmark-base",
                "command": [
                    "python",
                    "runner.py",
                    "--adapter",
                    "memstrata",
                    "--movie-dir",
                    "/old/checkout/assets/trackA/Dataset/Movie",
                    "--input-mode",
                    "name_anchored",
                    "--budget",
                    "16",
                ],
                "artifacts": {"selection": str(selection), "bank": str(bank)},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "chunk_ids": [0, 1],
        "dataset": "Dataset",
        "movie": "Movie",
        "input_mode": "name_anchored",
        "budget": 16,
        "benchmark_git": {"commit": "fixed-benchmark"},
    }

    class Adopter:
        name = "memstrata"

        def stage1_resume_identity(self):
            return {
                "method_git": {"commit": "fixed-method"},
                "repo_root": "/method",
                "recovery_policy": {"patch_ids": ["method-patch"], "paths": ["method.py"]},
            }

        def adopt_stage1_checkpoint(self, **kwargs):
            assert kwargs["bank_path"] == bank
            return {"bank_sha256": "frozen"}

    def fake_proof(**kwargs):
        return {
            "verdict": "PASS",
            "old_commit": kwargs["old_commit"],
            "new_commit": kwargs["current_commit"],
        }

    monkeypatch.setattr(runner, "_recovery_lineage_proof", fake_proof)
    checkpoint = run_dir / "stage1_checkpoint.json"
    records, state = runner._adopt_legacy_stage1_checkpoint(
        provenance_path=provenance,
        checkpoint_path=checkpoint,
        expected_config=config,
        selection_path=selection,
        adapter=Adopter(),
        work_dir=work_dir,
    )
    assert [record.chunk_id for record in records] == [0]
    assert state == {"bank_sha256": "frozen"}
    committed = json.loads(checkpoint.read_text())
    assert committed["adopted_from"] == str(provenance.resolve())
    assert committed["compatibility_verification"]["verdict"] == "PASS"
    assert committed["compatibility_verification"]["method"]["old_commit"] == "method-base"


def test_run_movie_resumes_only_committed_prefix_without_duplicate_writes(
    tmp_path, monkeypatch
):
    movie_dir = tmp_path / "bench" / "assets" / "trackA" / "Dataset" / "Movie"
    (movie_dir / "gold").mkdir(parents=True)
    (movie_dir / "gold" / "chunk_annotations.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {"chunk_id": cid, "seconds_span": [cid, cid + 1], "prompt": f"p{cid}"}
                    for cid in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    monkeypatch.setattr(runner, "_OUTPUT_ROOT", output)
    monkeypatch.setattr(runner, "_resolve_source_video", lambda _: source)
    monkeypatch.setattr(
        runner, "_git_state", lambda _: {"commit": "bench", "tracked_diff_sha256": "clean"}
    )

    def fake_cut(_ffmpeg, _src, out, _s0, _s1):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"segment")
        return out

    def write_selection(*, system, movie, records, out_dir, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"chunk_id": record.chunk_id, "prompt": "", "selected": []}
            for record in records
        ]
        (out_dir / f"{system}.json").write_text(
            json.dumps({"movie": movie.movie_id, "system": system, "chunks": rows}),
            encoding="utf-8",
        )
        return {"system": system, "chunks": len(rows)}

    def append_selection(*, system, movie, rec, out_dir, **kwargs):
        path = out_dir / f"{system}.json"
        previous = json.loads(path.read_text())["chunks"] if path.is_file() else []
        by_id = {int(row["chunk_id"]): row for row in previous}
        by_id[rec.chunk_id] = {"chunk_id": rec.chunk_id, "prompt": "", "selected": []}
        return write_selection(
            system=system,
            movie=movie,
            records=[
                runner.RetrievedMemory(chunk_id=cid) for cid in sorted(by_id)
            ],
            out_dir=out_dir,
            **kwargs,
        )

    monkeypatch.setattr(runner, "_cut_segment", fake_cut)
    monkeypatch.setattr(runner, "materialize_record_checkpoint", append_selection)
    monkeypatch.setattr(runner, "materialize_system", write_selection)
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner, "runtime_context", lambda: {})

    class FakeAdapter:
        name = "memstrata"

        def __init__(self, fail_on=None):
            self.fail_on = fail_on
            self.observed = []
            self.resume_state = None

        def reset(self, movie):
            self.resume_state = movie.extras.get("stage1_resume")

        def compose(self, req):
            return runner.RetrievedMemory(chunk_id=req.chunk_id)

        def observe_segment(self, obs):
            if obs.chunk_id == self.fail_on:
                raise RuntimeError("simulated crop failure")
            self.observed.append(obs.chunk_id)

        def stage1_checkpoint(self, segment_id):
            return {"bank_sha256": f"bank-through-{segment_id}"}

        def finalize(self):
            return {"assets": len(self.observed)}

    first = FakeAdapter(fail_on=2)
    with pytest.raises(RuntimeError, match="simulated crop failure"):
        runner.run_movie(
            first,
            movie_dir,
            ffmpeg="ffmpeg",
            fps=16,
            limit=None,
            budget=16,
        )
    assert first.observed == [0, 1]

    resumed = FakeAdapter()
    summary = runner.run_movie(
        resumed,
        movie_dir,
        ffmpeg="ffmpeg",
        fps=16,
        limit=None,
        budget=16,
        resume=True,
    )
    assert resumed.resume_state["completed_chunk_ids"] == [0, 1]
    assert resumed.observed == [2]
    assert summary["chunks"] == 3
    selection = json.loads(
        (
            output
            / "memstrata__B16"
            / "Dataset"
            / "Movie"
            / "visual_selections"
            / "memstrata__B16.json"
        ).read_text()
    )
    assert [row["chunk_id"] for row in selection["chunks"]] == [0, 1, 2]
