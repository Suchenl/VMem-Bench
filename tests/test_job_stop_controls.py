"""Console job stop-all / sample-activity helpers."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.servers.backend.jobs import JobStore


def _write_job(root: Path, job_id: str, payload: dict) -> None:
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(
        json.dumps({"job_id": job_id, **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_sample_activity_uses_progress_for_multi_movie_job(tmp_path: Path) -> None:
    store = JobStore(jobs_root=tmp_path / "jobs", data_root=tmp_path / "data")
    job_id = "job-batch"
    job_dir = store.jobs_root / job_id
    job_dir.mkdir(parents=True)
    samples = [
        {"dataset": "LSMDC", "movie_id": "movie_a"},
        {"dataset": "LSMDC", "movie_id": "movie_b"},
        {"dataset": "LSMDC", "movie_id": "movie_c"},
    ]
    (job_dir / "job.json").write_text(
        json.dumps(
            {"job_id": job_id, "status": "running", "pid": 0, "samples": samples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (job_dir / "progress.json").write_text(
        json.dumps(
            {
                "pending": [
                    {"dataset": "LSMDC", "movie_id": "movie_b"},
                    {"dataset": "LSMDC", "movie_id": "movie_c"},
                ],
                "running": [{"dataset": "LSMDC", "movie_id": "movie_a"}],
                "done": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    snap = store.jobs_snapshot()
    assert snap["sample_activity"]["LSMDC::movie_a"]["sample_state"] == "running"
    assert snap["sample_activity"]["LSMDC::movie_b"]["sample_state"] == "queued"
    assert snap["sample_activity"]["LSMDC::movie_c"]["sample_state"] == "queued"
    assert snap["sample_counts"] == {"running": 1, "queued": 2, "covered": 3}


def test_sample_activity_and_stop_all(tmp_path: Path) -> None:
    store = JobStore(jobs_root=tmp_path / "jobs", data_root=tmp_path / "data")
    _write_job(
        store.jobs_root,
        "job-running",
        {
            "status": "running",
            "pid": 0,
            "samples": [{"dataset": "LSMDC", "movie_id": "movie_a"}],
        },
    )
    _write_job(
        store.jobs_root,
        "job-queued",
        {
            "status": "queued",
            "samples": [
                {"dataset": "LSMDC", "movie_id": "movie_b"},
                {"dataset": "LSMDC", "movie_id": "movie_c"},
            ],
            "bdy_queue_job_id": "",
        },
    )
    _write_job(
        store.jobs_root,
        "job-done",
        {
            "status": "succeeded",
            "samples": [{"dataset": "LSMDC", "movie_id": "movie_d"}],
        },
    )

    snap = store.jobs_snapshot()
    assert {job["job_id"] for job in snap["active"]} == {"job-running", "job-queued"}
    assert snap["sample_activity"]["LSMDC::movie_a"]["sample_state"] == "running"
    assert snap["sample_activity"]["LSMDC::movie_b"]["sample_state"] == "queued"
    assert snap["sample_activity"]["LSMDC::movie_c"]["n_samples"] == 2
    assert "LSMDC::movie_d" not in snap["sample_activity"]

    stopped = store.stop_all_jobs()
    assert stopped["n_stopped"] == 2
    assert store.active_jobs() == []


def test_stop_sample_job(tmp_path: Path) -> None:
    store = JobStore(jobs_root=tmp_path / "jobs", data_root=tmp_path / "data")
    _write_job(
        store.jobs_root,
        "job-one",
        {
            "status": "running",
            "pid": 0,
            "samples": [{"dataset": "BlenderOpenMovies", "movie_id": "big_buck_bunny"}],
        },
    )
    result = store.stop_sample_job(dataset="BlenderOpenMovies", movie_id="big_buck_bunny")
    assert result["ok"] is True
    assert result["status"] in {"stopping", "stopped"}
    refreshed = store.refresh_job("job-one")
    assert refreshed["status"] in {"stopping", "stopped"}


def test_create_job_skips_busy_samples(tmp_path: Path, monkeypatch) -> None:
    store = JobStore(jobs_root=tmp_path / "jobs", data_root=tmp_path / "data")
    _write_job(
        store.jobs_root,
        "job-busy",
        {
            "status": "running",
            "pid": 0,
            "samples": [{"dataset": "LSMDC", "movie_id": "busy_movie"}],
        },
    )

    def fake_list_samples(**_kwargs):
        return [
            {
                "dataset": "LSMDC",
                "movie_id": "busy_movie",
                "movie_dir": str(tmp_path / "busy"),
                "source_video": str(tmp_path / "busy.mp4"),
                "source_video_exists": True,
                "has_vlm_output": True,
            },
            {
                "dataset": "LSMDC",
                "movie_id": "idle_movie",
                "movie_dir": str(tmp_path / "idle"),
                "source_video": str(tmp_path / "idle.mp4"),
                "source_video_exists": True,
                "has_vlm_output": True,
            },
        ]

    monkeypatch.setattr(
        "vmem_bench.annotation.pipeline.servers.backend.jobs.list_samples",
        fake_list_samples,
    )
    monkeypatch.setattr(
        "vmem_bench.annotation.pipeline.servers.backend.jobs.find_sample",
        lambda samples, dataset, movie_id: next(
            (s for s in samples if s["dataset"] == dataset and s["movie_id"] == movie_id),
            None,
        ),
    )
    monkeypatch.setattr(
        "vmem_bench.annotation.pipeline.servers.backend.jobs.manifest_from_sample",
        lambda sample: type(
            "M",
            (),
            {
                "dataset": sample["dataset"],
                "movie_id": sample["movie_id"],
                "to_dict": lambda self: {
                    "dataset": sample["dataset"],
                    "movie_id": sample["movie_id"],
                    "root": sample["movie_dir"],
                    "source_video": sample["source_video"],
                },
            },
        )(),
    )

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(
        "vmem_bench.annotation.pipeline.servers.backend.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeProc(),
    )

    payload = store.create_job(
        {
            "samples": [
                {"dataset": "LSMDC", "movie_id": "busy_movie"},
                {"dataset": "LSMDC", "movie_id": "idle_movie"},
            ],
            "reviewer": "passthrough",
            "grounder": "full-frame",
            "execution_target": "local",
        }
    )
    assert [s["movie_id"] for s in payload["samples"]] == ["idle_movie"]
    assert len(payload["skipped_busy"]) == 1
    assert payload["skipped_busy"][0]["movie_id"] == "busy_movie"
