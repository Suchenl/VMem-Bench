"""S7 review continuation stays synchronous and outside the job registry."""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline.servers.backend import jobs


def test_after_s6_continuation_freezes_without_creating_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    movie_dir = tmp_path / "movie"
    sample = {
        "dataset": "BlenderOpenMovies",
        "movie_id": "movie",
        "movie_dir": str(movie_dir),
    }
    store = jobs.JobStore(tmp_path / "jobs", tmp_path / "data")
    observed: dict[str, object] = {}

    monkeypatch.setattr(jobs, "list_samples", lambda **_: [sample])

    def fake_continue_after_s6(*, movie_dir: Path, automation_smoke: bool) -> dict[str, str]:
        observed["movie_dir"] = movie_dir
        observed["automation_smoke"] = automation_smoke
        return {
            "status": "human_reviewed_complete",
            "movie_dir": str(movie_dir),
            "gold": str(movie_dir / "gold"),
        }

    monkeypatch.setattr(jobs, "continue_after_s6", fake_continue_after_s6)

    result = store.create_continue_job(
        {
            "dataset": sample["dataset"],
            "movie_id": sample["movie_id"],
            "continue_from": "after_s6",
        }
    )

    assert result["ok"] is True
    assert result["synchronous"] is True
    assert result["continue_from"] == "after_s6"
    assert observed == {"movie_dir": movie_dir, "automation_smoke": False}
    assert list((tmp_path / "jobs").glob("*/job.json")) == []
