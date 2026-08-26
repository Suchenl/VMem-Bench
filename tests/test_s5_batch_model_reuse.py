"""S5 batch model lifetime tests."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.orchestration import batch


def test_batch_creates_s5_proposers_once_and_reuses_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "test",
                    "movie_id": f"movie_{index}",
                    "movie_dir": str(tmp_path / f"movie_{index}"),
                    "source_video": str(tmp_path / f"movie_{index}.mp4"),
                    "vlm_output": str(tmp_path / f"movie_{index}" / "vlm_output.json"),
                }
            )
            for index in range(2)
        ),
        encoding="utf-8",
    )
    created = {"sam3": 0, "dino": 0}
    observed: list[tuple[object, object]] = []

    class FakeSam3:
        def __init__(self) -> None:
            created["sam3"] += 1

    class FakeDino:
        def __init__(self) -> None:
            created["dino"] += 1

    def fake_run_one_movie(**kwargs):
        observed.append((kwargs["s5_segmenter"], kwargs["s5_detector"]))
        return {"movie_id": kwargs["item"].movie_id, "status": "ok"}

    monkeypatch.setattr(batch, "Sam3ConceptSegmenter", FakeSam3)
    monkeypatch.setattr(batch, "GroundingDinoProposer", FakeDino)
    monkeypatch.setattr(batch, "_run_one_movie", fake_run_one_movie)

    batch.run_batch(
        catalog_path=catalog,
        out_path=tmp_path / "result.json",
        skip_human=False,
        reviewer_mode="passthrough",
        reviewer_base_url="",
        reviewer_model="unused",
        grounder_mode="qwen",
        grounder_base_url="unused",
        grounder_model="unused",
        max_tasks=None,
        limit=None,
        proposer="fusion",
    )

    assert created == {"sam3": 1, "dino": 1}
    assert len(observed) == 2
    assert observed[0][0] is observed[1][0]
    assert observed[0][1] is observed[1][1]
