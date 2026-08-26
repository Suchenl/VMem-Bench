"""Materialize S2 and enqueue every selected segment into SharedClipQueue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.orchestration.contracts import MovieManifest
from vmem_bench.annotation.pipeline.stages.s2_annotation_postprocess.materialize import (
    postprocess_annotation,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.clip_queue import (
    ClipTask,
    SharedClipQueue,
)


def _catalog(path: Path) -> list[MovieManifest]:
    return [
        MovieManifest.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _segments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for scene in (annotation.get("screenplay") or {}).get("scenes") or []
        for segment in scene.get("visual_segments") or []
    ]


def _task_id(output_path: Path) -> str:
    return "clip-" + hashlib.sha256(str(output_path).encode("utf-8")).hexdigest()[:40]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    queue = SharedClipQueue(args.queue_root)
    rows: list[dict[str, Any]] = []
    for movie in _catalog(args.catalog):
        if movie.status != "source_ready":
            rows.append({"movie_id": movie.movie_id, "status": movie.status, "n_enqueued": 0})
            continue
        pipeline = movie.root / "tmp" / "pipeline"
        s2 = postprocess_annotation(
            Path(movie.vlm_output),
            pipeline / "s2_annotation_postprocess",
        )
        if s2["status"] != "ok":
            rows.append({"movie_id": movie.movie_id, "status": s2["status"], "n_enqueued": 0})
            continue
        annotation = json.loads(Path(s2["normalized_annotation"]).read_text(encoding="utf-8"))
        states: dict[str, int] = {}
        for segment in _segments(annotation):
            segment_id = str(segment["segment_id"])
            output_path = (
                pipeline
                / "s3_segment_auto_review_revise"
                / "clip_cache"
                / "prefetched"
                / segment_id
                / "segment.mp4"
            )
            state = queue.enqueue(
                ClipTask(
                    task_id=_task_id(output_path),
                    source_video=Path(movie.source_video),
                    output_path=output_path,
                    start_seconds=float(segment["start_seconds"]),
                    end_seconds=float(segment["end_seconds"]),
                    metadata={
                        "dataset": movie.dataset,
                        "movie_id": movie.movie_id,
                        "segment_id": segment_id,
                    },
                )
            )
            states[state] = states.get(state, 0) + 1
        rows.append({"movie_id": movie.movie_id, "status": "ok", "states": states})
    payload = {"queue_root": str(args.queue_root), "movies": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
