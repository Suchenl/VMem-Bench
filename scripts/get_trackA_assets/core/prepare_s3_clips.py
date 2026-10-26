"""Prepare reusable S3 clips for one deterministic movie shard.

Workers write clips to each movie's shared ``tmp/pipeline`` cache.  The S3
runner consumes those ready clips without re-encoding them and deletes them
after the movie's S3 stage finishes.
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.orchestration.contracts import MovieManifest
from vmem_bench.annotation.pipeline.stages.s2_annotation_postprocess.materialize import (
    postprocess_annotation,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.segment_media import (
    worker_clip,
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


def _in_shard(movie_id: str, shard_index: int, shard_count: int) -> bool:
    return (zlib.adler32(movie_id.encode("utf-8")) & 0x7FFFFFFF) % shard_count == shard_index


def _prepare_movie(movie: MovieManifest, workers: int) -> dict[str, Any]:
    pipeline = movie.root / "tmp" / "pipeline"
    s2_dir = pipeline / "s2_annotation_postprocess"
    s2 = postprocess_annotation(Path(movie.vlm_output), s2_dir)
    if s2["status"] != "ok":
        return {"movie_id": movie.movie_id, "status": s2["status"], "n_segments": 0}
    annotation = json.loads(Path(s2["normalized_annotation"]).read_text(encoding="utf-8"))
    segments = _segments(annotation)
    cache_root = pipeline / "s3_segment_auto_review_revise" / "clip_cache" / "prefetched"
    source_video = Path(movie.source_video)

    def prepare(segment: dict[str, Any]) -> str:
        segment_id = str(segment["segment_id"])
        target = cache_root / segment_id / "segment.mp4"
        if target.is_file() and target.stat().st_size > 0:
            return "reused"
        with worker_clip(
            source_video=source_video,
            cache_root=cache_root,
            worker_id=segment_id,
            start_seconds=float(segment["start_seconds"]),
            end_seconds=float(segment["end_seconds"]),
            output_path=target,
            remove_on_exit=False,
        ):
            pass
        return "prepared"

    counts = {"prepared": 0, "reused": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(prepare, segment) for segment in segments]
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception:  # noqa: BLE001 - keep remaining clips flowing
                counts["failed"] += 1
    return {"movie_id": movie.movie_id, "status": "ok", "n_segments": len(segments), **counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid workers or shard selection")
    movies = [
        item
        for item in _catalog(args.catalog)
        if item.status == "source_ready" and _in_shard(item.movie_id, args.shard_index, args.shard_count)
    ]
    results = [_prepare_movie(movie, args.workers) for movie in movies]
    payload = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": args.workers,
        "n_movies": len(movies),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if all(item["status"] == "ok" and not item.get("failed", 0) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
