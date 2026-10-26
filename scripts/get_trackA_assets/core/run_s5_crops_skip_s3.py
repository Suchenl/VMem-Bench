#!/usr/bin/env python3
"""Batch S5 crop acquisition, skipping S3/S4.

Uses annotation priority S4 > S3 > S2 (normalized). Default route:
``propose_and_pick`` + ``--proposer sam3``.

Example::

  CUDA_VISIBLE_DEVICES=0 \\
  PYTHONPATH=models/vendor/sam3_transformers59:src \\
  MEMSTRATA_SAM3_DEPS=$PWD/models/vendor/sam3_transformers59 \\
  python3 \\
    benchmarks/MemStrata/scripts/vmem_bench/core/run_s5_crops_skip_s3.py \\
    --grounder-base-url http://127.0.0.1:8113/v1 \\
    --out data/_runs/s5_skip_s3/batch_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vmem_bench.annotation.pipeline.orchestration.orchestrator import continue_after_s4

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "benchmarks" / "MemStrata" / "data"
BLENDER_VIDEOS = Path(
    "${VMEM_DATASETS_ROOT}/BlenderOpenMovies"
)
LSMDC_INDEX = Path(
    "${VMEM_DATASETS_ROOT}/LSMDC/complete_movies.json"
)


def _has_annotation(movie_dir: Path) -> bool:
    root = movie_dir / "tmp" / "pipeline"
    return any(
        (root / rel).is_file()
        for rel in (
            "s4_segment_sampling_human_review/human_revised_annotation.json",
            "s3_segment_auto_review_revise/auto_revised_annotation.json",
            "s2_annotation_postprocess/normalized_annotation.json",
        )
    )


def _blender_video(movie_id: str) -> Path | None:
    folder = BLENDER_VIDEOS / movie_id
    if not folder.is_dir():
        return None
    videos = sorted(folder.glob("*.mp4"))
    return videos[0] if videos else None


def _lsmdc_video(movie_id: str, by_id: dict) -> Path | None:
    meta = by_id.get(movie_id) or {}
    path = Path(str((meta.get("stitched") or {}).get("output_file") or ""))
    return path if path.is_file() else None


def build_jobs() -> list[dict[str, str]]:
    lsmdc = json.loads(LSMDC_INDEX.read_text(encoding="utf-8"))
    by_id = {str(m["movie_id"]): m for m in lsmdc.get("movies") or []}
    jobs: list[dict[str, str]] = []
    for dataset in ("BlenderOpenMovies", "LSMDC"):
        root = DATA / dataset
        if not root.is_dir():
            continue
        for movie_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if "copy" in movie_dir.name:
                continue
            if not _has_annotation(movie_dir):
                continue
            if dataset == "BlenderOpenMovies":
                video = _blender_video(movie_dir.name)
            else:
                video = _lsmdc_video(movie_dir.name, by_id)
            if video is None:
                continue
            jobs.append(
                {
                    "dataset": dataset,
                    "movie_id": movie_dir.name,
                    "movie_dir": str(movie_dir),
                    "source_video": str(video),
                }
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounder-base-url", required=True,
                        help="One URL or comma-separated picker pool")
    parser.add_argument("--grounder-model", default="qwen3-vl-8b")
    parser.add_argument("--crop-route", default="propose_and_pick")
    parser.add_argument("--proposer", default="sam3")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--task-mode",
        choices=("coverage", "per_slot"),
        default="coverage",
        help="coverage=capped library + ≤t slot bind (default); per_slot=legacy",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--movie-id", action="append", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA / "_runs" / "s5_skip_s3" / "batch_results.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=DATA / "_runs" / "s5_skip_s3" / "progress.jsonl",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, num-shards)")

    jobs = build_jobs()
    if args.movie_id:
        wanted = set(args.movie_id)
        jobs = [j for j in jobs if j["movie_id"] in wanted or j["movie_dir"].endswith(tuple(wanted))]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    catalog_path = args.out.with_name(f"{args.out.stem}_catalog_shard{args.shard_index}.json")
    catalog_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"shard {args.shard_index}/{args.num_shards}: scheduled {len(jobs)} movies -> {catalog_path}",
        flush=True,
    )

    records: list[dict] = []
    with args.progress.open("a", encoding="utf-8") as progress:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job['dataset']}/{job['movie_id']}", flush=True)
            try:
                result = continue_after_s4(
                    movie_dir=Path(job["movie_dir"]),
                    source_video=Path(job["source_video"]),
                    grounder_mode="qwen",
                    grounder_base_url=args.grounder_base_url,
                    grounder_model=args.grounder_model,
                    skip_human=True,
                    max_tasks=args.max_tasks,
                    crop_route=args.crop_route,
                    proposer=args.proposer,
                    task_mode=args.task_mode,
                )
                row = {**job, **result}
            except Exception as exc:  # isolate per movie
                row = {
                    **job,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:800],
                }
            records.append(row)
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  -> {row.get('status')} proposals={row.get('n_proposals')}", flush=True)

    print(f"done: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
