"""Batch S2--S7 orchestration over a shallow catalog JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.orchestration.contracts import (
    S4_MODES,
    MovieManifest,
    resolve_s4_mode,
)
from vmem_bench.annotation.pipeline.orchestration.orchestrator import (
    auto_accept_pending_s4,
    continue_after_s4,
    continue_after_s6,
    infer_resume_action,
    run_pipeline,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (
    DEFAULT_MAX_REVIEW_ROUNDS,
    DEFAULT_MAX_TOKENS,
    build_qwen_reviewer,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.grounding_dino import (
    GroundingDinoProposer,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_concept import (
    Sam3ConceptSegmenter,
)

_BEIJING = timezone(timedelta(hours=8))


def _now_stamp() -> str:
    return datetime.now(_BEIJING).strftime("%Y-%m-%d %H:%M:%S")


def _same_movie(row: dict[str, Any], key: dict[str, str]) -> bool:
    return (
        str(row.get("dataset") or "") == str(key.get("dataset") or "")
        and str(row.get("movie_id") or "") == str(key.get("movie_id") or "")
    )


class BatchProgress:
    """Durable per-movie running/queued/done markers for the console."""

    def __init__(self, path: Path | None, catalog: list[MovieManifest]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.pending = [
            {"dataset": item.dataset, "movie_id": item.movie_id}
            for item in catalog
            if item.status == "source_ready"
        ]
        self.running: list[dict[str, str]] = []
        self.done: list[dict[str, str]] = []
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        payload = {
            "updated_at": _now_stamp(),
            "pending": list(self.pending),
            "running": list(self.running),
            "done": list(self.done),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def mark_running(self, item: MovieManifest) -> None:
        key = {"dataset": item.dataset, "movie_id": item.movie_id}
        with self._lock:
            self.pending = [row for row in self.pending if not _same_movie(row, key)]
            self.running = [row for row in self.running if not _same_movie(row, key)]
            self.running.append(key)
            self._write()

    def mark_done(self, item: MovieManifest, status: str) -> None:
        key = {"dataset": item.dataset, "movie_id": item.movie_id, "status": status}
        with self._lock:
            self.pending = [row for row in self.pending if not _same_movie(row, key)]
            self.running = [row for row in self.running if not _same_movie(row, key)]
            self.done = [row for row in self.done if not _same_movie(row, key)]
            self.done.append(key)
            self._write()


def _catalog(path: Path) -> list[MovieManifest]:
    return [
        MovieManifest.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _shared_s5_proposers(
    *, crop_route: str, proposer: str
) -> tuple[Sam3ConceptSegmenter | None, GroundingDinoProposer | None]:
    """Create lazy S5 proposers once for the batch, never once per movie."""
    if crop_route != "propose_and_pick":
        return None, None
    return (
        Sam3ConceptSegmenter() if proposer in ("sam3", "fusion") else None,
        GroundingDinoProposer() if proposer in ("gdino", "fusion") else None,
    )


def _clear_movie_pipeline(movie_dir: Path) -> None:
    """Drop tmp/pipeline so a force-restart begins at S2."""
    pipeline = movie_dir / "tmp" / "pipeline"
    if pipeline.is_dir():
        shutil.rmtree(pipeline)


def _run_one_movie(
    *,
    item: MovieManifest,
    skip_human: bool,
    reviewer_mode: str,
    reviewer_base_url: str,
    reviewer_model: str,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    max_tasks: int | None,
    s4_mode: str,
    continue_from: str,
    resume: bool,
    max_review_rounds: int,
    max_review_workers: int | None,
    max_clip_workers: int | None,
    clip_queue_root: Path | None,
    max_tokens: int,
    crop_route: str,
    proposer: str,
    task_mode: str,
    auto_accept_s4: bool,
    force_restart: bool = False,
    progress: BatchProgress | None = None,
    reviewer_override=None,
    endpoint_pool_override=None,
    s5_segmenter: Sam3ConceptSegmenter | None = None,
    s5_detector: GroundingDinoProposer | None = None,
) -> dict:
    """Run or resume one catalog row; preserve per-movie stage progress when resume=True."""
    if progress is not None:
        progress.mark_running(item)
    try:
        if force_restart:
            _clear_movie_pipeline(item.root)
            resume = False
            continue_from = ""
        action = continue_from
        if resume and not continue_from:
            action = infer_resume_action(item.root)
            if action == "awaiting_human" and auto_accept_s4 and auto_accept_pending_s4(item.root):
                action = infer_resume_action(item.root)
            if action == "complete":
                result = {
                    "movie_id": item.movie_id,
                    "dataset": item.dataset,
                    "status": "already_complete",
                    "resume_action": action,
                }
                if progress is not None:
                    progress.mark_done(item, "already_complete")
                return result
            if action == "awaiting_human":
                result = {
                    "movie_id": item.movie_id,
                    "dataset": item.dataset,
                    "status": "awaiting_human",
                    "resume_action": action,
                }
                if progress is not None:
                    progress.mark_done(item, "awaiting_human")
                return result
            if action == "pipeline":
                action = ""

        if action in {"after_s4", "s5_only"}:
            continue_skip_human = skip_human or action == "s5_only"
            # Production S5 propose_and_pick requires qwen picker (same as console 续跑 S5).
            effective_grounder = "qwen" if action == "after_s4" else grounder_mode
            result = continue_after_s4(
                movie_dir=item.root,
                source_video=Path(item.source_video),
                grounder_mode=effective_grounder,
                grounder_base_url=grounder_base_url,
                grounder_model=grounder_model,
                skip_human=continue_skip_human,
                max_tasks=max_tasks,
                s4_mode=resolve_s4_mode(
                    s4_mode,
                    skip_human=continue_skip_human,
                ),
                crop_route=crop_route,
                proposer=proposer,
                task_mode=task_mode,
                s5_segmenter=s5_segmenter,
                s5_detector=s5_detector,
            )
        elif action == "after_s6":
            result = continue_after_s6(movie_dir=item.root, automation_smoke=False)
        else:
            result = run_pipeline(
                movie_dir=item.root,
                source_video=Path(item.source_video),
                reviewer_mode=reviewer_mode,
                reviewer_base_url=reviewer_base_url,
                reviewer_model=reviewer_model,
                grounder_mode=grounder_mode,
                grounder_base_url=grounder_base_url,
                grounder_model=grounder_model,
                skip_human=skip_human,
                max_tasks=max_tasks,
                s4_mode=s4_mode,
                max_review_rounds=max_review_rounds,
                max_review_workers=max_review_workers,
                max_clip_workers=max_clip_workers,
                clip_queue_root=clip_queue_root,
                max_tokens=max_tokens,
                crop_route=crop_route,
                proposer=proposer,
                task_mode=task_mode,
                resume=resume,
                auto_accept_s4=auto_accept_s4,
                reviewer_override=reviewer_override,
                endpoint_pool_override=endpoint_pool_override,
                s5_segmenter=s5_segmenter,
                s5_detector=s5_detector,
            )
        payload = {"movie_id": item.movie_id, "dataset": item.dataset, **result}
        if resume and not continue_from:
            payload["resume_action"] = action or "pipeline"
        if progress is not None:
            progress.mark_done(item, str(payload.get("status") or "ok"))
        return payload
    except Exception as exc:  # noqa: BLE001 - keep batch-level isolation
        if progress is not None:
            progress.mark_done(item, "failed")
        return {
            "movie_id": item.movie_id,
            "dataset": item.dataset,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_parallel_s3_batch(
    *,
    catalog: list[MovieManifest],
    out_path: Path,
    skip_human: bool,
    reviewer_mode: str,
    reviewer_base_url: str,
    reviewer_model: str,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    max_tasks: int | None,
    limit: int | None,
    s4_mode: str,
    continue_from: str,
    resume: bool,
    max_review_rounds: int,
    max_review_workers: int | None,
    max_clip_workers: int | None,
    clip_queue_root: Path | None,
    max_tokens: int,
    crop_route: str,
    proposer: str,
    task_mode: str,
    auto_accept_s4: bool,
    max_parallel_movies: int,
    s5_segmenter: Sam3ConceptSegmenter | None,
    s5_detector: GroundingDinoProposer | None,
    force_restart: bool = False,
    progress: BatchProgress | None = None,
) -> list[dict]:
    """Run multiple movies against one shared endpoint pool."""
    if reviewer_mode != "qwen":
        raise ValueError("parallel S3 batch requires reviewer=qwen")
    if not reviewer_base_url:
        raise ValueError("parallel S3 batch requires reviewer_base_url")
    reviewer, endpoint_pool = build_qwen_reviewer(
        base_urls=reviewer_base_url,
        model=reviewer_model,
        max_tokens=max_tokens,
    )
    records: list[dict] = []
    scheduled: list[MovieManifest] = []
    for item in catalog:
        if item.status != "source_ready":
            records.append({"movie_id": item.movie_id, "dataset": item.dataset, "status": item.status})
        elif limit is not None and len(scheduled) >= limit:
            records.append({"movie_id": item.movie_id, "dataset": item.dataset, "status": "not_scheduled"})
        else:
            scheduled.append(item)

    def run_one(item: MovieManifest) -> dict:
        return _run_one_movie(
            item=item,
            skip_human=skip_human,
            reviewer_mode=reviewer_mode,
            reviewer_base_url=reviewer_base_url,
            reviewer_model=reviewer_model,
            grounder_mode=grounder_mode,
            grounder_base_url=grounder_base_url,
            grounder_model=grounder_model,
            max_tasks=max_tasks,
            s4_mode=s4_mode,
            continue_from=continue_from,
            resume=resume,
            max_review_rounds=max_review_rounds,
            max_review_workers=max_review_workers,
            max_clip_workers=max_clip_workers,
            clip_queue_root=clip_queue_root,
            max_tokens=max_tokens,
            crop_route=crop_route,
            proposer=proposer,
            task_mode=task_mode,
            auto_accept_s4=auto_accept_s4,
            force_restart=force_restart,
            progress=progress,
            reviewer_override=reviewer,
            endpoint_pool_override=endpoint_pool,
            s5_segmenter=s5_segmenter,
            s5_detector=s5_detector,
        )

    outcomes: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_parallel_movies) as executor:
        futures = {executor.submit(run_one, item): item.movie_id for item in scheduled}
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    records.extend(outcomes[item.movie_id] for item in scheduled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def run_batch(
    *,
    catalog_path: Path,
    out_path: Path,
    skip_human: bool,
    reviewer_mode: str,
    reviewer_base_url: str,
    reviewer_model: str,
    grounder_mode: str,
    grounder_base_url: str,
    grounder_model: str,
    max_tasks: int | None,
    limit: int | None,
    s4_mode: str = "auto",
    continue_from: str = "",
    resume: bool = False,
    max_review_rounds: int = DEFAULT_MAX_REVIEW_ROUNDS,
    max_review_workers: int | None = None,
    max_clip_workers: int | None = None,
    clip_queue_root: Path | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    crop_route: str = "propose_and_pick",
    proposer: str = "sam3",
    task_mode: str = "coverage",
    auto_accept_s4: bool = False,
    max_parallel_movies: int = 1,
    force_restart: bool = False,
    progress_path: Path | None = None,
) -> list[dict]:
    """Run ready items, explicitly preserving skipped/incomplete movies."""
    if max_parallel_movies < 1:
        raise ValueError("max_parallel_movies must be >= 1")
    catalog = _catalog(catalog_path)
    progress = BatchProgress(progress_path, catalog)
    s5_segmenter, s5_detector = _shared_s5_proposers(
        crop_route=crop_route,
        proposer=proposer,
    )
    if max_parallel_movies > 1 and not continue_from:
        return _run_parallel_s3_batch(
            catalog=catalog,
            out_path=out_path,
            skip_human=skip_human,
            reviewer_mode=reviewer_mode,
            reviewer_base_url=reviewer_base_url,
            reviewer_model=reviewer_model,
            grounder_mode=grounder_mode,
            grounder_base_url=grounder_base_url,
            grounder_model=grounder_model,
            max_tasks=max_tasks,
            limit=limit,
            s4_mode=s4_mode,
            continue_from=continue_from,
            resume=resume,
            max_review_rounds=max_review_rounds,
            max_review_workers=max_review_workers,
            max_clip_workers=max_clip_workers,
            clip_queue_root=clip_queue_root,
            max_tokens=max_tokens,
            crop_route=crop_route,
            proposer=proposer,
            task_mode=task_mode,
            auto_accept_s4=auto_accept_s4,
            max_parallel_movies=max_parallel_movies,
            s5_segmenter=s5_segmenter,
            s5_detector=s5_detector,
            force_restart=force_restart,
            progress=progress,
        )
    records: list[dict] = []
    ready = 0
    for item in catalog:
        if item.status != "source_ready":
            records.append({"movie_id": item.movie_id, "dataset": item.dataset, "status": item.status})
            continue
        if limit is not None and ready >= limit:
            records.append({"movie_id": item.movie_id, "dataset": item.dataset, "status": "not_scheduled"})
            continue
        ready += 1
        records.append(
            _run_one_movie(
                item=item,
                skip_human=skip_human,
                reviewer_mode=reviewer_mode,
                reviewer_base_url=reviewer_base_url,
                reviewer_model=reviewer_model,
                grounder_mode=grounder_mode,
                grounder_base_url=grounder_base_url,
                grounder_model=grounder_model,
                max_tasks=max_tasks,
                s4_mode=s4_mode,
                continue_from=continue_from,
                resume=resume,
                max_review_rounds=max_review_rounds,
                max_review_workers=max_review_workers,
                max_clip_workers=max_clip_workers,
                clip_queue_root=clip_queue_root,
                max_tokens=max_tokens,
                crop_route=crop_route,
                proposer=proposer,
                task_mode=task_mode,
                auto_accept_s4=auto_accept_s4,
                force_restart=force_restart,
                progress=progress,
                s5_segmenter=s5_segmenter,
                s5_detector=s5_detector,
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument("--s4-mode", choices=S4_MODES, default="auto")
    parser.add_argument(
        "--continue-from",
        choices=("", "after_s4", "s5_only", "after_s6"),
        default="",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Per-movie continue: keep S3 audit progress, or jump to after_s4/after_s6 "
            "when that sample is ready; skip samples awaiting human review"
        ),
    )
    parser.add_argument(
        "--auto-accept-s4",
        action="store_true",
        help="accept pending and newly generated S4 queues, then continue to S5",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="clear each movie tmp/pipeline before running (S2 restart)",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="write per-movie running/queued/done progress JSON for the console",
    )
    parser.add_argument("--reviewer", choices=("passthrough", "qwen"), default="passthrough")
    parser.add_argument(
        "--reviewer-base-url",
        default="",
        help="Qwen reviewer URL, or comma-separated H800 endpoint pool",
    )
    parser.add_argument("--reviewer-model", default="qwen3-vl-32b")
    parser.add_argument("--max-review-rounds", type=int, default=DEFAULT_MAX_REVIEW_ROUNDS)
    parser.add_argument("--max-review-workers", type=int, default=None)
    parser.add_argument("--max-clip-workers", type=int, default=None)
    parser.add_argument("--clip-queue-root", type=Path, default=None)
    parser.add_argument("--max-parallel-movies", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--grounder", choices=("full-frame", "qwen"), default="full-frame")
    parser.add_argument("--grounder-base-url", default="")
    parser.add_argument("--grounder-model", default="qwen3-vl-32b")
    parser.add_argument(
        "--crop-route",
        choices=("vlm_sam_refine", "propose_and_pick"),
        default="propose_and_pick",
    )
    parser.add_argument("--proposer", choices=("sam3", "gdino", "fusion"), default="sam3")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--task-mode",
        choices=("coverage", "per_slot"),
        default="coverage",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_batch(
        catalog_path=args.catalog,
        out_path=args.out,
        skip_human=args.skip_human,
        reviewer_mode=args.reviewer,
        reviewer_base_url=args.reviewer_base_url,
        reviewer_model=args.reviewer_model,
        grounder_mode=args.grounder,
        grounder_base_url=args.grounder_base_url,
        grounder_model=args.grounder_model,
        max_tasks=args.max_tasks,
        limit=args.limit,
        s4_mode=args.s4_mode,
        continue_from=args.continue_from,
        resume=bool(args.resume),
        max_review_rounds=args.max_review_rounds,
        max_review_workers=args.max_review_workers,
        max_clip_workers=args.max_clip_workers,
        clip_queue_root=args.clip_queue_root,
        max_tokens=args.max_tokens,
        crop_route=args.crop_route,
        proposer=args.proposer,
        task_mode=args.task_mode,
        auto_accept_s4=bool(args.auto_accept_s4),
        max_parallel_movies=args.max_parallel_movies,
        force_restart=bool(args.force_restart),
        progress_path=args.progress,
    )


if __name__ == "__main__":
    main()
