#!/usr/bin/env python
"""Track A Stage-2 background scorer with pooled VLM endpoint dispatch.

This keeps the benchmark metric in ``vmem_bench.scoring.visual_coverage`` and
only changes execution shape: discover completed Stage-1 outputs, score segments
through a pooled judge API/fleet, and write a small progress file that a detached
launcher can watch without SSH polling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = BENCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vmem_bench.scoring.visual_coverage import (  # noqa: E402
    DEFAULT_API,
    DEFAULT_FFMPEG,
    DEFAULT_MODEL,
    build_judge_api,
    run as run_visual_coverage,
)

DEFAULT_PUBLIC_MODELS_ROOT = "${PUBLIC_MODELS_ROOT}"
DEFAULT_SYSTEMS = (
    "memstrata",
    "longlive_rag",
    "memflow",
    "memflow_sma",
    "iamflow",
    "retrieval_frame_text_ablation",
    "retrieval_seg_uniform_ablation",
    "retrieval_seg_dinokey_ablation",
    "retrieval_seg_framererank_ablation",
)
DEFAULT_MOVIES = {
    "big_buck_bunny": BENCH_ROOT / "assets/trackA/BlenderOpenMovies/big_buck_bunny",
    "0022_Reservoir_Dogs": BENCH_ROOT / "assets/trackA/LSMDC/0022_Reservoir_Dogs",
}
MODE_SUFFIX = {"name_anchored": "", "description_provided": "__descprov"}
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")


@dataclass(slots=True)
class ScoreTask:
    system: str
    movie_key: str
    movie_dir: str
    video: str
    out_dir: str

    @property
    def label(self) -> str:
        return f"{self.system}/{self.movie_key}"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _split_csvish(value: str) -> list[str]:
    return [part for piece in str(value or "").replace(";", ",").split(",") for part in piece.split()]


def _read_dataset_dirs() -> dict[str, Path]:
    path = BENCH_ROOT / "assets/trackA/dataset_dirs.txt"
    out: dict[str, Path] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        dataset, _, root = line.partition(":")
        if dataset.strip() and root.strip():
            out[dataset.strip()] = Path(root.strip()).expanduser()
    return out


def resolve_video(movie_dir: Path) -> Path:
    for child in sorted(movie_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in VIDEO_EXTS:
            return child
    base = movie_dir.name
    for ext in VIDEO_EXTS:
        sibling = movie_dir.parent / f"{base}{ext}"
        if sibling.is_file():
            return sibling
    external_root = _read_dataset_dirs().get(movie_dir.parent.name)
    if external_root:
        for ext in VIDEO_EXTS:
            candidate = external_root / f"{base}{ext}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"could not resolve source video for {movie_dir}")


def _tracka_run_dir(movie_dir: Path, system: str) -> Path:
    return BENCH_ROOT / "outputs/evaluation/trackA" / system / movie_dir.parent.name / movie_dir.name


def _selection_exists(movie_dir: Path, system: str) -> bool:
    new_path = _tracka_run_dir(movie_dir, system) / "visual_selections" / f"{system}.json"
    legacy_path = movie_dir / "benchmark_run/visual_selections" / f"{system}.json"
    return new_path.is_file() or legacy_path.is_file()


def discover_tasks(*, systems: list[str], movies: list[str], modes: list[str]) -> tuple[list[ScoreTask], list[str]]:
    tasks: list[ScoreTask] = []
    skipped: list[str] = []
    for system_base in systems:
        for movie_key in movies:
            movie_dir = DEFAULT_MOVIES[movie_key]
            video = resolve_video(movie_dir)
            for mode in modes:
                system = f"{system_base}{MODE_SUFFIX[mode]}"
                if not _selection_exists(movie_dir, system):
                    skipped.append(f"{system}/{movie_key}: no visual_selections")
                    continue
                run_dir = _tracka_run_dir(movie_dir, system)
                tasks.append(
                    ScoreTask(
                        system=system,
                        movie_key=movie_key,
                        movie_dir=str(movie_dir),
                        video=str(video),
                        out_dir=str(run_dir / "_visual_score"),
                    )
                )
    return tasks, skipped


def _stage1_done_count(log_dir: Path, *, systems: list[str], movies: list[str], modes: list[str]) -> int:
    count = 0
    for system in systems:
        for movie in movies:
            for mode in modes:
                path = log_dir / f"{system}__{movie}__{mode}.log"
                if not path.is_file():
                    continue
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        if any(line.startswith("EXIT:") for line in handle):
                            count += 1
                except OSError:
                    continue
    return count


def wait_for_stage1(
    log_dir: Path,
    *,
    systems: list[str],
    movies: list[str],
    modes: list[str],
    expected: int,
    interval: float,
    timeout_min: float,
) -> None:
    deadline = time.monotonic() + timeout_min * 60 if timeout_min > 0 else None
    while True:
        done = _stage1_done_count(log_dir, systems=systems, movies=movies, modes=modes)
        print(f"[stage2-service] {_now()} stage1 sentinels={done}/{expected}", flush=True)
        if done >= expected:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for Stage-1 sentinels in {log_dir}")
        time.sleep(interval)


def _aggregate(log_dir: Path) -> int:
    out_dir = log_dir / "_agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SRC_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(BENCH_ROOT / "scripts/evaluate_baselines/trackA/aggregate_two_movie_run.py"), "--out", str(out_dir)],
        cwd=str(BENCH_ROOT),
        env=env,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--api-list", default=os.environ.get("JUDGE_APIS", ""))
    parser.add_argument("--fleet", action="store_true", default=os.environ.get("STAGE2_FLEET", "0") == "1")
    parser.add_argument("--fleet-root", type=Path, default=None)
    parser.add_argument("--fleet-role", default=os.environ.get("STAGE2_FLEET_ROLE", "reviewer"))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", DEFAULT_FFMPEG))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STAGE2_WORKERS", "0") or 0))
    parser.add_argument(
        "--endpoint-slots",
        type=int,
        default=int(os.environ.get("STAGE2_ENDPOINT_SLOTS", "1") or 1),
        help="concurrent judge requests per endpoint (default 1). Raise ONLY when the "
             "vLLM replicas run a matching MAX_NUM_SEQS; scheduling-only, does not "
             "change prompts, sampling, or metrics",
    )
    parser.add_argument("--score-gpu", default=os.environ.get("SCORE_GPU", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--movies", default=",".join(DEFAULT_MOVIES))
    parser.add_argument("--modes", default="name_anchored")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--wait-stage1-log-dir", type=Path, default=None)
    parser.add_argument("--wait-stage1-expected", type=int, default=0)
    parser.add_argument("--wait-stage1-interval", type=float, default=60.0)
    parser.add_argument("--wait-stage1-timeout-min", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.score_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.score_gpu)
    os.environ.setdefault("PUBLIC_MODELS_ROOT", DEFAULT_PUBLIC_MODELS_ROOT)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = (args.log_dir or (BENCH_ROOT / "_tgpu_run" / f"stage2_service_{stamp}")).resolve()
    progress_path = (args.progress or (log_dir / "progress.json")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    systems = _split_csvish(args.systems)
    movies = _split_csvish(args.movies)
    modes = _split_csvish(args.modes)
    unknown_movies = sorted(set(movies) - set(DEFAULT_MOVIES))
    unknown_modes = sorted(set(modes) - set(MODE_SUFFIX))
    if unknown_movies:
        raise ValueError(f"unknown movies: {unknown_movies}; choose from {sorted(DEFAULT_MOVIES)}")
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}; choose from {sorted(MODE_SUFFIX)}")

    if args.wait_stage1_log_dir:
        wait_for_stage1(
            args.wait_stage1_log_dir.resolve(),
            systems=systems,
            movies=movies,
            modes=modes,
            expected=args.wait_stage1_expected or (len(systems) * len(movies) * len(modes)),
            interval=args.wait_stage1_interval,
            timeout_min=args.wait_stage1_timeout_min,
        )

    tasks, skipped = discover_tasks(systems=systems, movies=movies, modes=modes)
    judge_api = build_judge_api(
        api=args.api,
        api_list=args.api_list,
        use_fleet=args.fleet,
        fleet_root=args.fleet_root,
        fleet_role=args.fleet_role,
        model=args.model,
        endpoint_slots=args.endpoint_slots,
    )
    workers = args.workers if args.workers > 0 else int(getattr(judge_api, "size", 1) or 1)
    endpoint_urls = list(getattr(judge_api, "base_urls", [str(judge_api)]))

    progress: dict[str, Any] = {
        "status": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "log_dir": str(log_dir),
        "progress_path": str(progress_path),
        "model": args.model,
        "workers": workers,
        "endpoint_slots": args.endpoint_slots,
        "endpoint_urls": endpoint_urls,
        "total": len(tasks),
        "done": 0,
        "failed": 0,
        "skipped": skipped,
        "tasks": [asdict(task) | {"status": "pending"} for task in tasks],
    }
    _write_json_atomic(progress_path, progress)
    print(f"[stage2-service] tasks={len(tasks)} skipped={len(skipped)} workers={workers}", flush=True)
    print(f"[stage2-service] progress={progress_path}", flush=True)

    failures = 0
    for idx, task in enumerate(tasks):
        progress["tasks"][idx]["status"] = "running"
        progress["tasks"][idx]["started_at"] = _now()
        progress["updated_at"] = _now()
        _write_json_atomic(progress_path, progress)
        print(f"[stage2-service] START {idx + 1}/{len(tasks)} {task.label}", flush=True)
        try:
            summary = run_visual_coverage(
                Path(task.movie_dir),
                task.system,
                Path(task.video),
                Path(task.out_dir),
                judge_api,
                args.model,
                args.ffmpeg,
                args.limit,
                workers=workers,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue unless fail-fast
            failures += 1
            progress["failed"] = failures
            progress["tasks"][idx]["status"] = "failed"
            progress["tasks"][idx]["error"] = repr(exc)
            progress["tasks"][idx]["finished_at"] = _now()
            progress["updated_at"] = _now()
            _write_json_atomic(progress_path, progress)
            print(f"[stage2-service] FAILED {task.label}: {exc!r}", flush=True)
            if args.fail_fast:
                break
            continue
        progress["done"] = int(progress.get("done", 0)) + 1
        progress["tasks"][idx]["status"] = "done"
        progress["tasks"][idx]["finished_at"] = _now()
        progress["tasks"][idx]["summary"] = summary
        progress["updated_at"] = _now()
        _write_json_atomic(progress_path, progress)
        print(f"[stage2-service] DONE {task.label}", flush=True)

    if args.aggregate:
        rc = _aggregate(log_dir)
        progress["aggregate_return_code"] = rc
        progress["aggregate_out"] = str(log_dir / "_agg")
        if rc != 0:
            failures += 1

    progress["status"] = "failed" if failures else "done"
    progress["finished_at"] = _now()
    progress["updated_at"] = _now()
    _write_json_atomic(progress_path, progress)
    print(f"[stage2-service] {progress['status'].upper()} progress={progress_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
