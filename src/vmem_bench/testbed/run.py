"""CLI of the Track-B shared-generator testbed: ``python -m vmem_bench.testbed.run``.

Example (paper text-frame row, 3 seeds):

    python -m vmem_bench.testbed.run --variant frame_text --budget 3 --backend flux2_wan22 \\
        --seed 2026 2027 2028 --run-tag trackb_testbed --out-root <dir> --stories all

Seed 2026 reproduces the seeds of the published Track-B runs; the standard is 3 seeds per condition.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
from pathlib import Path

from vmem_bench.testbed.backends import BACKENDS, load_backend
from vmem_bench.testbed.runner import (
    PAPER_SEED,
    PROMPTS_DIR,
    VARIANTS,
    RunConfig,
    all_stories,
    run,
)


def _parse_opts(items: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--backend-opt expects key=value, got {item!r}")
        try:
            opts[key] = json.loads(value)
        except ValueError:
            opts[key] = value
    return opts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=[*VARIANTS, "oracle"],
                    help="memory condition; 'oracle' is the deprecated name of last_seen")
    ap.add_argument("--budget", type=int, default=3, help="max reference stills per segment (none forces 0)")
    ap.add_argument("--backend", default="flux2_wan22", choices=sorted(BACKENDS))
    ap.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="backend constructor option (JSON value or string); repeatable")
    ap.add_argument("--seed", type=int, nargs="+", default=[PAPER_SEED],
                    help=f"run seed(s); {PAPER_SEED} = the published runs; standard is 3 seeds")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--stories", nargs="+", required=True, help="story ids (e.g. 0001_lighthouse_keeper) or 'all'")
    ap.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    ap.add_argument("--segments", type=int, default=0, help="first N segments only (probes; 0 = whole story)")
    ap.add_argument("--text-provider", default="qwen3_embedding")
    ap.add_argument("--frame-provider", default="siglip2")
    ap.add_argument("--keyframe-provider", default="dinov3")
    ap.add_argument("--resume", action="store_true", help="continue after the closed prefix of progress.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-services", action="store_true", help="leave backend servers running at exit")
    args = ap.parse_args(argv)

    stories = all_stories(args.prompts_dir) if args.stories == ["all"] else list(args.stories)
    service_dir = args.out_root / "_services" / f"{args.backend}_{socket.gethostname()}_{os.getpid()}"
    backend = load_backend(args.backend, _parse_opts(args.backend_opt), defaults={"service_dir": service_dir})
    base = RunConfig(variant=args.variant, budget=args.budget, backend=args.backend, run_tag=args.run_tag,
                     out_root=args.out_root, segments=args.segments, resume=args.resume, overwrite=args.overwrite,
                     text_provider=args.text_provider, frame_provider=args.frame_provider,
                     keyframe_provider=args.keyframe_provider)
    backend.start()
    try:
        for seed in args.seed:
            cfg = dataclasses.replace(base, seed=seed)
            for manifest in run(cfg, stories, backend, prompts_dir=args.prompts_dir, keep_services=True):
                print(json.dumps({k: manifest[k] for k in ("story_id", "variant", "seed", "segments_ok",
                                                           "segments_failed", "run_dir")}, ensure_ascii=False))
    finally:
        if not args.keep_services:
            backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
