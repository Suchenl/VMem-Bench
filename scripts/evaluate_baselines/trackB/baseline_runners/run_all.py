#!/usr/bin/env python3
"""Convenience launcher for Track B baseline runners.

This is intentionally thin: each system keeps its own runner and environment
knobs. For real GPU runs, launch this from a tmux/tgpu session.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEMS = ["memstrata", "memflow", "memflow_sma", "longlive_rag", "iamflow"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS), help="Comma-separated systems")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--run-tag", default="bench")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--slotmem-ref-image-path", type=Path, default=None)
    args, passthrough = parser.parse_known_args(argv)

    rc_all = 0
    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        runner = ROOT / system / "run.py"
        if not runner.is_file():
            print(f"[trackB-run-all] skip unknown system={system}: {runner}", file=sys.stderr)
            rc_all = max(rc_all, 2)
            continue
        cmd = [sys.executable, str(runner), "--run-tag", args.run_tag]
        if args.prompts:
            cmd += ["--prompts", str(args.prompts)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.resume:
            cmd.append("--resume")
        if system == "slotmem":
            if not args.slotmem_ref_image_path:
                print("[trackB-run-all] slotmem requires --slotmem-ref-image-path; skipping", file=sys.stderr)
                rc_all = max(rc_all, 2)
                continue
            cmd += ["--ref-image-path", str(args.slotmem_ref_image_path)]
        cmd += passthrough
        print("[trackB-run-all]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        rc_all = max(rc_all, int(rc))
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
