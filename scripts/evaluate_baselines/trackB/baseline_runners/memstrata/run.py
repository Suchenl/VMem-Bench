#!/usr/bin/env python3
"""Run MemStrata on VMem-Bench Track B prompt streams.

This runner does not read TrackB GT. It converts ``sut_prompts`` into a
prompt-only production-screenplay wrapper and invokes ``memstrata.production.run``
in bench-mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    add_common_args,
    command_env,
    copy_prompt_stream,
    find_memstrata_src,
    iter_requested_streams,
    output_dir_for,
    prepare_output_dir,
    prompt_stream_to_minimal_screenplay,
    run_command,
    write_json,
    write_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_system="memstrata")
    parser.add_argument("--backend", default="wan22_i2v_a14b_lightx2v_4step")
    parser.add_argument("--decompose", choices=["crop_server", "none"], default="crop_server")
    parser.add_argument("--crop-acq-device", default="")
    parser.add_argument("--mllm-gpu", default="0")
    parser.add_argument("--mllm-port", default="8000")
    parser.add_argument("--no-autoserve", action="store_true")
    parser.add_argument("--stop-services", action="store_true")
    parser.add_argument("--no-flux", action="store_true")
    parser.add_argument("--force-recompose", action="store_true", default=True)
    parser.add_argument("--no-force-recompose", dest="force_recompose", action="store_false")
    parser.add_argument("--discovery", action="store_true", default=True)
    parser.add_argument("--no-discovery", dest="discovery", action="store_false")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--angle-classifier", default="")
    # "mllm" lets MemStrata's own VLM bind each shot's prompt names to entities it confirms in the
    # generated frames. Without it a first appearance is banked under a synthetic label the
    # name-authoritative read path can never resolve, so the identity hard cases measure noise.
    parser.add_argument("--write-naming", choices=["perception", "mllm"], default="perception")
    parser.add_argument("--segments", type=int, default=0,
                        help="limit shots (0 = whole story); for probes, not for reported results")
    args = parser.parse_args(argv)

    method_src = find_memstrata_src()
    method_root = method_src.parent

    rc_all = 0
    for stream in iter_requested_streams(args):
        out_dir = output_dir_for(args.system, stream, args.output_root, args.run_tag)
        prepare_output_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
        copy_prompt_stream(stream, out_dir)
        screenplay_path = write_json(
            out_dir / "input" / "prompt_only_screenplay.json",
            prompt_stream_to_minimal_screenplay(stream),
        )
        cmd = [
            str(args.python),
            "-u",
            "-m",
            "memstrata.production.run",
            "--screenplay",
            str(screenplay_path),
            "--backend",
            args.backend,
            "--system",
            args.system,
            "--run-dir",
            str(out_dir),
            "--decompose",
            args.decompose,
            "--mllm-gpu",
            str(args.mllm_gpu),
            "--mllm-port",
            str(args.mllm_port),
            "--embedder",
            args.embedder,
        ]
        if args.crop_acq_device:
            cmd += ["--crop-acq-device", args.crop_acq_device]
        if args.no_autoserve:
            cmd.append("--no-autoserve")
        if args.stop_services:
            cmd.append("--stop-services")
        if args.no_flux:
            cmd.append("--no-flux")
        # Resuming is the cheap path for an interrupted story: MemStrata's memory is external, so the
        # producer reopens the persisted bank and only generates the shots that are still missing.
        if args.resume:
            cmd.append("--resume")
        if args.force_recompose:
            cmd.append("--force-recompose")
        if args.discovery:
            cmd.append("--discovery")
        if args.angle_classifier:
            cmd += ["--angle-classifier", args.angle_classifier]
        cmd += ["--write-naming", args.write_naming]
        if args.segments:
            cmd += ["--segments", str(args.segments)]

        env = command_env({"PYTHONPATH": f"{method_src}:{method_root}:{os.environ.get('PYTHONPATH', '')}"})
        rc = run_command(
            cmd,
            cwd=method_root,
            log_path=out_dir / "logs" / "memstrata_run.log",
            env=env,
            dry_run=args.dry_run,
        )
        long_video = out_dir / "review" / "long_video.mp4"
        manifest = out_dir / "run_manifest.json"
        write_manifest(
            out_dir,
            system=args.system,
            stream=stream,
            command=cmd,
            status="dry_run" if args.dry_run else ("done" if rc == 0 else "failed"),
            exit_code=rc,
            artifacts={
                "long_video": str(long_video) if long_video.is_file() else None,
                "segments_dir": str(out_dir / "review" / "segments"),
                "memstrata_run_manifest": str(manifest) if manifest.is_file() else None,
                "prompt_only_screenplay": str(screenplay_path),
            },
            notes=[
                "Input is TrackB sut_prompts only; no TrackB gt JSON is read by this runner.",
                "The prompt-only wrapper leaves main_entities empty; memory must come from generated pixels/discovery.",
            ],
        )
        rc_all = max(rc_all, rc)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
