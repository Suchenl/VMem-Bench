#!/usr/bin/env python3
"""Run LongLive-RAG interactive generation on Track B prompt streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    REPO_ROOT,
    add_common_args,
    command_env,
    copy_prompt_stream,
    iter_requested_streams,
    output_dir_for,
    prepare_output_dir,
    run_command,
    write_manifest,
    write_prompt_list,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_system="longlive_rag")
    parser.set_defaults(python="${CONDA_ENVS_ROOT}/wan2_1/bin/python")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--frames-per-segment", type=int, default=39)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--cuda-visible-devices", default="")
    args = parser.parse_args(argv)

    repo = REPO_ROOT / "baselines" / "Causal" / "LongLive-RAG"
    if not repo.is_dir():
        raise SystemExit(f"missing LongLive-RAG checkout: {repo}")
    config = args.config_path or (repo / "configs" / "longlive_latentmem.yaml")

    rc_all = 0
    for stream in iter_requested_streams(args):
        out_dir = output_dir_for(args.system, stream, args.output_root, args.run_tag)
        prepare_output_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
        copy_prompt_stream(stream, out_dir)
        prompts_path = write_prompt_list(out_dir / "input" / "prompts.json", stream.prompts)
        output_path = out_dir / "review" / "long_video.mp4"
        cmd = [
            str(args.python),
            "-u",
            str(repo / "interactive_inference.py"),
            "--config_path",
            str(config),
            "--prompts_file",
            str(prompts_path),
            "--frames_per_segment",
            str(args.frames_per_segment),
            "--output_path",
            str(output_path),
            "--fps",
            str(args.fps),
        ]
        env_extra = {}
        if args.cuda_visible_devices:
            env_extra["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        rc = run_command(
            cmd,
            cwd=repo,
            log_path=out_dir / "logs" / "longlive_rag_run.log",
            env=command_env(env_extra),
            dry_run=args.dry_run,
        )
        write_manifest(
            out_dir,
            system=args.system,
            stream=stream,
            command=cmd,
            status="dry_run" if args.dry_run else ("done" if rc == 0 else "failed"),
            exit_code=rc,
            artifacts={
                "long_video": str(output_path) if output_path.is_file() else None,
                "prompts_json": str(prompts_path),
                "config": str(config),
                "memory_log": str(output_path).replace(".mp4", "_memory_log.json"),
            },
            notes=["Uses LongLive-RAG interactive_inference.py; one long mp4 per TrackB story."],
        )
        rc_all = max(rc_all, rc)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
