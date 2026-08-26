#!/usr/bin/env python3
"""Run MemFlow interactive generation on Track B prompt streams."""

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
    merge_rc,
    output_dir_for,
    patch_simple_yaml,
    prepare_output_dir,
    run_command,
    shell_rc,
    write_json,
    write_manifest,
    write_prompt_jsonl,
)


def _command(args: argparse.Namespace, config_path: Path, repo: Path) -> list[str]:
    script = repo / "interactive_inference.py"
    if int(args.nproc_per_node) > 1:
        return [
            str(args.python),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "--master_port",
            str(args.master_port),
            str(script),
            "--config_path",
            str(config_path),
        ]
    return [str(args.python), str(script), "--config_path", str(config_path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_system="memflow")
    parser.set_defaults(python="python3")
    parser.add_argument("--frames-per-segment", type=int, default=39)
    parser.add_argument("--config-template", type=Path, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", default="29501")
    parser.add_argument("--cuda-visible-devices", default="")
    args = parser.parse_args(argv)

    repo = REPO_ROOT / "baselines" / "Causal" / "MemFlow"
    if not repo.is_dir():
        raise SystemExit(f"missing MemFlow checkout: {repo}")
    template = args.config_template or (repo / "configs" / "interactive_inference.yaml")

    rc_all = 0
    for stream in iter_requested_streams(args):
        out_dir = output_dir_for(args.system, stream, args.output_root, args.run_tag)
        prepare_output_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
        copy_prompt_stream(stream, out_dir)
        prompts_path = write_prompt_jsonl(out_dir / "input" / "prompts.jsonl", stream.prompts)
        n = len(stream.segments)
        num_output_frames = int(args.frames_per_segment) * n
        switch_indices = ", ".join(str(int(args.frames_per_segment) * (i + 1)) for i in range(n - 1))
        yaml_switch_indices = switch_indices if switch_indices else '""'
        config_text = patch_simple_yaml(
            template,
            {
                "data_path": str(prompts_path),
                "output_folder": str(out_dir / "review"),
                "num_output_frames": str(num_output_frames),
                "switch_frame_indices": yaml_switch_indices,
                "save_with_index": "true",
                "model_kwargs.SMA": "False",
            },
        )
        config_path = out_dir / "input" / "interactive_inference.trackb.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        write_json(
            out_dir / "input" / "trackb_generation_params.json",
            {
                "frames_per_segment": args.frames_per_segment,
                "num_output_frames": num_output_frames,
                "switch_frame_indices": switch_indices,
            },
        )

        cmd = _command(args, config_path, repo)
        env_extra = {}
        if args.cuda_visible_devices:
            env_extra["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        rc = run_command(
            cmd,
            cwd=repo,
            log_path=out_dir / "logs" / "memflow_run.log",
            env=command_env(env_extra),
            dry_run=args.dry_run,
        )
        videos = sorted(str(p) for p in (out_dir / "review").glob("*.mp4"))
        write_manifest(
            out_dir,
            system=args.system,
            stream=stream,
            command=cmd,
            status="dry_run" if args.dry_run else ("done" if rc == 0 else "failed"),
            exit_code=rc,
            artifacts={"long_videos": videos, "config": str(config_path), "prompts_jsonl": str(prompts_path)},
            notes=[
                "Uses MemFlow interactive_inference.py with one JSONL sample containing all TrackB segment prompts.",
                "Stage-2 scorer should read the generated long mp4 from review/.",
            ],
        )
        rc_all = merge_rc(rc_all, rc)
    return shell_rc(rc_all)


if __name__ == "__main__":
    raise SystemExit(main())
