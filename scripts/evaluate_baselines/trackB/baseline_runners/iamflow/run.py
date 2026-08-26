#!/usr/bin/env python3
"""Run IAMFlow on Track B prompt streams."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_system="iamflow")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--pretrained-root", default="${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/IAMFlow")
    parser.add_argument("--wan-model-path", default="${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--llm-model-path", default="${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vlm-model-path", default="${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dit-quantized-ckpt", default="")
    parser.add_argument("--max-memory-frames", type=int, default=3)
    parser.add_argument("--frames-per-segment", type=int, default=39)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--use-tinyvae", action="store_true")
    args = parser.parse_args(argv)

    repo = REPO_ROOT / "baselines" / "Causal" / "IAMFlow"
    if not repo.is_dir():
        raise SystemExit(f"missing IAMFlow checkout: {repo}")
    config = args.config_path or (repo / "configs" / "iamflow.yaml")
    if args.dit_quantized_ckpt:
        dit_ckpt = args.dit_quantized_ckpt
    else:
        pretrained_root = Path(args.pretrained_root)
        root_ckpt = pretrained_root / "iamflow_fp8.safetensors"
        nested_ckpt = pretrained_root / "iamflow_models" / "iamflow_fp8.safetensors"
        dit_ckpt = str(root_ckpt if root_ckpt.is_file() else nested_ckpt)

    rc_all = 0
    for sample_idx, stream in enumerate(iter_requested_streams(args)):
        out_dir = output_dir_for(args.system, stream, args.output_root, args.run_tag)
        prepare_output_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
        copy_prompt_stream(stream, out_dir)
        prompts_path = write_prompt_jsonl(out_dir / "input" / "prompts.jsonl", stream.prompts)
        n = len(stream.segments)
        num_output_frames = int(args.frames_per_segment) * n
        switch_indices = ", ".join(str(int(args.frames_per_segment) * (i + 1)) for i in range(n - 1))
        yaml_switch_indices = switch_indices if switch_indices else '""'
        config_path = out_dir / "input" / "iamflow.trackb.yaml"
        config_path.write_text(
            patch_simple_yaml(
                config,
                {
                    "data_path": str(prompts_path),
                    "output_folder": str(out_dir / "review"),
                    "num_output_frames": str(num_output_frames),
                    "switch_frame_indices": yaml_switch_indices,
                },
            ),
            encoding="utf-8",
        )
        write_json(
            out_dir / "input" / "trackb_generation_params.json",
            {
                "frames_per_segment": args.frames_per_segment,
                "num_output_frames": num_output_frames,
                "switch_frame_indices": switch_indices,
            },
        )
        save_dir = out_dir / "agent_frames"
        mapping_path = out_dir / "review" / "mapping.json"
        cmd = [
            str(args.python),
            "-u",
            "-m",
            "iamflow.run_iamflow",
            "--config_path",
            str(config_path),
            "--data_path",
            str(prompts_path),
            "--output_folder",
            str(out_dir / "review"),
            "--dit_quantized_ckpt",
            str(dit_ckpt),
            "--llm_model_path",
            str(args.llm_model_path),
            "--vlm_model_path",
            str(args.vlm_model_path),
            "--max_memory_frames",
            str(args.max_memory_frames),
            "--save_dir",
            str(save_dir),
            "--mapping_path",
            str(mapping_path),
            "--sample_index_offset",
            str(args.sample_index_offset + sample_idx),
        ]
        if args.use_tinyvae:
            cmd.append("--use_tinyvae")
        env = command_env(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "WAN_MODEL_PATH": args.wan_model_path,
                "PRETRAINED_ROOT": args.pretrained_root,
                "PYTHONPATH": f"{repo}:{repo / 'iamflow'}",
                "MASTER_ADDR": "127.0.0.1",
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_LOOPBACK_IP": "127.0.0.1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
        )
        rc = run_command(
            cmd,
            cwd=repo,
            log_path=out_dir / "logs" / "iamflow_run.log",
            env=env,
            dry_run=args.dry_run,
        )
        videos = sorted(str(p) for p in (out_dir / "review").glob("*.mp4"))
        write_json(out_dir / "input" / "iamflow_params.json", {"dit_quantized_ckpt": dit_ckpt})
        write_manifest(
            out_dir,
            system=args.system,
            stream=stream,
            command=cmd,
            status="dry_run" if args.dry_run else ("done" if rc == 0 else "failed"),
            exit_code=rc,
            artifacts={
                "long_videos": videos,
                "mapping": str(mapping_path) if mapping_path.is_file() else None,
                "prompts_jsonl": str(prompts_path),
                "config": str(config_path),
                "agent_frames": str(save_dir),
            },
            notes=["Uses IAMFlow native prompt-switch generation; stage-2 scoring consumes review/*.mp4."],
        )
        rc_all = merge_rc(rc_all, rc)
    return shell_rc(rc_all)


if __name__ == "__main__":
    raise SystemExit(main())
