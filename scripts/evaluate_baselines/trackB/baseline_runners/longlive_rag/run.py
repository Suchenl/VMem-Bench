#!/usr/bin/env python3
"""Run LongLive-RAG interactive generation on Track B prompt streams."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    REPO_ROOT,
    acquire_node_slot,
    add_common_args,
    command_env,
    copy_prompt_stream,
    iter_requested_streams,
    merge_rc,
    output_dir_for,
    prepare_output_dir,
    run_command,
    shell_rc,
    write_manifest,
    write_prompt_list,
)

# LongLive-RAG's DynamicSwap text encoder pins ~69GB CPU RAM and spikes another
# ~69GB when it moves to GPU. Several LongLive jobs loading on one node at once
# blow past physical RAM and the kernel OOM-killer SIGKILLs them together (-9).
# Cap concurrent LongLive jobs PER NODE with a /dev/shm semaphore. Defaults are
# safe for a crowded shared node; the sweep can widen via env once memflow/iamflow
# have drained. wait<=0 means "fast-defer" (don't hold a GPU slot idle waiting).
LONGLIVE_NODE_SLOTS = int(os.environ.get("TRACKB_LONGLIVE_NODE_SLOTS", "1"))
LONGLIVE_SLOT_WAIT = float(os.environ.get("TRACKB_LONGLIVE_SLOT_WAIT", "15"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_system="longlive_rag")
    parser.set_defaults(python="python3")
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

        # Cap concurrent LongLive jobs on this physical node (RAM safety). If no
        # slot frees within the wait window, defer instead of risking a node-wide
        # OOM: record a non-zero manifest so --resume re-runs it later (e.g. in a
        # dedicated LongLive sweep once the node is quieter).
        gate = None if args.dry_run else acquire_node_slot(
            "mave_trackb_longlive", slots=LONGLIVE_NODE_SLOTS, wait_sec=LONGLIVE_SLOT_WAIT
        )
        if not args.dry_run and gate is None:
            write_manifest(
                out_dir,
                system=args.system,
                stream=stream,
                command=cmd,
                status="deferred",
                exit_code=75,
                artifacts={"long_video": None, "prompts_json": str(prompts_path), "config": str(config)},
                notes=[f"Deferred: node LongLive concurrency cap ({LONGLIVE_NODE_SLOTS}/node) busy; "
                       "re-run via --resume when the node is quieter."],
            )
            rc_all = merge_rc(rc_all, 75)
            continue
        try:
            rc = run_command(
                cmd,
                cwd=repo,
                log_path=out_dir / "logs" / "longlive_rag_run.log",
                env=command_env(env_extra),
                dry_run=args.dry_run,
            )
        finally:
            if gate is not None:
                gate.close()
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
        rc_all = merge_rc(rc_all, rc)
    return shell_rc(rc_all)


if __name__ == "__main__":
    raise SystemExit(main())
