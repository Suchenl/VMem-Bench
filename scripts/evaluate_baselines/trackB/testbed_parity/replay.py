#!/usr/bin/env python3
"""Replay the testbed's reference selection on a frozen paper run's own clips (teacher forcing).

A fresh rollout cannot reproduce retrieved ids past the first clip because Wan2.2 is not
bit-deterministic (identical keyframe/prompt/seed give ~22 dB PSNR between runs). This removes Wan from
the comparison: for each shot k the testbed's ``ReferencePolicy`` (the runner's selection path) reads
a store fed with the frozen run's clips 0..k-1, and its reference ids are compared with the ids the
frozen run handed to FLUX at shot k (``generator_logs/segment_<k>/media_generation_task.json``).

    replay.py --variant frame_text --frozen <run dir> [-n 10] [--story ...] [--work <dir>] [--out <json>]
Encoders as in the paper: Qwen3-Embedding-4B (text), SigLIP2 (frames), DINOv3 (keyframes).

Frozen producer defect: when a frozen retrieval run was resumed (``[prod] RESUME: N/...`` in
``logs/memstrata_run.log``) its retrieval store restarted empty at shot N (clock 0; earlier shots were
never re-observed). ``--emulate-resumes`` reproduces that reset so the remaining ids can still be
compared; the testbed itself re-observes closed shots on ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from vmem_bench.testbed.runner import (
    ReferencePolicy,
    RunConfig,
    load_prompt_stream,
    story_prompt_path,
    usable_references,
)
from vmem_bench.testbed.retrieval.retrievers import RetrieverConfig
from vmem_bench.testbed.store import TrackBGeneratedStore


def _media(run: Path, video: str) -> str:
    p = Path(video)
    return str(p) if p.is_file() else str(run / video[video.index("/media/") + 1:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--story", default="0001_lighthouse_keeper")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--emulate-resumes", action="store_true")
    args = ap.parse_args()

    stream = load_prompt_stream(story_prompt_path(args.story))
    prog = {int(s["segment_id"]): s for s in json.loads((args.frozen / "progress.json").read_text())["segments"]}
    log = args.frozen / "logs" / "memstrata_run.log"
    resumes = sorted({int(m) for m in re.findall(r"\[prod\] RESUME: (\d+)/", log.read_text(errors="replace"))}) \
        if log.is_file() else []
    work = args.work or Path(tempfile.mkdtemp(prefix=f"replay_{args.variant}_"))
    cfg = RunConfig(variant=args.variant, budget=3)

    def fresh(tag: str):
        store = TrackBGeneratedStore(RetrieverConfig(text_provider=cfg.text_provider, frame_provider=cfg.frame_provider,
                                                     keyframe_provider=cfg.keyframe_provider), work_dir=work / tag)
        return store, ReferencePolicy(cfg, store, plan=None, run_dir=work)

    store, policy = fresh("start")
    rows = []
    for seg in stream.segments[: args.n]:
        if args.emulate_resumes and seg.index in resumes:
            store, policy = fresh(f"resume_{seg.index}")
        task = args.frozen / "generator_logs" / f"segment_{seg.index:03d}" / "media_generation_task.json"
        ctl = json.loads(task.read_text()).get("controls", {})
        frozen_ids = [str(c.get("asset_id", "")).removeprefix("retr:") for c in ctl.get("source_crops") or []]
        ids = [r.representation_id for r in usable_references(policy.select(seg))]
        rows.append({"shot": seg.index, "testbed": ids, "frozen": frozen_ids, "equal": ids == frozen_ids})
        print(seg.index, "equal" if ids == frozen_ids else "DIFF", ids, frozen_ids, flush=True)
        store.observe_generated_segment(segment_id=seg.index, prompt_text=seg.prompt,
                                        segment_video=_media(args.frozen, prog[seg.index]["video"]),
                                        description=seg.prompt)
    res = {"variant": args.variant, "frozen": str(args.frozen), "frozen_resumes": resumes,
           "emulate_resumes": args.emulate_resumes, "shots": rows,
           "equal_shots": sum(r["equal"] for r in rows), "n": len(rows)}
    print(json.dumps({k: res[k] for k in ("variant", "frozen_resumes", "equal_shots", "n")}))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
