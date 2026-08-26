#!/usr/bin/env python3
"""Execute a console-created annotation job locally on a BDY node.

The console writes a JSON spec to shared storage, then submits this runner
through ``scripts/tgpu_fs.py``. The runner keeps video decoding and VLM HTTP
calls inside the BDY node by using local ``127.0.0.1`` VLM endpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import urllib.request
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _local_vlm_urls() -> str:
    urls: list[str] = []
    for port in range(8110, 8118):
        url = f"http://127.0.0.1:{port}/v1"
        request = urllib.request.Request(f"{url}/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                if 200 <= int(response.status) < 300:
                    urls.append(url)
        except Exception:  # noqa: BLE001 - unavailable ranks are excluded locally
            continue
    if not urls:
        raise RuntimeError("no local BDY VLM endpoint passed /v1/models")
    return ",".join(urls)


def _run_s3_probe(*, spec: dict, local_urls: str, runner_log: Path, status_path: Path) -> int:
    """Run one short, node-local S3 video request for executor validation."""
    from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (
        QwenVideoReviewer,
    )

    catalog = Path(spec["catalog_path"])
    row = json.loads(next(line for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()))
    source = Path(row["source_video"])
    probe_dir = Path(spec["job_dir"]) / "bdy_s3_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    clip = probe_dir / "segment_0_10s.mp4"
    extract = subprocess.run(
        ["ffmpeg", "-y", "-ss", "0", "-t", "10", "-i", str(source), "-c:v", "libx264", "-an", str(clip)],
        capture_output=True,
        text=True,
        check=False,
    )
    if extract.returncode != 0:
        _write_text(runner_log, extract.stdout + extract.stderr)
        _write_text(status_path, f"{extract.returncode}\n")
        return int(extract.returncode)
    reviewer = QwenVideoReviewer(
        base_url=local_urls.split(",")[0],
        model=str((spec.get("options") or {}).get("reviewer_model") or "qwen3-vl-8b"),
        max_tokens=256,
        timeout_seconds=180,
        fps=2.0,
    )
    review = reviewer.review(
        clip=clip,
        segment={
            "segment_id": "bdy_probe_segment_0",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "present_entity_ids": [],
            "_seed_present_entity_ids": [],
            "action": "Review the visible opening scene.",
        },
        roster=[],
    )
    result_path = Path(spec["job_dir"]) / "bdy_s3_probe_result.json"
    _write_text(result_path, json.dumps(asdict(review), ensure_ascii=False, indent=2) + "\n")
    _write_text(runner_log, json.dumps({"clip": str(clip), "review": asdict(review)}, ensure_ascii=False, indent=2) + "\n")
    _write_text(status_path, "0\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    job_dir = Path(spec["job_dir"])
    catalog = Path(spec["catalog_path"])
    output = Path(spec["out_path"])
    status_path = Path(spec["status_path"])
    runner_log = job_dir / "bdy_runner.log"
    options = dict(spec.get("options") or {})

    rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(row.get("source_video") or "") for row in rows if not Path(str(row.get("source_video") or "")).is_file()]
    if missing:
        message = "BDY source video missing: " + ", ".join(missing)
        _write_text(runner_log, message + "\n")
        _write_text(status_path, "1\n")
        return 1

    repo = Path(spec["repo"])
    python = str(spec["python"])
    local_urls = _local_vlm_urls()
    if options.get("bdy_mode") == "s3_probe":
        return _run_s3_probe(
            spec=spec,
            local_urls=local_urls,
            runner_log=runner_log,
            status_path=status_path,
        )
    command = [
        python,
        "-m",
        "vmem_bench.annotation.pipeline.batch",
        "--catalog",
        str(catalog),
        "--out",
        str(output),
        "--reviewer",
        "qwen",
        "--reviewer-base-url",
        local_urls,
        "--reviewer-model",
        str(options.get("reviewer_model") or "qwen3-vl-8b"),
        "--grounder",
        str(options.get("grounder") or "full-frame"),
        "--grounder-base-url",
        local_urls if str(options.get("grounder") or "") == "qwen" else "",
        "--grounder-model",
        str(options.get("grounder_model") or "qwen3-vl-8b"),
        "--s4-mode",
        str(options.get("s4_mode") or "blocking"),
        "--crop-route",
        str(options.get("crop_route") or "propose_and_pick"),
        "--proposer",
        str(options.get("proposer") or "fusion"),
        "--task-mode",
        str(options.get("task_mode") or "coverage"),
    ]
    if options.get("continue_from"):
        command.extend(["--continue-from", str(options["continue_from"])])
    if options.get("resume") and not options.get("continue_from"):
        command.append("--resume")
    if options.get("skip_human"):
        command.append("--skip-human")
    if options.get("max_tasks") not in (None, ""):
        command.extend(["--max-tasks", str(int(options["max_tasks"]))])
    if options.get("max_review_rounds") not in (None, ""):
        command.extend(["--max-review-rounds", str(int(options["max_review_rounds"]))])

    env = dict(os.environ)
    src = str(repo / "benchmarks" / "MemStrata" / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    runner_log.parent.mkdir(parents=True, exist_ok=True)
    with runner_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(repo),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return_code = process.wait()
    _write_text(status_path, f"{return_code}\n")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
