#!/usr/bin/env python3
"""Track B Stage-1 multi-node fan-out (English pool).

Optional cluster helper — **not** the public quickstart. Requires a launcher
binary on PATH (``MEMSTRATA_TGPU``, default name ``tgpu``) with ``-c`` / ``-node``.
The README path is a single-node ``causal/runner.py`` / Track B baseline_runner.

Generates the three baseline long-video systems for the VMem-Bench Track B
English prompt streams across several remote a800 nodes, one story per GPU, with a
shared job queue so every GPU stays saturated until all jobs finish.

Systems / step counts (denoising_step_list length):
  * memflow      @ 5 steps
  * longlive_rag @ 10 steps
  * iamflow      @ 5 steps (FULL VLM: per-node vLLM services for Qwen3-4B + Qwen3-VL-2B)

Scheduling policy (matches the "finish whole stories, shortest first" request):
  stories are ordered by ascending segment count; a story's three system jobs are
  kept together in the queue and dispatched in order, so at any deadline the set of
  *fully* completed stories is a shortest-first prefix that has all three systems.

GPU layout per node (8x a800):
  gpu0..5 -> 6 DiT job slots ; gpu6 -> Qwen3-4B vLLM ; gpu7 -> Qwen3-VL-2B vLLM.
IAMFlow jobs reach the services over 127.0.0.1, so they land on the same node's
DiT slots as the services (every node runs both services).

Each job is a self-contained script on shared storage launched under
``$MEMSTRATA_TGPU ... setsid`` (or a binary named ``tgpu`` if that env is unset).
that writes its own ``EXIT:<rc>`` sentinel; this orchestrator only watches those
sentinels (no nested ssh babysitting). Safe to nohup.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent / "baseline_runners"
BENCH_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[5]
ASSETS_EN = BENCH_ROOT / "assets" / "trackB" / "en" / "sut_prompts"
OUT_ROOT = BENCH_ROOT / "outputs" / "evaluation" / "trackB"

PY_WAN = os.environ.get("WAN_PYTHON", "python3")
PY_VACE = os.environ.get("IAMFLOW_PYTHON", "python3")
PY_VLLM = os.environ.get("VLLM_PYTHON", "python3")

STEPS = {"memflow": 5, "longlive_rag": 10, "iamflow": 5}
# Per-story dispatch order (lighter first, VLM-heavy iamflow last).
SYSTEM_ORDER = ["memflow", "longlive_rag", "iamflow"]

BASE_CONFIG = {
    "memflow": REPO_ROOT / "baselines" / "Causal" / "MemFlow" / "configs" / "interactive_inference.yaml",
    "longlive_rag": REPO_ROOT / "baselines" / "Causal" / "LongLive-RAG" / "configs" / "longlive_latentmem.yaml",
    "iamflow": REPO_ROOT / "baselines" / "Causal" / "IAMFlow" / "configs" / "iamflow.yaml",
}

# vLLM service ports (loopback, identical on every node). Both small services
# (Qwen3-4B util 0.25 + Qwen3-VL-2B util 0.35 = ~0.6 card) share gpu7, leaving
# gpu0..6 (7 cards/node) for DiT jobs. DiT generation is compute-bound, so we
# keep 1 DiT job per card (packing >1 only time-slices, no throughput gain).
LLM_PORT = 8100
VLM_PORT = 8101
LLM_SERVICE_GPU = 7
VLM_SERVICE_GPU = 7
DIT_GPUS = [0, 1, 2, 3, 4, 5, 6]

LAUNCH_VLLM = RUNNER_ROOT / "iamflow" / "launch_vllm_services.sh"


# --------------------------------------------------------------------------- #
# Config generation                                                            #
# --------------------------------------------------------------------------- #
def denoising_steps(n: int) -> list[int]:
    stride = 1000 // n
    return [1000 - stride * i for i in range(n)]


def rewrite_config(src: Path, dst: Path, *, steps: int, system: str) -> Path:
    """Set denoising_step_list to the requested length; for iamflow pin every
    component to cuda:0 (single-GPU DiT job) and KEEP vlm_enabled true so the
    full VLM path runs against the vLLM HTTP endpoints. Nothing else is touched,
    so the run stays method-faithful."""
    values = denoising_steps(steps)
    iam_device_keys = ("dit_device", "vae_device", "text_encoder_device", "llm_device", "vlm_device")
    out: list[str] = []
    skip_list = False
    for line in src.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if skip_list:
            if stripped.startswith("-"):
                continue
            skip_list = False
        if line.startswith("denoising_step_list:"):
            out.append("denoising_step_list: [" + ", ".join(str(x) for x in values) + "]")
            skip_list = True
            continue
        if system == "iamflow":
            for key in iam_device_keys:
                if line.startswith(f"{key}:"):
                    line = f'{key}: "cuda:0"'
                    break
        out.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- #
# Story enumeration + queue                                                    #
# --------------------------------------------------------------------------- #
def story_segment_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return len(data.get("segments") or [])


def enumerate_stories(limit_stories: int) -> list[tuple[str, Path, int]]:
    rows: list[tuple[str, Path, int]] = []
    for p in sorted(ASSETS_EN.glob("*_name_anchored.json")):
        n = story_segment_count(p)
        if n <= 0:
            continue
        story_id = json.loads(p.read_text(encoding="utf-8")).get("story_id") or p.stem
        rows.append((story_id, p, n))
    rows.sort(key=lambda r: (r[2], r[0]))  # shortest first
    if limit_stories > 0:
        rows = rows[:limit_stories]
    return rows


def out_dir_for(system: str, story_id: str, register: str = "name_anchored") -> Path:
    tag = f"en_steps{STEPS[system]}"
    return OUT_ROOT / system / story_id / register / tag


def already_done(system: str, story_id: str) -> bool:
    manifest = out_dir_for(system, story_id) / "trackb_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "done"
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Job scripts                                                                  #
# --------------------------------------------------------------------------- #
COMMON_ENV = (
    "export NO_PROXY=localhost,127.0.0.1 PYTORCH_ALLOC_CONF=expandable_segments:True\n"
    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
    "export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false\n"
    "export MAVE_REQUIRE_A800_KEEPALIVE=1 MAVE_TASK_NICE=10 MAVE_FFMPEG_THREADS=1\n"
)


def build_jobsh(
    *,
    system: str,
    story_id: str,
    prompt: Path,
    gpu: int,
    cfg: Path,
    frames_per_segment: int,
    jobsh: Path,
    joblog: Path,
    service_env: Path,
    seg_progress: Path,
    longlive_node_slots: int = 1,
    longlive_slot_wait: float = 3600.0,
    longlive_pool: str = "disk",
    longlive_pool_dir: str = "/tmp/longlive_pool",
) -> None:
    runner = RUNNER_ROOT / system / "run.py"
    tag = f"en_steps{STEPS[system]}"
    py = PY_VACE if system == "iamflow" else PY_WAN
    cfg_flag = "--config-template" if system == "memflow" else "--config-path"
    lines = [
        "#!/bin/bash",
        f"cd {json.dumps(str(REPO_ROOT))} || exit 97",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        COMMON_ENV.rstrip("\n"),
        # Unified per-segment progress heartbeat: each baseline appends one JSONL
        # line per completed segment to this shared file (env-gated, O_APPEND-atomic).
        f"export TRACKB_SEG_PROGRESS={json.dumps(str(seg_progress))}",
        f"export TRACKB_SYSTEM={json.dumps(system)}",
        f"export TRACKB_STORY_ID={json.dumps(story_id)}",
    ]
    if system == "longlive_rag":
        # Per-node LongLive RAM concurrency cap (see runner). Wait long enough to
        # block for a slot rather than fast-defer, so a dedicated LongLive sweep
        # actually completes every story instead of skipping under contention.
        lines.append(f"export TRACKB_LONGLIVE_NODE_SLOTS={int(longlive_node_slots)}")
        lines.append(f"export TRACKB_LONGLIVE_SLOT_WAIT={longlive_slot_wait:g}")
        # Memory-safe + fast LongLive path (probe: experiments/results/probe/
        # longlive_rag_perseg_cost/RESULTS.md). The retrieval pool must retain EVERY
        # evicted latent frame, so it grows ~5.62 GB/segment and the host OOM-killer
        # SIGKILLs every full-length story once it exceeds free RAM.
        #   - "disk"     : back the pool with a MAP_SHARED file (torch.from_file);
        #                  the SAME bytes live as reclaimable page cache instead of
        #                  anonymous RAM, so resident RAM stays bounded and OOM never
        #                  fires. This is what lets one LongLive job run PER GPU (many
        #                  per node) instead of one-per-node. Output is byte-identical.
        #   - "prealloc" : one contiguous host RAM buffer per layer (old behavior,
        #                  1/node only). Kept for fallback / parity checks.
        # MEMO reuses retrieved K/V across a block's denoise steps; DETERMINISM=fast
        # drops torch.use_deterministic_algorithms (2.21x faster, output unchanged).
        if longlive_pool == "disk":
            lines.append("export LONGLIVE_CPU_POOL_DISK=1")
            lines.append(f"export LONGLIVE_POOL_DIR={json.dumps(longlive_pool_dir)}")
        else:
            lines.append("export LONGLIVE_CPU_POOL_PREALLOC=1")
        lines.append("export LONGLIVE_MEM_GATHER_MEMO=1")
        lines.append("export LONGLIVE_DETERMINISM=fast")
    if system == "iamflow":
        lines.append(f"source {json.dumps(str(service_env))}")
    cmd = [
        py, "-u", str(runner),
        "--prompts", str(prompt),
        "--run-tag", tag,
        "--overwrite",
        "--frames-per-segment", str(frames_per_segment),
        cfg_flag, str(cfg),
        "--python", py,
        "--cuda-visible-devices", str(gpu),
    ]
    lines.append(" ".join(json.dumps(x) for x in cmd) + f" > {json.dumps(str(joblog))} 2>&1")
    lines.append(f"echo EXIT:$? >> {json.dumps(str(joblog))}")
    jobsh.parent.mkdir(parents=True, exist_ok=True)
    jobsh.write_text("\n".join(lines) + "\n", encoding="utf-8")
    jobsh.chmod(0o755)


# --------------------------------------------------------------------------- #
# optional cluster launcher                                                    #
# --------------------------------------------------------------------------- #
def tgpu(cluster: str, node: int, script: str, *, dry_run: bool, log) -> int:
    launcher = os.environ.get("MEMSTRATA_TGPU", "tgpu")
    cmd = [launcher, "-c", cluster, "-node", str(node), "bash", "-lc", script]
    log(f"  {launcher} -c {cluster} -node {node} :: {script}")
    if dry_run:
        return 0
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def launch_services(cluster: str, node: int, run_dir: Path, *, dry_run: bool, log) -> None:
    shared = "IAMFLOW_ALLOW_SHARED_SERVICE_GPU=1 " if LLM_SERVICE_GPU == VLM_SERVICE_GPU else ""
    env = (
        f"{shared}IAMFLOW_VLLM_PY={PY_VLLM} "
        f"IAMFLOW_LLM_SERVICE_GPU={LLM_SERVICE_GPU} IAMFLOW_VLM_SERVICE_GPU={VLM_SERVICE_GPU} "
        f"IAMFLOW_LLM_PORT={LLM_PORT} IAMFLOW_VLM_PORT={VLM_PORT} "
    )
    script = f"cd {REPO_ROOT} && {env} bash {LAUNCH_VLLM}"
    tgpu(cluster, node, script, dry_run=dry_run, log=log)


def service_ready(cluster: str, node: int, *, dry_run: bool) -> bool:
    if dry_run:
        return True
    probe = (
        f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{LLM_PORT}/v1/models && "
        f"echo ' ' && curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{VLM_PORT}/v1/models"
    )
    try:
        res = subprocess.run(
            [os.environ.get("MEMSTRATA_TGPU", "tgpu"), "-c", cluster, "-node", str(node), "bash", "-lc", probe],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return False
    return res.stdout.count("200") >= 2


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", default="gpu-a800")
    ap.add_argument("--nodes", default="0,1,2,3,4", help="comma-separated tgpu node ids")
    ap.add_argument("--max-job-sec", type=int, default=18000,
                    help="free a slot if a job writes no EXIT sentinel within this many seconds")
    ap.add_argument("--frames-per-segment", type=int, default=21)
    ap.add_argument("--dit-gpus", default=",".join(str(g) for g in DIT_GPUS),
                    help="comma-separated per-node GPU ids to use as DiT job slots "
                         "(partition GPUs to run a second orchestrator without collision)")
    ap.add_argument("--longlive-node-slots", type=int, default=1,
                    help="max concurrent LongLive-RAG jobs per physical node (RAM cap)")
    ap.add_argument("--longlive-pool", choices=["disk", "prealloc"], default="disk",
                    help="LongLive retrieval-pool backing: 'disk' (MAP_SHARED file, "
                         "bounded RAM, one job PER GPU) or 'prealloc' (contiguous host "
                         "RAM, old 1/node behavior). Default disk.")
    ap.add_argument("--longlive-pool-dir", default="/tmp/longlive_pool",
                    help="node-local scratch dir for the disk-backed retrieval pool")
    ap.add_argument("--systems", default=",".join(SYSTEM_ORDER))
    ap.add_argument("--limit-stories", type=int, default=0, help=">0 to run only the N shortest stories (pilot)")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--stagger-sec", type=float, default=8.0,
                    help="minimum seconds between successive job launches on the SAME node, to "
                         "avoid a startup RAM/IO thundering-herd when many models load at once")
    ap.add_argument("--service-warmup-sec", type=int, default=240)
    ap.add_argument("--resume", action="store_true", help="skip stories whose manifest is already status=done")
    ap.add_argument("--dry-run", action="store_true", help="print the plan / commands, touch no GPUs")
    args = ap.parse_args(argv)

    nodes = [int(x) for x in args.nodes.split(",") if x.strip()]
    dit_gpus = [int(x) for x in args.dit_gpus.split(",") if x.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / "_stage1_multinode" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "_stage1_multinode" / "latest").unlink(missing_ok=True)
    (OUT_ROOT / "_stage1_multinode" / "latest").symlink_to(run_dir)
    master = run_dir / "master.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with master.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # Canonical iamflow service env (loopback, identical per node).
    service_env = run_dir / "iamflow_service.env"
    service_env.write_text(
        f"export IAMFLOW_LLM_ENDPOINT=http://127.0.0.1:{LLM_PORT}/v1\n"
        f"export IAMFLOW_VLM_ENDPOINT=http://127.0.0.1:{VLM_PORT}/v1\n"
        "export IAMFLOW_LLM_MODEL=Qwen3-4B-Instruct-2507\n"
        "export IAMFLOW_VLM_MODEL=Qwen3-VL-2B-Instruct\n"
        "export IAMFLOW_HTTP_TIMEOUT=900\n",
        encoding="utf-8",
    )

    # Unified per-segment progress heartbeat file (all systems + all stories append
    # one JSONL line per completed segment). Enables real-time in-flight seg progress
    # in progress_detail.py without waiting for whole-story completion.
    seg_progress = run_dir / "seg_progress.jsonl"
    seg_progress.touch(exist_ok=True)

    # Per-system step configs (generated once).
    cfg_dir = run_dir / "configs"
    configs = {
        sys_: rewrite_config(BASE_CONFIG[sys_], cfg_dir / f"{sys_}_steps{STEPS[sys_]}.yaml",
                             steps=STEPS[sys_], system=sys_)
        for sys_ in systems
    }

    stories = enumerate_stories(args.limit_stories)
    log(f"==== TRACKB STAGE-1 MULTINODE START stamp={stamp} dry_run={args.dry_run} ====")
    log(f"cluster={args.cluster} nodes={nodes} systems={systems} fps={args.frames_per_segment}")
    log(f"stories={len(stories)} (shortest first) total_segments={sum(n for _, _, n in stories)}")

    # Build the job queue: per story (shortest first), the systems in SYSTEM_ORDER.
    queue: list[dict] = []
    skipped = 0
    for story_id, prompt, nseg in stories:
        for sys_ in systems:
            if args.resume and already_done(sys_, story_id):
                skipped += 1
                continue
            queue.append({"system": sys_, "story_id": story_id, "prompt": prompt, "nseg": nseg,
                          "name": f"{sys_}__{story_id}"})
    log(f"queued {len(queue)} jobs; skipped {skipped} already-done")

    # DiT slots: (cluster, node, gpu).
    slots = [(args.cluster, nd, g) for nd in nodes for g in dit_gpus]
    log(f"DiT slots = {len(slots)} ({len(nodes)} nodes x {len(dit_gpus)} gpus={dit_gpus}); "
        f"longlive_node_slots={args.longlive_node_slots}")

    need_iamflow = any(j["system"] == "iamflow" for j in queue)
    if need_iamflow:
        log(f"launching per-node vLLM services (Qwen3-4B gpu{LLM_SERVICE_GPU}, Qwen3-VL-2B gpu{VLM_SERVICE_GPU})...")
        for nd in nodes:
            launch_services(args.cluster, nd, run_dir, dry_run=args.dry_run, log=log)

    if args.dry_run:
        # Print the first few concrete job scripts for inspection, then exit.
        preview = run_dir / "jobs_preview"
        for j in queue[:6]:
            cl, nd, g = slots[0]
            jobsh = preview / f"{j['name']}.sh"
            build_jobsh(system=j["system"], story_id=j["story_id"], prompt=j["prompt"], gpu=g,
                        cfg=configs[j["system"]], frames_per_segment=args.frames_per_segment,
                        jobsh=jobsh, joblog=run_dir / "logs" / f"{j['name']}.log", service_env=service_env,
                        seg_progress=seg_progress, longlive_node_slots=args.longlive_node_slots,
                        longlive_pool=args.longlive_pool, longlive_pool_dir=args.longlive_pool_dir)
        log(f"[dry-run] wrote {min(6, len(queue))} preview job scripts under {preview}")
        log(f"[dry-run] configs under {cfg_dir}")
        log("[dry-run] DONE")
        return 0

    # iamflow dispatch is gated per-node on service readiness (probed lazily in
    # can_dispatch); memflow/longlive dispatch immediately.
    iamflow_ready_nodes: set[int] = set()

    # Dispatch loop.
    free_slots = list(slots)
    running: dict[str, dict] = {}  # name -> {slot, joblog, job}
    done: dict[str, int] = {}
    last_launch_on_node: dict[int, float] = {}  # node -> last launch time (stagger)
    qi = 0
    logdir = run_dir / "logs"
    jobdir = run_dir / "jobs"
    logdir.mkdir(parents=True, exist_ok=True)
    jobdir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.json"

    def node_of(slot) -> int:
        return slot[1]

    def can_dispatch(job, slot) -> bool:
        if job["system"] != "iamflow":
            return True
        nd = node_of(slot)
        if nd in iamflow_ready_nodes:
            return True
        if service_ready(args.cluster, nd, dry_run=False):
            iamflow_ready_nodes.add(nd)
            return True
        return False

    def flush_summary() -> None:
        rows = []
        for name, rc in done.items():
            rows.append({"name": name, "rc": rc})
        summary.write_text(json.dumps({
            "stamp": stamp, "cluster": args.cluster, "nodes": nodes, "systems": systems,
            "fps": args.frames_per_segment, "n_jobs": len(queue),
            "done": len(done), "running": len(running), "results": rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    while qi < len(queue) or running:
        # Fill free slots.
        progressed = True
        while progressed and free_slots and qi < len(queue):
            progressed = False
            for k in range(qi, len(queue)):
                job = queue[k]
                # find a slot this job can use
                slot_idx = None
                for si, slot in enumerate(free_slots):
                    nd_ = node_of(slot)
                    cooling = (time.time() - last_launch_on_node.get(nd_, 0.0)) < args.stagger_sec
                    if can_dispatch(job, slot) and not cooling:
                        slot_idx = si
                        break
                if slot_idx is None:
                    continue  # iamflow waiting on services, or node still in stagger cooldown
                if k != qi:
                    # keep queue order stable but allow skipping a temporarily
                    # undispatchable head (iamflow waiting on services)
                    queue[qi], queue[k] = queue[k], queue[qi]
                    job = queue[qi]
                slot = free_slots.pop(slot_idx)
                cl, nd, g = slot
                joblog = logdir / f"{job['name']}.log"
                jobsh = jobdir / f"{job['name']}.sh"
                build_jobsh(system=job["system"], story_id=job["story_id"], prompt=job["prompt"], gpu=g,
                            cfg=configs[job["system"]], frames_per_segment=args.frames_per_segment,
                            jobsh=jobsh, joblog=joblog, service_env=service_env, seg_progress=seg_progress,
                            longlive_node_slots=args.longlive_node_slots,
                            longlive_pool=args.longlive_pool, longlive_pool_dir=args.longlive_pool_dir)
                joblog.write_text("", encoding="utf-8")
                tgpu(cl, nd, f"setsid bash {jobsh} </dev/null >/dev/null 2>&1 &", dry_run=False, log=log)
                last_launch_on_node[nd] = time.time()
                running[job["name"]] = {"slot": slot, "joblog": joblog, "job": job, "start": time.time()}
                log(f"LAUNCH {job['name']} -> {cl} node {nd} gpu {g} (nseg={job['nseg']})")
                qi += 1
                progressed = True
                break

        if not running:
            break
        time.sleep(args.poll_sec)

        # Reap sentinels.
        finished = []
        for name, info in running.items():
            jl = info["joblog"]
            if not jl.is_file():
                continue
            try:
                txt = jl.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\nEXIT:" in txt or txt.startswith("EXIT:"):
                rc = 0
                for ln in txt.splitlines():
                    if ln.startswith("EXIT:"):
                        try:
                            rc = int(ln.split(":", 1)[1].strip())
                        except ValueError:
                            rc = 1
                finished.append((name, rc))
            elif time.time() - info["start"] > args.max_job_sec:
                finished.append((name, 124))  # timeout watchdog; free the slot
                log(f"WATCHDOG timeout {name} after {args.max_job_sec}s; freeing slot")
        for name, rc in finished:
            info = running.pop(name)
            free_slots.append(info["slot"])
            done[name] = rc
            log(f"DONE  {name} rc={rc}  ({len(done)}/{len(queue)})")
            if rc != 0:
                tail = "\n".join(info["joblog"].read_text(errors="replace").splitlines()[-6:])
                log(f"   FAIL tail:\n{tail}")
        flush_summary()

    n_ok = sum(1 for rc in done.values() if rc == 0)
    log(f"==== STAGE-1 COMPLETE: {n_ok}/{len(queue)} ok, {len(done) - n_ok} failed ====")
    log(f"summary: {summary}")
    print(f"EXIT:{0 if n_ok == len(queue) else 1}", flush=True)
    return 0 if n_ok == len(queue) else 1


if __name__ == "__main__":
    raise SystemExit(main())
