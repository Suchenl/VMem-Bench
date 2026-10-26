"""Paper Track-B shared generator: FLUX.2 [klein] 9B-KV keyframe -> Wan2.2-TI2V-5B-Turbo clip.

Per segment: FLUX renders an 832x480 keyframe from the formatted prompt — multi-image composition from
the references the runner chose, text-to-image when there are none — and Wan animates it (I2V from that
keyframe) at its native 704x1280, delivering 121 frames at 24 fps downscaled to 480x832.

Seeds. ``request.seed`` is the run seed S. FLUX uses ``S + segment_id`` (the paper's keyframe composer
used ``2026 + shot``), Wan uses ``S`` for every segment (the paper's ``base_seed = 2026``). S = 2026
therefore reproduces the frozen paper runs' seeds exactly. FLUX text-to-image keyframes are then
byte-identical to the paper's; Wan is not bit-deterministic across runs/GPUs, so later clips (and
anything retrieved from them) can drift.

Both models live in resident servers (:mod:`flux_server`, :mod:`wan22_turbo_server`) started on
``start()`` or attached to when already alive in the given server dirs. Configuration (constructor
keyword or environment):

* ``flux_model``  ``VMEM_FLUX_MODEL``   default ``$PUBLIC_MODELS_ROOT/black-forest-labs/FLUX.2-klein-9b-kv``
* ``flux_python`` ``VMEM_FLUX_PYTHON``  an interpreter whose diffusers has ``Flux2KleinKVPipeline``
* ``wan_repo``    ``VMEM_WAN22_TURBO_REPO``  upstream Wan2.2-TI2V-5B-Turbo checkout (required)
* ``wan_checkpoint`` ``VMEM_WAN22_TURBO_CKPT`` default ``$PUBLIC_MODELS_ROOT/Wan-AI/Wan2.2-TI2V-5B-Turbo``
* ``wan_python``  ``VMEM_WAN_PYTHON``   an interpreter with torch + flash-attn + the repo's deps
* ``flux_gpu`` / ``wan_gpu``  ``VMEM_FLUX_GPU`` / ``VMEM_WAN_GPU``  physical GPU index for each server
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vmem_bench.testbed.backends.base import RealizationRequest, RealizationResult
from vmem_bench.testbed.backends.job_queue import request_stop, server_alive, submit_job
from vmem_bench.testbed.backends.prompt_format import flux_prompt, wan_prompt

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_BENCH_SRC = Path(__file__).resolve().parents[3]

KEYFRAME_WIDTH, KEYFRAME_HEIGHT, KEYFRAME_STEPS = 832, 480, 4
RENDER_HEIGHT, RENDER_WIDTH = 704, 1280
OUT_HEIGHT, OUT_WIDTH = 480, 832
NUM_FRAMES, FPS = 121, 24


def _models_root() -> str:
    root = os.environ.get("PUBLIC_MODELS_ROOT", "").strip()
    if not root:
        raise RuntimeError("PUBLIC_MODELS_ROOT is not set (FLUX/Wan weights resolve under it)")
    return root


def _sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def gpu_free_mib(index: str) -> int | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        idx, _, free = line.partition(",")
        if idx.strip() == str(index).strip():
            return int(free.strip())
    return None


class Flux2Wan22Backend:
    name = "flux2_wan22"

    def __init__(
        self,
        *,
        service_dir: str | Path | None = None,
        flux_model: str | None = None,
        flux_python: str | None = None,
        wan_repo: str | None = None,
        wan_checkpoint: str | None = None,
        wan_config: str | None = None,
        wan_python: str | None = None,
        flux_gpu: str | None = None,
        wan_gpu: str | None = None,
        flux_min_free_mib: int = 24000,
        wan_min_free_mib: int = 44000,
        ready_timeout: float = 1800.0,
        job_timeout: float = 1800.0,
        idle_timeout: float = 1800.0,
    ) -> None:
        env = os.environ.get
        self.service_dir = Path(service_dir or env("VMEM_TESTBED_SERVICE_DIR") or Path.cwd() / "_testbed_services")
        self.flux_model = flux_model or env("VMEM_FLUX_MODEL") or ""
        self.flux_python = flux_python or env("VMEM_FLUX_PYTHON") or sys.executable
        self.wan_repo = wan_repo or env("VMEM_WAN22_TURBO_REPO") or ""
        self.wan_checkpoint = wan_checkpoint or env("VMEM_WAN22_TURBO_CKPT") or ""
        self.wan_config = wan_config or ""
        self.wan_python = wan_python or env("VMEM_WAN_PYTHON") or sys.executable
        self.flux_gpu = flux_gpu if flux_gpu is not None else env("VMEM_FLUX_GPU")
        self.wan_gpu = wan_gpu if wan_gpu is not None else env("VMEM_WAN_GPU")
        self.flux_min_free_mib, self.wan_min_free_mib = int(flux_min_free_mib), int(wan_min_free_mib)
        self.ready_timeout, self.job_timeout, self.idle_timeout = ready_timeout, job_timeout, idle_timeout
        self.flux_dir = self.service_dir / "flux"
        self.wan_dir = self.service_dir / "wan22_turbo"
        self._launched: list[tuple[Path, subprocess.Popen]] = []

    # ------------------------------------------------------------------ services
    def _gpu_env(self, gpu: str | None, min_free: int, what: str) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{_BENCH_SRC}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if gpu is not None and str(gpu).strip() != "":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible:
            raise RuntimeError(f"{what} server needs exactly one GPU: set {what}_gpu / VMEM_{what.upper()}_GPU "
                               f"or a single CUDA_VISIBLE_DEVICES (got {visible!r})")
        free = gpu_free_mib(visible)
        if free is not None and free < min_free:
            raise RuntimeError(f"GPU {visible} has {free} MiB free; the {what} server needs >= {min_free} MiB")
        return env

    def _launch(self, server_dir: Path, command: list[str], env: dict[str, str]) -> None:
        if server_alive(server_dir):
            return
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "stop").unlink(missing_ok=True)
        (server_dir / "ready").unlink(missing_ok=True)
        log = open(server_dir / "server.log", "ab")  # noqa: SIM115 - owned by the child
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        self._launched.append((server_dir, proc))
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during startup (see {server_dir / 'server.log'})")
            if server_alive(server_dir):
                return
            time.sleep(2.0)
        os.killpg(proc.pid, signal.SIGTERM)
        raise TimeoutError(f"server not ready within {self.ready_timeout}s (see {server_dir / 'server.log'})")

    def start(self) -> None:
        # A server launched here runs in its own session and would outlive a failed start by its idle
        # timeout, pinning the card; stop what this call launched before re-raising.
        try:
            self._start()
        except BaseException:
            self.stop()
            raise

    def _start(self) -> None:
        flux_model = self.flux_model or f"{_models_root()}/black-forest-labs/FLUX.2-klein-9b-kv"
        wan_ckpt = self.wan_checkpoint or f"{_models_root()}/Wan-AI/Wan2.2-TI2V-5B-Turbo"
        if not server_alive(self.flux_dir):
            self._launch(self.flux_dir, [
                self.flux_python, "-m", "vmem_bench.testbed.backends.flux_server",
                "--model_path", flux_model, "--server_dir", str(self.flux_dir.resolve()),
                "--idle_timeout", str(self.idle_timeout),
            ], self._gpu_env(self.flux_gpu, self.flux_min_free_mib, "flux"))
        if not server_alive(self.wan_dir):
            if not self.wan_repo:
                raise RuntimeError("set wan_repo / VMEM_WAN22_TURBO_REPO to the upstream Wan2.2-TI2V-5B-Turbo checkout")
            repo = Path(self.wan_repo).resolve()
            self._launch(self.wan_dir, [
                self.wan_python, "-m", "vmem_bench.testbed.backends.wan22_turbo_server",
                "--repo", str(repo), "--checkpoint_folder", wan_ckpt,
                "--config_path", str(Path(self.wan_config) if self.wan_config else repo / "configs/inference/wan22.yaml"),
                "--server_dir", str(self.wan_dir.resolve()), "--idle_timeout", str(self.idle_timeout),
                "--height", str(RENDER_HEIGHT), "--width", str(RENDER_WIDTH),
                "--out_height", str(OUT_HEIGHT), "--out_width", str(OUT_WIDTH),
                "--num_frames", str(NUM_FRAMES), "--fps", str(FPS),
            ], self._gpu_env(self.wan_gpu, self.wan_min_free_mib, "wan"))

    def stop(self) -> None:
        for server_dir, proc in self._launched:
            request_stop(server_dir)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
        self._launched.clear()

    # ------------------------------------------------------------------ render
    def realize(self, request: RealizationRequest) -> RealizationResult:
        out_dir = Path(request.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        refs = [str(Path(p).resolve()) for p in request.references
                if Path(p).is_file() and Path(p).suffix.lower() in _IMAGE_SUFFIXES]
        flux_seed = int(request.seed) + int(request.segment_id)
        wan_seed = int(request.seed)
        fp, wp = flux_prompt(request.prompt), wan_prompt(request.prompt)

        t0 = time.time()
        kf = submit_job(self.flux_dir, {
            "prompt": fp, "height": KEYFRAME_HEIGHT, "width": KEYFRAME_WIDTH, "steps": KEYFRAME_STEPS,
            "seed": flux_seed, "references": refs, "save_file": str((out_dir / "keyframe.png").resolve()),
        }, timeout=self.job_timeout)
        if kf.get("status") != "ok":
            raise RuntimeError(f"FLUX keyframe failed: {kf.get('error')}")
        keyframe = Path(kf["out_image"])
        t1 = time.time()
        clip = submit_job(self.wan_dir, {
            "prompt": wp, "seed": wan_seed, "height": RENDER_HEIGHT, "width": RENDER_WIDTH,
            "out_height": OUT_HEIGHT, "out_width": OUT_WIDTH, "num_frames": NUM_FRAMES,
            "first_frame_path": str(keyframe), "save_file": str((out_dir / "clip.mp4").resolve()),
        }, timeout=self.job_timeout)
        if clip.get("status") != "ok":
            raise RuntimeError(f"Wan2.2-Turbo clip failed: {clip.get('error')}")
        t2 = time.time()
        return RealizationResult(
            clip=Path(clip["out_video"]),
            keyframe=keyframe,
            meta={
                "keyframe_mode": kf.get("mode"), "n_references": len(refs),
                "flux_seed": flux_seed, "wan_seed": wan_seed, "wan_mode": clip.get("mode"),
                "flux_prompt": fp, "wan_prompt": wp, "keyframe_sha16": _sha16(keyframe),
                "flux_s": round(t1 - t0, 3), "wan_s": round(t2 - t1, 3), "wan_denoise_s": clip.get("denoise_s"),
                "keyframe_size": [KEYFRAME_WIDTH, KEYFRAME_HEIGHT], "clip": {
                    "render": [RENDER_WIDTH, RENDER_HEIGHT], "size": [OUT_WIDTH, OUT_HEIGHT],
                    "frames": NUM_FRAMES, "fps": FPS},
            },
        )
