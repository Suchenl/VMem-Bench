"""FLUX.2 [klein] 9B-KV keyframe -> MiniMax-H3 FL2VA (first-frame) Turbo clip.

The paper pipeline (:mod:`flux2_wan22`) with the animator swapped: the FLUX keyframe step is identical
(same prompt formatting, ``S + segment_id`` seed, 832x480, 4 steps, multi-image composition from the
references the runner chose), and a resident H3 FL2VA server (``server --task fl2va``, FL2V Turbo 8-step
v1.0 LoRA by default) animates that keyframe as frame 0 with the release's i2va prompt layout
(:func:`build_h3_i2v_prompt`), seed ``S``, rendering at ``geometry`` and delivering 121 frames at 24 fps
downscaled to 480x832. The references reach H3 only through the keyframe, as they reach Wan in the paper.

FLUX is started on ``start()`` like in :mod:`flux2_wan22` (same options); H3 servers are started out of
band (``scripts/testbed/minimax_h3/serve.sh`` with ``H3_SERVER_ARGS="--task fl2va"``) and found through
``endpoints`` (URL list, endpoint dir, or ``$VMEM_H3_FL2V_ENDPOINTS``).
"""

from __future__ import annotations

import json
import os
import queue
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from vmem_bench.testbed.backends.base import RealizationRequest, RealizationResult
from vmem_bench.testbed.backends.flux2_wan22 import (
    _IMAGE_SUFFIXES,
    KEYFRAME_HEIGHT,
    KEYFRAME_STEPS,
    KEYFRAME_WIDTH,
    Flux2Wan22Backend,
    _models_root,
    _sha16,
)
from vmem_bench.testbed.backends.job_queue import server_alive, submit_job
from vmem_bench.testbed.backends.minimax_h3 import geometry as geo
from vmem_bench.testbed.backends.minimax_h3.client import discover_endpoints
from vmem_bench.testbed.backends.minimax_h3.prompt import build_h3_i2v_prompt
from vmem_bench.testbed.backends.prompt_format import flux_prompt

ENDPOINTS_ENV = "VMEM_H3_FL2V_ENDPOINTS"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Flux2H3Backend(Flux2Wan22Backend):
    name = "flux2_h3"

    def __init__(self, *, endpoints: str | Path | Sequence[str] | None = None, geometry: str = "544x960",
                 steps: int = 8, h3_timeout_sec: float = 1800.0, **flux_opts: Any) -> None:
        super().__init__(**flux_opts)
        self.endpoints_spec = endpoints if endpoints is not None else os.environ.get(ENDPOINTS_ENV)
        self.geometry = geo.geometry(geometry).name
        self.steps, self.h3_timeout_sec = int(steps), float(h3_timeout_sec)
        self._pool: queue.Queue[str] | None = None

    def _start(self) -> None:
        flux_model = self.flux_model or f"{_models_root()}/black-forest-labs/FLUX.2-klein-9b-kv"
        if not server_alive(self.flux_dir):
            self._launch(self.flux_dir, [
                self.flux_python, "-m", "vmem_bench.testbed.backends.flux_server",
                "--model_path", flux_model, "--server_dir", str(self.flux_dir.resolve()),
                "--idle_timeout", str(self.idle_timeout),
            ], self._gpu_env(self.flux_gpu, self.flux_min_free_mib, "flux"))
        if not self.endpoints_spec:
            raise RuntimeError(f"no MiniMax-H3 FL2VA endpoints: pass endpoints= or set ${ENDPOINTS_ENV}")
        urls = discover_endpoints(self.endpoints_spec, model_id=geo.FL2V_MODEL_ID)
        if not urls:
            raise RuntimeError(f"no healthy MiniMax-H3 FL2VA endpoint in {self.endpoints_spec!r}")
        self._pool = queue.Queue()
        for u in urls:
            self._pool.put(u)

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(url + "/v1/realize", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with _OPENER.open(req, timeout=self.h3_timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"MiniMax-H3 FL2VA {url} HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def realize(self, request: RealizationRequest) -> RealizationResult:
        if self._pool is None:
            self.start()
        assert self._pool is not None
        out_dir = Path(request.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        refs = [str(Path(p).resolve()) for p in request.references
                if Path(p).is_file() and Path(p).suffix.lower() in _IMAGE_SUFFIXES]
        flux_seed, h3_seed = int(request.seed) + int(request.segment_id), int(request.seed)
        fp, hp = flux_prompt(request.prompt), build_h3_i2v_prompt(request.prompt)

        t0 = time.time()
        kf = submit_job(self.flux_dir, {
            "prompt": fp, "height": KEYFRAME_HEIGHT, "width": KEYFRAME_WIDTH, "steps": KEYFRAME_STEPS,
            "seed": flux_seed, "references": refs, "save_file": str((out_dir / "keyframe.png").resolve()),
        }, timeout=self.job_timeout)
        if kf.get("status") != "ok":
            raise RuntimeError(f"FLUX keyframe failed: {kf.get('error')}")
        keyframe = Path(kf["out_image"])
        t1 = time.time()
        g = geo.geometry(self.geometry)
        body = {"prompt": hp, "references": [], "first_frame": str(keyframe.resolve()), "seed": h3_seed,
                "out": str((out_dir / "clip.mp4").resolve()), "num_frames": geo.NUM_FRAMES, "fps": geo.FPS,
                "geometry": g.name, "height": g.height, "width": g.width, "out_height": g.out_height,
                "out_width": g.out_width, "steps": self.steps, "ref_image_short_edge": geo.REF_IMAGE_SHORT_EDGE}
        url = self._pool.get()
        try:
            meta = self._post(url, body)
        finally:
            self._pool.put(url)
        clip = Path(meta.get("clip") or body["out"])
        if not clip.is_file() or clip.stat().st_size == 0:
            raise RuntimeError(f"MiniMax-H3 FL2VA {url} reported success but {clip} is missing or empty")
        return RealizationResult(clip=clip, keyframe=keyframe, meta={
            "keyframe_mode": kf.get("mode"), "n_references": len(refs), "flux_seed": flux_seed, "h3_seed": h3_seed,
            "flux_prompt": fp, "h3_prompt": hp, "keyframe_sha16": _sha16(keyframe), "flux_s": round(t1 - t0, 3),
            "h3_s": round(time.time() - t1, 3), "h3_endpoint": url, "h3_model": meta.get("model"),
            "geometry": {"name": g.name, "render": [g.height, g.width], "delivered": [g.out_height, g.out_width]},
            "steps": self.steps, "generate_sec": meta.get("generate_sec"),
        })
