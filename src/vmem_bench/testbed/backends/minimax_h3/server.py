"""Resident MiniMax-H3 Ref2V Turbo server (one GPU, weights loaded once, jobs serialised).

    python -m vmem_bench.testbed.backends.minimax_h3.server --port 18300 --endpoint-dir <dir>

Routes: ``GET /health``, ``GET /info``, ``POST /v1/realize`` (body: :class:`RealizeBody`). The request
already carries the final H3 prompt; the server renders, writes ``out`` (mp4) plus ``out``.json, and
returns that metadata. A repeated request whose fingerprint (prompt, reference bytes, seed, geometry
name and sizes, steps, model) matches the sidecar of an existing ``out`` returns the cached result
without rendering. The offload mode changes placement only, so it is recorded (``vram``) but not
fingerprinted. Launch through ``scripts/testbed/minimax_h3/serve.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vmem_bench.testbed.backends.minimax_h3 import geometry as geo
from vmem_bench.testbed.backends.minimax_h3.geometry import MODEL_ID


class RealizeBody(BaseModel):
    """``geometry`` names a preset (server ``--geometry`` when omitted) that fills any size left unset."""

    prompt: str
    references: list[str] = Field(default_factory=list)
    seed: int
    out: str
    num_frames: int = geo.NUM_FRAMES
    fps: int = geo.FPS
    geometry: str | None = None
    height: int | None = None
    width: int | None = None
    out_height: int | None = None
    out_width: int | None = None
    steps: int = geo.STEPS
    ref_image_short_edge: int = geo.REF_IMAGE_SHORT_EDGE
    first_frame: str | None = None   # FL2VA servers only (``--task fl2va``)


def resolve_geometry(body: dict[str, Any], default: str) -> dict[str, Any]:
    """Fill ``geometry`` and the render/delivery sizes from the named preset (explicit sizes win)."""
    g = geo.geometry(body.get("geometry") or default)
    out = {**body, "geometry": g.name}
    for key in ("height", "width", "out_height", "out_width"):
        if out.get(key) is None:
            out[key] = getattr(g, key)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request_fingerprint(body: dict[str, Any], *, model: dict[str, Any]) -> str:
    keys = ("prompt", "seed", "num_frames", "fps", "geometry", "height", "width", "out_height", "out_width",
            "steps", "ref_image_short_edge")
    payload = {k: body.get(k) for k in keys}
    payload["references"] = [file_sha256(Path(p)) for p in body.get("references", [])]
    if body.get("first_frame"):
        payload["first_frame"] = file_sha256(Path(body["first_frame"]))
    payload["model"] = model
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cached(out: Path, fingerprint: str) -> dict[str, Any] | None:
    side = out.with_suffix(".json")
    if not (out.is_file() and side.is_file()):
        return None
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {**meta, "cached": True} if meta.get("fingerprint") == fingerprint else None


def build_app(render: Callable[[dict[str, Any]], dict[str, Any]], *, model: dict[str, Any],
              default_geometry: str = geo.DEFAULT_GEOMETRY, task: str = "ref2va"):
    """``render(body) -> meta`` does the GPU work; everything else (locking, cache, validation) is here."""
    geo.geometry(default_geometry)
    MODEL_ID = geo.MODEL_ID if task == "ref2va" else geo.FL2V_MODEL_ID  # noqa: N806
    app = FastAPI(title=f"VMem-Bench MiniMax-H3 {task} Turbo")
    lock = threading.Lock()
    state = {"jobs": 0, "started": time.time()}

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID, "busy": lock.locked(), "jobs": state["jobs"],
                "default_geometry": default_geometry, "uptime_sec": round(time.time() - state["started"], 1)}

    @app.get("/info")
    def info():
        return {"ok": True, "model": MODEL_ID, **model, "default_geometry": default_geometry,
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "busy": lock.locked(), "jobs": state["jobs"]}

    @app.post("/v1/realize")
    def realize(req: RealizeBody):
        body = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        try:
            body = resolve_geometry(body, default_geometry)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not body["prompt"].strip():
            raise HTTPException(status_code=400, detail="empty prompt")
        if len(body["references"]) > 9:
            raise HTTPException(status_code=400, detail="Ref2V accepts at most 9 image references")
        for p in body["references"]:
            if not Path(p).is_file():
                raise HTTPException(status_code=400, detail=f"reference not found: {p}")
        if task == "fl2va":
            if body["references"] or not body.get("first_frame") or not Path(body["first_frame"]).is_file():
                raise HTTPException(status_code=400, detail="FL2VA takes one existing first_frame and no references")
        elif body.get("first_frame"):
            raise HTTPException(status_code=400, detail="Ref2VA takes references, not first_frame")
        if body["height"] % 32 or body["width"] % 32:
            raise HTTPException(status_code=400, detail="height/width must be multiples of 32")
        out = Path(body["out"])
        fingerprint = request_fingerprint(body, model=model)
        hit = _cached(out, fingerprint)
        if hit is not None:
            return hit
        t_wait = time.time()
        with lock:
            hit = _cached(out, fingerprint)
            if hit is not None:
                return hit
            state["jobs"] += 1
            t0 = time.time()
            try:
                meta = render(body)
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
            meta = {
                **meta,
                "clip": str(out),
                "geometry": {"name": body["geometry"], "render": [body["height"], body["width"]],
                             "delivered": [body["out_height"], body["out_width"]]},
                "fingerprint": fingerprint,
                "request": {k: v for k, v in body.items() if k != "prompt"},
                "prompt": body["prompt"],
                "model": {"id": MODEL_ID, **model},
                "queue_wait_sec": round(t0 - t_wait, 2),
                "elapsed_sec": round(time.time() - t0, 2),
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "job_index": state["jobs"],
                "cached": False,
            }
            out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            return meta

    return app


def _gpu_renderer(pipe, vram: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    import torch

    from vmem_bench.testbed.backends.minimax_h3 import pipeline as h3

    def render(body: dict[str, Any]) -> dict[str, Any]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        frames = h3.generate(
            pipe, prompt=body["prompt"], references=[Path(p) for p in body["references"]],
            seed=body["seed"], num_frames=body["num_frames"], height=body["height"], width=body["width"],
            steps=body["steps"], ref_image_short_edge=body["ref_image_short_edge"],
            first_frame=Path(body["first_frame"]) if body.get("first_frame") else None,
        )
        torch.cuda.synchronize()
        t_gen = time.time() - t0
        render_sha = h3.frames_sha256(frames)
        delivered = h3.write_clip(frames, Path(body["out"]), fps=body["fps"],
                                  out_height=body["out_height"], out_width=body["out_width"])
        return {
            "generate_sec": round(t_gen, 2),
            "rendered_frames": geo.aligned_frame_count(body["num_frames"]),
            "delivered_frames": len(delivered),
            "delivered_size": list(delivered[0].size),
            "render_frames_sha256": render_sha,
            "delivered_frames_sha256": h3.frames_sha256(delivered),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "reserved_vram_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
            "peak_reserved_vram_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            "vram": vram,
        }

    return render


def _reachable_ip() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except OSError:
        return socket.gethostname()


def write_endpoint(endpoint_dir: Path, *, port: int, extra: dict[str, Any], model_id: str = MODEL_ID) -> Path:
    ip = _reachable_ip()
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    path = endpoint_dir / f"{model_id}_{ip}_{port}.json"
    payload = {"model": model_id, "url": f"http://{ip}:{port}", "host": socket.gethostname(),
               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--endpoint-dir", default=None, help="write <model>_<ip>_<port>.json here after load")
    p.add_argument("--lora", default=None, help="Turbo LoRA (default: VMEM_H3_TURBO_LORA / PUBLIC_MODELS_ROOT)")
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--no-lora", action="store_true",
                   help="base Ref2VA without the Turbo LoRA (the released weights are CFG-distilled; use ~40 steps)")
    p.add_argument("--geometry", default=geo.DEFAULT_GEOMETRY, choices=sorted(geo.GEOMETRIES),
                   help="render/delivery preset for requests that do not name one")
    p.add_argument("--offload", default="none", choices=("none", "text_encoder", "all"),
                   help="park models on the host: none (~120 GiB VRAM), text_encoder, all")
    p.add_argument("--vram-limit-gib", type=float, default=None,
                   help="DiffSynth vram_limit (default: card total - 2 GiB)")
    p.add_argument("--task", default="ref2va", choices=("ref2va", "fl2va"),
                   help="ref2va: image references (MiniMax backend); fl2va: first frame (flux2_h3 backend)")
    args = p.parse_args(argv)

    from vmem_bench.testbed.backends.minimax_h3 import pipeline as h3

    lora = None if args.no_lora else (Path(args.lora) if args.lora else h3.turbo_lora_path(args.task))
    t0 = time.time()
    print(f"[h3-server] loading {h3.h3_root()}/{h3.TASK_DIRS[args.task]} + {lora} (offload={args.offload})", flush=True)
    pipe, audit, vram = h3.load_pipeline(lora=lora, lora_alpha=args.lora_alpha, offload=args.offload,
                                         vram_limit_gib=args.vram_limit_gib, task=args.task)
    load_sec = round(time.time() - t0, 1)
    print(f"[h3-server] loaded in {load_sec}s; lora {audit}; vram {vram}", flush=True)
    model = {"h3_root": str(h3.h3_root()), "lora": str(lora) if lora else None, "lora_alpha": args.lora_alpha,
             "lora_fused": audit.get("fused") if audit else None}
    if args.task != "ref2va":
        model["task"] = args.task
    app = build_app(_gpu_renderer(pipe, vram), model=model, default_geometry=args.geometry, task=args.task)
    path = None
    if args.endpoint_dir:
        path = write_endpoint(Path(args.endpoint_dir), port=args.port,
                              extra={"load_sec": load_sec, "default_geometry": args.geometry, "vram": vram},
                              model_id=geo.MODEL_ID if args.task == "ref2va" else geo.FL2V_MODEL_ID)
        print(f"[h3-server] endpoint {path}", flush=True)

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=3600)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
