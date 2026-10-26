"""Resident Wan2.2-TI2V-5B-Turbo server (4-step Self-Forcing distilled, CFG-free).

The Turbo checkpoint is a bare distilled DiT that only runs under its upstream repo's
``Wan22FewstepInferencePipeline``; pass that checkout as ``--repo`` (its ``wan_models/`` symlinks
resolve the base T5/VAE/transformer, which is why the server chdir()s into it). Weights load once.

Rendering happens at the checkpoint's native 704x1280 and the decoded tensor is downscaled once to
the delivered size (480x832): off-native renders visibly shake and ``seq_len`` is fixed upstream, so a
smaller render saves little. ``first_frame_path`` present -> I2V from that keyframe; absent -> T2V.
The request ``seed`` seeds the initial noise (``torch.Generator('cuda')``).

``python -m vmem_bench.testbed.backends.wan22_turbo_server --repo <checkout> --checkpoint_folder
<dir with model.pt> --config_path <repo>/configs/inference/wan22.yaml --server_dir <dir>``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from vmem_bench.testbed.backends.job_queue import serve_forever


def build_pipe(args: argparse.Namespace):
    repo = Path(args.repo).resolve()
    if not (repo / "pipeline" / "wan22_fewstep_inference.py").is_file():
        raise SystemExit(f"not a Wan2.2-TI2V-5B-Turbo checkout: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import torch
    from omegaconf import OmegaConf
    from pipeline.wan22_fewstep_inference import Wan22FewstepInferencePipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    t0 = time.time()
    pipe = Wan22FewstepInferencePipeline(OmegaConf.load(args.config_path))
    state = torch.load(Path(args.checkpoint_folder) / "model.pt", map_location="cpu")
    clean = {}
    for key, value in state.items():
        for junk in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            key = key.replace(junk, "")
        clean[key] = value
    _, unexpected = pipe.generator.load_state_dict(clean, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in Turbo checkpoint: {unexpected[:8]}")
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
    logging.info("weights resident after %.1fs", time.time() - t0)
    return pipe


def run_job(pipe, req: dict, args: argparse.Namespace) -> dict:
    import torch
    import torchvision.transforms.functional as TF
    from diffusers.utils import export_to_video
    from PIL import Image

    height, width = int(req.get("height") or args.height), int(req.get("width") or args.width)
    out_h = int(req.get("out_height") or args.out_height or height)
    out_w = int(req.get("out_width") or args.out_width or width)
    num_frames = int(req.get("num_frames") or args.num_frames)
    if num_frames % 4 != 1:
        raise ValueError(f"num_frames must be one more than a multiple of 4, got {num_frames}")
    if height % 16 or width % 16:
        raise ValueError(f"height/width must be multiples of 16, got {height}x{width}")
    out_path = Path(req["save_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_latent, mode = None, "t2v"
    ref = str(req.get("first_frame_path") or "").strip()
    if ref:
        img = Image.open(ref).convert("RGB").resize((width, height), Image.LANCZOS)
        tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        image_latent = pipe.vae.encode_to_latent(tensor.unsqueeze(0))
        mode = "i2v"

    noise = torch.randn(
        1, (num_frames - 1) // 4 + 1, 48, height // 16, width // 16,
        generator=torch.Generator(device="cuda").manual_seed(int(req["seed"])),
        dtype=torch.bfloat16, device="cuda",
    )
    torch.cuda.synchronize()
    t0 = time.time()
    video = pipe.inference(noise=noise, text_prompts=[str(req.get("prompt") or "")],
                           wan22_image_latent=image_latent)[0]
    torch.cuda.synchronize()
    denoise_s = time.time() - t0
    if (out_h, out_w) != (height, width):
        video = torch.nn.functional.interpolate(video, size=(out_h, out_w), mode="bilinear",
                                                align_corners=False, antialias=True)
    export_to_video(video.permute(0, 2, 3, 1).cpu().numpy(), str(out_path), fps=int(args.fps))
    return {"out_video": str(out_path), "mode": mode, "denoise_s": round(denoise_s, 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint_folder", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--server_dir", required=True)
    p.add_argument("--idle_timeout", type=float, default=1800)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--out_height", type=int, default=0)
    p.add_argument("--out_width", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir).resolve()
    pipe = build_pipe(args)
    serve_forever(server_dir, lambda req: run_job(pipe, req, args), idle_timeout=args.idle_timeout,
                  log=logging.info)


if __name__ == "__main__":
    main()
