"""Resident FLUX.2 [klein] 9B-KV keyframe server (text-to-image, or multi-image from references).

Loads ``Flux2KleinKVPipeline`` (bf16) once and serves jobs over :mod:`job_queue`. A request carries the
already formatted prompt (``prompt_format.flux_prompt``), the render size, step count, seed and the
reference image paths; with references FLUX composes from all of them (one reference = single-image
I2I), without it is text-to-image. The seed drives a ``torch.Generator`` on the render device.

Run in an env whose diffusers has ``Flux2KleinKVPipeline``:
``python -m vmem_bench.testbed.backends.flux_server --model_path <FLUX.2-klein-9b-kv> --server_dir <dir>``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from vmem_bench.testbed.backends.job_queue import serve_forever


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--server_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--idle_timeout", type=float, default=1800.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])

    import torch
    from diffusers import Flux2KleinKVPipeline
    from PIL import Image

    model_path = os.path.expandvars(os.path.expanduser(args.model_path))
    logging.info("loading Flux2KleinKVPipeline from %s", model_path)
    pipe = Flux2KleinKVPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(args.device)

    def handle(req: dict) -> dict:
        seed = req.get("seed")
        generator = torch.Generator(device=args.device).manual_seed(int(seed)) if seed is not None else None
        kwargs = {"prompt": req["prompt"], "height": int(req["height"]), "width": int(req["width"]),
                  "num_inference_steps": int(req["steps"]), "generator": generator}
        refs = req.get("references") or []
        if refs:
            images = [Image.open(r).convert("RGB") for r in refs]
            kwargs["image"] = images if len(images) > 1 else images[0]
        image = pipe(**kwargs).images[0]
        out = Path(req["save_file"])
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        return {"out_image": str(out), "mode": "flux_multi_i2i" if refs else "flux_t2i"}

    serve_forever(Path(args.server_dir), handle, idle_timeout=args.idle_timeout, log=logging.info)


if __name__ == "__main__":
    main()
