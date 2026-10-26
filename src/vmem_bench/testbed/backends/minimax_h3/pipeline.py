"""Server-side MiniMax-H3 Ref2V + Turbo-LoRA loading and generation.

Runs only inside the resident server (``server.py``), in an environment that has DiffSynth-Studio
with MiniMax-H3 support on ``PYTHONPATH``; the client never imports this module. Weights stay
GPU-resident unless an offload mode is chosen (see :data:`OFFLOAD_MODES`), and the public Turbo
LoRA is fused into the DiT once at load.

Paths: ``VMEM_H3_ROOT`` (MiniMax-H3 release, needs ``Ref2VA/``) and ``VMEM_H3_TURBO_LORA``
(``minimax_h3_ref2v_turbo_4step_*.safetensors``) override the defaults under
``$PUBLIC_MODELS_ROOT`` (``MiniMax-H3/`` and ``MiniMax-H3-Turbo/``).
"""

from __future__ import annotations

import glob
import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from vmem_bench.testbed.backends.minimax_h3.geometry import aligned_frame_count

TURBO_LORA_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
# FL2V Turbo 8-step v1.0: 544p mixed-AR training, shifts 12/3 (the pipeline defaults), 8 or 4 steps.
FL2V_TURBO_LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
TASK_DIRS = {"ref2va": "Ref2VA", "fl2va": "FL2VA"}
_QKV = ("to_q", "to_k", "to_v")


def _public_root() -> Path:
    from vmem_bench.install_models import public_root

    return public_root()


def h3_root() -> Path:
    env = os.environ.get("VMEM_H3_ROOT")
    return Path(env).expanduser() if env else _public_root() / "MiniMax-H3"


def turbo_lora_path(task: str = "ref2va") -> Path:
    env = os.environ.get("VMEM_H3_TURBO_LORA")
    name = TURBO_LORA_NAME if task == "ref2va" else FL2V_TURBO_LORA_NAME
    return Path(env).expanduser() if env else _public_root() / "MiniMax-H3-Turbo" / name


def offline() -> None:
    for key, value in (("DIFFSYNTH_SKIP_DOWNLOAD", "True"), ("HF_HUB_OFFLINE", "1"),
                       ("TRANSFORMERS_OFFLINE", "1"), ("MODELSCOPE_OFFLINE", "1")):
        os.environ.setdefault(key, value)


def _model_config(path_or_glob: str, **vram):
    from diffsynth.core.loader.config import ModelConfig

    files = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not files or not Path(files[0]).exists():
        raise FileNotFoundError(f"missing local checkpoint: {path_or_glob}")
    return ModelConfig(path=files if len(files) > 1 else files[0], skip_download=True, **vram)


# ---------------------------------------------------------------- VRAM placement
# DiffSynth parks a model on ``offload_device`` while another model runs, moves it to
# ``onload_device`` when its stage starts, and during forward promotes layers to ``preparing_device``
# while device-wide VRAM use stays under ``vram_limit``; layers past the limit are copied to the GPU
# per call. ``none`` keeps everything resident (~120 GiB peak, B200/H200). ``text_encoder`` parks
# only the ~62 GiB text encoder (used once per segment) on the host; ``all`` parks every model there.
OFFLOAD_MODES = ("none", "text_encoder", "all")
_MODEL_DIRS = ("text_encoder", "transformer", "video_vae", "audio_vae")


def vram_placement(offload: str) -> dict[str, str]:
    """Parking device ("cuda" or "cpu") for each of the four Ref2VA models."""
    if offload not in OFFLOAD_MODES:
        raise ValueError(f"unknown offload mode {offload!r}; choices: {OFFLOAD_MODES}")
    host = {"none": (), "text_encoder": ("text_encoder",), "all": _MODEL_DIRS}[offload]
    return {m: ("cpu" if m in host else "cuda") for m in _MODEL_DIRS}


def _vram_config(parking: str = "cuda") -> dict[str, Any]:
    import torch

    return {
        "offload_dtype": torch.bfloat16, "offload_device": parking,
        "onload_dtype": torch.bfloat16, "onload_device": parking,
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }


# ---------------------------------------------------------------- Turbo LoRA (PEFT -> DiffSynth)
# The Turbo release is a rank-128 PEFT adapter keyed by Diffusers names
# (``transformer_blocks.N.attn.to_q``); DiffSynth's MiniMaxH3DiT uses fused ``blocks.N.attn.qkv_proj``
# laid out as ``view(..., num_heads, 3, head_dim)``, so Q/K/V deltas are interleaved per head.

def _rename_linear(stem: str) -> str | None:
    for src, dst in (("transformer_blocks.", "blocks."), ("token_refiner.refiner_blocks.", "token_refiner.blocks.")):
        if stem.startswith(src):
            idx, _, tail = stem[len(src):].partition(".")
            mapping = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
            if tail in mapping:
                return f"{dst}{idx}.{mapping[tail]}"
            if tail.startswith("attn.to_"):
                return f"{dst}{idx}.attn.qkv_proj"
            return None
    return None


def _interleave_qkv(dq, dk, dv, num_heads: int):
    out, inn = dq.shape
    if dk.shape != (out, inn) or dv.shape != (out, inn) or out % num_heads:
        raise ValueError(f"bad QKV delta shapes {tuple(dq.shape)} {tuple(dk.shape)} {tuple(dv.shape)}")
    hd = out // num_heads
    import torch

    return torch.stack([d.view(num_heads, hd, inn) for d in (dq, dk, dv)], dim=1).reshape(3 * out, inn)


def fuse_turbo_lora(dit, path: Path, *, alpha: float = 8.0, num_heads: int = 56) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")

    def delta(stem: str):
        a = state[f"{stem}.lora_A.default.weight"]
        b = state[f"{stem}.lora_B.default.weight"]
        return (float(alpha) / float(a.shape[0])) * torch.mm(b.float(), a.float())

    stems = sorted({k.rsplit(".lora_", 1)[0] for k in state if k.endswith(".lora_A.default.weight")})
    groups: dict[str, dict[str, str]] = {}
    for stem in stems:
        head, _, leaf = stem.rpartition(".")
        if leaf in _QKV:
            groups.setdefault(head, {})[leaf] = stem
        else:
            groups.setdefault(stem, {})["linear"] = stem
    modules = dict(dit.named_modules())
    fused, missing = 0, []
    for parts in groups.values():
        if all(q in parts for q in _QKV):
            target = _rename_linear(parts["to_q"])
            d = _interleave_qkv(*(delta(parts[q]) for q in _QKV), num_heads=num_heads)
        else:
            target = _rename_linear(parts["linear"])
            d = delta(parts["linear"])
        mod = modules.get(target) if target else None
        if mod is None or tuple(mod.weight.shape) != tuple(d.shape):
            missing.append(target or str(parts))
            continue
        mod.weight.data.add_(d.to(device=mod.weight.device, dtype=mod.weight.dtype))
        fused += 1
    if missing:
        raise RuntimeError(f"Turbo LoRA fuse incomplete: fused={fused} missing={missing[:8]}")
    return {"fused": fused, "alpha": alpha, "path": str(path)}


# ---------------------------------------------------------------- load / generate

def load_pipeline(*, lora: Path | None = None, lora_alpha: float = 8.0, offload: str = "none",
                  vram_limit_gib: float | None = None, task: str = "ref2va"):
    """Load Ref2VA or FL2VA (+ fused Turbo LoRA). Returns ``(pipe, lora_audit, vram_audit)``.

    ``vram_limit_gib`` defaults to the card's total minus 2 GiB; with host parking it bounds how many
    layers stay promoted on the GPU, so leave room for activations on 72-80 GB cards.
    """
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline

    offline()
    root = h3_root() / TASK_DIRS[task]
    if not root.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 {TASK_DIRS[task]} weights not found under {root}")
    placement = vram_placement(offload)
    total_gib = torch.cuda.mem_get_info("cuda")[1] / 1024 ** 3
    limit = float(vram_limit_gib) if vram_limit_gib else total_gib - 2
    files = {"text_encoder": "text_encoder/model*.safetensors", "transformer": "transformer/model*.safetensors",
             "video_vae": "video_vae/source/model.safetensors", "audio_vae": "audio_vae/model.safetensors"}
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[_model_config(str(root / files[m]), **_vram_config(placement[m])) for m in _MODEL_DIRS],
        processor_config=_model_config(str(root / "processor")),
        vram_limit=limit,
    )
    audit = fuse_turbo_lora(pipe.dit, lora, alpha=lora_alpha) if lora else None
    vram_audit = {"offload": offload, "parking": placement, "vram_limit_gib": round(limit, 2),
                  "gpu_total_gib": round(total_gib, 2), "gpu_name": torch.cuda.get_device_name()}
    return pipe, audit, vram_audit


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_request_state(pipe) -> None:
    """Drop per-request state the text encoder keeps between calls.

    transformers' ``Qwen3VLModel`` caches ``rope_deltas`` from the last prompt that carried images and
    applies it to later text-only prompts, so without this a zero-reference segment depends on
    whichever reference request the server served before it (measured 2026-09-27: mean |diff| ~6/255).
    """
    model = getattr(getattr(pipe, "text_encoder", None), "model", None)
    if model is not None and hasattr(model, "rope_deltas"):
        model.rope_deltas = None


def generate(pipe, *, prompt: str, references: list[Path], seed: int, num_frames: int,
             height: int, width: int, steps: int, ref_image_short_edge: int, first_frame: Path | None = None) -> list:
    """Return ``num_frames`` PIL frames (H3 renders the 17n+5 superset; the tail is trimmed).

    ``first_frame`` (FL2VA) conditions frame 0 on that image; it excludes ``references`` (Ref2VA)."""
    from PIL import Image

    reset_request_state(pipe)
    seed_everything(int(seed))
    refs = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in references]
    kf = {"keyframes": [Image.open(first_frame).convert("RGB")], "keyframe_indices": [0]} if first_frame else {}
    video, _audio = pipe(
        **kf,
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_frames=aligned_frame_count(num_frames),
        num_inference_steps=int(steps),
        seed=int(seed),
        rand_device="cpu",
        cfg_scale=1.0,
        references=refs or None,
        ref_image_short_edge=int(ref_image_short_edge),
    )
    frames = list(video)[: int(num_frames)]
    if len(frames) != int(num_frames):
        raise RuntimeError(f"H3 returned {len(frames)} frames, expected >= {num_frames}")
    return frames


def frames_sha256(frames) -> str:
    h = hashlib.sha256()
    for f in frames:
        h.update(np.asarray(f, dtype=np.uint8).tobytes())
    return h.hexdigest()


def write_clip(frames, path: Path, *, fps: int, out_height: int | None, out_width: int | None) -> list:
    """Downscale to the delivery size if requested and write an H.264 mp4 (no audio)."""
    import imageio.v2 as imageio
    from PIL import Image

    if out_height and out_width and frames[0].size != (out_width, out_height):
        frames = [f.resize((int(out_width), int(out_height)), Image.Resampling.LANCZOS) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.mp4")
    writer = imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for f in frames:
            writer.append_data(np.asarray(f.convert("RGB")))
    finally:
        writer.close()
    os.replace(tmp, path)
    return frames
