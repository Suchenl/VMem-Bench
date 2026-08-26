"""LongLive-RAG causal adapter (new protocol, real segments).

Native memory: a growing pool of per-latent **AE descriptors** over the SUT's own
observed history. Native retrieval: **cosine top-k** of the latest descriptor
against the eligible pool (after ``sink_size``, before ``recent_exclude``), the
published ``latentmem`` rule. We reuse LongLive-RAG's own ``WanVAEWrapper`` (to
encode real frames to Wan latents) and ``ae.model.LatentAE`` (the retrieval AE);
no generator forward is needed because retrieval is a pure descriptor operation.

Difference vs the retired gold path: descriptors come from the SUT VAE-encoding
the **real segment** (not gold latents), and hits are returned as temporal items
(absolute source seconds) for frame materialization -- never mapped to gold ids.

Env: LongLive-RAG's own env (diffusers==0.31.0, transformers>=4.49, torch+cu, av).
Weights (place under the vendored repo root):
  baselines/Causal/LongLive-RAG/wan_models/Wan2.1-T2V-1.3B/{Wan2.1_VAE.pth,...}
  baselines/Causal/LongLive-RAG/checkpoints/ae_latent_mem.pt
Run via runner.py --adapter longlive_rag.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _video_io import latent_local_seconds, read_segment_pixels

_REPO = Path(__file__).resolve().parents[7] / "baselines" / "Causal" / "LongLive-RAG"

# Published latentmem retrieval knobs (configs/*_latentmem.yaml).
MEMORY_SIZE = 6
RECENT_EXCLUDE = 5
SINK_SIZE = 1


@contextlib.contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class LongLiveRAGAdapter:
    name = "longlive_rag"

    def __init__(
        self,
        *,
        memory_size: int = MEMORY_SIZE,
        recent_exclude: int = RECENT_EXCLUDE,
        sink_size: int = SINK_SIZE,
        ae_ckpt: str | None = None,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.memory_size = int(memory_size)
        self.recent_exclude = int(recent_exclude)
        self.sink_size = int(sink_size)
        self.ae_ckpt = Path(ae_ckpt) if ae_ckpt else (_REPO / "checkpoints" / "ae_latent_mem.pt")
        self.ffmpeg = ffmpeg
        self._vae = None
        self._vae_mean = None
        self._vae_std = None
        self._ae = None
        self._device = None
        # Global descriptor pool over observed history.
        self._descs: list = []            # list of [D] tensors (L2-normalizable)
        self._desc_seconds: list[float] = []  # absolute source seconds per descriptor

    def reset(self, movie: MovieContext) -> None:
        import torch

        if not _REPO.is_dir():
            raise FileNotFoundError(f"LongLive-RAG checkout missing: {_REPO}")
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._descs = []
        self._desc_seconds = []
        self._movie = movie

        if self._vae is None or self._ae is None:
            # Import the VAE and AE directly (not utils.wan_wrapper) so we avoid the
            # DiT / flash-attn import chain -- retrieval only needs VAE encode + AE.
            # Both load weights by paths relative to the repo root.
            with _cwd(_REPO):
                from wan.modules.vae import _video_vae
                from ae.config import AEConfig
                from ae.model import LatentAE

                # WanVAEWrapper normalization constants (utils/wan_wrapper.py:63-78).
                mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
                        1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
                        0.2503, -0.2921]
                std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
                self._vae_mean = torch.tensor(mean, dtype=torch.float32, device=self._device)
                self._vae_std = torch.tensor(std, dtype=torch.float32, device=self._device)
                self._vae = _video_vae(
                    pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", z_dim=16,
                ).eval().requires_grad_(False).to(self._device)
                if not self.ae_ckpt.is_file():
                    raise FileNotFoundError(f"AE checkpoint not found: {self.ae_ckpt}")
                ckpt = torch.load(self.ae_ckpt, map_location="cpu")
                cfg = AEConfig(**{k: v for k, v in ckpt["config"].items()
                                  if k in {f.name for f in dataclasses.fields(AEConfig)}})
                ae = LatentAE(cfg)
                ae.load_state_dict(ckpt["model"], strict=False)
                self._ae = ae.eval().to(self._device)

    def _encode_to_latent(self, pixel):
        """Reproduce WanVAEWrapper.encode_to_latent (utils/wan_wrapper.py:80-94)."""
        import torch

        dtype = self._vae_mean.dtype
        scale = [self._vae_mean, 1.0 / self._vae_std]
        out = [self._vae.encode(u.unsqueeze(0).to(dtype), scale).float().squeeze(0)
               for u in pixel]
        out = torch.stack(out, dim=0)               # [B, C, T, H, W]
        return out.permute(0, 2, 1, 3, 4)           # [B, T, C, H, W]

    def observe_segment(self, obs: SegmentObservation) -> None:
        import torch

        pixel, _n = read_segment_pixels(obs.segment_video, ffmpeg=self.ffmpeg)
        pixel = pixel.to(self._device)
        t0 = float(obs.seconds_span[0])
        with torch.no_grad():
            latents = self._encode_to_latent(pixel)
            latents = latents[0]  # [T, C, H, W]
            for li in range(int(latents.shape[0])):
                desc = self._ae.encode(latents[li : li + 1].float().to(self._device))  # [1, D]
                self._descs.append(desc.squeeze(0).detach().float().cpu())
                self._desc_seconds.append(t0 + latent_local_seconds(li, obs.fps))

    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        import torch

        rec = RetrievedMemory(chunk_id=req.chunk_id)
        n = len(self._descs)
        if n == 0 or self.memory_size <= 0:
            return rec
        # Eligible pool: after sink, before the most-recent recent_exclude.
        lo = self.sink_size
        hi = n - self.recent_exclude
        if hi <= lo:
            return rec
        pool = torch.stack(self._descs[lo:hi], dim=0)  # [P, D]
        query = self._descs[-1].unsqueeze(0)           # [1, D] latest observed
        pool = pool / (pool.norm(dim=-1, keepdim=True) + 1e-8)
        q = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (pool @ q.t()).squeeze(-1)              # [P]
        k = min(self.memory_size, int(sims.numel()))
        top = torch.topk(sims, k=k)
        for local_i, sc in zip(top.indices.tolist(), top.values.tolist()):
            gidx = lo + int(local_i)
            rec.items.append(RetrievedItem(
                evidence_kind="latent",
                source_seconds=self._desc_seconds[gidx],
                score=float(sc),
                raw_ref=f"desc[{gidx}]",
            ))
        return rec

    def finalize(self) -> dict[str, Any]:
        return {
            "retrieval": "ae_cosine_topk_on_self_encoded_real_latents",
            "memory_size": self.memory_size,
            "recent_exclude": self.recent_exclude,
            "sink_size": self.sink_size,
            "ae_ckpt": str(self.ae_ckpt),
            "n_descriptors": len(self._descs),
        }


def build_adapter() -> LongLiveRAGAdapter:
    return LongLiveRAGAdapter()
