"""DINOv3 crop embedder for cross-chunk consolidation (lazy singleton)."""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.model_weights import public_models_root

# Stay on ViT-S until the re-ID thresholds are recalibrated: ViT-L has a different cosine
# distribution and, measured on BBB v8 (2026-07-12), the uncalibrated swap SPLIT clusters
# (13 characters / 16 singletons vs ViT-S's 6 / 4). Bigger is only better after calibration.
DEFAULT_MODEL_PATH = "facebook/dinov3-vits16-pretrain-lvd1689m"  # resolved under public models root


class DinoV3Embedder:
    def __init__(self, model_path: str | None = None, *, device: str | None = None) -> None:
        if model_path is None:
            try:
                local = public_models_root() / DEFAULT_MODEL_PATH
                model_path = str(local) if local.is_dir() else DEFAULT_MODEL_PATH
            except RuntimeError:
                model_path = DEFAULT_MODEL_PATH
        self.model_path = model_path
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()  # lazy load races under parallel QA branches

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from vmem_bench.common.model_weights import hf_cache_dir
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache = hf_cache_dir()
        self._processor = AutoImageProcessor.from_pretrained(self.model_path, cache_dir=cache)
        self._model = (AutoModel.from_pretrained(self.model_path, cache_dir=cache)
                       .to(self._device).eval())

    def embed_image(self, image: Path) -> list[float]:
        self._ensure_loaded()
        from PIL import Image
        pil = Image.open(image).convert("RGB")
        with self._lock:  # serialize GPU inference across parallel QA branches
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                out = self._model(**inputs)
            cls = out.last_hidden_state[0, 0]
            cls = cls / cls.norm().clamp_min(1e-8)
            return [float(v) for v in cls.tolist()]

    def embed_batch(self, images: list[Path]) -> list[list[float]]:
        """Embed many crops in ONE forward pass (real GPU batching). A chunk's located entities
        are embedded together instead of N serialized single-image calls — the per-call lock +
        Python overhead and the GPU idle gap between calls dominate when N is small but repeated
        across chunks*branches. Falls back to [] for an empty list. Lock-protected like
        embed_image so parallel QA branches still serialize on the GPU (Pitfall_Notes: F5).
        Ponytail ceiling: grounder.ground is NOT batched here — GroundingDINO跨图异 phrase 的
        batch 语义复杂且易错，per-call lock 序列化保留；如需批量定位，升级路径是改用支持
        batched open-vocab detection 的后端。"""
        if not images:
            return []
        self._ensure_loaded()
        from PIL import Image
        pils = [Image.open(p).convert("RGB") for p in images]
        with self._lock:
            inputs = self._processor(images=pils, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                out = self._model(**inputs)
            cls = out.last_hidden_state[:, 0]  # (N, D) CLS tokens
            cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return [[float(v) for v in row.tolist()] for row in cls]
