"""DINOv3 crop embedder for S5 identity (owned copy; no track_first import)."""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.model_weights import hf_cache_dir, public_models_root

# ViT-S until re-ID thresholds are recalibrated (same choice as track_first).
DEFAULT_MODEL_SUBDIR = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DinoV3Embedder:
    def __init__(self, model_path: str | None = None, *, device: str | None = None) -> None:
        if model_path is None:
            local = public_models_root() / DEFAULT_MODEL_SUBDIR
            model_path = str(local) if local.is_dir() else DEFAULT_MODEL_SUBDIR
        self.model_path = model_path
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._torch = torch
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            cache = hf_cache_dir()
            local_only = Path(self.model_path).is_dir()
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_path, cache_dir=cache, local_files_only=local_only
            )
            self._model = (
                AutoModel.from_pretrained(
                    self.model_path, cache_dir=cache, local_files_only=local_only
                )
                .to(self._device)
                .eval()
            )

    def embed_image(self, image: Path) -> list[float]:
        self._ensure_loaded()
        from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
            load_crop_rgb_for_model,
        )

        pil = load_crop_rgb_for_model(image)
        with self._lock:
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                out = self._model(**inputs)
            cls = out.last_hidden_state[0, 0]
            cls = cls / cls.norm().clamp_min(1e-8)
            return [float(v) for v in cls.tolist()]

    def embed_batch(self, images: list[Path]) -> list[list[float]]:
        if not images:
            return []
        self._ensure_loaded()
        from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.crop_io import (
            load_crop_rgb_for_model,
        )

        pils = [load_crop_rgb_for_model(path) for path in images]
        with self._lock:
            inputs = self._processor(images=pils, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                out = self._model(**inputs)
            cls = out.last_hidden_state[:, 0]
            cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return [[float(v) for v in row.tolist()] for row in cls]
