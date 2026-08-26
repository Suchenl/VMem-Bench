"""Minimal GroundingDINO proposer for S5 route B (owned copy; no track_first import)."""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.model_weights import hf_cache_dir

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


class GroundingDinoProposer:
    """Open-vocab detection returning [ymin,xmin,ymax,xmax] 0-1000 boxes."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
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
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._torch = torch
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            cache = hf_cache_dir()
            # Offline GPU nodes: never hit huggingface.co; hub snapshot lives under
            # Montage models/model_weights/hub (see common.model_weights.repo_root).
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=cache, local_files_only=True
            )
            self._model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.model_id, cache_dir=cache, local_files_only=True
                )
                .to(self._device)
                .eval()
            )

    def detect_all(self, image: Path, phrase: str) -> list[tuple[list[int], float]]:
        """All boxes for ``phrase`` above threshold, score-desc."""
        self._ensure_loaded()
        from PIL import Image

        pil = Image.open(image).convert("RGB")
        text = phrase.strip().lower().rstrip(".") + "."
        with self._lock:
            inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
        width, height = pil.size
        out: list[tuple[list[int], float]] = []
        for score, box in zip(results["scores"], results["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box)
            bbox = [
                int(round(y0 / height * 1000)),
                int(round(x0 / width * 1000)),
                int(round(y1 / height * 1000)),
                int(round(x1 / width * 1000)),
            ]
            out.append((bbox, float(score)))
        out.sort(key=lambda item: item[1], reverse=True)
        return out
