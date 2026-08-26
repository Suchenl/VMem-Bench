"""SAM3 open-vocab concept segmentation (route B proposer).

Mirrored from pipeline_track_first; owned by S5 so the new pipeline stays decoupled.
"""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_common import (
    import_sam3_classes,
)
from vmem_bench.common.model_weights import public_models_root


class Sam3ConceptSegmenter:
    """``segment(image, concept)`` -> [(bbox_px xyxy, score, mask)] score-desc."""

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model_dir = model_dir or str(public_models_root() / "facebook/sam3")
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        Sam3Model, Sam3Processor = import_sam3_classes()
        import torch

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Absolute local dir under PUBLIC_MODELS_ROOT; force offline so HF_HUB_OFFLINE
        # does not attempt hub HEAD when tokenizer/processor wiring is incomplete.
        self._model = Sam3Model.from_pretrained(
            self.model_dir, dtype=torch.bfloat16, local_files_only=True
        ).to(self._device).eval()
        self._processor = Sam3Processor.from_pretrained(
            self.model_dir, local_files_only=True
        )

    def segment(self, image: Path, concept: str) -> list[tuple[list[float], float, object]]:
        return self.segment_multi(image, [concept])[concept]

    def segment_multi(
        self, image: Path, concepts: list[str]
    ) -> dict[str, list[tuple[list[float], float, object]]]:
        self._ensure_loaded()
        from PIL import Image

        uniq = list(dict.fromkeys(c for c in concepts if c and c.strip()))
        out: dict[str, list[tuple[list[float], float, object]]] = {c: [] for c in concepts}
        if not uniq:
            return out
        with self._lock:
            pil = Image.open(image).convert("RGB")
            inputs = self._processor(
                images=[pil] * len(uniq), text=uniq, return_tensors="pt"
            ).to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=[pil.size[::-1]] * len(uniq),
            )
        for concept, res in zip(uniq, results):
            boxes, scores, masks = res.get("boxes"), res.get("scores"), res.get("masks")
            found: list[tuple[list[float], float, object]] = []
            for i in range(0 if boxes is None else len(boxes)):
                mask = masks[i].cpu().numpy().astype(bool) if masks is not None else None
                found.append(([float(v) for v in boxes[i]], float(scores[i]), mask))
            found.sort(key=lambda item: -item[1])
            out[concept] = found
        return out
