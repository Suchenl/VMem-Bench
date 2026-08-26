"""SAM3 geometric refinement of VLM-proposed crop boxes and points."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.geometry import (
    mask_to_bbox_norm,
    norm_to_px_xyxy,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.sam3_common import (
    import_sam3_classes,
)
from vmem_bench.common.model_weights import public_models_root


@dataclass(slots=True)
class RefinedMask:
    bbox_norm: list[int]
    score: float
    mask: object
    point_inside_mask: bool


class Sam3BoxPointRefiner:
    """Refine a VLM box using SAM3 and select a mask with the positive point.

    HF SAM3 currently exposes box prompts.  The VLM point is still consumed:
    after segmentation it selects the mask containing that point, falling back
    to the highest-score mask if no predicted mask contains it.
    """

    def __init__(
        self,
        *,
        model_dir: str | None = None,
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
        with self._lock:
            if self._model is not None:
                return
            Sam3Model, Sam3Processor = import_sam3_classes()
            import torch
            self._torch = torch
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = Sam3Model.from_pretrained(
                self.model_dir, dtype=torch.bfloat16, local_files_only=True
            ).to(self._device).eval()
            self._processor = Sam3Processor.from_pretrained(
                self.model_dir, local_files_only=True
            )

    def refine(self, *, image: Path, bbox_norm: list[int], point_norm: list[int]) -> RefinedMask | None:
        if len(bbox_norm) != 4 or len(point_norm) != 2:
            raise ValueError("bbox_norm and point_norm must contain 4 and 2 values")
        self._ensure_loaded()
        from PIL import Image

        pil = Image.open(image).convert("RGB")
        width, height = pil.size
        box = norm_to_px_xyxy(bbox_norm, width, height)
        # HF Sam3Processor expects [batch, num_boxes, xyxy] in pixel coordinates.
        with self._lock:
            inputs = self._processor(
                images=pil,
                input_boxes=[[box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(self._device)
            # Box prompts come in as float32; the model weights are bf16.
            model_dtype = next(self._model.parameters()).dtype
            for key, value in list(inputs.items()):
                if hasattr(value, "dtype") and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)
            try:
                with self._torch.no_grad():
                    outputs = self._model(**inputs)
                result = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=[pil.size[::-1]],
                )[0]
            except RuntimeError:
                return None
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or scores is None or len(scores) == 0:
            return None
        point_y = min(height - 1, max(0, round(point_norm[0] / 1000 * height)))
        point_x = min(width - 1, max(0, round(point_norm[1] / 1000 * width)))
        candidates: list[tuple[bool, float, object, list[int]]] = []
        for score, mask in zip(scores, masks):
            array = mask.cpu().numpy().astype(bool)
            tight = mask_to_bbox_norm(array)
            if tight is None:
                continue
            candidates.append((bool(array[point_y, point_x]), float(score), array, tight))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        inside, score, mask, tight = candidates[0]
        return RefinedMask(bbox_norm=tight, score=score, mask=mask, point_inside_mask=inside)
