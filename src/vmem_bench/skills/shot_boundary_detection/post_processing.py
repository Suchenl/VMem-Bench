from __future__ import annotations

import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vmem_bench.common.model_weights import hf_cache_dir

logger = logging.getLogger(__name__)


class DinoV3Refiner:
    """Refiner using DINOv3-ViT-S/16 to align candidate boundaries to exact frames."""

    def __init__(
        self,
        model_id: str = "${PUBLIC_MODELS_ROOT}/facebook/dinov3-vits16-pretrain-lvd1689m",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device_str = device
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel
        
        cache_dir = hf_cache_dir()
        logger.info(f"Loading DINOv3-ViT-S/16 from {self.model_id}...")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(self.model_id, cache_dir=cache_dir).to(self.device).eval()
        self.num_register_tokens = int(getattr(self._model.config, "num_register_tokens", 0) or 0)

    def refine_boundary(
        self,
        video_path: str | Path,
        candidate_frame_idx: int,
        fps: float,
        total_frames: int,
        window_size: int = 5,
        mode: str = "cls",
    ) -> int:
        """Refines a candidate frame index using local patch+cls similarity and first-order difference."""
        self._ensure_loaded()
        
        start_frame = max(0, candidate_frame_idx - window_size)
        end_frame = min(total_frames - 1, candidate_frame_idx + window_size)
        actual_window_len = end_frame - start_frame + 1
        
        if actual_window_len < 3:
            return candidate_frame_idx

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Failed to open video for refinement: {video_path}")
            return candidate_frame_idx

        frames_bgr = []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(actual_window_len):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
        finally:
            cap.release()

        if len(frames_bgr) < 3:
            return candidate_frame_idx

        resized = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR) for f in frames_bgr]
        batch = np.stack(resized, axis=0)
        tensor = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor[:, [2, 1, 0], :, :]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        inputs = {"pixel_values": tensor}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state
        B, T_tokens, D = last_hidden_state.shape
        
        cls_tokens = last_hidden_state[:, 0, :]
        cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
        
        patch_start = 1 + self.num_register_tokens
        patch_tokens = last_hidden_state[:, patch_start:, :]
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        
        cls_sims = torch.sum(cls_tokens[:-1] * cls_tokens[1:], dim=-1)
        patch_sims = torch.sum(patch_tokens[:-1] * patch_tokens[1:], dim=-1)
        patch_sims_mean = patch_sims.mean(dim=-1)
        
        if mode == "cls":
            similarities_tensor = cls_sims
        elif mode == "patch":
            similarities_tensor = patch_sims_mean
        elif mode == "combined":
            similarities_tensor = 0.3 * cls_sims + 0.7 * patch_sims_mean
        else:
            raise ValueError(f"Invalid refinement mode: {mode}.")
            
        similarities = similarities_tensor.cpu().tolist()

        disruption_scores = []
        for i in range(len(similarities)):
            sim_curr = similarities[i]
            sim_left = similarities[i - 1] if i > 0 else 1.0
            sim_right = similarities[i + 1] if i < len(similarities) - 1 else 1.0
            
            diff_left = max(0.0, sim_left - sim_curr)
            diff_right = max(0.0, sim_right - sim_curr)
            disruption = diff_left + diff_right
            disruption_scores.append(disruption)

        best_offset = int(np.argmax(disruption_scores))
        refined_frame_idx = start_frame + best_offset + 1
        
        logger.info(
            f"DINOv3 Refinement for candidate frame {candidate_frame_idx}: "
            f"Similarities: {[round(s, 4) for s in similarities]} -> "
            f"Refined to frame {refined_frame_idx}"
        )
        
        return refined_frame_idx
