"""SAM3 concept-segmentation wrapper (route B's only model dependency).

Loads Meta's SAM3 via HuggingFace transformers >= 5.9. The annotation client environments pin
older transformers (vLLM constraint), so a vendored copy is injected onto ``sys.path`` FIRST:
``MEMSTRATA_SAM3_DEPS`` env var, else ``<repo>/models/vendor/sam3_transformers59`` (gitignored;
rebuild: ``pip install --target <dir> --no-deps transformers==5.9.0 'huggingface_hub>=1.0'
mistral_common``). Weights resolve under the public models root (external weights allowed,
external code vendored — Principle 7).

Probe evidence (experiments/results/probes/sam3_exemplar_bbb): generic-concept segmentation
("animal") recovered 4/4 shots where GroundingDINO phrase detection yielded zero.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from vmem_bench.common.model_weights import public_models_root

_DEFAULT_MODEL_SUBDIR = "facebook/sam3"


def vendored_deps_dir() -> str:
    """Vendored SAM3-capable transformers dir (env override, else repo-local vendor)."""
    deps = os.environ.get("MEMSTRATA_SAM3_DEPS")
    if deps:
        return deps
    # src/vmem_bench/annotation/perception/ -> repo root is 6 up.
    repo = Path(__file__).resolve().parents[6]
    candidate = repo / "models" / "vendor" / "sam3_transformers59"
    return str(candidate) if candidate.is_dir() else ""


def _import_sam3_classes():
    """Import (Sam3Model, Sam3Processor), requiring a PROCESS-LEVEL consistent transformers.

    Hot-swapping transformers versions inside a live process corrupts model init (mixed
    modeling_utils across versions), so we refuse it: either the process already runs a
    SAM3-capable transformers (route B launcher prepends the vendored dir to PYTHONPATH),
    or transformers was never imported yet and we can safely prepend now."""
    try:
        from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415
        return Sam3Model, Sam3Processor
    except ImportError:
        pass
    deps = vendored_deps_dir()
    if not deps:
        raise RuntimeError(
            "SAM3 needs transformers>=5.9. Vendor it (pip install --target "
            "models/vendor/sam3_transformers59 --no-deps transformers==5.9.0 "
            "'huggingface_hub>=1.0' mistral_common) or set MEMSTRATA_SAM3_DEPS.")
    if "transformers" in sys.modules:
        raise RuntimeError(
            "This process already imported an older transformers; SAM3 cannot hot-swap it. "
            f"Launch with PYTHONPATH={deps} prepended (run_annotation.sh does this for "
            "PERCEPTION_BACKEND=sam3_track).")
    sys.path.insert(0, deps)
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415
    return Sam3Model, Sam3Processor


class Sam3ConceptSegmenter:
    """One loaded SAM3 model; ``segment(image, concept)`` -> [(bbox_px, score, mask)] desc.

    ``bbox_px`` is [x0, y0, x1, y1] in image pixels; ``mask`` is a bool HxW numpy array.
    Deterministic given the same weights/threshold. Thread-safe (single lock like GroundingDino).
    """

    def __init__(self, model_dir: str | None = None, *, threshold: float = 0.4,
                 mask_threshold: float = 0.5, device: str | None = None) -> None:
        self.model_dir = model_dir or str(public_models_root() / _DEFAULT_MODEL_SUBDIR)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        Sam3Model, Sam3Processor = _import_sam3_classes()
        import torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Sam3Model.from_pretrained(
            self.model_dir, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        self._processor = Sam3Processor.from_pretrained(
            self.model_dir, local_files_only=True)
        self._torch = torch
        self._device = device

    def segment(self, image: Path, concept: str) -> list[tuple[list[float], float, "object"]]:
        """All instances of ``concept`` in ``image``: [(bbox_px, score, bool mask)], score desc."""
        return self.segment_multi(image, [concept])[concept]

    def segment_multi(self, image: Path,
                      concepts: list[str]) -> dict[str, list[tuple[list[float], float, object]]]:
        """All instances of EVERY concept in ONE batched forward (frame decoded once).

        Per-shot tracking queries ~10 concepts per frame; independent forwards would dominate
        the run's wall time (triad: speed), while a batch of duplicated pixel tensors is cheap
        for an 886M model."""
        self._ensure_loaded()
        from PIL import Image
        uniq = list(dict.fromkeys(c for c in concepts if c and c.strip()))
        out: dict[str, list[tuple[list[float], float, object]]] = {c: [] for c in concepts}
        if not uniq:
            return out
        with self._lock:
            pil = Image.open(image).convert("RGB")
            inputs = self._processor(images=[pil] * len(uniq), text=uniq,
                                     return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
                target_sizes=[pil.size[::-1]] * len(uniq))
        for concept, res in zip(uniq, results):
            boxes, scores, masks = res.get("boxes"), res.get("scores"), res.get("masks")
            found: list[tuple[list[float], float, object]] = []
            for i in range(0 if boxes is None else len(boxes)):
                mask = masks[i].cpu().numpy().astype(bool) if masks is not None else None
                found.append(([float(v) for v in boxes[i]], float(scores[i]), mask))
            found.sort(key=lambda t: -t[1])
            out[concept] = found
        return out
