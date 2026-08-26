"""Self-gating face cue for re-ID (annotation_tracking_internals.md §3.7).

Key design (addresses the user's concern about "classify is-it-a-face, THEN call ArcFace"):
there is NO separate "is this a face?" classifier and NO ``if style == live_action`` branch. We
run one face detector+encoder (InsightFace ``buffalo_l``: RetinaFace detect + ArcFace 512d in one
forward) directly on a *character* crop. The detector returning zero boxes IS the gate -- a crop
with no (or a stylized/undetectable) face yields ``None`` and the face cue simply drops out of the
re-ID fusion (reid.fuse_similarity renormalizes over the remaining cues). So:
  - efficiency: one model call, only on character crops (props/locations never call it);
  - robustness: a mis-detected face only adds/removes one *weighted cue*, it never flips control
    flow the way a wrong "is-face" boolean would; body appearance stays the main cue.

Runs on GPU (H800). Weights load under ``weights_root()/human_face_embedding`` (Principle 7:
external weights allowed). The face-selection policy is a pure function, unit-testable offline.
"""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.model_weights import weights_root


def _largest_face(faces: list):
    """Pick the largest-area detected face (most likely the crop's subject, most pixels for ArcFace).

    ``faces`` are InsightFace Face objects with ``.bbox = [x0,y0,x1,y1]`` and ``.normed_embedding``.
    Returns the chosen face or None if the list is empty. Pure -> offline-testable with fakes."""
    if not faces:
        return None
    def area(f):
        x0, y0, x1, y1 = f.bbox
        return max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    return max(faces, key=area)


class FaceEncoder:
    """InsightFace detect-is-the-gate encoder: crop -> largest face's ArcFace vector, or None."""

    def __init__(self, model_name: str = "buffalo_l", *, device: str | None = None,
                 det_size: int = 640) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self._device = device
        self._app = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            from insightface.app import FaceAnalysis
            root = str(weights_root() / "human_face_embedding")
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if self._device != "cpu" else ["CPUExecutionProvider"])
            app = FaceAnalysis(name=self.model_name, root=root, providers=providers)
            ctx = 0 if self._device != "cpu" else -1
            app.prepare(ctx_id=ctx, det_size=(self.det_size, self.det_size))
            self._app = app

    def encode(self, crop_path: Path) -> list[float] | None:
        """Largest detected face's normed ArcFace embedding, or None when no face is detected."""
        self._ensure_loaded()
        import cv2
        img = cv2.imread(str(crop_path))
        if img is None:
            return None
        with self._lock:
            faces = self._app.get(img)
        face = _largest_face(faces)
        if face is None:
            return None
        emb = face.normed_embedding
        return [float(v) for v in emb]
