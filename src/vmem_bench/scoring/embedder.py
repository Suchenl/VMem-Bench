"""Bench-owned pinned image embedder for the v3 VisualFidelity headline (spec §4.1).

Self-contained on purpose: the scoring embedder is a *benchmark* artifact, pinned by the
benchmark and identical for every system on a scorecard. It must NOT be borrowed from any
SUT package — ``vmem_bench`` never imports ``memstrata`` (AGENTS.md Rule 2). The only
allowed out-of-tree read is the model-weights cache (source-code-decoupling rule §1), so a
local DINOv3 snapshot dir is loaded when available and otherwise the HF id is used.

Default: ``facebook/dinov3-vitb16-pretrain-lvd1689m`` (ViT-B/16, 768-d CLS token, unit-norm).
A dependency-free ``HashScoringEmbedder`` fallback keeps CPU smoke tests and offline runs
deterministic (its scores are meaningless as a visual metric — it exists only so the harness
never crashes when torch/weights are absent).
"""

from __future__ import annotations

import math
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

Vector = list[float]

DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Relative path of the pinned snapshot under a public model-weights root.
_DINOV3_VITB16_REL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _l2_normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(c * c for c in vec))
    return [c / norm for c in vec] if norm else list(vec)


class HashScoringEmbedder:
    """Deterministic, dependency-free fallback (NOT a real visual metric)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = "hash-scoring-fallback"

    def embed_image(self, image: str | Path) -> Vector:
        path = Path(image)
        seed = path.read_bytes() if path.exists() else str(image).encode("utf-8")
        out: Vector = []
        counter = 0
        while len(out) < self.dim:
            digest = sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(digest), 2):
                if len(out) >= self.dim:
                    break
                out.append(int.from_bytes(digest[i:i + 2], "big") / 65535.0 * 2.0 - 1.0)
            counter += 1
        return _l2_normalize(out)


class DinoV3ScoringEmbedder:
    """Pinned DINOv3 CLS-token embedder (transformers AutoModel), unit-norm output."""

    def __init__(self, model_id: str = DEFAULT_DINOV3_MODEL_ID, *, device: str | None = None) -> None:
        self.model_id = model_id
        self.name = f"dinov3-scoring:{model_id}"
        self._device = device
        self._model = None
        self._processor = None
        self._num_register_tokens = 0
        self.dim = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
        self.dim = int(self._model.config.hidden_size)
        self._num_register_tokens = int(getattr(self._model.config, "num_register_tokens", 0) or 0)

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        from PIL import Image

        torch = self._torch
        pil = Image.open(str(image)).convert("RGB")
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        cls = outputs.last_hidden_state[0][0]
        return _l2_normalize(cls.float().cpu().tolist())


# --- weights-root resolution (mirrors memstrata.lib.weights WITHOUT importing the SUT) ---
# AGENTS.md Rule 2 forbids importing ``memstrata`` here, so the model-weights cache is the
# only shared read (source-code-decoupling rule §1). We locate the real weights dir robustly
# by walking up to the first existing ``models/model_weights`` rather than trusting an
# AGENTS.md marker (both the subproject and the repo root carry one).

DEFAULT_SIGLIP2_MODEL_ID = os.environ.get(
    "MEMSTRATA_SIGLIP2_MODEL", "google/siglip2-base-patch16-224")
_SIGLIP2_REL = "google/siglip2-base-patch16-224"


def _weights_root() -> Path:
    override = os.environ.get("MEMSTRATA_WEIGHTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        cand = parent / "models" / "model_weights"
        if cand.is_dir():
            return cand
    return cur.parents[5] / "models" / "model_weights"


def _configure_torch_hub() -> None:
    root = _weights_root()
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))
    try:
        import torch
        torch.hub.set_dir(str(root / "torch_hub"))
    except Exception:  # noqa: BLE001 - torch may be absent on CPU smoke hosts
        pass


class Siglip2ScoringEmbedder:
    """Pinned SigLIP2 image-tower embedder (vision-language semantic axis), unit-norm.

    Orthogonal to DINOv3's self-supervised visual axis: SigLIP2 embeds in a language-aligned
    space, so it is the second general-purpose VisualFidelity column (decision D2)."""

    def __init__(self, model_id: str = DEFAULT_SIGLIP2_MODEL_ID, *, device: str | None = None) -> None:
        self.model_id = model_id
        self.name = f"siglip2-scoring:{model_id}"
        self._device = device
        self._model = None
        self._processor = None
        self.dim = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        from PIL import Image

        torch = self._torch
        pil = Image.open(str(image)).convert("RGB")
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = _l2_normalize(feats[0].float().cpu().tolist())
        self.dim = len(vec)
        return vec


class ArcFaceScoringEmbedder:
    """Pinned ArcFace (InsightFace) face-identity embedder for LSMDC real-person characters
    (decision D2). Raises on a face-less crop; the VisualScorer treats that as an N/A slot."""

    def __init__(self, pack: str = "buffalo_l", *, root: str | None = None, det_size: int = 640) -> None:
        self.pack = pack
        self.root = root
        self.det_size = det_size
        self.name = f"arcface-scoring:{pack}"
        self.dim = 512
        self._app = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # noqa: TRY003
            raise RuntimeError(
                "ArcFaceScoringEmbedder needs insightface + onnxruntime "
                "(pip install insightface onnxruntime)."
            ) from exc
        root = self.root or str(_weights_root() / "human_face_embedding")
        app = FaceAnalysis(name=self.pack, root=root)
        app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        self._app = app

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        import numpy as np
        from PIL import Image

        pil = Image.open(str(image)).convert("RGB")
        bgr = np.asarray(pil)[:, :, ::-1]
        faces = self._app.get(np.ascontiguousarray(bgr))
        if not faces:
            raise ValueError(f"no face detected in {image}")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return _l2_normalize([float(v) for v in face.normed_embedding])


class MegaLocScoringEmbedder:
    """Pinned MegaLoc (VPR) place embedder for the ``location`` consistency column (decision
    D2). Mirrors ``memstrata.encoders.place.vpr`` local-load, reading only the weights cache."""

    def __init__(self, *, weights: str | None = None, image_size: int = 322,
                 device: str | None = None) -> None:
        self.weights = weights
        self.image_size = image_size
        self.name = "megaloc-scoring"
        self.dim = 0
        self._device = device
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import sys as _sys

        import torch
        from safetensors.torch import load_file

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        _configure_torch_hub()
        wr = _weights_root()
        repo_dir = wr / "torch_hub" / "gmberton_MegaLoc_main"
        weights_file = Path(self.weights) if self.weights else (
            wr / "location_embedding" / "MegaLoc" / "model.safetensors")
        if not weights_file.is_file():
            base = wr / "hub" / "models--gberton--MegaLoc" / "snapshots"
            cands = list(base.glob("*/model.safetensors")) if base.is_dir() else []
            if cands:
                weights_file = cands[0]
        if not (repo_dir.is_dir() and weights_file.is_file()):
            raise RuntimeError(
                f"MegaLoc weights/repo not found under {wr} "
                "(need torch_hub/gmberton_MegaLoc_main + location_embedding/MegaLoc/model.safetensors)."
            )
        if str(repo_dir) not in _sys.path:
            _sys.path.insert(0, str(repo_dir))
        from megaloc_model import MegaLoc

        model = MegaLoc()
        model.load_state_dict(load_file(str(weights_file)))
        self._model = model.to(self._device).eval()

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        import numpy as np
        from PIL import Image

        torch = self._torch
        pil = Image.open(str(image)).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype="float32") / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
        std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        with torch.no_grad():
            descriptor = self._model(tensor)
        vec = _l2_normalize(descriptor.squeeze(0).float().cpu().tolist())
        self.dim = len(vec)
        return vec


def resolve_local_siglip2(model_id_or_weights: str | None = None) -> str:
    """Local SigLIP2 snapshot dir (offline-safe), mirroring DINOv3 local resolution.

    ``vmem_bench`` forces ``HF_HUB_OFFLINE`` on import, so a repo-id that is only in the
    HF cache (not an explicit local dir) still triggers an offline API reach and gets silently
    dropped from the VisualFidelity ``extra`` column. Resolve to a loadable local dir instead.
    Order: explicit local path/id arg → ``$PUBLIC_MODELS_ROOT/google/siglip2-…`` → the local
    HF hub snapshot under the weights cache → the bare HF id (needs network)."""
    if model_id_or_weights:
        cand = Path(model_id_or_weights)
        if cand.exists():
            return str(cand)
        # A non-default HF id the caller asked for explicitly: honor it verbatim.
        if model_id_or_weights != DEFAULT_SIGLIP2_MODEL_ID:
            return model_id_or_weights
    root = os.environ.get("PUBLIC_MODELS_ROOT")
    if root:
        cand = Path(root) / _SIGLIP2_REL
        if (cand / "config.json").is_file():
            return str(cand)
    hub = _weights_root() / "hub" / "models--google--siglip2-base-patch16-224" / "snapshots"
    if hub.is_dir():
        snaps = sorted(p for p in hub.glob("*") if (p / "config.json").is_file())
        if snaps:
            return str(snaps[0])
    return DEFAULT_SIGLIP2_MODEL_ID


def resolve_pinned_weights(weights: str | None = None) -> str | None:
    """Local DINOv3-ViT-B/16 snapshot dir, from explicit arg → $PUBLIC_MODELS_ROOT →
    $MEMSTRATA_SCORING_EMBEDDER_WEIGHTS → repo-local ``models/model_weights``.

    The final repo-local fallback matters: ``vmem_bench`` forces ``HF_HUB_OFFLINE`` on
    import, so returning None (→ bare HF id) makes ``from_pretrained`` crash offline and the
    scorer then silently drops ``redundancy_sim`` (this is exactly how the DINO column ended
    up empty on hosts where ``$PUBLIC_MODELS_ROOT`` was unset or pointed at a root without the
    snapshot). The weights cache is always mounted alongside the code, so prefer it over None."""
    if weights:
        return weights
    root = os.environ.get("PUBLIC_MODELS_ROOT")
    if root:
        cand = Path(root) / _DINOV3_VITB16_REL
        if (cand / "config.json").is_file():
            return str(cand)
    env_w = os.environ.get("MEMSTRATA_SCORING_EMBEDDER_WEIGHTS")
    if env_w:
        return env_w
    local = _weights_root() / _DINOV3_VITB16_REL
    if (local / "config.json").is_file():
        return str(local)
    return None


SCORING_PROVIDERS = ("dinov3", "siglip2", "arcface", "megaloc", "hash")


def build_scoring_embedder(provider: str = "dinov3", *, model_id: str | None = None,
                           weights: str | None = None) -> Any:
    """Construct a pinned scoring embedder by provider (decision D2 multi-embedder routing).

    ``dinov3`` / ``siglip2`` = general visual + vision-language axes; ``arcface`` = LSMDC
    real-person face identity; ``megaloc`` = location/VPR; ``hash`` = deterministic offline
    fallback. Each is bench-owned and identical for every system on a scorecard."""
    p = (provider or "dinov3").lower()
    if p == "hash":
        return HashScoringEmbedder()
    if p == "siglip2":
        return Siglip2ScoringEmbedder(model_id=resolve_local_siglip2(model_id or weights))
    if p == "arcface":
        return ArcFaceScoringEmbedder(root=weights) if weights else ArcFaceScoringEmbedder()
    if p == "megaloc":
        return MegaLocScoringEmbedder(weights=weights)
    if p != "dinov3":
        raise ValueError(f"unknown scoring provider {provider!r}; choose one of {SCORING_PROVIDERS}")
    resolved = resolve_pinned_weights(weights)
    mid = resolved or model_id or DEFAULT_DINOV3_MODEL_ID
    return DinoV3ScoringEmbedder(model_id=str(mid))
