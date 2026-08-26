"""Self-contained encoder substrate for the causal retrieval-family baselines.

This is a bench-owned port of the small subset of ``memstrata.encoders`` that the
segment/frame retrieval variants need, so ``vmem_bench`` never imports the SUT
package (``memstrata``) for baseline code (AGENTS.md Rule 2 + the benchmark
self-containment requirement). Only the model-weights cache is read out of tree
(source-code-decoupling rule §1); everything else lives here.

Providers
---------
* text   : ``hash`` (default, dependency-free) | ``qwen3_embedding`` (Qwen3-Embedding-4B,
           local snapshot or OpenAI-compatible ``/embeddings`` endpoint)
           | ``siglip2`` (cross-modal text tower)
* image  : ``hash`` (default) | ``siglip2`` (cross-modal image tower)
           | ``dinov3`` (keyframe diversity; reuses the bench scoring DINOv3 embedder)

All heavy backends are lazy (imported on first use) and process-cached by
``(kind, provider, model)`` so repeated pipeline construction does not reload weights.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    name: str
    dim: int

    def embed_image(self, image: str | Path) -> Vector: ...


@runtime_checkable
class TextEmbeddingModel(Protocol):
    name: str
    dim: int

    def embed_text(self, text: str) -> Vector: ...


def l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must share the same dimensionality")
    return sum(a * b for a, b in zip(left, right))


def _as_vector(value: Any) -> Vector:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [float(v) for v in value]


def _public_model_root() -> Path:
    root = os.environ.get("PUBLIC_MODELS_ROOT", "").strip()
    if not root:
        raise FileNotFoundError(
            "PUBLIC_MODELS_ROOT is not set; heavy embedding providers resolve local "
            "weights under this root. Set PUBLIC_MODELS_ROOT or pass an explicit model path."
        )
    return Path(root)


def _resolve_local_model(*, provider: str, model: str | None, default_rel: str, env_var: str) -> str:
    explicit = os.environ.get(env_var)
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"{provider} weights not found: {path}. Set {env_var} to a local snapshot.")
    if model and Path(model).expanduser().exists():
        return str(Path(model).expanduser())
    rel = model or default_rel
    cand = _public_model_root() / rel
    if cand.exists():
        return str(cand)
    if os.environ.get("MEMSTRATA_ALLOW_HF_DOWNLOAD") == "1":
        return rel
    raise FileNotFoundError(
        f"{provider} weights not found under PUBLIC_MODELS_ROOT: {cand}. "
        f"Set {env_var}, pass an explicit model path, or set MEMSTRATA_ALLOW_HF_DOWNLOAD=1."
    )


class HashEmbedding:
    """Deterministic, dependency-free fallback (text + image); NOT a real metric."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = "hash-fallback"

    def embed_image(self, image: str | Path) -> Vector:
        text = str(image)
        path = Path(text) if text else None
        seed = path.read_bytes() if (path is not None and path.is_file()) else text.encode("utf-8")
        components: Vector = []
        counter = 0
        while len(components) < self.dim:
            digest = sha256(seed + counter.to_bytes(4, "big")).digest()
            for index in range(0, len(digest), 2):
                if len(components) >= self.dim:
                    break
                raw = int.from_bytes(digest[index:index + 2], "big")
                components.append((raw / 65535.0) * 2.0 - 1.0)
            counter += 1
        return l2_normalize(components)

    def embed_text(self, text: str) -> Vector:
        return self.embed_image(f"text:{text}")

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)


class OpenAITextEmbedding:
    """OpenAI-compatible ``/embeddings`` client for a server-backed text encoder."""

    def __init__(self, *, endpoint: str, model: str, api_key: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = f"openai-embedding:{model}"
        self.dim = 0

    def _embed_many(self, texts: list[str]) -> list[Vector]:
        url = self.endpoint if self.endpoint.endswith("/embeddings") else f"{self.endpoint}/embeddings"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key or 'EMPTY'}"},
            method="POST",
        )
        timeout = float(os.environ.get("MEMSTRATA_EMBED_TIMEOUT", "120"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = sorted(data.get("data", []), key=lambda row: int(row.get("index", 0)))
        vectors = [l2_normalize([float(v) for v in row["embedding"]]) for row in rows]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors

    def embed_text(self, text: str) -> Vector:
        return self._embed_many([text])[0]

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)


class Qwen3TextEmbedding:
    """Qwen3-Embedding text encoder (last-token pooling + query instruct); lazy heavy imports."""

    INSTRUCT = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:"
    )

    def __init__(self, *, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.name = f"qwen3-embedding@{self.device}"
        self.dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)

    def _encode(self, texts: list[str]) -> list[Vector]:
        torch = self._torch
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            emb = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
        return [_as_vector(row) for row in emb]

    def embed_text(self, text: str) -> Vector:
        return self._encode([text])[0]

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(self.INSTRUCT + text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)


class SigLIP2Embedding:
    """SigLIP2 shared text/image encoder (cross-modal text→frame retrieval); lazy imports."""

    def __init__(self, *, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.name = f"siglip2@{self.device}"
        cfg = getattr(self.model, "config", None)
        self.dim = int(
            getattr(getattr(cfg, "text_config", None), "hidden_size", 0)
            or getattr(cfg, "projection_dim", 0)
            or 0
        )

    def embed_text(self, text: str) -> Vector:
        torch = self._torch
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length",
                                truncation=True).to(self.device)
        with torch.no_grad():
            feat = torch.nn.functional.normalize(self.model.get_text_features(**inputs).float(), dim=-1)
        return _as_vector(feat)

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)

    def embed_image(self, image: str | Path) -> Vector:
        torch = self._torch
        from PIL import Image

        try:
            img = Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001 - missing/corrupt crop -> stable hash handle
            return HashEmbedding(dim=self.dim or 64).embed_image(image)
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = torch.nn.functional.normalize(self.model.get_image_features(**inputs).float(), dim=-1)
        return _as_vector(feat)


_EMBEDDING_CACHE: dict[tuple[str, str, str | None], Any] = {}


def _siglip2_model_id(model: str | None) -> str:
    return _resolve_local_model(
        provider="siglip2", model=model,
        default_rel="google/siglip2-base-patch16-512", env_var="MEMSTRATA_SIGLIP2_WEIGHTS")


def build_text_embedding(*, provider: str = "hash", model: str | None = None) -> TextEmbeddingModel:
    backend = (provider or "hash").lower()
    key = ("text", backend, model)
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    if backend in {"qwen3", "qwen3_embedding", "qwen3-embedding"}:
        endpoint = os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT", "").strip()
        if endpoint:
            inst: Any = OpenAITextEmbedding(
                endpoint=endpoint,
                model=model or os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
                api_key=os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_API_KEY", ""),
            )
        else:
            mid = _resolve_local_model(
                provider="qwen3_embedding", model=model,
                default_rel="Qwen/Qwen3-Embedding-4B", env_var="MEMSTRATA_QWEN3_EMBEDDING_WEIGHTS")
            inst = Qwen3TextEmbedding(model_id=mid)
    elif backend == "siglip2":
        inst = SigLIP2Embedding(model_id=_siglip2_model_id(model))
    else:
        inst = HashEmbedding()
    _EMBEDDING_CACHE[key] = inst
    return inst


def build_image_embedding(*, provider: str = "hash", model: str | None = None) -> EmbeddingModel:
    backend = (provider or "hash").lower()
    key = ("image", backend, model)
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    if backend == "siglip2":
        inst: Any = SigLIP2Embedding(model_id=_siglip2_model_id(model))
    elif backend in {"dinov3", "dinov2"}:
        # Reuse the bench-owned pinned DINOv3 embedder (also weights-cache only, no SUT import).
        from vmem_bench.scoring.embedder import build_scoring_embedder
        inst = build_scoring_embedder("dinov3", model_id=model)
    else:
        inst = HashEmbedding()
    _EMBEDDING_CACHE[key] = inst
    return inst
