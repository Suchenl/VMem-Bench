"""Per-crop QA: drop mixed-class representations before they pollute the asset bank (§3.3, step 5).

Replaces the old per-crop VLM audit (one VLM call per crop) with a deterministic zero-shot check.
The two methods use FUNDAMENTALLY DIFFERENT model families and answer DIFFERENT questions -- they
are NOT interchangeable and must not be conflated:

  - ``prototype`` (default, cheapest-reliable-tool) -- IMAGE<->IMAGE, self-supervised DINOv3.
    Question: "does this crop's *appearance* look like the OTHER crops we already grouped under
    entity X, or is it closer to entity Y's crops?" This is a pure vision-only similarity between
    the crop embedding and each entity's mean-crop prototype (no text ever enters). It catches
    re-ID over-merges / detection false-positives -- an *identity/appearance* error. DINO is the
    right tool here precisely because the comparison is image-to-image. Reuses the crop embeddings
    re-ID already computed; no new model, fully offline-testable.

  - ``siglip`` (alternative/ablation) -- IMAGE<->TEXT, SigLIP2 (a CLIP-family image-text aligner).
    Question: "does this crop's *semantics* match the roster's text phrase for entity X (e.g. 'a
    rabbit') better than another phrase?" This scores the crop image against text labels, so it
    catches *class/label* errors (a 'dog' crop filed under the 'cat' phrase). This is a text-image
    task, so a DINO-style image encoder is the WRONG tool -- you need an aligned image-text model
    (SigLIP2 / CLIP / OpenCLIP / jina-clip-v2). GPU/weights, isolated behind a lazy singleton.

Rule of thumb: use DINO (image<->image) to answer "same instance/appearance?"; use SigLIP/CLIP
(image<->text) to answer "matches this word?". Both are ~2 orders of magnitude cheaper than a VLM
call and reproducible. VLM is only an optional gray-zone tiebreak (not wired by default).
"""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.vecmath import cosine_similarity
from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first.reid import _entity_signature


def nearest_prototype(vec: list[float], prototypes: dict[str, list[float]]
                      ) -> tuple[str | None, float]:
    """(argmax entity_id, cosine) of ``vec`` over ``prototypes``; (None, -1) when empty. Pure."""
    best_eid, best = None, -2.0
    for eid, proto in prototypes.items():
        s = cosine_similarity(vec, proto)
        if s > best:
            best, best_eid = s, eid
    return best_eid, (best if best_eid is not None else -1.0)


def reassign_label(ranked: list[tuple[str, float]], current: str, margin: float) -> str:
    """Return the top label iff it beats ``current`` by at least ``margin``; else ``current``.

    ``ranked`` is ``[(label, prob), ...]`` sorted desc (as from SiglipCropClassifier.classify).
    Pure / unit-testable; does not mutate inputs.
    """
    if not ranked:
        return current
    top_label, top_prob = ranked[0]
    if top_label == current:
        return current
    cur_prob = next((p for lab, p in ranked if lab == current), 0.0)
    if top_prob - cur_prob >= margin:
        return top_label
    return current


def audit_registry_crops(registry: Registry, *, margin: float = 0.05) -> list[str]:
    """Flag grounded representations that look more like another same-kind entity (§3.3).

    Prototype = mean of an entity's stored body embeddings (includes the rep under test, so a
    singleton entity can never be flagged against itself -- correct: there is nothing to compare).
    A rep is flagged iff the nearest same-kind prototype belongs to a DIFFERENT entity AND beats
    the rep's own-entity cosine by more than ``margin``. Only grounded crops (tracker/grounding_dino
    /sam3) are audited; full-frame location crops carry scene content, not an instance, so their
    cosine is a scene-similarity, not an identity-similarity. Deterministic -> unit-testable.

    ponytail: the own-entity prototype includes the rep under test, mildly inflating its self-score
    (a leave-one-out prototype would be stricter); adequate because a truly mixed crop is still
    closer to the other entity's multi-member prototype. Upgrade path: leave-one-out signatures.
    """
    grounded = {"grounding_dino", "tracker", "sam3"}
    # Precompute per-kind entity prototypes once.
    proto_by_kind: dict[str, dict[str, list[float]]] = {}
    for eid, ent in registry.entities.items():
        sig = _entity_signature(ent, registry.embeddings)
        if sig is not None:
            proto_by_kind.setdefault(ent.kind, {})[eid] = sig

    flagged: list[str] = []
    for eid, ent in registry.entities.items():
        protos = proto_by_kind.get(ent.kind, {})
        if len(protos) < 2:  # nothing to be confused with in this kind
            continue
        own_proto = protos.get(eid)
        if own_proto is None:
            continue
        for rep in ent.representations:
            if rep.bbox_source not in grounded:
                continue
            vec = registry.embeddings.get(rep.embedding_key)
            if vec is None:
                continue
            best_eid, best = nearest_prototype(vec, protos)
            own = cosine_similarity(vec, own_proto)
            if best_eid is not None and best_eid != eid and best - own > margin:
                flagged.append(rep.representation_id)
    return flagged


class SiglipCropClassifier:
    """Zero-shot image-text crop classifier (SigLIP2). Alternative to the prototype auditor.

    ``classify(crop, labels)`` returns [(label, prob), ...] sorted desc. Lazy singleton; loads
    weights via the HF cache under models/model_weights (Principle 7: external weights allowed)."""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-512",
                 *, device: str | None = None) -> None:
        self.model_id = model_id
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
            from transformers import AutoModel, AutoProcessor
            from vmem_bench.common.model_weights import hf_cache_dir
            self._torch = torch
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            cache = hf_cache_dir()
            self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache)
            self._model = AutoModel.from_pretrained(self.model_id, cache_dir=cache).to(
                self._device).eval()

    def classify(self, crop: Path, labels: list[str]) -> list[tuple[str, float]]:
        self._ensure_loaded()
        from PIL import Image
        pil = Image.open(crop).convert("RGB")
        texts = [f"a photo of {l}" for l in labels]
        with self._lock:
            inputs = self._processor(text=texts, images=pil, return_tensors="pt",
                                     padding="max_length").to(self._device)
            with self._torch.no_grad():
                logits = self._model(**inputs).logits_per_image[0]
                probs = self._torch.sigmoid(logits)
        ranked = sorted(zip(labels, (float(p) for p in probs)), key=lambda t: t[1], reverse=True)
        return ranked
