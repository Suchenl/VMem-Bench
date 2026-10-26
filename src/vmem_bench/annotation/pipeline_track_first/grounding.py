"""Open-vocabulary grounding via GroundingDINO (workflow step 4).

The VLM proposes entity phrases; localization is delegated to a dedicated detector
(VLM bboxes are unreliable). Lazy singleton, loaded once per run (persistent-serving rule).
"""

from __future__ import annotations

import threading
from pathlib import Path

from vmem_bench.common.model_weights import hf_cache_dir

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


def _norm_phrase(phrase: str) -> str:
    return phrase.strip().lower().rstrip(".")


def _phrase_groups(phrases: list[str], max_chars: int = 320) -> list[list[str]]:
    """Split phrases into caption groups under a char budget (GroundingDINO's BERT caps ~256 tokens).

    Rosters here are small so this is usually one group / one forward; huge rosters degrade to a
    handful of forwards instead of one-per-phrase. Deterministic greedy packing."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for p in phrases:
        add = len(p) + 2  # ". "
        if cur and cur_len + add > max_chars:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += add
    if cur:
        groups.append(cur)
    return groups


def _match_phrase(label: str, phrases: list[str]) -> str | None:
    """Map a GroundingDINO per-box text label back to the roster phrase with max word overlap.

    Joint captioning returns a matched substring per box ("rabbit" for "white rabbit"); we attribute
    it to the phrase sharing the most lowercased tokens. Ties -> first phrase (deterministic). Zero
    overlap -> None (box dropped, unattributable). Ceiling: phrases sharing head nouns ("red squirrel"
    vs "flying squirrel") can tie on "squirrel"; keep entity phrases distinct to avoid it."""
    lab = set(label.lower().split())
    if not lab:
        return None
    best, best_score = None, 0
    for p in phrases:
        overlap = len(lab & set(p.split()))
        if overlap > best_score:
            best, best_score = p, overlap
    return best if best_score > 0 else None


class GroundingDino:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, device: str | None = None,
                 box_threshold: float = 0.30, text_threshold: float = 0.25) -> None:
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()  # lazy load races under parallel QA branches

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache = hf_cache_dir()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=cache, local_files_only=True
        )
        self._model = (AutoModelForZeroShotObjectDetection
                       .from_pretrained(self.model_id, cache_dir=cache, local_files_only=True)
                       .to(self._device).eval())

    def ground(self, image: Path, phrase: str) -> tuple[list[int], float] | None:
        """Best box for ``phrase`` as ([ymin, xmin, ymax, xmax] normalized to 0-1000, score)."""
        self._ensure_loaded()
        from PIL import Image
        pil = Image.open(image).convert("RGB")
        # GroundingDINO expects lower-cased phrases terminated by a period.
        text = phrase.strip().lower().rstrip(".") + "."
        with self._lock:  # serialize GPU inference across parallel QA branches
            inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[pil.size[::-1]])[0]
        if len(results["scores"]) == 0:
            return None
        best = int(results["scores"].argmax())
        score = float(results["scores"][best])
        x0, y0, x1, y1 = (float(v) for v in results["boxes"][best])
        w, h = pil.size
        bbox = [int(round(y0 / h * 1000)), int(round(x0 / w * 1000)),
                int(round(y1 / h * 1000)), int(round(x1 / w * 1000))]
        return bbox, score

    def detect_all(self, image: Path, phrase: str) -> list[tuple[list[int], float]]:
        """ALL boxes for ``phrase`` above threshold (not just argmax), for multi-instance tracking.

        ``ground``/``ground_batch`` return one best box (single-instance assumption); a tracking
        backend needs every instance in the frame (two squirrels -> two boxes -> two tracklets).
        Returns ``[(bbox 0-1000, score), ...]`` sorted by descending score (deterministic)."""
        self._ensure_loaded()
        from PIL import Image
        pil = Image.open(image).convert("RGB")
        text = phrase.strip().lower().rstrip(".") + "."
        with self._lock:
            inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[pil.size[::-1]])[0]
        w, h = pil.size
        out: list[tuple[list[int], float]] = []
        for score, box in zip(results["scores"], results["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box)
            bbox = [int(round(y0 / h * 1000)), int(round(x0 / w * 1000)),
                    int(round(y1 / h * 1000)), int(round(x1 / w * 1000))]
            out.append((bbox, float(score)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def detect_all_multi(self, image: Path,
                         phrases: list[str]) -> dict[str, list[tuple[list[int], float]]]:
        """ALL boxes for MANY phrases in ONE forward per frame (cross-entity batching).

        The old ``gdino_track`` path called ``detect_all`` once per phrase -> ``N_phrases`` GPU
        forwards per frame, which was the per-shot bottleneck (128 shots x ~15 frames x ~20 phrases).
        GroundingDINO natively grounds a multi-phrase caption ("a. b. c.") in a single forward and
        returns a matched ``text_labels`` per box; we map each box back to the roster phrase by word
        overlap (``_match_phrase``). Returns ``{original_phrase: [(bbox 0-1000, score), ...] desc}``,
        one entry per input phrase (phrases that normalize alike share the same box list, matching the
        old per-phrase behavior). Lock-protected like ``detect_all``. Ceiling: caption is split into
        groups under a token budget (``_phrase_groups``) so huge rosters cost a few forwards, not one
        per phrase."""
        self._ensure_loaded()
        from PIL import Image
        # Preserve caller's exact keys; dedup by normalized form so one forward covers repeats.
        norm = {p: _norm_phrase(p) for p in phrases if p and p.strip()}
        uniq = list(dict.fromkeys(norm.values()))
        by_norm: dict[str, list[tuple[list[int], float]]] = {n: [] for n in uniq}
        result: dict[str, list[tuple[list[int], float]]] = {p: [] for p in phrases}
        if not uniq:
            return result
        pil = Image.open(image).convert("RGB")
        w, h = pil.size
        for group in _phrase_groups(uniq):
            text = " ".join(p + "." for p in group)
            with self._lock:
                inputs = self._processor(images=pil, text=text,
                                         return_tensors="pt").to(self._device)
                with self._torch.no_grad():
                    outputs = self._model(**inputs)
                res = self._processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.box_threshold,
                    text_threshold=self.text_threshold, target_sizes=[pil.size[::-1]])[0]
            labels = res.get("text_labels")
            if labels is None:
                labels = res.get("labels", [])
            for score, box, label in zip(res["scores"], res["boxes"], labels):
                phrase = _match_phrase(str(label), group)
                if phrase is None:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box)
                bbox = [int(round(y0 / h * 1000)), int(round(x0 / w * 1000)),
                        int(round(y1 / h * 1000)), int(round(x1 / w * 1000))]
                by_norm[phrase].append((bbox, float(score)))
        for n in by_norm:
            by_norm[n].sort(key=lambda t: t[1], reverse=True)
        for orig, n in norm.items():
            result[orig] = list(by_norm[n])
        return result

    def ground_batch(self, images: list[Path], phrase: str) -> list[tuple[list[int], float] | None]:
        """Best box for ``phrase`` across many images in ONE forward pass.

        HF GroundingDINO supports batched images + per-image text via
        ``processor(images=list, text=list, padding=True)`` — do NOT set ``padding_side='left'``,
        it breaks batched text of different lengths (HF issue #34346). Same phrase across frames
        is the common ``_ground_and_crop`` pattern (one entity across N sampled frames); this cuts
        N serialized single-image forwards per entity to one. Returns one ``(bbox, score) | None``
        per input image, aligned order. Lock-protected like ``ground`` so parallel QA branches
        still serialize on the GPU. Ponytail ceiling: cross-ENTITY batching (M entities × N frames
        in one forward) would need decoupling grounding from per-entity crop logic in
        ``_ground_and_crop``; left for later — per-entity frame batching already removes the
        N× per-entity forward cost."""
        if not images:
            return []
        self._ensure_loaded()
        from PIL import Image
        pils = [Image.open(p).convert("RGB") for p in images]
        text = phrase.strip().lower().rstrip(".") + "."
        with self._lock:
            inputs = (self._processor(images=pils, text=[text] * len(pils), padding=True,
                                      return_tensors="pt").to(self._device))
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1] for pil in pils])
        out: list[tuple[list[int], float] | None] = []
        for pil, res in zip(pils, results):
            if len(res["scores"]) == 0:
                out.append(None)
                continue
            best = int(res["scores"].argmax())
            score = float(res["scores"][best])
            x0, y0, x1, y1 = (float(v) for v in res["boxes"][best])
            w, h = pil.size
            bbox = [int(round(y0 / h * 1000)), int(round(x0 / w * 1000)),
                    int(round(y1 / h * 1000)), int(round(x1 / w * 1000))]
            out.append((bbox, score))
        return out
