"""Route B perception backend: exemplar-grounded discriminative tracking (segment-then-match).

Why this exists (probe: experiments/results/probes/sam3_exemplar_bbb): route A hands a
VLM-invented phrase to GroundingDINO, and when the two models' vocabularies disagree the entity
silently vanishes (BBB v10: "red fox" -> 0 tracklets while the creature filled the frame).
Route B removes language from perception entirely:

1. SAM3 segments every instance of a GENERIC concept per frame ("animal" for characters —
   category words carry no per-individual identity, so they cannot mis-name anyone).
2. Each candidate crop is embedded with DINOv3 and matched against the roster entities'
   exemplar crops; argmax similarity above a floor assigns the instance, else it is dropped.
3. Assigned detections reuse the shared ByteTrack-style association (``tracking.track_shot``)
   so both routes emit identical Tracklet structures to re-ID and everything downstream.

Props keep their class-level head noun as the SAM3 concept (a prop's identity IS its class);
locations are scene-level and not tracked (same as route A).
"""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet, track_shot
from vmem_bench.common.vecmath import cosine_similarity

# Generic concept words for character discovery. Category-level on purpose: SAM3 enumerates
# instances, exemplar similarity decides WHO each one is. "person" covers live-action films.
DEFAULT_CHARACTER_CONCEPTS = ("animal", "person")


def head_noun(phrase: str) -> str:
    """Last word of a grounding phrase — the class word ("red apple" -> "apple")."""
    words = [w for w in phrase.replace("_", " ").split() if w]
    return words[-1] if words else phrase


def mask_to_bbox(mask) -> list[int] | None:
    """Tight [ymin,xmin,ymax,xmax] (0-1000 normalized) of a boolean HxW mask, or None if empty.

    Pure geometry (numpy), deterministic -> unit-testable without SAM3."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    h, w = m.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return [round(y0 / h * 1000), round(x0 / w * 1000),
            round(y1 / h * 1000), round(x1 / w * 1000)]


def px_to_norm_bbox(box_px: list[float], width: int, height: int) -> list[int]:
    """SAM3 pixel [x0,y0,x1,y1] -> the pipeline's [ymin,xmin,ymax,xmax] 0-1000 convention."""
    x0, y0, x1, y1 = box_px
    return [round(y0 / height * 1000), round(x0 / width * 1000),
            round(y1 / height * 1000), round(x1 / width * 1000)]


def assign_by_exemplar(candidate_vec, exemplars: dict[str, list],
                       *, sim_floor: float) -> tuple[str, float] | None:
    """argmax cosine over exemplar vectors; None when nothing clears the floor.

    Each entity may carry SEVERAL exemplar vectors (multi-view anchors); its similarity is the
    max over them, so a back-view anchor rescues candidates a front-view anchor would miss."""
    best_phrase, best_sim = None, -2.0
    for phrase, vecs in exemplars.items():
        if vecs and not isinstance(vecs[0], (list, tuple)):
            vecs = [vecs]  # single flat vector (legacy callers/tests)
        sim = max((cosine_similarity(list(candidate_vec), list(v)) for v in vecs),
                  default=-2.0)
        if sim > best_sim:
            best_phrase, best_sim = phrase, sim
    if best_phrase is None or best_sim < sim_floor:
        return None
    return best_phrase, best_sim


class Sam3ExemplarTrackBackend:
    """frames + roster (with exemplar crops) -> tracklets, language-free identity."""

    name = "sam3_track"

    def __init__(self, embedder, *, crop_dir: Path, segmenter=None,
                 character_concepts: tuple[str, ...] = DEFAULT_CHARACTER_CONCEPTS,
                 exemplar_sim_floor: float = 0.28, seg_threshold: float = 0.4,
                 track_iou_threshold: float = 0.3, track_min_len: int = 2,
                 appearance_gate: float = 0.0, max_miss: int = 1,
                 high_score: float = 0.0, use_motion: bool = True,
                 min_box_px: int = 12) -> None:
        self.embedder = embedder
        self.crop_dir = Path(crop_dir)
        self.character_concepts = tuple(character_concepts)
        self.exemplar_sim_floor = exemplar_sim_floor
        self.track_iou_threshold = track_iou_threshold
        self.track_min_len = track_min_len
        self.appearance_gate = appearance_gate
        self.max_miss = max_miss
        self.high_score = high_score
        self.use_motion = use_motion
        self.min_box_px = min_box_px
        if segmenter is None:
            from vmem_bench.annotation.pipeline_track_first.perception.sam3_seg import Sam3ConceptSegmenter
            segmenter = Sam3ConceptSegmenter(threshold=seg_threshold)
        self.segmenter = segmenter
        self._exemplar_cache: dict[str, list[float]] = {}

    def _exemplar_vectors(self, roster: list[RosterEntry]) -> dict[str, list[list[float]]]:
        """phrase -> exemplar embeddings. ``exemplar_crop`` may hold several ';'-separated
        multi-view anchor paths (see vlm_grounding.augment_exemplars)."""
        out: dict[str, list[list[float]]] = {}
        for entry in roster:
            if entry.kind == "location" or not entry.exemplar_crop:
                continue
            vecs: list[list[float]] = []
            for raw in entry.exemplar_crop.split(";"):
                path = Path(raw.strip())
                if not raw.strip() or not path.is_file():
                    continue
                key = str(path)
                if key not in self._exemplar_cache:
                    self._exemplar_cache[key] = list(self.embedder.embed_image(path))
                vecs.append(self._exemplar_cache[key])
            if vecs:
                out[entry.grounding_phrase] = vecs
        return out

    def track_shot(self, frames: list[Frame], roster: list[RosterEntry], *,
                   next_track_id: int = 0) -> list[Tracklet]:
        from PIL import Image
        characters = [e for e in roster if e.kind == "character"]
        props = [e for e in roster if e.kind == "prop"]
        char_exemplars = self._exemplar_vectors(characters)
        prop_exemplars = self._exemplar_vectors(props)
        # One concept -> the prop phrases sharing that head noun (class identity).
        prop_concepts: dict[str, list[str]] = {}
        for e in props:
            prop_concepts.setdefault(head_noun(e.grounding_phrase), []).append(e.grounding_phrase)

        detections_by_frame: list[list[Detection]] = []
        for fr in frames:
            pil = Image.open(fr.path).convert("RGB")
            w, h = pil.size
            dets: list[Detection] = []
            seen_boxes: list[tuple[int, ...]] = []

            def add(bbox_px, score, tag, *, exemplars=None, fallback_phrase=None,
                    sim_floor=None) -> None:
                """Crop+embed one candidate; resolve identity via ``exemplars`` (argmax cosine),
                else use ``fallback_phrase``; drop candidates that resolve to nothing."""
                x0, y0, x1, y1 = bbox_px
                if (x1 - x0) < self.min_box_px or (y1 - y0) < self.min_box_px:
                    return
                bbox = px_to_norm_bbox(bbox_px, w, h)
                key = tuple(round(v / 20) for v in bbox)  # coarse dedup across concepts
                if key in seen_boxes:
                    return
                seen_boxes.append(key)
                crop = self.crop_dir / f"f{fr.frame_index:06d}_{tag}.jpg"
                crop.parent.mkdir(parents=True, exist_ok=True)
                pil.crop((int(x0), int(y0), int(x1), int(y1))).save(crop)
                emb = list(self.embedder.embed_image(crop))
                phrase = fallback_phrase
                if exemplars:
                    match = assign_by_exemplar(
                        emb, exemplars,
                        sim_floor=self.exemplar_sim_floor if sim_floor is None else sim_floor)
                    if match is not None:
                        phrase = match[0]
                    elif fallback_phrase is None:
                        return  # character candidate nobody claims -> background/extra, drop
                if phrase is None:
                    return
                dets.append(Detection(frame_index=fr.frame_index, bbox=bbox, score=float(score),
                                      phrase=phrase, embedding=emb, crop_path=str(crop)))

            char_concepts = list(self.character_concepts) if (characters and char_exemplars) \
                else []
            all_concepts = char_concepts + list(prop_concepts)
            if hasattr(self.segmenter, "segment_multi"):
                by_concept = self.segmenter.segment_multi(fr.path, all_concepts)
            else:  # test fakes / simple segmenters
                by_concept = {c: self.segmenter.segment(fr.path, c) for c in all_concepts}
            for ci, concept in enumerate(char_concepts):
                for bi, (bbox_px, score, _mask) in enumerate(by_concept.get(concept, [])):
                    add(bbox_px, score, f"c{ci}_{bi}", exemplars=char_exemplars)
            for concept, phrases in prop_concepts.items():
                # Class identity: the head noun IS the prop's identity. With several same-noun
                # phrases and exemplars, similarity picks among them (floor-free: it is a choice
                # between known props, not an accept/reject gate).
                subset = {p: v for p, v in prop_exemplars.items() if p in phrases}
                for bi, (bbox_px, score, _mask) in enumerate(by_concept.get(concept, [])):
                    add(bbox_px, score, f"p{head_noun(concept)}_{bi}",
                        exemplars=(subset if len(phrases) > 1 and subset else None),
                        fallback_phrase=phrases[0], sim_floor=-1.0)
            # Fusion hook (v12): extra proposers contribute candidates through the SAME
            # discriminative assignment — a proposal's origin never decides identity.
            for bi, (bbox_px, score, is_char, fallback) in enumerate(
                    self._extra_proposals(fr, characters, props)):
                if is_char:
                    add(bbox_px, score, f"x{bi}", exemplars=char_exemplars)
                else:
                    add(bbox_px, score, f"x{bi}", fallback_phrase=fallback)
            detections_by_frame.append(dets)

        return track_shot(detections_by_frame, iou_threshold=self.track_iou_threshold,
                          appearance_gate=self.appearance_gate, max_miss=self.max_miss,
                          min_len=self.track_min_len, next_track_id=next_track_id,
                          high_score=self.high_score, use_motion=self.use_motion)

    def _extra_proposals(self, fr: Frame, characters: list[RosterEntry],
                         props: list[RosterEntry],
                         ) -> list[tuple[list[float], float, bool, str | None]]:
        """Fusion hook: [(bbox_px, score, is_character, prop_fallback_phrase)]. Base: none."""
        return []


# Back-compat alias: factory historically imported Sam3TrackBackend.
Sam3TrackBackend = Sam3ExemplarTrackBackend
