"""Route A perception backend: GroundingDINO detection + self-contained class-aware tracking.

Modality-agnostic (§3.0): open-vocab detection + IoU/appearance association works the same on
animation and live-action; style differences are parameters (box/text threshold, track_fps), not
code branches. Reuses this benchmark's own ``grounding.py`` (detect_all: all instances per frame),
``embedding.py`` (DINOv3 crop signature), and ``tracking.py`` (association) -- no external code.

Runs on GPU (H800); the model calls are isolated in the injected ``detector``/``embedder`` so the
association plumbing stays deterministic and testable with fakes.
"""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet, track_shot


def _keep_box(bbox: list[int], score: float, w: int, h: int,
              min_score: float, min_box_px: int) -> bool:
    """Drop junk detections before cropping/embedding: low score or too-small a crop in PIXELS.

    bbox is [ymin,xmin,ymax,xmax] on a 0-1000 grid; we convert to real pixels with the frame's
    (w, h) and require BOTH sides >= ``min_box_px``. Absolute px (not a frame fraction) is used on
    purpose: "too low-res to be a useful crop" is an absolute property of the cutout, so a 24 px box
    is junk on both 480p and 4K, whereas a fixed area-fraction would keep tiny boxes on huge frames
    and over-prune on small ones."""
    if score < min_score:
        return False
    y0, x0, y1, x1 = bbox
    box_w = max(0, x1 - x0) / 1000.0 * w
    box_h = max(0, y1 - y0) / 1000.0 * h
    return box_w >= min_box_px and box_h >= min_box_px


def _crop(pil, bbox: list[int], out_path: Path) -> Path:
    """Cut a [ymin,xmin,ymax,xmax] (0-1000) box from an ALREADY-open PIL frame. Returns out_path.

    Takes the decoded image (not a path) so a frame is decoded once per frame, not once per box."""
    w, h = pil.size
    y0, x0, y1, x1 = bbox
    box = (int(x0 / 1000 * w), int(y0 / 1000 * h), int(x1 / 1000 * w), int(y1 / 1000 * h))
    box = (min(box[0], w - 1), min(box[1], h - 1), max(box[2], box[0] + 1), max(box[3], box[1] + 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.crop(box).save(out_path)
    return out_path


class GdinoTrackBackend:
    """frames + roster -> tracklets via GroundingDINO(detect_all) + DINOv3 crop embeddings."""

    name = "gdino_track"

    def __init__(self, detector, embedder, *, crop_dir: Path,
                 track_iou_threshold: float = 0.3, track_min_len: int = 2,
                 appearance_gate: float = 0.0, max_miss: int = 1,
                 high_score: float = 0.5, use_motion: bool = True,
                 min_score: float = 0.0, min_box_px: int = 0) -> None:
        self.detector = detector      # grounding.GroundingDino (or a fake with .detect_all_multi)
        self.embedder = embedder      # embedding.DinoV3Embedder (or a fake with .embed_image)
        self.crop_dir = Path(crop_dir)
        self.track_iou_threshold = track_iou_threshold
        self.track_min_len = track_min_len
        self.appearance_gate = appearance_gate
        self.max_miss = max_miss
        self.high_score = high_score  # ByteTrack two-stage split threshold
        self.use_motion = use_motion  # constant-velocity prediction (off -> old greedy IoU behavior)
        self.min_score = min_score      # extra score floor beyond GDINO's own box_threshold
        self.min_box_px = min_box_px    # drop crops whose either pixel side < this (absolute px)

    def track_shot(self, frames: list[Frame], roster: list[RosterEntry], *,
                   next_track_id: int = 0) -> list[Tracklet]:
        from PIL import Image
        # Locations are scene-level, not tracked (§3.4); everything else grounds in ONE forward/frame.
        tracked = [e for e in roster if e.kind != "location"]
        phrases = [e.grounding_phrase for e in tracked]
        detections_by_frame: list[list[Detection]] = []
        for fr in frames:
            boxes_by_phrase = self.detector.detect_all_multi(fr.path, phrases)
            pil = Image.open(fr.path).convert("RGB")  # decode once per frame for all crops
            w, h = pil.size
            dets: list[Detection] = []
            for e_idx, entry in enumerate(tracked):
                for i, (bbox, score) in enumerate(boxes_by_phrase.get(entry.grounding_phrase, [])):
                    if not _keep_box(bbox, score, w, h, self.min_score, self.min_box_px):
                        continue  # skip junk before the expensive crop+embed
                    crop = _crop(pil, bbox,
                                 self.crop_dir / f"f{fr.frame_index:06d}_{e_idx}_{i}.jpg")
                    emb = self.embedder.embed_image(crop)
                    dets.append(Detection(frame_index=fr.frame_index, bbox=bbox, score=score,
                                          phrase=entry.grounding_phrase, embedding=emb,
                                          crop_path=str(crop)))
            detections_by_frame.append(dets)
        return track_shot(detections_by_frame, iou_threshold=self.track_iou_threshold,
                          appearance_gate=self.appearance_gate, max_miss=self.max_miss,
                          min_len=self.track_min_len, next_track_id=next_track_id,
                          high_score=self.high_score, use_motion=self.use_motion)
