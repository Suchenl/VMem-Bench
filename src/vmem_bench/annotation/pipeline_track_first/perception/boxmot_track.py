"""Ablation-only perception backend: GroundingDINO detection + industrial BoT-SORT (boxmot).

This exists ONLY for the paper's ablation "our deterministic bytetrack_local vs industrial BoT-SORT".
It is NEVER the default gold path: boxmot's BoT-SORT runs its own ReID model over full frames, so it
is nondeterministic and unfit for reproducible gold (design_principles §6/II, Q1 decision).

boxmot is an optional, pip-installed dependency (not vendored: it is stable third-party code we do
not modify -> importable per design_principles §7). If it is absent the backend raises a clear
install hint instead of failing obscurely. Detection + crop + DINOv3 embedding reuse the exact same
path as gdino_track so downstream re-ID/crop-QA are identical; only the ID association differs.
"""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import _crop, _keep_box
from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet, iou


class BoxmotBotsortBackend:
    """frames + roster -> tracklets via GroundingDINO(detect_all) + boxmot BoT-SORT id association."""

    name = "boxmot_botsort"

    def __init__(self, detector, embedder, *, crop_dir: Path, track_min_len: int = 2,
                 reid_weights: str | None = None, device: str = "cuda", half: bool = True,
                 min_score: float = 0.0, min_box_px: int = 0) -> None:
        self.detector = detector
        self.embedder = embedder
        self.crop_dir = Path(crop_dir)
        self.track_min_len = track_min_len
        self.reid_weights = reid_weights
        self.device = device
        self.half = half
        self.min_score = min_score
        self.min_box_px = min_box_px
        self._tracker = None

    def _ensure_tracker(self):
        if self._tracker is not None:
            return self._tracker
        try:
            from boxmot import BotSort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "perception_backend/tracker=boxmot_botsort needs the optional `boxmot` package "
                "(ablation only). Install on the GPU host: `pip install boxmot`. This backend is "
                "intentionally excluded from the default deterministic gold path."
            ) from exc
        import torch
        from pathlib import Path as _P
        weights = _P(self.reid_weights) if self.reid_weights else _P("osnet_x0_25_msmt17.pt")
        self._tracker = BotSort(reid_weights=weights, device=self.device,
                                half=self.half and torch.cuda.is_available())
        return self._tracker

    def track_shot(self, frames: list[Frame], roster: list[RosterEntry], *,
                   next_track_id: int = 0) -> list[Tracklet]:
        import numpy as np
        from PIL import Image
        tracker = self._ensure_tracker()
        # boxmot ids are per-tracker-instance; reset per shot so ids do not bleed across shots.
        if hasattr(tracker, "reset"):
            tracker.reset()
        by_id: dict[int, Tracklet] = {}
        id_offset = next_track_id
        tracked = [e for e in roster if e.kind != "location"]
        phrases = [e.grounding_phrase for e in tracked]
        for fr in frames:
            pil = Image.open(fr.path).convert("RGB")  # decode once per frame
            img = np.array(pil)
            h, w = img.shape[:2]
            boxes_by_phrase = self.detector.detect_all_multi(fr.path, phrases)  # one forward/frame
            dets: list[Detection] = []
            rows = []  # xyxy conf cls (pixel)
            for cls_idx, entry in enumerate(tracked):
                for i, (bbox, score) in enumerate(boxes_by_phrase.get(entry.grounding_phrase, [])):
                    if not _keep_box(bbox, score, w, h, self.min_score, self.min_box_px):
                        continue
                    crop = _crop(pil, bbox,
                                 self.crop_dir / f"f{fr.frame_index:06d}_{cls_idx}_{i}.jpg")
                    emb = self.embedder.embed_image(crop)
                    d = Detection(frame_index=fr.frame_index, bbox=bbox, score=score,
                                  phrase=entry.grounding_phrase, embedding=emb, crop_path=str(crop))
                    y0, x0, y1, x1 = bbox
                    rows.append([x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h,
                                 float(score), float(cls_idx)])
                    dets.append(d)
            if not dets:
                tracker.update(np.empty((0, 6)), img)
                continue
            tracks = tracker.update(np.asarray(rows, dtype=float), img)  # -> Nx(>=7): xyxy,id,conf,cls,...
            for row in tracks:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                # Match this output box back to an input Detection by IoU (version-robust).
                obox = [int(y1 / h * 1000), int(x1 / w * 1000), int(y2 / h * 1000), int(x2 / w * 1000)]
                best = max(dets, key=lambda d: iou(d.bbox, obox), default=None)
                if best is None or iou(best.bbox, obox) <= 0.0:
                    continue
                gid = id_offset + tid
                tl = by_id.get(gid)
                if tl is None:
                    tl = Tracklet(track_id=gid, phrase=best.phrase, detections=[])
                    by_id[gid] = tl
                tl.detections.append(best)
        out = [t for t in by_id.values() if len(t.detections) >= self.track_min_len]
        out.sort(key=lambda t: t.track_id)
        return out
