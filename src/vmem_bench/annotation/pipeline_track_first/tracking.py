"""Intra-shot multi-object tracking (track-first redesign, docs/benchmark/annotation_tracking_internals.md step 3).

Within a shot the frames are contiguous, so identity is established *deterministically* by
tracking detections across sampled frames into tracklets -- one tracklet = one object instance
in that shot, with a stable local ``track_id``. This replaces the old "VLM re-discovers and
re-names entities every chunk" path that fragmented identity (Pitfall_Notes / baseline probe:
23% duplicate ids).

Self-contained deterministic tracker (ponytail: no new dependency). It absorbs the two ideas from
ByteTrack / BoT-SORT that actually matter for our low-fps film case, WITHOUT taking their
dependency (which would run its own ReID model on full frames -> nondeterministic, unfit for gold):

  1. Two-stage association (ByteTrack): first match tracks to HIGH-score detections, then recover
     with the LOW-score leftovers. Low-score boxes extend existing tracks but never start new ones,
     so a briefly low-confidence frame no longer breaks a track (fewer fragments for re-ID to fix).
  2. Constant-velocity motion prediction (the Kalman idea, linearised): a track's box for the next
     sampled frame is predicted from its last two boxes, and IoU is taken against that PREDICTION.
     At ~3 fps a moving subject can shift a lot between samples; matching to the raw last box breaks
     (IoU->0), matching to the predicted box holds.
  3. Appearance fusion (BoT-SORT): DINOv3 crop cosine stays an association gate + tiebreak.

Association is class-constrained (the open-vocab detector tags each box with its phrase, so a
"grey rabbit" track only ever eats "grey rabbit" boxes -- the hardest cross-class error is
structurally impossible) and greedy by descending match score (deterministic; O(dets^2)/frame,
fine at few-fps). An external industrial BoT-SORT (boxmot) is available only as an *ablation*
backend (perception/boxmot_track.py), never the default gold path.

bbox convention matches grounding.py: ``[ymin, xmin, ymax, xmax]`` normalized to 0-1000.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vmem_bench.common.vecmath import cosine_similarity

Bbox = list[int]  # [ymin, xmin, ymax, xmax], 0-1000 normalized


def iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two [ymin, xmin, ymax, xmax] boxes."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0, iy1 - iy0), max(0, ix1 - ix0)
    inter = ih * iw
    if inter == 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class Detection:
    """One open-vocab detection in one sampled frame."""

    frame_index: int
    bbox: Bbox
    score: float
    phrase: str  # the roster grounding_phrase this box was detected for (the track's class)
    embedding: list[float] | None = None  # appearance vector of the crop (grounded)
    crop_path: str | None = None  # saved crop on disk (best detection becomes the representation)


@dataclass(slots=True)
class Tracklet:
    """A sequence of detections of one object instance within a single shot."""

    track_id: int
    phrase: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def frame_span(self) -> tuple[int, int]:
        return self.detections[0].frame_index, self.detections[-1].frame_index

    def mean_embedding(self) -> list[float] | None:
        """Mean of member crop embeddings (the tracklet's appearance signature for re-ID)."""
        vecs = [d.embedding for d in self.detections if d.embedding is not None]
        if not vecs:
            return None
        dim = len(vecs[0])
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    def best_detection(self) -> Detection:
        """Highest-scoring member detection (used to pick the representative crop)."""
        return max(self.detections, key=lambda d: d.score)


@dataclass(slots=True)
class _ActiveTrack:
    tracklet: Tracklet
    misses: int = 0


def _clamp(v: int, lo: int = 0, hi: int = 1000) -> int:
    return lo if v < lo else hi if v > hi else v


def predict_bbox(tracklet: Tracklet) -> Bbox:
    """Constant-velocity prediction of a tracklet's box for the next sampled frame.

    Uses the last two member boxes (velocity = last - prev); a single-box track predicts itself.
    Clamped to the 0-1000 canvas. Pure -> offline-testable. This is the linearised Kalman idea:
    at low fps the raw last box no longer overlaps the moved subject, but the predicted one does."""
    dets = tracklet.detections
    if len(dets) < 2:
        return list(dets[-1].bbox)
    a, b = dets[-2].bbox, dets[-1].bbox
    return [_clamp(int(2 * b[i] - a[i])) for i in range(4)]


def _associate(active: list[_ActiveTrack], predicted: list[Bbox], dets: list[Detection],
               free_tracks: list[int], iou_threshold: float, appearance_gate: float
               ) -> tuple[dict[int, int], set[int]]:
    """Greedy class-constrained IoU(+appearance) matching of ``free_tracks`` to ``dets``.

    IoU is against each track's *predicted* box. Returns (track_idx->det_idx, matched_det_idxs).
    Deterministic: sorted by descending (iou*score, cosine), ties by (track,det) index."""
    pairs: list[tuple[float, float, int, int]] = []
    for ti in free_tracks:
        at = active[ti]
        temb = at.tracklet.last.embedding
        for di, det in enumerate(dets):
            if det.phrase != at.tracklet.phrase:
                continue
            ov = iou(predicted[ti], det.bbox)
            if ov < iou_threshold:
                continue
            cos = 1.0
            if appearance_gate > 0.0 and temb is not None and det.embedding is not None:
                cos = cosine_similarity(temb, det.embedding)
                if cos < appearance_gate:
                    continue
            pairs.append((ov * det.score, cos, ti, di))
    pairs.sort(key=lambda p: (p[0], p[1], -p[2], -p[3]), reverse=True)
    track_to_det: dict[int, int] = {}
    matched_dets: set[int] = set()
    for _score, _cos, ti, di in pairs:
        if ti in track_to_det or di in matched_dets:
            continue
        track_to_det[ti] = di
        matched_dets.add(di)
    return track_to_det, matched_dets


def track_shot(
    detections_by_frame: list[list[Detection]],
    *,
    iou_threshold: float = 0.3,
    appearance_gate: float = 0.0,
    max_miss: int = 1,
    min_len: int = 2,
    next_track_id: int = 0,
    high_score: float = 0.5,
    use_motion: bool = True,
) -> list[Tracklet]:
    """Two-stage, motion-predicted, class-constrained tracklet association within one shot.

    ``detections_by_frame`` is ordered by sampled-frame time. Per frame:
      - split detections into HIGH (``score >= high_score``) and LOW (below) sets (ByteTrack);
      - predict each active track's next box (constant velocity);
      - **stage 1**: match active tracks to HIGH dets by IoU(pred, det) >= ``iou_threshold``
        (and cosine >= ``appearance_gate`` when both carry embeddings), greedily by descending
        ``iou*score``;
      - **stage 2**: match still-unmatched tracks to LOW dets (recover brief low-confidence);
      - unmatched tracks accrue a miss, retired after ``max_miss`` consecutive misses;
      - unmatched HIGH dets start new tracks; unmatched LOW dets are dropped (never seed tracks).
    Tracklets shorter than ``min_len`` are discarded as likely single-frame false positives.

    Returns tracklets with ``track_id`` from ``next_track_id`` (movie-level caller keeps ids
    unique across shots). Deterministic: ties broken by (score, cosine, track idx, det idx)."""
    active: list[_ActiveTrack] = []  # tracks alive going INTO the current frame
    finished: list[Tracklet] = []
    tid = next_track_id

    for dets in detections_by_frame:
        high = [d for d in dets if d.score >= high_score]
        low = [d for d in dets if d.score < high_score]
        predicted = ([predict_bbox(at.tracklet) for at in active] if use_motion
                     else [list(at.tracklet.last.bbox) for at in active])

        # Stage 1: all active tracks vs high-score detections.
        s1, matched_high = _associate(active, predicted, high, list(range(len(active))),
                                      iou_threshold, appearance_gate)
        # Stage 2: tracks still unmatched vs low-score detections (recovery).
        free = [ti for ti in range(len(active)) if ti not in s1]
        s2, _matched_low = _associate(active, predicted, low, free, iou_threshold, appearance_gate)

        for ti, di in s1.items():
            active[ti].tracklet.detections.append(high[di])
            active[ti].misses = 0
        for ti, di in s2.items():
            active[ti].tracklet.detections.append(low[di])
            active[ti].misses = 0

        matched_tracks = set(s1) | set(s2)
        survivors: list[_ActiveTrack] = []
        for ti, at in enumerate(active):
            if ti in matched_tracks:
                survivors.append(at)
                continue
            at.misses += 1
            if at.misses <= max_miss:
                survivors.append(at)
            else:
                finished.append(at.tracklet)
        # Only unmatched HIGH detections seed new tracks (ByteTrack: low boxes never start tracks).
        for di, det in enumerate(high):
            if di in matched_high:
                continue
            survivors.append(_ActiveTrack(Tracklet(track_id=tid, phrase=det.phrase,
                                                   detections=[det])))
            tid += 1
        active = survivors

    finished.extend(at.tracklet for at in active)
    finished.sort(key=lambda t: t.track_id)
    return [t for t in finished if len(t.detections) >= min_len]
