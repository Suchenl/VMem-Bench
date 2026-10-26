"""Build a perception backend (and, for the in-process path, the models it wraps) in ONE place.

Two callers share this so backend wiring never drifts:
  - run.py: builds the backend from either resident-service clients or in-process singletons.
  - track_parallel.py: each multi-GPU worker builds a fully in-process perception stack on its card.

``backend_from`` takes an already-built ``detector``/``embedder`` (client OR in-process) and returns
the configured tracker backend. ``build_inprocess_perception`` loads GroundingDINO + DINOv3 (+ face)
onto the currently-visible GPU and returns ``(backend, embedder, face_encoder)``; the caller must set
CUDA_VISIBLE_DEVICES *before* calling (weights load onto the visible card).
"""

from __future__ import annotations

from pathlib import Path


def backend_from(config, detector, embedder, crop_dir: Path):
    """Configured perception backend for ``config`` wrapping the given detector/embedder."""
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    if config.perception_backend == "gdino_track":
        if config.tracker == "boxmot_botsort":
            from vmem_bench.annotation.pipeline_track_first.perception.boxmot_track import BoxmotBotsortBackend
            return BoxmotBotsortBackend(detector, embedder, crop_dir=crop_dir,
                                        track_min_len=config.track_min_len,
                                        min_score=config.track_det_min_score,
                                        min_box_px=config.track_det_min_box_px)
        from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import GdinoTrackBackend
        # tracker == "iou" reproduces the old single-stage greedy IoU (no motion prediction).
        use_motion = config.tracker != "iou"
        high = config.track_high_score if config.tracker == "bytetrack_local" else 0.0
        return GdinoTrackBackend(
            detector, embedder, crop_dir=crop_dir,
            track_iou_threshold=config.track_iou_threshold, track_min_len=config.track_min_len,
            appearance_gate=config.track_appearance_gate, max_miss=config.track_max_miss,
            high_score=high, use_motion=use_motion,
            min_score=config.track_det_min_score, min_box_px=config.track_det_min_box_px)
    route_b_kwargs = dict(
        crop_dir=crop_dir,
        character_concepts=tuple(config.sam3_character_concepts),
        exemplar_sim_floor=config.sam3_exemplar_sim_floor,
        seg_threshold=config.sam3_seg_threshold,
        track_iou_threshold=config.track_iou_threshold, track_min_len=config.track_min_len,
        appearance_gate=config.track_appearance_gate, max_miss=config.track_max_miss,
        high_score=(config.track_high_score if config.tracker == "bytetrack_local" else 0.0),
        use_motion=(config.tracker != "iou"), min_box_px=config.track_det_min_box_px)
    if config.perception_backend == "fusion_track":
        from vmem_bench.annotation.pipeline_track_first.perception.fusion_track import FusionTrackBackend
        return FusionTrackBackend(embedder, detector=detector, **route_b_kwargs)
    from vmem_bench.annotation.pipeline_track_first.perception.sam3_track import Sam3ExemplarTrackBackend
    return Sam3ExemplarTrackBackend(embedder, **route_b_kwargs)


def build_inprocess_perception(config, crop_dir: Path):
    """Load an in-process perception stack (backend + embedder + face encoder) on the visible GPU."""
    from vmem_bench.annotation.pipeline_track_first.embedding import DinoV3Embedder
    embedder = DinoV3Embedder()
    detector = None
    if config.perception_backend in ("gdino_track", "fusion_track"):  # pure route B skips GDINO
        from vmem_bench.annotation.pipeline_track_first.grounding import GroundingDino
        detector = GroundingDino()
    backend = backend_from(config, detector, embedder, crop_dir)
    face_encoder = None
    if config.use_face:
        from vmem_bench.annotation.pipeline_track_first.face import FaceEncoder
        face_encoder = FaceEncoder()
    return backend, embedder, face_encoder
