"""CLI for the offline annotation pipeline.

Examples (on a GPU node, with the vLLM judger server running):

    PY=python3
    PYTHONPATH=benchmarks/MemStrata/src "$PY" -m vmem_bench.annotation.pipeline_track_first.run \
        --video ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/big_buck_bunny_720p/big_buck_bunny_720p_h264.mp4 \
        --out benchmarks/MemStrata/data/blender_open_movies/big_buck_bunny \
        --movie-id big_buck_bunny

    # after human review:
    ... -m vmem_bench.annotation.pipeline_track_first.run --out <dir> --apply-patch <patch.json> --freeze
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemStrata-Bench offline annotation pipeline. Input must be one single-source, "
                    "time-continuous long video; SBD creates shots and shots are aggregated into chunks. "
                    "Pre-stitched short clips with unknown temporal continuity are unsupported.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--movie-id", default=None)
    parser.add_argument("--roster-seed", type=Path,
                        help="human-confirmed canonical roster JSON required for production gold")
    parser.add_argument("--proposal-only", action="store_true",
                        help="allow automatic roster discovery for diagnostics/proposals only; "
                             "the resulting draft is not production gold")
    parser.add_argument("--min-frames", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--max-sampled-frames", type=int, default=12)
    parser.add_argument("--draft-crops-per-entity", type=int, default=3,
                        help="current-chunk evidence crops per present entity for prompt drafting")
    parser.add_argument("--draft-max-entity-crops", type=int, default=12,
                        help="maximum total entity evidence crops sent to the drafter per chunk")
    parser.add_argument("--qa-rounds", type=int, default=2)
    parser.add_argument("--vlm-base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--judge-base-url", default=None,
                        help="optional LARGE judge-model endpoint for roster/naming/identity "
                             "resolution (cluster verify + cross-cluster merge)/adjudication/"
                             "auto-review (hybrid serving); drafting stays on --vlm-base-url. "
                             "Comma-separated URLs stand up multiple replicas (e.g. on different "
                             "GPUs) round-robined via RoundRobinRole for real multi-GPU "
                             "parallelism on identity_resolution's concurrent calls")
    parser.add_argument("--judge-model", default="qwen3-vl-32b")
    parser.add_argument("--verifier-base-url", default=None,
                        help="optional separate endpoint for the verifier role (parallel QA)")
    parser.add_argument("--annotator-urls", default=None,
                        help="comma-separated annotator endpoints (one parallel branch each)")
    parser.add_argument("--verifier-urls", default=None,
                        help="comma-separated verifier endpoints")
    parser.add_argument("--vlm-model", default="qwen3-vl-8b")
    parser.add_argument("--verifier-model", default=None,
                        help="different model family for verification (reduces correlated errors; "
                             "default reuses --vlm-model)")
    parser.add_argument("--perception-backend", default="gdino_track",
                        choices=["gdino_track", "sam3_track", "fusion_track"],
                        help="perception route: A=language-grounded GDINO, B=exemplar-grounded "
                             "SAM3 (discriminative), fusion=union proposals + one identity judge")
    parser.add_argument("--track-fps", type=float, default=3.0)
    parser.add_argument("--seed-assignment-min-similarity", type=float, default=0.30,
                        help="reject an individual tracklet when its best seed-exemplar cosine is "
                             "below this floor")
    parser.add_argument("--seed-assignment-min-margin", type=float, default=0.04,
                        help="reject an individual tracklet when best-vs-second seed margin is "
                             "below this ambiguity floor")
    parser.add_argument("--tracker", default="bytetrack_local",
                        choices=["iou", "bytetrack_local", "boxmot_botsort"],
                        help="intra-shot tracker (Q1): bytetrack_local=two-stage+motion+DINO "
                             "(default, deterministic); iou=old greedy; boxmot_botsort=industrial "
                             "BoT-SORT (ablation only, nondeterministic, needs `pip install boxmot`)")
    parser.add_argument("--identity-resolution-mode", default="cluster_vlm",
                        choices=["seeded", "cluster_vlm", "greedy"],
                        help="seeded: canonical roster ids own identity (automatically selected by "
                             "--roster-seed). cluster_vlm: deterministic complete/average-link "
                             "pre-clustering + VLM cluster verification/merge (--judge-base-url "
                             "role is authoritative for identity, not gray-zone). greedy: restore "
                             "the old online reid_assign nearest-neighbor path (fallback/ablation)")
    parser.add_argument("--precluster-linkage", default="complete", choices=["complete", "average"],
                        help="cluster_vlm pre-clustering linkage (complete resists single-link "
                             "chaining through one noisy embedding pair; see identity_clustering.py)")
    parser.add_argument("--identity-verify-max-crops", type=int, default=8)
    parser.add_argument("--identity-merge-max-images", type=int, default=24)
    parser.add_argument("--roster-completeness-min-observations", type=int, default=3)
    parser.add_argument("--identity-resolution-max-workers", type=int, default=8)
    parser.add_argument("--reid-threshold", type=float, default=0.55)
    parser.add_argument("--det-min-score", type=float, default=0.0,
                        help="extra detection score floor beyond GDINO box_threshold (0 = off); "
                             "drops low-confidence boxes before crop+embed")
    parser.add_argument("--det-min-box-px", type=int, default=24,
                        help="drop a detection when either pixel side < this (absolute px, "
                             "resolution-robust); cuts tiny junk crops from tmp/candidates")
    parser.add_argument("--keep-scratch", dest="prune_scratch", action="store_false", default=True,
                        help="keep tmp/frames + tmp/candidates after commit (default: prune "
                             "them and any legacy derived/ tree; gold references only assets/ + gold/*)")
    parser.add_argument("--no-reassign-by-class", dest="reassign_by_class", action="store_false",
                        default=True,
                        help="disable SigLIP tracklet-label reassignment before roster lookup "
                             "(default ON; degrades gracefully if SigLIP fails to load)")
    parser.add_argument("--text-embed-base-url", default=None,
                        help="OpenAI-compatible endpoint for Qwen3-Embedding-4B (roster dedup + "
                             "prompt-completeness check). Unset -> text-embedding features off.")
    parser.add_argument("--text-embed-model", default="Qwen3-Embedding-4B")
    parser.add_argument("--no-face", dest="use_face", action="store_false", default=True,
                        help="disable the self-gating face cue on character crops (§3.7)")
    parser.add_argument("--branches-per-chunk", type=int, default=1,
                        help="per-chunk candidate branches; the endpoint pool is round-robined "
                             "separately across chunks/attempts")
    parser.add_argument("--diversity-temperature", type=float, default=0.3,
                        help="sampling temperature for QA retry (attempt>=2) and redundancy "
                             "branches (branch>=1); branch 0 / attempt 1 stays 0.0 (deterministic). "
                             "Decorrelates parallel candidates (principle #11); scoring unaffected.")
    parser.add_argument("--grounding-min-frames", type=int, default=2,
                        help="temporal consistency: min sampled frames a non-location entity must "
                             "be detected in (default 2 for animation; 1 = single best detection)")
    parser.add_argument("--grounding-score-threshold", type=float, default=0.35)
    parser.add_argument("--grounding-temporal-iou-threshold", type=float, default=0.10)
    parser.add_argument("--grounding-temporal-center-threshold", type=float, default=0.35)
    parser.add_argument("--max-non-location-bbox-area", type=float, default=0.95)
    parser.add_argument("--same-chunk-bbox-iou-threshold", type=float, default=0.95)
    parser.add_argument("--static-overlap-threshold", type=float, default=0.75)
    parser.add_argument("--crop-audit-score-threshold", type=float, default=0.60)
    parser.add_argument("--verifier-video-for-retry", dest="verifier_video_for_retry",
                        action="store_true", default=False,
                        help="on QA retry, audit the chunk video clip (default off; can be slow)")
    parser.add_argument("--no-verifier-video", dest="verifier_video_for_retry",
                        action="store_false", help="disable chunk-video verification on retry")
    parser.add_argument("--review-spot-check", type=int, default=10)
    parser.add_argument("--review-seed", type=int, default=None,
                        help="spot-check RNG seed (None = random each run)")
    parser.add_argument("--known-entity-limit", type=int, default=60,
                        help="cap the discovery known-entity prior to the N most-recent entities "
                             "(prevents prompt-bloat on long videos)")
    parser.add_argument("--checkpoint-embedding-interval", type=int, default=5,
                        help="write the checkpoint embeddings sidecar every N chunks (and once at "
                             "run_done); 1 = every chunk. JSON checkpoint files are always written "
                             "every chunk. On --resume a lagging sidecar truncates the resume point.")
    parser.add_argument("--resume", action="store_true",
                        help="resume annotation RESULTS from tmp/checkpoint/ (reload finished "
                             "roster / per-shot tracklets / per-entity names / per-chunk drafts and "
                             "only fill the gap; deterministic steps are recomputed)")
    parser.add_argument("--track-workers", type=int, default=1,
                        help="parallelize per-shot tracking across the N most-free GPUs (one "
                             "in-process perception worker per card). 1 = single-GPU (default)")
    parser.add_argument("--track-devices", default=None,
                        help="explicit GPU indices for per-shot tracking, e.g. 0,1,2,3 (overrides "
                             "--track-workers)")
    parser.add_argument("--dashboard-port", type=int, default=7863,
                        help="auto-start the live web dashboard on this port (0 = off)")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--services-manifest", type=Path, default=None,
                        help="tmp/services.json from the resident launcher; models are called over "
                             "HTTP instead of loaded in-process (default: <out>/tmp/services.json "
                             "if it exists; legacy build/services.json still accepted)")
    parser.add_argument("--no-services", action="store_true",
                        help="force in-process model singletons even if a services.json exists")
    parser.add_argument("--apply-patch", type=Path, help="apply review_patch.json to gold/")
    parser.add_argument("--freeze", action="store_true", help="mark gold human_reviewed")
    parser.add_argument("--review-only", action="store_true", help="only regenerate review.html")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.apply_patch or args.freeze or args.review_only:
        from vmem_bench.annotation.pipeline_track_first.review import apply_patch, freeze, generate_review_html
        if args.apply_patch:
            apply_patch(args.out / "gold", args.apply_patch)
            generate_review_html(args.out)
        if args.review_only:
            generate_review_html(args.out)
        if args.freeze:
            freeze(args.out / "gold")
        return

    if not args.video:
        parser.error("--video is required for annotation runs")
    if not args.roster_seed and not args.proposal_only:
        parser.error("production annotation requires --roster-seed; use --proposal-only only for "
                     "automatic roster diagnostics that must not be frozen as gold")
    if args.roster_seed and args.proposal_only:
        parser.error("--roster-seed and --proposal-only are mutually exclusive")

    from vmem_bench.judger.vlm import VlmJudger
    from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
    from vmem_bench.annotation.pipeline_track_first.embedding import DinoV3Embedder
    from vmem_bench.annotation.pipeline_track_first.grounding import GroundingDino
    from vmem_bench.annotation.pipeline_track_first import annotate_movie_track_first
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import AnnotatorRole, RoundRobinRole

    run_movie_id = args.movie_id or args.video.stem
    if args.roster_seed:
        from vmem_bench.annotation.pipeline_track_first.roster_seed import load_roster_seed
        # Validate before loading GPU models or creating output artifacts.
        load_roster_seed(args.roster_seed, expected_movie_id=run_movie_id, require_confirmed=True)
    config = AnnotationConfig(
        video=args.video, out_dir=args.out,
        movie_id=run_movie_id,
        min_frames_per_chunk=args.min_frames,
        max_frames_per_chunk=args.max_frames,
        max_sampled_frames=args.max_sampled_frames,
        draft_crops_per_entity=args.draft_crops_per_entity,
        draft_max_entity_crops=args.draft_max_entity_crops,
        vlm_model=args.vlm_model,
        static_overlap_threshold=args.static_overlap_threshold,
        review_spot_check=args.review_spot_check,
        review_seed=args.review_seed,
        perception_backend=args.perception_backend,
        track_fps=args.track_fps,
        tracker=args.tracker,
        track_det_min_score=args.det_min_score,
        track_det_min_box_px=args.det_min_box_px,
        prune_scratch=args.prune_scratch,
        reassign_by_class=args.reassign_by_class,
        reid_threshold=args.reid_threshold,
        roster_seed_path=args.roster_seed,
        production_mode=not args.proposal_only,
        seed_assignment_min_similarity=args.seed_assignment_min_similarity,
        seed_assignment_min_margin=args.seed_assignment_min_margin,
        identity_resolution_mode=("seeded" if args.roster_seed else args.identity_resolution_mode),
        precluster_linkage=args.precluster_linkage,
        identity_verify_max_crops=args.identity_verify_max_crops,
        identity_merge_max_images=args.identity_merge_max_images,
        roster_completeness_min_observations=args.roster_completeness_min_observations,
        identity_resolution_max_workers=args.identity_resolution_max_workers,
        use_face=args.use_face,
        text_embed_base_url=args.text_embed_base_url,
        text_embed_model=args.text_embed_model,
        tracker_name=args.tracker,
        text_embedder_name=(args.text_embed_model if args.text_embed_base_url else None))
    if not args.no_dashboard and args.dashboard_port:
        _spawn_dashboard(args.out, args.dashboard_port)

    # Track-first roles all live on AnnotatorRole (roster discovery, per-entity naming, prompt
    # drafting). One VLM endpoint is enough; identity resolution no longer relies on a *separate*
    # verifier ensemble -- it now goes through the judge_role below (verify_cluster / cross-cluster
    # merge / adjudication), same as naming.
    role = AnnotatorRole(VlmJudger(base_url=args.vlm_base_url, model=args.vlm_model))

    # Hybrid serving (annotation-system triad): high-volume drafting stays on the fast model;
    # low-frequency judgment roles (roster discovery, naming, IDENTITY RESOLUTION -- cluster
    # verification + cross-cluster merge, identity_resolution_mode=cluster_vlm's authoritative
    # decisions -- identity adjudication, auto-review votes) go to the LARGE judge model when one
    # is served. Without --judge-base-url everything uses the single --vlm-base-url endpoint,
    # exactly as before (identity decisions then run on the fast model too).
    judge_role = role
    if args.judge_base_url:
        judge_urls = [u.strip() for u in args.judge_base_url.split(",") if u.strip()]
        judge_roles = [AnnotatorRole(VlmJudger(base_url=u, model=args.judge_model))
                       for u in judge_urls]
        # Multiple replicas (comma-separated --judge-base-url) -> round-robin real GPU parallelism
        # for identity_resolution's concurrent verify_cluster/group_same_individuals calls (extreme
        # parallelization, principle #8), instead of queueing behind one server's batch scheduler.
        judge_role = judge_roles[0] if len(judge_roles) == 1 else RoundRobinRole(judge_roles)
        logging.info("hybrid VLM: drafter=%s@%s judge=%s@%s (%d replica(s))",
                     args.vlm_model, args.vlm_base_url, args.judge_model, args.judge_base_url,
                     len(judge_roles))

    # Resident-service path: if a tmp/services.json exists (from the launcher) and --no-services
    # was not set, call every model over HTTP (persistent-model-serving). Otherwise fall back to
    # in-process singletons (the models load once per run and are reused across chunks).
    from vmem_bench.common.paths import MovieDirs
    dirs = MovieDirs(args.out)
    manifest_path = args.services_manifest or dirs.services_manifest
    clients: dict = {}
    if not args.no_services and manifest_path.exists():
        import json as _json
        from vmem_bench.services import clients_from_manifest
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        clients = clients_from_manifest(manifest)
        logging.info("using resident services from %s: %s", manifest_path, sorted(clients))

    embedder = clients.get("embedder") or DinoV3Embedder()  # always needed (roster keyframes)
    text_embed_fn = None
    if "text_embed" in clients:
        text_embed_fn = clients["text_embed"].embed
    elif config.use_text_embed and args.text_embed_base_url:
        from vmem_bench.services.clients import EmbedClient
        text_embed_fn = EmbedClient(args.text_embed_base_url, args.text_embed_model).embed

    crop_dir = MovieDirs(args.out, write=True).candidates
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Multi-GPU per-shot tracking: --track-devices 0,1,.. (explicit) or --track-workers N (auto-pick
    # the N most-free cards). >1 device -> track_parallel spawns one in-process worker per card and
    # the main process skips loading perception (workers own it); <=1 -> single in-process backend.
    devices = _resolve_track_devices(args)
    if len(devices) > 1:
        backend, face_encoder = None, None
        logging.info("multi-GPU per-shot tracking on devices %s", devices)
    else:
        from vmem_bench.annotation.pipeline_track_first.perception.factory import backend_from
        detector = None
        if config.perception_backend in ("gdino_track", "fusion_track"):
            detector = clients.get("detector") or GroundingDino()
        backend = backend_from(config, detector, embedder, crop_dir)
        face_encoder = clients.get("face_encoder")
        if face_encoder is None and config.use_face:
            from vmem_bench.annotation.pipeline_track_first.face import FaceEncoder
            face_encoder = FaceEncoder()

    crop_classifier = None
    if args.reassign_by_class:
        try:
            import os
            from vmem_bench.annotation.pipeline_track_first.crop_classify import SiglipCropClassifier
            # Prefer the shared local snapshot (models/model_weights/local_paths.md); training
            # nodes have no internet, so an HF model id would fail at first classify().
            local_siglip = (Path(os.environ.get("PUBLIC_MODELS_ROOT", "${PUBLIC_MODELS_ROOT}"))
                            / "google" / "siglip2-base-patch16-512")
            if local_siglip.is_dir():
                crop_classifier = SiglipCropClassifier(model_id=str(local_siglip))
            else:
                crop_classifier = SiglipCropClassifier()
        except Exception:  # noqa: BLE001 — graceful degrade when SigLIP weights/deps unavailable
            logging.warning("SigLIP crop classifier failed to construct; "
                            "continuing with reassign_by_class disabled", exc_info=True)
            crop_classifier = None
            args.reassign_by_class = False
            config.reassign_by_class = False

    # Roster DISCOVERY deliberately stays on the fast model: measured on BBB v9 (2026-07-12),
    # the 32B judge is too thorough as a discoverer (every butterfly wing becomes a grounding
    # phrase -> 55 entities / 26 singletons vs 22 / 4). Judgment roles keep the judge model.
    summary = annotate_movie_track_first(
        config, roster_vlm=role, namer_vlm=judge_role, drafter_vlm=role,
        backend=backend, embedder=embedder, face_encoder=face_encoder,
        text_embed_fn=text_embed_fn, resume=args.resume, track_devices=devices,
        crop_classifier=crop_classifier)
    print(summary)


def _resolve_track_devices(args) -> list[int]:
    """GPU indices for multi-GPU per-shot tracking. Explicit --track-devices wins; else auto-pick
    the --track-workers most-free cards (query_gpus honors CUDA_VISIBLE_DEVICES). [] / one card ->
    single in-process tracking. Worker device ids are physical, so each worker's CUDA_VISIBLE_DEVICES
    pin is unambiguous."""
    if args.track_devices:
        return [int(x) for x in args.track_devices.split(",") if x.strip()]
    if args.track_workers and args.track_workers > 1:
        from vmem_bench.services.placement import query_gpus
        gpus = sorted(query_gpus(), key=lambda g: (g.free_mib, -g.index), reverse=True)
        return [g.index for g in gpus[:args.track_workers]]
    return []


def _spawn_dashboard(out_dir: Path, port: int) -> None:
    """Start the live dashboard as a daemon subprocess if the port is free (idempotent)."""
    import socket
    import subprocess
    import sys
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            logging.info("dashboard already running on :%d", port)
            return
    subprocess.Popen(
        [sys.executable, "-m", "vmem_bench.web.server",
         "--out", str(out_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    logging.info("dashboard started on http://127.0.0.1:%d", port)


if __name__ == "__main__":
    main()
