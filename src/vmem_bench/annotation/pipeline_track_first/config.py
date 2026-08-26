"""Configuration for the offline annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnnotationConfig:
    video: Path
    out_dir: Path
    movie_id: str

    # Chunking (D2): cut shots first, then concatenate; chunk frame count in [min, max].
    min_frames_per_chunk: int = 120
    max_frames_per_chunk: int = 360
    sbd_method: str = "transnetv2"
    min_scene_len_sec: float = 1.0

    # Opening/ending credits exclusion (annotation/credits.py): deterministic dark-card prefilter
    # in the head/tail window + one VLM confirmation batch. Excluded segments are recorded in
    # chunk_index.json/manifest.json so evaluation never hands those spans to the SUT.
    exclude_credits: bool = True
    credits_head_tail_ratio: float = 0.08
    credits_max_luminance: float = 45.0

    # Per-chunk frame sampling for VLM/grounding.
    max_sampled_frames: int = 12
    # Prompt drafting is evidence-constrained: each present entity contributes at most this many
    # current-chunk crops, subject to the chunk-wide visual-token budget below.
    draft_crops_per_entity: int = 3
    draft_max_entity_crops: int = 12

    # Cross-chunk consolidation (dual threshold + VLM adjudication).
    high_threshold: float = 0.80
    low_threshold: float = 0.40
    # Static-attribute identity gate (principle: static-identity vs dynamic-state decoupling).
    # Two observations of the same (kind, name) are only reused if their static_attributes overlap
    # by at least this fraction of shared keys; a conflict (e.g. species fox vs bird) forces a new
    # entity even when names match. The embedding-match path is gated the same way.
    static_overlap_threshold: float = 0.75

    # Grounding.
    # Must be >= GroundingDino.box_threshold so it actually selects (otherwise it is a no-op
    # duplicate of the detector's own post-filter).
    grounding_score_threshold: float = 0.35
    # Temporal-consistency floor: a non-location entity must be detected in at least this many
    # sampled frames to be accepted as a real crop (guards against one-frame grounding false
    # positives). Default 2 is intentionally conservative for animation where entities persist
    # across frames.
    grounding_min_frames: int = 2
    # Multiple detections must describe roughly the same object before the best-scoring crop is
    # accepted. We allow either low IoU or nearby centers because characters move between sampled
    # frames and camera cuts can shift composition.
    grounding_temporal_iou_threshold: float = 0.10
    grounding_temporal_center_threshold: float = 0.35
    # Deterministic asset-bank gates before registry commit: non-location crops that cover too
    # much of the frame, or two different entities in one chunk sharing the same bbox, are
    # blocking identity/crop failures and must not enter assets/.
    max_non_location_bbox_area: float = 0.95
    same_chunk_bbox_iou_threshold: float = 0.95

    # Self-verification loop (principle #11).
    qa_max_rounds: int = 2
    # Per-chunk candidate branches. Decoupled from the annotator/verifier endpoint pool: the pool
    # is for dataset-level throughput (round-robin across chunks/attempts), this is the per-chunk
    # redundancy knob. Default 1 = one annotator+verifier branch per chunk.
    branches_per_chunk: int = 1
    # Sampling temperature for diversity. 0.0 = deterministic (the first attempt of branch 0
    # always runs at 0.0 so the canonical run is reproducible). The pipeline raises it to this
    # value on QA retry (attempt >= 2) and on redundancy branches (branch >= 1) so parallel
    # candidates decorrelate instead of returning near-identical outputs — temperature=0 across
    # branches would defeat branches_per_chunk>1 (correlated errors, principle #11). Scoring is
    # unaffected: bench metrics are deterministic set operations over gold, never over VLM output.
    diversity_temperature: float = 0.3
    # Verifier independence: a different model family for verification reduces correlated errors
    # (workflow step 8). None reuses ``vlm_model``.
    verifier_model: str | None = None
    # On QA retry (attempt >= 2), let the verifier audit the chunk VIDEO clip instead of sparse
    # frames — sparse sampling structurally cannot catch actions between frames. Only takes effect
    # when the verifier backend exposes verify_chunk_video; falls back to frames otherwise.
    # Clips materialize into derived/clips/ (gitignored, never shipped), never out/chunks.
    verifier_video_for_retry: bool = False
    # Skip per-crop VLM audit (crop_match) for location-kind, full-frame, and high-grounding-score
    # crops (Pitfall_Notes: low value, high cost).
    crop_audit_score_threshold: float = 0.60

    # Random spot-check sample size for the human review page.
    review_spot_check: int = 10
    # Spot-check RNG seed. None = system-entropy random each run (different reviewers spot-check
    # different chunks); an int reproduces a specific review page (Pitfall_Notes: fixed seed=0
    # made the spot check deterministic across runs/reviewers).
    review_seed: int | None = None

    # Discovery prompt budget: cap the known-entity prior to the most-recently-introduced N
    # entities so the discovery prompt does not grow without bound on long videos. Older entities
    # are dropped (the VLM re-names them; consolidation reconciles via embedding/static-attr
    # matching). Pitfall_Notes: prompt-bloat on long videos.
    known_entity_limit: int = 60

    # Checkpoint embeddings write cadence (Pitfall_Notes: F6). checkpoint.json +
    # checkpoint_registry.json are written every chunk (cheap), but the embeddings sidecar
    # (safetensors, full rewrite) is only written every N chunks AND once at run_done —
    # rewriting O(embeddings) safetensors every chunk is the dominant checkpoint cost on long
    # videos. On --resume, if the sidecar lags behind last_chunk_id, the resume point is
    # truncated to the sidecar's covered chunk so embeddings stay complete (re-running <= N
    # chunks is cheap). 1 = write every chunk (original behavior, safe but slow on long videos).
    checkpoint_embedding_interval: int = 5

    # --- Track-first redesign (docs/benchmark/annotation_tracking_internals.md) --------------------------------
    # Pluggable perception backend (§3.1b): gdino_track (route A, language-grounded) |
    # sam3_track (route B, exemplar-grounded discriminative — probe-validated on BBB where
    # route A phrase mismatches lost every second-half character).
    perception_backend: str = "gdino_track"
    # Route B knobs: generic concept words for character candidate enumeration (category-level
    # by design — identity comes from exemplar similarity, never from these words), SAM3
    # instance threshold, and the DINOv3 cosine floor for exemplar assignment.
    sam3_character_concepts: tuple[str, ...] = ("animal", "person")
    sam3_seg_threshold: float = 0.4
    sam3_exemplar_sim_floor: float = 0.28
    # Exemplar-anchored characters: the phrase is already an individual identity (assigned by
    # exemplar similarity), so re-ID within one anchored phrase merges permissively — it is
    # reconciling viewpoints of ONE individual, not discovering individuals (v11: strict
    # thresholds split one flying squirrel into a dozen entities).
    anchored_reid_threshold: float = 0.35
    anchored_cluster_min_similarity: float = 0.20
    # Global cast roster keyframe selection (§3.1a): candidate fps per shot -> dedup -> per-shot
    # medoid -> FPS to a global budget -> VLM discovery in batches.
    roster_candidate_fps: float = 2.0
    roster_per_shot_k: int = 1
    # Keyframe count adapts to the film: floor = max(roster_global_budget,
    # roster_budget_min_ratio * total_frames); cap = roster_budget_max; between them, frames are
    # added only while residual DINOv3 novelty (cosine distance to the picked set) stays >=
    # roster_novelty_threshold. Set roster_budget_max = 0 to restore the fixed legacy budget.
    roster_global_budget: int = 32
    roster_budget_max: int = 96
    roster_budget_min_ratio: float = 0.002
    roster_novelty_threshold: float = 0.32
    roster_vlm_batch: int = 8
    # Production gold no longer lets automatic discovery define benchmark identities.  A
    # human-confirmed canonical roster supplies stable ids, detector phrases, identity scope, and
    # exemplar crops.  ``production_mode`` is set by the CLI; direct/unit-test callers stay in
    # proposal mode unless they opt in explicitly.
    roster_seed_path: Path | None = None
    production_mode: bool = False
    # In seeded production, names/descriptions/ids are source-of-truth fields and must not be
    # overwritten or re-slugged from a later VLM naming response.
    lock_seed_identity: bool = True
    # Closed-set tracklet -> canonical entity assignment. Individual identities compare against
    # every same-kind seed's multi-view exemplars; weak/ambiguous matches become unknown/reject.
    seed_assignment_min_similarity: float = 0.30
    seed_assignment_min_margin: float = 0.04
    # Intra-shot tracking (§3.1): sample each shot at track_fps, associate into tracklets.
    track_fps: float = 3.0
    track_min_len: int = 2
    track_iou_threshold: float = 0.3
    track_appearance_gate: float = 0.0
    track_max_miss: int = 1
    # Tracker choice (Q1): iou (old greedy) | bytetrack_local (default: two-stage + motion predict +
    # DINO appearance, deterministic) | boxmot_botsort (ablation only: industrial BoT-SORT via boxmot,
    # nondeterministic -> never the default gold path).
    tracker: str = "bytetrack_local"
    track_high_score: float = 0.5   # ByteTrack two-stage split: >= is stage-1 (seeds tracks), < is recovery-only
    # Detection crop hygiene: drop boxes before cropping/embedding to cut derived/candidates clutter
    # (tiny/junk boxes are never useful representations). GDINO already thresholds at box_threshold;
    # min_score is an extra floor (0.0 = rely on GDINO). min_box_px drops a crop when EITHER pixel
    # side is below it -- an ABSOLUTE px size (robust across resolutions: a 24 px cutout is junk on
    # 480p and 4K alike; a frame-fraction would keep tiny boxes on big frames and over-prune small).
    track_det_min_score: float = 0.0
    track_det_min_box_px: int = 24
    # After naming/commit, delete the entire derived/ scratch tree (frames + candidates + clips).
    # Gold references only assets/ + gold/*, so scratch is disposable; keep it (--keep-scratch)
    # for debugging.
    prune_scratch: bool = True

    # Text embedding (Q2): Qwen3-Embedding-4B via an OpenAI-compatible resident service. Used for
    # (A) roster semantic dedup and (B) prompt-completeness / naming-consistency checks. Text<->text
    # only (CLIP text encoders are unfit for this); image<->text stays with SigLIP (crop_classify).
    use_text_embed: bool = True
    text_embed_model: str = "Qwen3-Embedding-4B"
    text_embed_base_url: str | None = None      # e.g. http://127.0.0.1:8003/v1 ; None -> feature off
    roster_dedup_threshold: float = 0.82        # merge two roster entries when name+desc cosine >= this
    prompt_completeness_threshold: float = 0.5  # flag a chunk when a present entity's name/desc cosine < this
    # Post-naming merge PROPOSALS (text + body cosine); written to build/merge_proposals.json only.
    merge_text_threshold: float = 0.85
    merge_body_threshold: float = 0.5
    # Machine-assisted review (auto_review): suspicion scoring + two-tier merge split.
    auto_review: bool = True
    auto_apply_safe_merges: bool = True
    merge_auto_text_threshold: float = 0.92
    merge_auto_body_threshold: float = 0.75
    # Location clustering uses full-frame scene vectors independently from character re-ID.
    # Calibrated on the cached BBB checkpoints: 0.40 yields 8 clusters (vs 28 per-shot entities).
    # Calibrated on BBB v13 scene vectors (threshold sweep): 0.32 collapsed the film into two
    # mega-clusters; 0.38 restores v10 granularity (8 clusters). Greedy centroid clustering has
    # NO good regime on low-diversity footage (one forest palette) — a temporal-adjacency
    # hierarchical method is the known-limitation upgrade path; until then prefer over- to
    # under-segmentation (mislabeled mega-places poison every chunk they cover).
    loc_scene_cluster_threshold: float = 0.38
    # Deprecated compatibility knobs retained for external configs; clustered locations no longer
    # depend on character tracklets or only the immediately preceding shot.
    loc_skip_closeup: bool = True
    loc_closeup_coverage: float = 0.45
    # A shot with no tracked characters/props may reuse the immediately preceding location when
    # its full-scene embedding agrees. This deliberately sits below reid_threshold: it prevents
    # per-shot location fragmentation during untracked establishing shots without broad matching.
    loc_no_tracklet_similarity: float = 0.48
    # Cross-shot re-ID fused threshold + multi-cue weights (§3.7).
    reid_threshold: float = 0.55
    reid_w_body: float = 1.0
    reid_w_face: float = 0.6
    reid_w_class: float = 0.3
    face_strong: float = 0.5
    # Prevent a high-scoring cluster centroid from absorbing a visually incompatible crop.
    # The roster phrase gate is applied by the track-first caller; this is the per-cluster guard.
    reid_cluster_min_similarity: float = 0.35
    use_face: bool = True   # run the self-gating face cue on character crops (§3.7)

    # --- Identity resolution v2 (VLM-primary batch clustering, docs/benchmark/annotation_tracking_internals.md
    # "identity resolution v2" + Pitfall_Notes) -------------------------------------------------
    # "cluster_vlm" (default): deterministic complete/average-link pre-clustering + VLM cluster
    # verification (authoritative) + VLM cross-cluster merge, replacing the online greedy
    # nearest-neighbor reid_assign hot path for character/prop identity. "greedy" restores the old
    # online reid_assign path verbatim -- kept as an explicit fallback/ablation switch for very
    # large-scale runs where the VLM cluster-verification budget is a concern, and so the BBB
    # before/after comparison has a same-codebase toggle rather than relying on old archived runs.
    identity_resolution_mode: str = "cluster_vlm"
    # complete-link (default): a merge requires the WORST cross-pair similarity to clear the
    # threshold -- resists single-link "chaining" through one noisy embedding pair (the diagnosed
    # root cause of DINOv3/SigLIP not being a fine-grained instance-re-id embedding). average-link
    # is the looser alternative; never plain connected-components (see identity_clustering.py).
    precluster_linkage: str = "complete"
    # Reuses reid_threshold as the pre-clustering cut (kept as ONE tuned knob rather than a second
    # near-duplicate threshold): complete-link at the same cut is at least as conservative as the
    # old absolute-accept threshold, and VLM verification catches any residual over-clustering.
    # Per-cluster VLM verification image budget (§verify_cluster): clusters larger than this send a
    # diverse SUBSET (excluded members are deterministically reattached by body cosine afterward).
    identity_verify_max_crops: int = 8
    # Cross-cluster VLM merge image budget per kind (one representative crop per cluster; mirrors
    # identity_adjudication.py's existing max_images=24, aligned to the vLLM --limit-mm-per-prompt).
    identity_merge_max_images: int = 24
    # A final entity group with >= this many tracklet observations whose members never matched a
    # roster phrase is flagged as a likely MISSED roster entry (finding), not silently
    # force-classified into the nearest (possibly wrong) roster candidate.
    roster_completeness_min_observations: int = 3
    # Thread-pool width for the independent per-cluster VLM verify calls and per-kind merge calls
    # (extreme parallelization principle #8); a single vLLM endpoint's continuous batching absorbs
    # concurrent requests, so this is a throughput knob even without multiple physical endpoints.
    identity_resolution_max_workers: int = 8
    # Per-crop QA (§3.3, step 5): drop mixed-class representations after re-ID.
    use_crop_classify: bool = True
    crop_classify_method: str = "prototype"   # prototype (DINOv3, no new model) | siglip
    crop_classify_margin: float = 0.05        # flag a rep when another same-kind proto beats own by >margin
    crop_classifier_model: str = "google/siglip2-base-patch16-512"
    # Optional SigLIP tracklet-label reassignment (default OFF): re-score crop vs roster phrases.
    reassign_by_class: bool = False
    class_reassign_margin: float = 0.15

    # Persistent-service placement (services_and_time_design.md §1.3): fastest (one service per card)
    # | packed (bin-pack by peak VRAM) | none (in-process singleton fallback, --no-services).
    service_placement: str = "fastest"

    # Provenance strings recorded into gold.
    vlm_model: str = "qwen3-vl-8b"
    # ViT-L regressed cluster cohesion without threshold recalibration (BBB v8); see embedding.py.
    embedder_name: str = "dinov3-vits16"
    tracker_name: str = "bytetrack_local"
    reid_name: str = "dinov3-vits16"
    face_encoder_name: str = "insightface-buffalo_l"
    text_embedder_name: str | None = None
    pipeline_version: str = "3.0.0"

    # Source-video download recipe (written into manifest.json; the bench ships no video bytes).
    source_dataset: str = "BlenderOpenMovies"
    source_url: str | None = None

    extra: dict = field(default_factory=dict)
