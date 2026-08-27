"""MemStrata causal adapter (new protocol, real segments) — our own system.

MemStrata's native four-step loop is intent → compose → generate → decompose →
curate. Under the causal bench protocol the bench substitutes the **real segment**
for the generator output, so this adapter drives MemStrata's own *write* path
(perception → curate) on the real segment and its own *read* path (intent →
compose) on the prompt, and never sees gold:

* ``observe_segment`` (memory WRITE): sample a frame from the real segment and run
  MemStrata's perception (S5-derived SAM3 concept proposal + DINOv3 identity/novelty
  selection, the ``crop_acquisition`` skill / ``ProposeIdentifyCropper`` core) to
  isolate entity crops, then admit them into the stratified ``AssetBank`` via the
  real ``MemoryUpdater`` (name-anchored identity, angle strata, novelty dedup). Every
  stored representation is tagged with the **absolute source-video seconds** of the
  frame it was cut from, so a retrieved rep carries a temporal identity.
* ``compose`` (memory READ): run the real ``IntentInterpreter`` (name-anchored,
  model-free FAST path) + model-free ``compose`` over the CURRENT bank; each selected
  representation is returned as a temporal :class:`RetrievedItem` at its stored
  source seconds plus its own crop path. The bench-side ``frame_materializer`` uses
  that guarded crop directly — MemStrata never reads or writes gold.

Protocol-faithfulness notes
---------------------------
* The perception model set (SAM3 concept segmenter + DINOv3 embedder) is MemStrata's
  own write-path perception, run **in-process** here (no crop_server / file queue and
  no GroundingDINO phrase detector — GDINO weights are absent in this environment, and
  the orchestrator runs SAM3-concept-only when ``detector=None``). This is exactly the
  ``orchestrator.acquire_entity_crop`` path used by ``ProposeIdentifyCropper``.
* Entity naming: under the causal protocol the SUT gets no roster or gold. The default
  ``perception`` path proposes per-type candidates (character/prop/location) with no
  names, leaving identity to reconciliation. The ``mllm`` path additionally lets
  MemStrata's own VLM bind the user-visible segment prompt's names to entities it
  visually confirms in the sampled frames (the paper's *requested* observations, Sec.
  Evidence Acquisition), while unnamed/visible-only entities stay *discovered* candidates.
  Neither path reads benchmark annotations (roster, present sets, entity registry): naming
  uses only what every SUT sees (the prompt) plus MemStrata's own perception. No
  hand-tuned, eval-set-fitted prompt lexicons are ever used in this file.

Env: run the runner under a Python that has torch + transformers (>=5.9 for SAM3, via
the vendored ``models/vendor/sam3_transformers59`` bundle prepended on PYTHONPATH) and
can import ``memstrata`` (sibling ``../MemStrata/src``, or ``MEMSTRATA_SRC``).
Weights resolve under ``PUBLIC_MODELS_ROOT`` (SAM3 at
``facebook/sam3``, DINOv3 at ``facebook/dinov3-vitb16-pretrain-lvd1689m``).
Run via runner.py --adapter memstrata.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _local_roots import find_memstrata_src

_SRC = find_memstrata_src()
_DEFAULT_PUBLIC_MODELS_ROOT = ""
# WeDetect-Ref is the DEFAULT crop backend (describe->bbox); the isolated service normally
# listens here. Override with MEMSTRATA_WEDETECT_URL; set it to "" only to force it off.
_DEFAULT_WEDETECT_URL = "http://127.0.0.1:8710"

# NOTE: no hand-written prompt-name lexicons. The old code typed prompt tokens via
# hand-tuned word lists fitted to the eval set (e.g. 守塔人/灯塔/考古学家, carrot/
# lighthouse); that is eval-set fitting and is prohibited by the fairness contract.
# Prompt-name *use* is allowed only through MemStrata's own VLM (name_source="mllm"),
# which binds a visible-prompt name to an entity it visually confirms in the frames —
# a method capability applied identically to every movie, never a hand-tuned lexicon.


def _ensure_real_memstrata_pkg() -> None:
    """Make ``import memstrata`` resolve to the real package, not this adapter file.

    ``runner.py`` loads adapters by module name, so this file is imported as the
    top-level module ``memstrata`` (runner puts the causal dir on ``sys.path[0]``),
    which shadows the ``memstrata`` **package** under ``src/``. We put ``src`` ahead of
    the causal dir and drop the shadowing binding so ``from memstrata.bank import ...``
    finds the package. The already-built adapter instance keeps working via closures.
    """
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    else:
        sys.path.remove(str(_SRC))
        sys.path.insert(0, str(_SRC))
    mod = sys.modules.get("memstrata")
    if mod is not None and getattr(mod, "__file__", "").endswith("/causal/memstrata.py"):
        sys.modules.pop("memstrata", None)


class MemStrataAdapter:
    name = "memstrata"

    def __init__(
        self,
        *,
        public_models_root: str | None = None,
        ffmpeg: str = os.environ.get("FFMPEG", "ffmpeg"),
        device: str = "",
        enable_perception: bool = True,
        identity_threshold: float = 0.25,
        frame_pos: float = 0.5,
        name_source: str = "perception",  # "perception" (unnamed candidates) | "mllm" (VLM binds visible-prompt names)
        max_reps_per_asset: int = 5,
        decompose_frames: int = 3,
        decompose_fps: float = 2.0,
    ) -> None:
        self.public_models_root = str(
            public_models_root
            or os.environ.get("PUBLIC_MODELS_ROOT")
            or _DEFAULT_PUBLIC_MODELS_ROOT
        )
        if self.public_models_root:
            os.environ.setdefault("PUBLIC_MODELS_ROOT", self.public_models_root)
        self.ffmpeg = ffmpeg
        self.device = str(device)
        self.enable_perception = bool(enable_perception)
        self.identity_threshold = float(identity_threshold)
        self.frame_pos = min(max(float(frame_pos), 0.0), 1.0)
        self.name_source = str(name_source)
        self.max_reps_per_asset = int(max_reps_per_asset)
        # Crop backend policy. WeDetect-Ref (describe->bbox) is the default; SAM3 concept
        # segmentation is only loaded when it is actually needed, to save GPU + startup:
        #   MEMSTRATA_ENABLE_SAM3 = auto (default) -> load SAM3 only if WeDetect is down OR
        #                            the perception naming path (no descriptions) is in use
        #                         = on  -> always load SAM3 (grounder still authoritative when up)
        #                         = off -> never load SAM3 (WeDetect-only; misses if it is down)
        self._sam3_mode = os.environ.get("MEMSTRATA_ENABLE_SAM3", "auto").strip().lower()
        if self._sam3_mode not in ("auto", "on", "off"):
            self._sam3_mode = "auto"
        # mllm branch: uniformly sample at decompose_fps, then DINO-select this many
        # diverse keyframes for the decomposer. More/diverse views make naming and
        # description more robust; the SAM3 crop is taken from the representative keyframe.
        self.decompose_frames = max(1, int(decompose_frames))
        self.decompose_fps = float(decompose_fps)
        # Read-side fast→slow cascade: when fast name+description matching misses, spend
        # one bounded MLLM resolver call to bridge aliases / coreference / cross-lingual
        # references to stored assets (still prompt + bank only, no gold). Defaults on for
        # the mllm naming branch; override with MEMSTRATA_TRACKA_READ_SLOW_FALLBACK=0/1.
        _rsf = os.environ.get("MEMSTRATA_TRACKA_READ_SLOW_FALLBACK", "").strip().lower()
        if _rsf in ("1", "true", "on", "yes"):
            self.read_slow_fallback = True
        elif _rsf in ("0", "false", "off", "no"):
            self.read_slow_fallback = False
        else:
            self.read_slow_fallback = self.name_source == "mllm"

        # Read-side reps-per-matched-asset (breadth-first composition). We used to return up
        # to 3 reps/asset for "equal-budget parity" with retrieval baselines that fill the
        # benchmark ceiling -- but visual-coverage recall is scored PER ENTITY (any one rep
        # depicting an entity covers it), so extra same-entity reps add zero recall while
        # they are near-duplicate DINOv3 crops that spike the redundancy metric (measured
        # redundancy_sim 0.67, 45% of refs were >=2nd copy of one entity) and dilute
        # precision/selection-efficiency. Default is now 1 rep/asset: one good rep per
        # matched entity maximises DISTINCT-entity breadth within the budget. Raise it only
        # to probe multi-angle behaviour. Override with MEMSTRATA_TRACKA_READ_MAX_REPS /
        # MEMSTRATA_TRACKA_READ_CONTEXT_BUDGET.
        try:
            self.read_max_reps_per_asset = max(
                1, int(os.environ.get("MEMSTRATA_TRACKA_READ_MAX_REPS", "1") or 1)
            )
        except ValueError:
            self.read_max_reps_per_asset = 1
        _rcb = os.environ.get("MEMSTRATA_TRACKA_READ_CONTEXT_BUDGET", "").strip()
        try:
            self.read_context_budget = int(_rcb) if _rcb else None
        except ValueError:
            self.read_context_budget = None

        # Snapshot export cadence. Default ON per segment for long-run monitoring: memory.json
        # is refreshed after every segment so the membank can be inspected live while a movie
        # runs. Note this is O(bank) per segment (≈quadratic over a movie) and sits inside the
        # write-side timing window, so a CLEAN write-latency measurement should turn it off with
        # MEMSTRATA_TRACKA_SNAPSHOT_EACH_SEGMENT=0 (finalize() still exports the final snapshot).
        self.snapshot_each_segment = os.environ.get(
            "MEMSTRATA_TRACKA_SNAPSHOT_EACH_SEGMENT", "1"
        ).strip().lower() in ("1", "true", "on", "yes")

        # Built in reset().
        self._movie: MovieContext | None = None
        self._bank = None
        self._curator = None
        self._interpreter_cls = None
        self._compose_fn = None
        self._intent_resolver = None
        self._asset_type = None
        self._observation_cls = None
        self._segmenter = None
        self._embedder = None
        self._grounder = None  # WeDetect-Ref describe->bbox grounder (None => SAM3 path)
        self._acquire = None
        self._vlm_decomposer = None  # built in reset() only when name_source="mllm"
        # Crop-attribute classifier for the mllm write branch (built once in reset()).
        # Default null offline; a real VLM when MEMSTRATA_CROP_ATTR_CLASSIFIER is set.
        self._crop_attr_classifier = None
        self._work_dir: Path | None = None
        # representation_id -> absolute source seconds (our temporal-identity table).
        self._rep_seconds: dict[str, float] = {}
        self._counts = {"char_hits": 0, "prop_hits": 0, "scene": 0, "scene_hits": 0, "segments": 0}
        self._retrieval_sources = {"name": 0, "description": 0, "recency": 0, "mllm": 0, "miss": 0}

    # ---- lifecycle --------------------------------------------------------
    def reset(self, movie: MovieContext) -> None:
        _ensure_real_memstrata_pkg()
        from memstrata.bank import AssetBank, AssetType
        from memstrata.encoders import HashEmbedding
        from memstrata.skills.memory_update.curator import MemoryUpdater, MemoryPolicy
        from memstrata.skills.intent_understanding import IntentInterpreter, MllmIntentResolver
        from memstrata.skills.composition.compose import compose as compose_fn
        from memstrata.skills.decomposition.decomposer import Observation

        self._movie = movie
        self._asset_type = AssetType
        self._observation_cls = Observation
        self._interpreter_cls = IntentInterpreter
        self._compose_fn = compose_fn

        # Read-side slow-fallback resolver (MemStrata's own MLLM intent path over the bank
        # listing). Built lazily-cheap: MllmPlanner connects only when the cascade fires.
        self._intent_resolver = None
        if self.read_slow_fallback:
            from memstrata.mllm.planner import MllmPlanner
            self._intent_resolver = MllmIntentResolver(MllmPlanner())

        self._bank = AssetBank()
        # Perception (DINOv3 always + WeDetect grounder by default + SAM3 only when needed)
        # is built first so the bank can share its DINOv3 encoder.
        if self.enable_perception:
            self._build_perception()

        # Bank identity/dedup embedder. Name-anchored *requested* assets don't need it, but
        # *discovered* observations are merged across segments by visual identity (χ in
        # curator._reconcile_identity), which is meaningless under the content-hash fallback:
        # two crops of the same rabbit hash to near-orthogonal vectors, so nothing ever merges
        # and the same entity fragments into 棕色兔子/棕色动物/橙色兔子/… . When perception is
        # active we already load DINOv3 for crop acquisition — reuse that SAME instance for the
        # bank (no extra GPU memory) so reconciliation is genuinely semantic; this is what the
        # run's MEMSTRATA_GENERAL_EMBEDDER_PROVIDER=dinov3 already asks for. Offline / no
        # perception falls back to HashEmbedding, and the default policy's cohesion floor stays
        # 0 for a non-semantic encoder either way (no silent gate activation).
        emb = self._embedder if (self.enable_perception and self._embedder is not None) else HashEmbedding()
        # ``attributes_when_angles_known=False`` lets crop-attribute packs the mllm branch
        # attaches (known angles) pass through curate untouched; the perception path is
        # unaffected because its observations always carry UNKNOWN angles.
        # Apply the paper's long-video PRODUCTION preset (curator.py MemoryPolicy.production):
        # per-type calibrated β_τ/γ_τ (character merges only ≥0.75, prop ≥0.21, not a flat
        # 0.55), the cohesion admission floor + per-segment self-audit, the crop-quality gate,
        # and the 512-rep global budget. Without it the curator ran on library defaults, so the
        # "stratified / self-audited / budgeted" bank the paper describes was inert. The VLM-first
        # identity judge that production enables stays inert unless MEMSTRATA_IDENTITY_JUDGE
        # injects a real judge, so this does not silently add model calls. Explicit kwargs still
        # win: ``attributes_when_angles_known=False`` keeps the mllm branch's own crop-attribute
        # packs from being re-classified. The cohesion floor only activates because ``emb`` is the
        # semantic DINOv3 encoder (curator refuses a preset floor under a non-semantic embedder).
        self._curator = MemoryUpdater(
            self._bank, emb,
            policy=MemoryPolicy.production(),
            max_reps_per_asset=self.max_reps_per_asset,
            attributes_when_angles_known=False,
        )

        self._rep_seconds = {}
        self._counts = {"char_hits": 0, "prop_hits": 0, "scene": 0, "scene_hits": 0, "segments": 0}
        self._retrieval_sources = {"name": 0, "description": 0, "recency": 0, "mllm": 0, "miss": 0}
        self._work_dir = Path(movie.work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Write-side naming via MemStrata's own multimodal model (Stage: Decompose).
        self._vlm_decomposer = None
        self._crop_attr_classifier = None
        if self.name_source == "mllm":
            from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer
            from memstrata.mllm.crop_attributes import build_crop_attribute_classifier

            self._vlm_decomposer = VlmEntityDecomposer()
            # One classifier for the whole movie: null offline, real VLM when
            # MEMSTRATA_CROP_ATTR_CLASSIFIER (or MEMSTRATA_ANGLE_CLASSIFIER) selects it.
            self._crop_attr_classifier = build_crop_attribute_classifier()

    def _build_perception(self) -> None:
        from memstrata.skills.crop_acquisition.embedding import DinoV3Embedder
        from memstrata.skills.crop_acquisition.orchestrator import acquire_entity_crop
        from memstrata.skills.crop_acquisition.wedetect_client import WeDetectRefGrounder

        dev = self.device or None
        # DINOv3 is always needed (identity gate, bank reconciliation, keyframe selection).
        self._embedder = DinoV3Embedder(device=dev)
        self._embedder._ensure_loaded()
        self._acquire = acquire_entity_crop

        # WeDetect-Ref describe->bbox grounder is the DEFAULT crop backend: it grounds the
        # crop from the entity DESCRIPTION, fixing the SAM3-concept "most-salient-wins"
        # mis-crop (a "squirrel" query returning the salient rabbit). Default URL; set
        # MEMSTRATA_WEDETECT_URL="" to force it off. None when the service is unreachable.
        url = os.environ.get("MEMSTRATA_WEDETECT_URL", _DEFAULT_WEDETECT_URL).strip()
        self._grounder = None
        if url:
            try:
                score = float(os.environ.get("MEMSTRATA_WEDETECT_SCORE_THRE", "0.25") or 0.25)
            except ValueError:
                score = 0.25
            try:
                topk = int(os.environ.get("MEMSTRATA_WEDETECT_TOPK", "5") or 5)
            except ValueError:
                topk = 5
            cand = WeDetectRefGrounder(url, score_thre=score, topk=topk)
            self._grounder = cand if cand.healthy() else None

        # SAM3 concept segmentation is loaded ONLY when needed (saves GPU + startup): when
        # forced on, or (auto) when the grounder is down OR the perception naming path is in
        # use (generic names carry no description for the grounder). mllm + grounder-up => no SAM3.
        load_sam3 = self._sam3_mode == "on" or (
            self._sam3_mode == "auto"
            and (self._grounder is None or self.name_source == "perception")
        )
        self._segmenter = None
        if load_sam3:
            from memstrata.skills.crop_acquisition.sam3_concept import Sam3ConceptSegmenter

            self._segmenter = Sam3ConceptSegmenter(device=dev)
            self._segmenter._ensure_loaded()

    # ---- helpers ----------------------------------------------------------
    def _probe_duration(self, segment_video: str) -> float:
        try:
            return float(subprocess.run(
                [self.ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(segment_video)],
                check=True, capture_output=True, text=True).stdout.strip() or 0.0)
        except Exception:
            return 0.0

    def _cut_frame_at(self, segment_video: str, out_path: Path, pos: float) -> Path | None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dur = self._probe_duration(segment_video)
        ss = max(0.0, min(max(pos, 0.0), 1.0) * dur) if dur > 0 else 0.0
        proc = subprocess.run(
            [self.ffmpeg, "-y", "-ss", f"{ss:.3f}", "-i", str(segment_video),
             "-threads", os.environ.get("VMEM_FFMPEG_THREADS", "1"),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            return out_path
        return None

    def _cut_frame(self, segment_video: str, out_path: Path) -> Path | None:
        # Representative frame at frame_pos (used for cropping + as the primary view).
        return self._cut_frame_at(segment_video, out_path, self.frame_pos)

    def _decode_uniform_frames(self, segment_video: str, out_dir: Path, fps: float) -> list[str]:
        """Uniformly sample frames across the segment at ``fps`` (ffmpeg -vf fps)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / "cand_%03d.png")
        proc = subprocess.run(
            [self.ffmpeg, "-y", "-i", str(segment_video),
             "-threads", os.environ.get("VMEM_FFMPEG_THREADS", "1"),
             "-vf", f"fps={max(0.1, float(fps))}", "-q:v", "2", pattern],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            return []
        return sorted(str(p) for p in out_dir.glob("cand_*.png") if p.stat().st_size > 0)

    def _keyframes(self, segment_video: str, segment_dir: Path, fallback: Path) -> list[str]:
        """Uniform-sample then DINO farthest-point select up to ``decompose_frames`` frames.

        Pipeline: ffmpeg uniform decode at ``decompose_fps`` -> DINOv3 diverse-keyframe
        selection (representative frame first). This shows the decomposer diverse views
        and within-segment state changes without near-duplicate frames, and the first
        keyframe is a good crop source. Falls back to the single representative
        ``fallback`` frame when decode/embedder are unavailable. Naming-side only — no
        gold/recency signal enters memory.
        """
        from memstrata.skills.crop_acquisition.keyframes import select_diverse_keyframes

        k = self.decompose_frames
        candidates = self._decode_uniform_frames(
            segment_video, segment_dir / "cand", self.decompose_fps)
        if not candidates:
            return [str(fallback)]
        if self._embedder is None or k <= 1:
            return candidates[:k] if k > 1 else [str(fallback)]
        picked = select_diverse_keyframes(candidates, self._embedder, k=k)
        return picked or [str(fallback)]

    def _entity_reference_vectors(self, asset_id: str) -> tuple[list, list]:
        """(exemplar_vectors, existing_rep_vectors) for the DINOv3 identity/novelty gate.

        Reads the embeddings the curator already stored on each rep
        (``rep.annotations["embedding"]``, same DINOv3 route), instead of re-encoding the
        crop files every segment — the old ``_entity_images`` returned paths and the caller
        ran ``embed_batch`` on the whole existing set on EVERY acquisition, i.e. O(reps) wasted
        DINOv3 forwards per entity per segment. Only reps missing a cached vector (rare:
        pre-embedding placeholders) are embedded here as a fallback.
        """
        asset = self._bank.get_asset(asset_id)
        if asset is None:
            return [], []
        reps_info: list[list] = []  # [embedding_or_None, uri, anchor_ok]
        to_embed: list[str] = []
        for rep in asset.representations:
            if getattr(rep, "deprecated", False) or not rep.object_uri:
                continue
            ann = rep.annotations or {}
            emb = ann.get("embedding")
            anchor_ok = ann.get("identity_anchor_eligible") is not False
            reps_info.append([emb, str(rep.object_uri), anchor_ok])
            if not emb:
                to_embed.append(str(rep.object_uri))
        if to_embed and self._embedder is not None:
            try:
                vecs = self._embedder.embed_batch([Path(p) for p in to_embed])
            except Exception:
                vecs = []
            if len(vecs) == len(to_embed):
                by_uri = dict(zip(to_embed, vecs))
                for info in reps_info:
                    if not info[0]:
                        info[0] = by_uri.get(info[1])
        exemplar: list = []
        existing: list = []
        for emb, _uri, anchor_ok in reps_info:
            if not emb:
                continue
            existing.append(emb)
            if anchor_ok:
                exemplar.append(emb)
        return exemplar, existing

    def _acquire_kind(
        self,
        frame_path: Path,
        *,
        name: str,
        kind: str,
        out_dir: Path,
        identity_key: str | None = None,
        category: str = "",
        description: str = "",
    ) -> str | None:
        # Need SOME crop backend: WeDetect grounder OR SAM3 segmenter. (SAM3 may be unloaded
        # by default now; the grounder alone is a valid backend.)
        if self._acquire is None or (self._segmenter is None and self._grounder is None):
            return None
        exemplar_vecs, existing_vecs = self._entity_reference_vectors(identity_key or name)
        # The decomposer's per-entity English ``category`` drives the open-vocab SAM3 concept
        # (generic "object" finds no props); kind concepts stay as fallback inside the
        # orchestrator. ``description`` feeds the GDINO phrase path when a detector exists.
        concepts = (category.strip(),) if category and category.strip() else None
        try:
            res = self._acquire(
                frame_path,
                entity_name=name,
                entity_kind=kind,
                entity_description=description,
                concepts=concepts,
                exemplar_vectors=exemplar_vecs,
                existing_rep_vectors=existing_vecs,
                out_dir=out_dir,
                segmenter=self._segmenter,
                detector=None,  # GroundingDINO absent; SAM3 concept proposals only.
                grounder=self._grounder,  # WeDetect-Ref describe->bbox (authoritative when up).
                embedder=self._embedder,
                identity_threshold=self.identity_threshold,
            )
        except Exception:
            return None
        return res.get("crop_path") if res else None

    # ---- WRITE ------------------------------------------------------------
    def observe_segment(self, obs: SegmentObservation) -> None:
        cid = int(obs.chunk_id)
        t0, t1 = float(obs.seconds_span[0]), float(obs.seconds_span[1])
        source_seconds = t0 + self.frame_pos * max(0.0, t1 - t0)
        self._counts["segments"] += 1

        segment_dir = self._work_dir / f"segment_{cid:05d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        frame_path = self._cut_frame(obs.segment_video, segment_dir / "frame.png")
        if frame_path is None:
            return

        AssetType = self._asset_type
        Observation = self._observation_cls
        from memstrata.skills.decomposition.decomposer import SOURCE_DISCOVERED, SOURCE_REQUESTED
        observations = []
        wrote_via_mllm = False

        # WRITE-SIDE NAMING SOURCE ---------------------------------------------------------
        # name_source="mllm": MemStrata's OWN multimodal model (skills.decomposition
        # VlmEntityDecomposer) decomposes sampled frames of the realized segment into typed,
        # labeled entities. Naming binds the user-visible segment prompt's names to entities
        # the VLM visually confirms in those frames (the paper's *requested* observations,
        # Sec. Evidence Acquisition); visible-only entities get descriptive labels and stay
        # *discovered* candidates. This uses only what every SUT sees (prompt + frames) plus
        # MemStrata's own perception — no gold/roster/registry. Cross-segment identity is then
        # routed by that requested/discovered split: a *requested* entity (label verbatim in
        # the prompt → entity_id set) is name-anchored in the bank, while a *discovered* entity
        # (entity_id None, source=discovered) is merged by the real MemoryUpdater's visual
        # reconciliation (χ over the shared DINOv3 encoder) so the same entity under drifting
        # descriptive labels collapses into one record instead of fragmenting into
        # 棕色兔子/棕色动物/橙色兔子/….
        # target description per acquired crop for the new-entity verification (path C):
        # a decomposer entity whose name matches no existing bank record but that carries
        # a description. Keyed by id(observation) so it survives the batch reordering.
        mllm_targets: dict[int, str] = {}
        if self.name_source == "mllm" and self._vlm_decomposer is not None:
            prompt_ctx = str(getattr(obs, "prompt_text", "") or "")
            naming_frames = self._keyframes(obs.segment_video, segment_dir, frame_path)
            # Crop from the representative keyframe (first) rather than a fixed time.
            mllm_frame = Path(naming_frames[0]) if naming_frames else frame_path
            mllm_has_location = False
            for ent in self._vlm_decomposer.propose(frames=naming_frames, prompt=prompt_ctx):
                # Requested (label verbatim in prompt) → anchor by name; discovered → leave
                # entity_id None + source=discovered so curate reconciles it visually (χ)
                # into the right existing record instead of spawning a new name-keyed entity.
                ent_source = SOURCE_REQUESTED if ent.entity_id else SOURCE_DISCOVERED
                if ent.kind == AssetType.LOCATION:
                    # Ground the location DESCRIPTION to a scene-region crop instead of banking
                    # the raw full frame. The old full-frame rep is dominated by the salient
                    # foreground character, so the visual-coverage judge attributed it to that
                    # character ('none' for the location) -- locations were covered ~0 despite
                    # being in the bank. WeDetect-Ref grounds a scene phrase ("sunny green
                    # meadow…") to the background region (e.g. the grass band), which the judge
                    # can credit as the location. Fall back to the full frame when grounding
                    # yields nothing (service down / no box) so location recall never regresses.
                    loc_crop = self._acquire_kind(
                        mllm_frame, name=ent.name, kind="location",
                        out_dir=segment_dir / "location", identity_key=ent.name,
                        category=ent.category, description=ent.description,
                    )
                    loc_image = loc_crop or str(mllm_frame)
                    if loc_crop:
                        self._counts["scene_hits"] += 1
                    loc_obs = Observation(
                        observation_id=f"{ent.name}@s{cid:05d}",
                        kind=AssetType.LOCATION, name=ent.name, image_path=loc_image,
                        entity_id=ent.entity_id, source=ent_source,
                        temporal_tag=f"segment_{cid}", description=ent.description,
                        source_frame_path=str(mllm_frame),
                    )
                    if ent.description and self._is_new_entity(ent.name, ent.kind):
                        mllm_targets[id(loc_obs)] = ent.description
                    observations.append(loc_obs)
                    self._counts["scene"] += 1
                    mllm_has_location = True
                    wrote_via_mllm = True
                    continue
                # char/prop need an isolated crop: SAM3 concept segments by the generic kind,
                # identity is keyed by the VLM label so reps of the same named entity group.
                concept = "character" if ent.kind == AssetType.CHARACTER else "prop"
                crop = self._acquire_kind(mllm_frame, name=ent.name, kind=concept,
                                          out_dir=segment_dir / concept, identity_key=ent.name,
                                          category=ent.category, description=ent.description)
                if crop:
                    self._counts["char_hits" if ent.kind == AssetType.CHARACTER else "prop_hits"] += 1
                    ent_obs = Observation(
                        observation_id=f"{ent.name}@s{cid:05d}",
                        kind=ent.kind, name=ent.name, image_path=str(crop),
                        entity_id=ent.entity_id, source=ent_source,
                        temporal_tag=f"segment_{cid}", description=ent.description,
                        source_frame_path=str(mllm_frame),
                    )
                    if ent.description and self._is_new_entity(ent.name, ent.kind):
                        mllm_targets[id(ent_obs)] = ent.description
                    observations.append(ent_obs)
                    wrote_via_mllm = True
            if wrote_via_mllm:
                self._retrieval_sources["mllm"] += 1
                if not mllm_has_location:
                    # Keep the recurring place stratum even if the VLM named no location.
                    observations.append(Observation(
                        observation_id=f"scene@s{cid:05d}",
                        kind=AssetType.LOCATION, name="scene", image_path=str(mllm_frame),
                        entity_id="scene", temporal_tag=f"segment_{cid}",
                    ))
                    self._counts["scene"] += 1
                # Crop-attribute VLM pass: one classify_batch over every acquired crop,
                # attaching real occlusion/state/appearance and running path-C verification.
                observations = self._classify_and_attach_mllm(observations, mllm_targets, cid)

        # PERCEPTION DEFAULT / FALLBACK ----------------------------------------------------
        # Faithful to the paper's write path (Sec. Evidence Acquisition): the SAM3-concept
        # proposer proposes per-type candidates and NEVER infers names; identity is decided
        # downstream by reconciliation. This is the default (name_source="perception") and
        # the fallback whenever the MLLM decomposer produced nothing (e.g. server down).
        if not wrote_via_mllm:
            if self.enable_perception:
                for name, kind_str, kind in (("character", "character", AssetType.CHARACTER),
                                              ("prop", "prop", AssetType.PROP)):
                    crop = self._acquire_kind(frame_path, name=name, kind=kind_str,
                                              out_dir=segment_dir / kind_str, identity_key=name)
                    if crop:
                        self._counts[f"{'char' if kind_str == 'character' else 'prop'}_hits"] += 1
                        observations.append(Observation(
                            observation_id=f"{name}@s{cid:05d}",
                            kind=kind, name=name, image_path=str(crop), entity_id=name,
                            temporal_tag=f"segment_{cid}",
                            source_frame_path=str(frame_path),
                        ))

            # Scene / location: bank the whole frame as the recurring scene identity
            # (continuity-of-place, MemStrata's location stratum).
            observations.append(Observation(
                observation_id=f"scene@s{cid:05d}",
                kind=AssetType.LOCATION, name="scene", image_path=str(frame_path),
                entity_id="scene", temporal_tag=f"segment_{cid}",
            ))
            self._counts["scene"] += 1

        touched = self._curator.curate_observations(observations, segment_id=cid)
        # Record the temporal identity of every rep this segment contributed.
        for aid in touched:
            asset = self._bank.get_asset(aid)
            if asset is None:
                continue
            for rep in asset.representations:
                if int(rep.origin_segment_id) == cid:
                    self._rep_seconds[rep.representation_id] = source_seconds
                    rep.annotations["source_seconds"] = source_seconds

        # Emit the curated memory product. Off the per-segment hot path by default
        # (finalize() always exports); opt in for long-run monitoring.
        if self.snapshot_each_segment:
            self._export_snapshot()

    # ---- crop-attribute + snapshot helpers --------------------------------
    def _is_new_entity(self, name: str, kind: Any) -> bool:
        """Whether ``name`` matches no existing bank record yet (new-entity path C)."""
        if self._bank is None:
            return True
        if self._bank.get_asset(name) is not None:
            return False
        try:
            if self._bank.find_by_name(name, kind=kind) is not None:
                return False
        except Exception:
            pass
        return True

    def _classify_and_attach_mllm(
        self, observations: list, targets_by_id: dict[int, str], cid: int
    ) -> list:
        """One crop-attribute ``classify_batch`` over the segment's mllm crops.

        Attaches each pack's flattened annotations onto its Observation (occlusion,
        state/spatial angle, appearance description, shot size, ...). When a crop carried
        a new-entity target description (path C) and the pack reports ``matches_target``
        False, the observation is dropped before curate — the first-sighting verification.
        Fully defensive: any failure returns the observations unchanged (no attributes).
        """
        if self._crop_attr_classifier is None or not observations:
            return observations
        path_obs = [o for o in observations if getattr(o, "image_path", "")]
        if not path_obs:
            return observations
        try:
            targets = [targets_by_id.get(id(o)) for o in path_obs]
            items = [
                {"image_path": o.image_path, "kind": o.kind.value,
                 "name": o.name, "segment_id": cid}
                for o in path_obs
            ]
            packs = self._crop_attr_classifier.classify_batch(
                items, target_descriptions=targets if any(targets) else None
            )
        except Exception:
            return observations
        if len(packs) != len(path_obs):
            return observations
        pack_by_id = {id(o): p for o, p in zip(path_obs, packs)}
        kept = []
        for o in observations:
            pack = pack_by_id.get(id(o))
            if pack is None:
                kept.append(o)
                continue
            # Path C: drop only on an explicit non-match (keep when key absent/None).
            if pack.extra.get("matches_target") is False:
                continue
            try:
                o.spatial_angle = pack.spatial_angle
                o.state_angle = pack.state_angle
                o.angle_meta = pack.to_annotations()
                if pack.description and not o.description:
                    o.description = pack.description
            except Exception:
                pass
            kept.append(o)
        return kept

    def _export_snapshot(self) -> None:
        """Write ``<work_dir>/memory.json`` (curated memory product). Skips gracefully."""
        if self._bank is None or self._work_dir is None:
            return
        try:
            from memstrata.skills.memory_update.snapshot import export_memory_snapshot

            movie_id = str(getattr(self._movie, "movie_id", "") or "") if self._movie else ""
            export_memory_snapshot(
                self._bank, self._work_dir, movie_id=movie_id, video_path=None
            )
        except Exception:
            pass

    # ---- READ -------------------------------------------------------------
    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        rec = RetrievedMemory(chunk_id=req.chunk_id)
        if self._bank is None:
            return rec
        cid = int(req.chunk_id)
        interpreter = self._interpreter_cls(
            self._bank, resolver=self._intent_resolver, mode="fast",
            slow_on_miss=self.read_slow_fallback,
            max_reps_per_asset=self.read_max_reps_per_asset,
            context_rep_budget=self.read_context_budget,
        )
        request, _calls = interpreter.interpret(req.prompt_text, segment_id=cid)
        ctx = self._compose_fn(self._bank, request, as_of_segment_id=cid)
        source = getattr(request, "intent_resolution_source", getattr(ctx, "intent_resolution_source", "recency"))
        rec.extras["intent_resolution_source"] = source
        rec.extras["intent_asset_ids"] = [ref.asset_id for ref in request.references]

        seen: set[str] = set()
        for aid in ctx.asset_ids:
            for rid in ctx.representation_ids.get(aid, []):
                sec = self._seconds_for_rep(rid)
                if sec is None or sec >= float(req.seconds_span[0]):
                    continue
                image_path = self._image_path_for_rep(rid)
                rec.items.append(RetrievedItem(
                    evidence_kind="reference_image", source_seconds=sec,
                    raw_ref=f"memstrata:{aid}:{rid}", image_path=image_path))
                seen.add(rid)

        self._retrieval_sources[source] = self._retrieval_sources.get(source, 0) + 1
        return rec

    def _seconds_for_rep(self, rep_id: str) -> float | None:
        if rep_id in self._rep_seconds:
            return self._rep_seconds[rep_id]
        found = self._bank.find_representation(rep_id)
        if found is not None:
            sec = (found[1].annotations or {}).get("source_seconds")
            if sec is not None:
                return float(sec)
        return None

    def _image_path_for_rep(self, rep_id: str) -> str | None:
        found = self._bank.find_representation(rep_id)
        if found is None:
            return None
        uri = str(getattr(found[1], "object_uri", "") or "").strip()
        if not uri:
            return None
        path = Path(uri).expanduser()
        if not path.is_absolute() and self._work_dir is not None:
            path = self._work_dir / path
        try:
            return str(path.resolve())
        except OSError:
            return str(path.absolute())

    def finalize(self) -> dict[str, Any]:
        # Final curated memory product (whole-movie view).
        self._export_snapshot()
        n_assets = len(self._bank.assets) if self._bank is not None else 0
        reps = {aid: len(a.representations) for aid, a in self._bank.assets.items()} if self._bank else {}
        # Authoritative crop backend actually wired up this run (grounder wins over SAM3 fallback).
        crop_backend = (
            "wedetect_ref" + ("+sam3_fallback" if self._segmenter is not None else "")
            if self._grounder is not None
            else ("sam3_concept" if self._segmenter is not None else "none")
        )
        # Human-readable crop stage that mirrors `crop_backend` (never a hard-coded backend name).
        crop_label = (
            "WeDetect-Ref (describe->bbox) crop"
            if self._grounder is not None
            else ("SAM3-concept crop" if self._segmenter is not None else "no crop")
        )
        return {
            "system": "memstrata",
            "read_path": (
                "IntentInterpreter(FAST name/description) + slow-on-miss MLLM cascade + compose"
                if self.read_slow_fallback
                else "IntentInterpreter(name_anchored, model-free FAST) + compose"
            ),
            "write_path": (
                f"DINO-keyframes + VlmEntityDecomposer(Qwen3.5-9B) + {crop_label} + DINOv3 novelty + MemoryUpdater"
                if self.name_source == "mllm"
                else f"{crop_label} + DINOv3 novelty (ProposeIdentify core) + MemoryUpdater"
            ),
            "name_source": self.name_source,
            "crop_backend": crop_backend,
            "read_slow_fallback": self.read_slow_fallback,
            "read_max_reps_per_asset": self.read_max_reps_per_asset,
            "read_context_budget": self.read_context_budget,
            "decompose_frames": self.decompose_frames,
            "decompose_fps": self.decompose_fps,
            "perception_enabled": self.enable_perception,
            "identity_threshold": self.identity_threshold,
            "public_models_root": self.public_models_root,
            "n_assets": n_assets,
            "representations": reps,
            "counts": dict(self._counts),
            "retrieval_sources": dict(self._retrieval_sources),
            "n_rep_seconds": len(self._rep_seconds),
        }


def build_adapter() -> MemStrataAdapter:
    # Track A selects the write-side naming path via env (runner calls build_adapter()
    # with no args, like RETR_BUDGET). "mllm" uses MemStrata's own VlmEntityDecomposer to
    # name entities from the observed segment + visible prompt (method-side memory build,
    # no gold/roster leakage); "perception" is the SAM3/DINOv3 generic-name default.
    name_source = os.environ.get("MEMSTRATA_TRACKA_NAME_SOURCE", "perception").strip().lower()
    if name_source not in ("perception", "mllm"):
        raise SystemExit(
            f"MEMSTRATA_TRACKA_NAME_SOURCE must be 'perception' or 'mllm', got {name_source!r}"
        )
    return MemStrataAdapter(name_source=name_source)
