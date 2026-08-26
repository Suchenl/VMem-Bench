"""Track-first annotation orchestrator (docs/benchmark/annotation_tracking_internals.md §3.5).

Replaces the VLM-first discover->ground->draft->verify->per-crop-audit loop (pipeline.py) with a
deterministic identity backbone:

    1. SBD + chunking                          (run_chunking, unchanged)
    2. global cast roster                       (roster.select_roster_keyframes + RosterVlm, ONCE)
    3. per-shot detect+track -> tracklets       (PerceptionBackend: gdino_track | sam3_track)
    4. cross-shot re-ID -> global entity_id      (reid_assign, multi-cue: body + self-gated face)
    5. presence / first_appearance              (DETERMINISTIC: tracklet/​shot span ∩ chunk span)
    6. per-entity naming                         (NamerVlm, ONCE per entity)
    7. per-chunk prompt draft + state events     (AnnotatorRole.draft_chunk)
    8. gold instructions/forbidden/tags + persist + review   (drafting.py + pipeline._persist)

VLM calls drop from ~chunks×rounds×branches×(discover+draft+verify+per_crop) to
1(roster) + N_entities(naming) + chunks(draft). Identity is decided by appearance, not by the
VLM name, so the `white_rabbit` vs `white_rabbit_character` fragmentation is structurally gone.

Model-bound steps (backend, embedder, VLM roles, face encoder, frame extraction) are injected, so
the deterministic core (presence, first-appearance, chunk assembly) is unit-testable without GPU
(see tests/test_pipeline_track_first.py). Self-contained: imports only this benchmark + stdlib.
"""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vmem_bench.common.media import extract_frame, sample_frame_indices
from vmem_bench.common.paths import (
    MovieDirs, asset_crop_relpath, entity_asset_dir, entity_asset_relprefix)
from vmem_bench.annotation.pipeline_track_first.config import AnnotationConfig
from vmem_bench.annotation.pipeline_track_first.chunking import run_chunking
from vmem_bench.annotation.pipeline_track_first.events import EventLog
from vmem_bench.annotation.pipeline_track_first.consolidation import (
    Registry, _KIND_PREFIX, normalize_entity_name, present_payload, slugify)
from vmem_bench.annotation.pipeline_track_first.reid import (
    _entity_signature, commit_tracklet_observation, reid_assign)
from vmem_bench.annotation.pipeline_track_first import identity_resolution
from vmem_bench.annotation.pipeline_track_first.crop_classify import audit_registry_crops, reassign_label
from vmem_bench.annotation.pipeline_track_first.drafting import (
    build_chunk_annotation, filter_state_events,
    gold_instructions_for, materialize_forbidden, scenario_tags_for,
    state_events_from_draft)
from vmem_bench.annotation.pipeline_track_first import resume as resume_mod
from vmem_bench.annotation.pipeline_track_first import roster as roster_mod
from vmem_bench.annotation.pipeline_track_first import track_parallel
from vmem_bench.annotation.pipeline_track_first.perception.base import RosterEntry
from vmem_bench.annotation.pipeline_track_first.pipeline import _persist, _rel, _best_cover_rep
from vmem_bench.common.vecmath import cosine_similarity
from vmem_bench.common.schemas import Entity

logger = logging.getLogger(__name__)


# --- deterministic core (unit-testable, no GPU/VLM) ----------------------------------------

def is_closeup_shot(bboxes: Sequence[Sequence[int]], *, coverage_threshold: float) -> bool:
    """True iff any 0-1000-normalized box [x0,y0,x1,y1] covers more than ``coverage_threshold`` of the frame."""
    for box in bboxes:
        if len(box) < 4:
            continue
        x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        if ((x1 - x0) * (y1 - y0) / 1e6) > coverage_threshold:
            return True
    return False


def should_reuse_location_without_tracklets(
    scene_vec: Sequence[float],
    previous_location,
    embeddings: dict[str, list[float]],
    *,
    similarity_threshold: float,
) -> bool:
    """Return whether an untracked shot continues its immediately preceding location.

    A location's body store contains its full-scene embeddings. Comparing only against
    the previous location makes this a conservative temporal continuation rule, rather
    than a lower-threshold global re-ID match.
    """
    signature = _entity_signature(previous_location, embeddings)
    if signature is None or len(scene_vec) != len(signature):
        return False
    return cosine_similarity(list(scene_vec), signature) >= similarity_threshold


def cluster_scene_locations(
    scene_vectors: Sequence[Sequence[float]], *, similarity_threshold: float
) -> list[list[int]]:
    """Greedily cluster shot scene vectors by their running centroid.

    This is intentionally separate from character re-ID: a full frame changes with
    foreground actors and composition, so it is useful for coarse place clustering
    but not individual identity. Input order is retained for deterministic cluster IDs.
    """
    clusters: list[dict[str, Any]] = []
    for index, vector in enumerate(scene_vectors):
        vec = list(vector)
        best_index: int | None = None
        best_score = -1.0
        for cluster_index, cluster in enumerate(clusters):
            score = cosine_similarity(vec, cluster["centroid"])
            if score > best_score:
                best_index, best_score = cluster_index, score
        if best_index is not None and best_score >= similarity_threshold:
            cluster = clusters[best_index]
            cluster["indices"].append(index)
            members = cluster["vectors"]
            members.append(vec)
            cluster["centroid"] = [
                sum(member[dim] for member in members) / len(members)
                for dim in range(len(vec))
            ]
        else:
            clusters.append({"indices": [index], "vectors": [vec], "centroid": vec})
    return [cluster["indices"] for cluster in clusters]


def reslug_entities(registry: Registry, assets_dir: Path,
                    locked_ids: set[str] | None = None) -> dict[str, str]:
    """Re-slug entity_ids from final VLM names; rename asset dirs + embedding keys. Returns {old: new}.

    On-disk ``assets/{new_id}/`` from a prior run (``--resume`` after a successful reslug) is
    adopted rather than keeping the provisional id — otherwise gold keeps phrase-based ids while
    cached chunk annotations still reference the name-based ids from the first run.
    """
    import shutil

    new_entities: dict[str, Any] = {}
    idmap: dict[str, str] = {}
    locked_ids = set(locked_ids or ())
    for old_id, entity in registry.entities.items():
        if old_id in locked_ids:
            new_entities[old_id] = entity
            continue
        slug = slugify(entity.name)
        if slug in ("", "unnamed"):
            new_entities[old_id] = entity
            continue
        base = f"{_KIND_PREFIX.get(entity.kind, entity.kind)}_{slug}"
        candidate, n = base, 1
        while candidate in new_entities:
            n += 1
            candidate = f"{base}_{n:02d}"
        new_id = candidate
        if new_id == old_id:
            new_entities[old_id] = entity
            continue
        old_dir = entity_asset_dir(assets_dir, old_id, entity.kind)
        new_dir = entity_asset_dir(assets_dir, new_id, entity.kind)
        for rep in entity.representations:
            new_rid = new_id + rep.representation_id[len(old_id):]
            old_key = rep.embedding_key or rep.representation_id
            for store in (registry.embeddings, registry.face_embeddings, registry.class_embeddings):
                if old_key in store:
                    store[new_rid] = store.pop(old_key)
            rep.representation_id = new_rid
            rep.embedding_key = new_rid
            old_prefix = entity_asset_relprefix(old_id, entity.kind)
            if rep.crop_path.startswith(old_prefix):
                rep.crop_path = asset_crop_relpath(
                    new_id, entity.kind, rep.crop_path[len(old_prefix):])
            else:
                # Absolute/scratch paths or empty: point at the committed asset if present.
                leaf = new_rid.split("@")[-1].replace(".", "_") + ".jpg"
                if (new_dir / leaf).is_file() or (old_dir / leaf).is_file():
                    rep.crop_path = asset_crop_relpath(new_id, entity.kind, leaf)
        if old_dir.is_dir() and not new_dir.exists():
            old_dir.rename(new_dir)
        elif old_dir.is_dir() and new_dir.exists() and old_dir.resolve() != new_dir.resolve():
            for f in old_dir.iterdir():
                if f.is_file() and not (new_dir / f.name).exists():
                    shutil.copyfile(f, new_dir / f.name)
        entity.entity_id = new_id
        new_entities[new_id] = entity
        idmap[old_id] = new_id
    registry.entities = new_entities
    return idmap




def shots_from_boundaries_csv(out: Path) -> list[tuple[int, int]]:
    """Read shot boundaries csv -> list of closed-inclusive shot spans ``(first, last)``.

    Written by run_chunking (rows: shot_idx, start_frame, last_frame). Resolves
    ``gold/shot_boundaries.csv`` or legacy ``layout/boundaries.csv`` via MovieDirs."""
    from vmem_bench.common.paths import MovieDirs
    path = MovieDirs(Path(out), write=False).shot_boundaries
    shots: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shots.append((int(row["start_frame"]), int(row["last_frame"])))
    return shots


def frame_to_chunk_fn(chunks: Sequence[dict]) -> Callable[[int], int]:
    """Map an absolute frame index to its chunk_id (by closed-inclusive frame_span).

    Clamps out-of-range frames to the nearest chunk (a sampled frame can sit one past the last
    frame due to rounding). Chunks are assumed sorted by frame_span, which run_chunking guarantees.
    """
    spans = [(int(c["frame_span"][0]), int(c["frame_span"][1]), int(c["chunk_id"])) for c in chunks]

    def _lookup(frame_index: int) -> int:
        for f0, f1, cid in spans:
            if f0 <= frame_index <= f1:
                return cid
        if frame_index < spans[0][0]:
            return spans[0][2]
        return spans[-1][2]

    return _lookup


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def merge_spans(spans: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Merge overlapping/adjacent closed-inclusive frame spans -> sorted disjoint [start,end] list."""
    ordered = sorted((int(a), int(b)) for a, b in spans if b >= a)
    out: list[list[int]] = []
    for a, b in ordered:
        if out and a <= out[-1][1] + 1:      # overlap or touch -> extend
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def entity_time_metadata(spans: Sequence[tuple[int, int]], fps: float) -> dict[str, Any]:
    """Deterministic per-entity time metadata (Q3, §4.1) from its presence frame spans.

    Returns presence_spans (merged), first/last frame+seconds, screen_time_seconds (union duration),
    and max_absence (longest gap between consecutive spans = the memory distance the entity tests).
    Pure -> unit-testable. Empty spans -> all-None/empty."""
    merged = merge_spans(spans)
    if not merged or fps <= 0:
        return {"presence_spans": merged, "first_frame": None, "first_seconds": None,
                "last_frame": None, "last_seconds": None, "screen_time_seconds": None,
                "max_absence_frames": None, "max_absence_seconds": None}
    first_f, last_f = merged[0][0], merged[-1][1]
    screen_frames = sum(b - a + 1 for a, b in merged)
    gaps = [merged[i + 1][0] - merged[i][1] - 1 for i in range(len(merged) - 1)]
    max_gap = max(gaps) if gaps else 0
    return {"presence_spans": merged,
            "first_frame": first_f, "first_seconds": round(first_f / fps, 2),
            "last_frame": last_f, "last_seconds": round(last_f / fps, 2),
            "screen_time_seconds": round(screen_frames / fps, 2),
            "max_absence_frames": max_gap, "max_absence_seconds": round(max_gap / fps, 2)}


def presence_for_chunks(chunks: Sequence[dict],
                        entity_spans: dict[str, list[tuple[int, int]]]
                        ) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Deterministic presence (§3.4): an entity is present in a chunk iff any of its tracklet/​shot
    spans intersects the chunk's frame_span. Returns (present_by_chunk, first_appearance_chunk).

    ``present_by_chunk[cid]`` is sorted for reproducibility. ``first_appearance_chunk[eid]`` is the
    smallest chunk_id where the entity is present. Pure -> unit-testable."""
    present_by_chunk: dict[int, list[str]] = {}
    first_appearance: dict[str, int] = {}
    for c in chunks:
        cid = int(c["chunk_id"])
        cspan = (int(c["frame_span"][0]), int(c["frame_span"][1]))
        present = sorted(eid for eid, spans in entity_spans.items()
                         if any(_overlaps(cspan, s) for s in spans))
        present_by_chunk[cid] = present
        for eid in present:
            if eid not in first_appearance:
                first_appearance[eid] = cid
    return present_by_chunk, first_appearance


# --- orchestrator --------------------------------------------------------------------------

def annotate_movie_track_first(
    config: AnnotationConfig, *,
    roster_vlm, namer_vlm, drafter_vlm,
    backend=None, embedder, face_encoder=None,
    shots: list[tuple[int, int]] | None = None,
    max_workers: int = 8,
    resume: bool = False,
    track_devices: list[int] | None = None,
    extract_frame_fn: Callable[[Path, Path, int, float], Path] | None = None,
    text_embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    crop_classifier=None,
) -> dict[str, Any]:
    """Run the track-first pipeline end-to-end. ``backend`` defaults to the configured perception
    backend (needs a grounder/embedder wired by the caller); pass an explicit backend for tests."""
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = MovieDirs(out, write=True)
    dirs.mkdirs()
    crops_dir = dirs.candidates
    assets_dir = dirs.assets
    frames_dir = dirs.frames
    for d in (crops_dir, assets_dir, frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    evlog = EventLog(dirs.events)
    roster_seed = None
    if config.roster_seed_path is not None:
        from vmem_bench.annotation.pipeline_track_first.roster_seed import load_roster_seed
        roster_seed = load_roster_seed(
            config.roster_seed_path,
            expected_movie_id=config.movie_id,
            require_confirmed=config.production_mode,
        )
    elif config.production_mode:
        raise ValueError(
            "production annotation requires a human-confirmed roster_seed_path; "
            "automatic discovery is proposal-only")
    evlog.emit("run_start", movie_id=config.movie_id, video=str(config.video),
               backend=config.perception_backend, stage="chunking",
               roster_mode="seeded" if roster_seed else "proposal")

    # Resume safety: credits exclusion involves a VLM confirmation, so a resumed run must reuse
    # the persisted segments instead of re-detecting (the frozen layout depends on them).
    import json as _json
    prior_excluded: list[dict] = []
    if resume and dirs.chunk_index.is_file():
        try:
            prior_excluded = list(_json.loads(
                dirs.chunk_index.read_text(encoding="utf-8")).get("excluded_segments") or [])
        except (OSError, ValueError):
            prior_excluded = []

    index = run_chunking(config.video, out, min_frames=config.min_frames_per_chunk,
                         max_frames=config.max_frames_per_chunk, shots=shots,
                         sbd_method=config.sbd_method, min_scene_len_sec=config.min_scene_len_sec)
    fps = float(index["fps"])
    shots = shots or shots_from_boundaries_csv(out)

    _extract = extract_frame_fn or extract_frame
    n_frames = int(index.get("total_frames") or 0)
    # Tail-guarded frame extractor, shared with the multi-GPU workers (track_parallel.make_frame_path)
    # so pipeline and workers decode/cache frames identically.
    _frame_path = track_parallel.make_frame_path(config.video, frames_dir, fps, n_frames,
                                                 extract_fn=_extract)

    # --- step 1b: drop opening/ending credits BEFORE any annotation sees them ------------------
    if config.exclude_credits:
        from vmem_bench.annotation.pipeline_track_first import credits as credits_mod
        segments = prior_excluded or credits_mod.detect_credit_segments(
            shots, total_frames=n_frames, fps=fps, frame_path=_frame_path,
            confirm_fn=(roster_vlm.classify_credit_frames
                        if hasattr(roster_vlm, "classify_credit_frames") else None),
            head_tail_ratio=config.credits_head_tail_ratio,
            max_luminance=config.credits_max_luminance)
        if segments:
            shots = credits_mod.filter_shots(shots, segments)
            # Re-chunk over the kept shots only; run_chunking expects half-open spans.
            index = run_chunking(config.video, out, min_frames=config.min_frames_per_chunk,
                                 max_frames=config.max_frames_per_chunk,
                                 shots=[(s, e + 1) for s, e in shots],
                                 sbd_method=config.sbd_method,
                                 min_scene_len_sec=config.min_scene_len_sec,
                                 excluded_segments=segments)
            evlog.emit("credits_excluded", stage="chunking", segments=segments,
                       n_excluded_frames=sum(seg["frame_span"][1] - seg["frame_span"][0] + 1
                                             for seg in segments))

    chunks = index["chunks"]
    to_chunk = frame_to_chunk_fn(chunks)
    evlog.emit("layout", n_chunks=len(chunks), n_shots=len(shots), fps=fps,
               layout_hash=index["layout_hash"], stage="roster",
               excluded_segments=index.get("excluded_segments") or [],
               chunks=[{"chunk_id": int(c["chunk_id"]), "frame_span": list(c["frame_span"]),
                        "shot_span": list(c["shot_span"])} for c in chunks])

    def _embed_indices(idxs: list[int]) -> list[list[float]]:
        paths = [_frame_path(i) for i in idxs]
        if hasattr(embedder, "embed_batch"):
            return embedder.embed_batch(paths)
        return [embedder.embed_image(p) for p in paths]

    # --- step 2: canonical roster ---------------------------------------------------------------
    # Production identities come from a human-confirmed seed. Automatic VLM discovery remains a
    # proposal/debug path and is never allowed to define production gold.
    roster = roster_seed.to_roster() if roster_seed is not None else (
        resume_mod.load_roster(out) if resume else None)
    key_indices: list[int] = []
    if roster_seed is not None:
        resume_mod.save_roster(out, roster)
        n_keyframes = 0
        evlog.emit("roster_seed_loaded", stage="roster", n_entities=len(roster),
                   seed=str(roster_seed.source_path), human_confirmed=roster_seed.human_confirmed)
    elif roster is None:
        per_shot = roster_mod.shot_candidate_indices(shots, fps=fps,
                                                     candidate_fps=config.roster_candidate_fps)
        key_indices = roster_mod.select_roster_keyframes(
            per_shot, _embed_indices, per_shot_k=config.roster_per_shot_k,
            budget=config.roster_global_budget,
            budget_max=(config.roster_budget_max or None),
            novelty_threshold=config.roster_novelty_threshold,
            min_ratio=config.roster_budget_min_ratio, total_frames=n_frames)
        evlog.emit("roster_start", stage="roster",
                   n_keyframes_budget=config.roster_global_budget,
                   n_keyframes_max=config.roster_budget_max,
                   n_keyframes_selected=len(key_indices),
                   vlm_batch=max(1, config.roster_vlm_batch))
        key_frames = [_frame_path(i) for i in key_indices]
        batch_size = max(1, config.roster_vlm_batch)
        n_batches = (len(key_frames) + batch_size - 1) // batch_size
        batches: list[list[dict]] = []
        known: list[dict] = []
        for i in range(0, len(key_frames), batch_size):
            batch_frames = key_frames[i:i + batch_size]
            found = roster_vlm.discover_roster(batch_frames, known)
            batches.append(found)
            known = roster_mod.merge_roster(batches)  # feed cumulative roster forward to dedupe
            evlog.emit("roster_progress", stage="roster",
                       done=len(batches), total=n_batches, n_known=len(known))
        roster = roster_mod.merge_roster(batches)
        if text_embed_fn is not None and config.use_text_embed and len(roster) > 1:
            from vmem_bench.annotation.pipeline_track_first.text_match import semantic_dedup
            before = len(roster)
            roster = semantic_dedup(roster, text_embed_fn, threshold=config.roster_dedup_threshold)
            evlog.emit("roster_semantic_dedup", stage="roster", before=before, after=len(roster),
                       merged=before - len(roster))
        resume_mod.save_roster(out, roster)
        n_keyframes = len(key_indices)
    else:
        n_keyframes = 0
        evlog.emit("roster_resumed", stage="roster", n_entities=len(roster))
    # Deterministic location guard for automatic proposals only. Human-confirmed seeds are already
    # the ontology source of truth and must never be silently retyped.
    # "Locations" whose head noun is an in-frame object (trunk,
    # branch, canopy...) are roster mislabels — demote to prop, where the story-prop gate
    # decides tracking. Applies to resumed rosters too (older checkpoints predate the guard).
    roster, demoted = (roster_mod.demote_object_locations(roster)
                       if roster_seed is None else (roster, []))
    if demoted:
        evlog.emit("locations_demoted", stage="roster", n=len(demoted), names=sorted(demoted))
    if not key_indices and roster_seed is None:  # proposal resume: rebuild exemplar keyframes
        per_shot = roster_mod.shot_candidate_indices(shots, fps=fps,
                                                     candidate_fps=config.roster_candidate_fps)
        key_indices = roster_mod.select_roster_keyframes(
            per_shot, _embed_indices, per_shot_k=config.roster_per_shot_k,
            budget=config.roster_global_budget,
            budget_max=(config.roster_budget_max or None),
            novelty_threshold=config.roster_novelty_threshold,
            min_ratio=config.roster_budget_min_ratio, total_frames=n_frames) \
            if config.perception_backend in ("sam3_track", "fusion_track") else []
    roster_entries: list[RosterEntry] = []
    for e in roster:
        if e["kind"] not in ("character", "prop"):
            continue
        phrases = list(e.get("grounding_phrases") or [e["grounding_phrase"]])
        for phrase in phrases:
            roster_entries.append(RosterEntry(
                name=e["name"], kind=e["kind"], grounding_phrase=phrase,
                static_attributes=e.get("static_attributes") or {},
                exemplar_crop=str((e.get("exemplar_crops") or [""])[0]),
                canonical_entity_id=str(e.get("entity_id") or ""),
                identity_scope=str(e.get("identity_scope") or "individual"),
                aliases=tuple(e.get("aliases") or ()),
                exemplar_crops=tuple(e.get("exemplar_crops") or ()),
                allowed_state_events=tuple(e.get("allowed_state_events") or ())))
    roster_by_phrase = {e.grounding_phrase: e for e in roster_entries}
    seed_exemplar_embeddings: dict[str, list[list[float]]] = {}
    if roster_seed is not None:
        exemplar_items = [
            (entity.entity_id, Path(path))
            for entity in roster_seed.entities
            if entity.identity_scope == "individual"
            for path in entity.exemplar_crops
        ]
        if exemplar_items:
            exemplar_paths = [path for _entity_id, path in exemplar_items]
            vectors = (embedder.embed_batch(exemplar_paths)
                       if hasattr(embedder, "embed_batch")
                       else [embedder.embed_image(path) for path in exemplar_paths])
            for (entity_id, _path), vector in zip(exemplar_items, vectors):
                seed_exemplar_embeddings.setdefault(entity_id, []).append(list(vector))
        evlog.emit("seed_exemplars_loaded", stage="roster",
                   n_entities=len(seed_exemplar_embeddings),
                   n_exemplars=sum(map(len, seed_exemplar_embeddings.values())))
    evlog.emit("cast_roster", stage="roster", n_keyframes=n_keyframes, n_entities=len(roster),
               entities=[{"name": e["name"], "kind": e["kind"]} for e in roster])

    # --- step 2b (routes B/fusion): anchor each character to a visual exemplar crop -----------
    # Identity then flows from pixels (SAM3 candidates + judge multiple-choice + DINOv3
    # similarity), never from detector language — the route A failure mode (BBB v10 "red fox"
    # -> 0 tracklets) is structurally impossible here.
    if config.perception_backend in ("sam3_track", "fusion_track") and roster_seed is None:
        from vmem_bench.annotation.pipeline_track_first.vlm_grounding import collect_exemplars
        from vmem_bench.annotation.pipeline_track_first.perception.sam3_seg import Sam3ConceptSegmenter
        ex_manifest = dirs.tmp / "exemplars" / "manifest.json"
        if resume and ex_manifest.is_file():
            import json as _j
            exemplars = {name: str(rec["crop"])
                         for name, rec in _j.loads(ex_manifest.read_text()).items()}
            evlog.emit("exemplars_resumed", n=len(exemplars), stage="roster")
        else:
            exemplars = collect_exemplars(
                out, [_frame_path(i) for i in key_indices], roster,
                segmenter=Sam3ConceptSegmenter(
                    threshold=config.sam3_seg_threshold),
                embed_image=lambda p: list(embedder.embed_image(p)),
                judge_role=namer_vlm,
                concepts=tuple(config.sam3_character_concepts))
            evlog.emit("exemplars_collected", stage="roster", n=len(exemplars),
                       anchored=sorted(exemplars))
        for entry in roster_entries:
            if entry.name in exemplars:
                entry.exemplar_crop = exemplars[entry.name]
        # Story-prop gate (v11 evidence: concept-enumerating background classes entity-ized
        # every boulder -> 122 props). Background dressing stays visible to the drafter through
        # full frames; it just is not an entity. Judge failure degrades to tracking everything.
        props = [e for e in roster if e.get("kind") == "prop"]
        if props and hasattr(namer_vlm, "classify_prop_relevance"):
            try:
                relevance = namer_vlm.classify_prop_relevance(props)
            except Exception:  # noqa: BLE001
                logger.exception("prop relevance classification failed; keeping all props")
                relevance = {}
            # The judge echoes listing lines ("grey rock: rough, grey stone...") as the phrase
            # (BBB v12: exact match removed ZERO props -> 122 boulders). Normalize: strip any
            # ": description" suffix, lowercase, and match prefix-tolerantly.
            def _norm(s: str) -> str:
                return s.split(":", 1)[0].strip().lower()
            background = {_norm(ph) for ph, rel in relevance.items() if rel == "background"}
            if background:
                gated = [e for e in roster_entries
                         if e.kind == "prop" and _norm(e.grounding_phrase) in background]
                roster_entries = [e for e in roster_entries if e not in gated]
                roster_by_phrase = {e.grounding_phrase: e for e in roster_entries}
                evlog.emit("props_gated", stage="roster", n_background=len(gated),
                           background=sorted(e.grounding_phrase for e in gated))

    # Note: `backend` may be None here in the multi-GPU path -- each track_parallel worker builds its
    # own in-process perception stack, so the main process never needs a backend for tracking. In the
    # single-GPU path run.py always passes a fully-built backend (with detector).

    # --- step 3: per-shot tracklets (GPU-heavy; sharded across GPUs + checkpointed per shot) ---
    # Each shot's detect+track+embed is independent, so it is computed by track_parallel (one worker
    # per GPU when track_devices has >1 card, else in-process) and checkpointed. On --resume, shots
    # with an existing checkpoint are skipped. Re-ID stays sequential below (order-dependent, CPU).
    weights = {"body": config.reid_w_body, "face": config.reid_w_face, "class": config.reid_w_class}
    n_shots = len(shots)
    devices = list(track_devices or [])
    roster_data = [{"name": e.name, "kind": e.kind, "grounding_phrase": e.grounding_phrase,
                    "static_attributes": dict(e.static_attributes),
                    "exemplar_crop": e.exemplar_crop,
                    "canonical_entity_id": e.canonical_entity_id,
                    "identity_scope": e.identity_scope,
                    "aliases": list(e.aliases),
                    "exemplar_crops": list(e.exemplar_crops),
                    "allowed_state_events": list(e.allowed_state_events)}
                   for e in roster_entries]
    todo = [i for i in range(n_shots) if not (resume and resume_mod.shot_done(out, i))]
    base_done = n_shots - len(todo)
    evlog.emit("tracking_start", n_shots=n_shots, n_pending=len(todo),
               n_devices=max(1, len(devices)), stage="tracking")
    _track_t0 = time.time()

    def _progress(done_in_todo: int, *, shot_idx: int | None = None,
                  n_tracklets: int = 0) -> None:
        elapsed = time.time() - _track_t0
        eta = round(elapsed / done_in_todo * (len(todo) - done_in_todo), 1) if done_in_todo else None
        evlog.emit("track_progress", shot=base_done + done_in_todo, n_shots=n_shots,
                   shot_idx=shot_idx, n_tracklets=n_tracklets,
                   elapsed=round(elapsed, 1), eta_seconds=eta, stage="tracking")

    failures = track_parallel.compute_tracklets(
        shots, todo, config=config, roster_entries=roster_entries, roster_data=roster_data,
        out=out, video=config.video, frames_dir=frames_dir, crop_dir=crops_dir, fps=fps,
        n_frames=n_frames, devices=devices, backend=backend, embedder=embedder,
        face_encoder=face_encoder, frame_path=_frame_path, progress=_progress)
    for si, err in failures:
        evlog.emit("track_error", shot=si, error=err)

    # --- step 4: cross-shot re-ID -- character/prop identity is resolved by ONE of two modes
    # (config.identity_resolution_mode): "cluster_vlm" (default, identity_resolution.py: batch
    # deterministic pre-clustering + VLM cluster verification/merge, authoritative) or "greedy"
    # (the original online reid_assign nearest-neighbor path, kept as an explicit fallback/ablation
    # switch). Either way this loop first extracts each tracklet's identity-relevant features;
    # "greedy" commits to the registry immediately (order-dependent, as before), "cluster_vlm"
    # instead collects TrackletObservations and resolves/commits them ALL AT ONCE after the loop
    # (order-independent grouping -- see identity_resolution.py module docstring for why this fixes
    # the diagnosed root causes: closed-set-as-open-set clustering, embedding chaining, irreversible
    # online errors). Location clustering (scene-level, not per-object) is unaffected either way. ---
    registry = Registry()
    # (rep_id, span) per entity so crop-QA pruning can drop a flagged rep AND its presence span.
    rep_spans: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    # A detector phrase is a cheap, stable identity prior. It prevents an observation labelled
    # "grey rabbit" from entering an existing "purple bird" cluster; visual re-ID still decides
    # whether appearances within one phrase represent the same individual. Only used by "greedy".
    identity_group_by_entity: dict[str, str] = {}
    use_seeded_identity = roster_seed is not None
    use_cluster_vlm = config.identity_resolution_mode == "cluster_vlm" and not use_seeded_identity
    cluster_observations: list[identity_resolution.TrackletObservation] = []
    cluster_frame_spans: dict[int, tuple[int, int]] = {}
    anchored_groups: set[str] = set()
    seed_findings: list[dict[str, Any]] = []
    seen_seed_ids: set[str] = set()
    next_tid = 0
    # Full-frame scene vectors are collected first and clustered after character/prop re-ID.
    # A location is a place-level state, not one identity decision per camera shot.
    scene_observations: list[tuple[int, int, int, list[float]]] = []
    for shot_idx, (first, last) in enumerate(shots):
        payload = resume_mod.load_shot(out, shot_idx)
        if payload is None:  # failed/missing shot -> no tracklets (already reported); skip
            continue
        tracklets, face_sigs = resume_mod.shot_tracklets(payload)
        for tk, face_sig in zip(tracklets, face_sigs):
            tk.track_id = next_tid
            next_tid += 1
            best = tk.best_detection()
            phrase = tk.phrase
            if (config.reassign_by_class and crop_classifier is not None and best.crop_path):
                crop_p = Path(best.crop_path)
                if not crop_p.is_absolute():
                    crop_p = out / crop_p
                try:
                    ranked = crop_classifier.classify(
                        crop_p, [e.grounding_phrase for e in roster_entries])
                except Exception as exc:  # noqa: BLE001 — one bad crop/model hiccup must not
                    # abort the whole run; disable the optional reassign cue and continue.
                    logger.exception("crop classify failed; disabling reassign_by_class")
                    evlog.emit("reassign_error", shot=shot_idx, error=str(exc))
                    crop_classifier = None
                    ranked = []
                new_phrase = reassign_label(ranked, tk.phrase, config.class_reassign_margin)
                if new_phrase != tk.phrase:
                    evlog.emit("tracklet_label_reassigned", shot=shot_idx,
                               old=tk.phrase, new=new_phrase)
                    phrase = new_phrase
            entry = roster_by_phrase.get(phrase)
            kind = entry.kind if entry else "prop"
            name = entry.name if entry else phrase
            static = dict(entry.static_attributes) if entry else {}
            description = entry.static_attributes.get("description", "") if entry else ""
            if use_seeded_identity:
                from vmem_bench.annotation.pipeline_track_first.roster_seed import assign_closed_set
                assignment = assign_closed_set(
                    tk.mean_embedding(),
                    kind=kind,
                    phrase_owner_id=(entry.canonical_entity_id if entry is not None else ""),
                    seed=roster_seed,
                    exemplar_embeddings=seed_exemplar_embeddings,
                    min_similarity=config.seed_assignment_min_similarity,
                    min_margin=config.seed_assignment_min_margin,
                )
                canonical_id = assignment.entity_id
                if not canonical_id:
                    fingerprint = f"{phrase}@{tk.frame_span[0]}-{tk.frame_span[1]}"
                    ignored = fingerprint in set(roster_seed.ignored_tracks)
                    seed_findings.append({
                        "code": ("unknown_track_ignored" if ignored
                                 else "unknown_track_rejected"),
                        "shot_idx": shot_idx,
                        "chunk_id": to_chunk(best.frame_index),
                        "track_id": tk.track_id,
                        "phrase": phrase,
                        "reason": assignment.reason,
                        "best_score": assignment.best_score,
                        "margin": assignment.margin,
                        "candidate_scores": {
                            eid: round(score, 4) for eid, score in assignment.scores.items()},
                        "fingerprint": fingerprint,
                        "frame_span": list(tk.frame_span),
                        "crop_path": best.crop_path or "",
                    })
                    continue
                seed_entity = roster_seed.by_id[canonical_id]
                matched = registry.entities.get(canonical_id)
                is_new = matched is None
                if matched is None:
                    matched = Entity(
                        entity_id=seed_entity.entity_id,
                        kind=seed_entity.kind,
                        name=seed_entity.name,
                        description=seed_entity.description,
                        first_chunk=to_chunk(best.frame_index),
                        static_attributes={
                            **seed_entity.static_attributes,
                            "identity_scope": seed_entity.identity_scope,
                            "allowed_state_events": ",".join(seed_entity.allowed_state_events),
                        },
                    )
                    registry.entities[canonical_id] = matched
                entity, rep, _ = commit_tracklet_observation(
                    registry, matched, is_new=False,
                    kind=seed_entity.kind, name=seed_entity.name,
                    description=seed_entity.description,
                    static_attributes=matched.static_attributes,
                    chunk_id=to_chunk(best.frame_index),
                    crop_path=best.crop_path or "", bbox=list(best.bbox),
                    bbox_source="tracker", frame_index=best.frame_index,
                    grounding_score=best.score, track_id=tk.track_id,
                    signature=tk.mean_embedding(), face_signature=face_sig,
                    extra_qa={"identity_assignment": "canonical_seed",
                              "identity_scope": seed_entity.identity_scope,
                              "assignment_reason": assignment.reason,
                              "assignment_score": assignment.best_score,
                              "assignment_margin": assignment.margin},
                )
                seen_seed_ids.add(canonical_id)
                rep_spans.setdefault(entity.entity_id, []).append(
                    (rep.representation_id, tk.frame_span))
                continue
            # Group by the phrase's HEAD NOUN, not the full phrase: the roster VLM invents
            # near-synonym phrases for one individual across batches ("grey rabbit" vs "white
            # rabbit" under different lighting), and full-phrase grouping made those tracklets
            # permanently unmergeable. Within a head-noun group, visual re-ID (+ static-attribute
            # overlap gate) still separates genuinely different individuals; the VLM cross-cluster
            # merge pass (cluster_vlm mode) additionally reconciles different head-noun groups that
            # turn out to be one recurring character.
            normalized_phrase = normalize_entity_name(phrase).lower()
            identity_group = normalized_phrase.rsplit("_", 1)[-1] or normalized_phrase
            # Exemplar-anchored characters (routes B/fusion): the phrase already IS an
            # individual-level identity (exemplar similarity assigned it), so the phrase is the
            # cluster seed — same-phrase tracklets merge under a permissive threshold instead of
            # re-discovering the individual bottom-up (v11: viewpoint spread split one flying
            # squirrel into a dozen entities). Static-attribute and conflict gates still apply.
            anchored = bool(entry is not None and entry.kind == "character"
                            and entry.exemplar_crop)
            effective_threshold = (config.anchored_reid_threshold if anchored
                                   else config.reid_threshold)
            if anchored:
                identity_group = f"anchored:{normalized_phrase}"
                anchored_groups.add(identity_group)

            if use_cluster_vlm:
                obs_index = len(cluster_observations)
                cluster_observations.append(identity_resolution.TrackletObservation(
                    index=obs_index, kind=kind, name=name, description=description,
                    static_attributes=static, signature=tk.mean_embedding(),
                    crop_path=best.crop_path or "", bbox=list(best.bbox),
                    frame_index=best.frame_index, chunk_id=to_chunk(best.frame_index),
                    grounding_score=best.score, track_id=tk.track_id, bbox_source="tracker",
                    identity_group=identity_group, roster_matched=(entry is not None),
                    face_signature=face_sig))
                cluster_frame_spans[obs_index] = tk.frame_span
            else:
                allowed_ids = {
                    entity_id for entity_id, group in identity_group_by_entity.items()
                    if group == identity_group
                }

                def _report_conflict(
                    candidate, fused_score: float, min_body_score: float
                ) -> None:
                    evlog.emit(
                        "identity_conflict",
                        shot=shot_idx,
                        phrase=phrase,
                        candidate=candidate.entity_id,
                        fused_score=round(fused_score, 4),
                        min_body_score=round(min_body_score, 4),
                    )

                entity, rep, _new = reid_assign(
                    registry, chunk_id=to_chunk(best.frame_index), kind=kind, name=name,
                    description=description, static_attributes=static,
                    signature=tk.mean_embedding(), face_signature=face_sig, weights=weights,
                    face_strong=config.face_strong, crop_path=best.crop_path or "",
                    bbox=best.bbox, frame_index=best.frame_index, grounding_score=best.score,
                    track_id=tk.track_id, reid_threshold=effective_threshold,
                    static_overlap_threshold=config.static_overlap_threshold,
                    bbox_source="tracker", allowed_entity_ids=allowed_ids,
                    cluster_min_similarity=(config.anchored_cluster_min_similarity if anchored
                                            else config.reid_cluster_min_similarity),
                    conflict_hook=_report_conflict)
                identity_group_by_entity.setdefault(entity.entity_id, identity_group)
                rep_spans.setdefault(entity.entity_id, []).append(
                    (rep.representation_id, tk.frame_span))

        scene_observations.append((
            first, last, int(payload["scene_frame"]), list(payload["scene_vec"])))

    identity_resolution_findings: list[dict[str, Any]] = list(seed_findings)
    if use_seeded_identity:
        for seed_entity in roster_seed.entities:
            if seed_entity.kind in ("character", "prop") and seed_entity.entity_id not in seen_seed_ids:
                identity_resolution_findings.append({
                    "code": "seed_entity_missing_evidence",
                    "entity_id": seed_entity.entity_id,
                    "name": seed_entity.name,
                    "kind": seed_entity.kind,
                })
        _write_seed_assignment_artifact(
            dirs.tmp, roster_seed, seen_seed_ids, identity_resolution_findings)
        evlog.emit("identity_resolution", stage="identity", mode="seeded",
                   n_seed_entities=len(roster_seed.entities),
                   n_seen=len(seen_seed_ids), n_findings=len(identity_resolution_findings))
    if use_cluster_vlm and cluster_observations:
        def _threshold_for_bucket(_kind: str, group: str) -> float:
            return (config.anchored_reid_threshold if group in anchored_groups
                    else config.reid_threshold)

        t_identity0 = time.time()
        resolution = identity_resolution.resolve_identities(
            cluster_observations, judge_vlm=namer_vlm, weights=weights,
            precluster_threshold=config.reid_threshold, linkage=config.precluster_linkage,
            static_overlap_threshold=config.static_overlap_threshold, out_root=out,
            verify_max_crops=config.identity_verify_max_crops,
            merge_max_images=config.identity_merge_max_images,
            roster_completeness_min_observations=config.roster_completeness_min_observations,
            max_workers=config.identity_resolution_max_workers,
            threshold_for_bucket=_threshold_for_bucket)
        group_entity_id, rep_id_by_obs = identity_resolution.commit_groups_to_registry(
            registry, cluster_observations, resolution.final_groups)
        entity_id_by_obs_index: dict[int, str] = {}
        for gi, group in enumerate(resolution.final_groups):
            eid = group_entity_id.get(gi)
            if eid is None:
                continue
            for i in group:
                entity_id_by_obs_index[i] = eid
        for obs in cluster_observations:
            eid = entity_id_by_obs_index.get(obs.index)
            rid = rep_id_by_obs.get(obs.index)
            span = cluster_frame_spans.get(obs.index)
            if eid and rid and span:
                rep_spans.setdefault(eid, []).append((rid, span))
        identity_resolution_findings = resolution.findings
        evlog.emit("identity_resolution", stage="identity",
                   n_observations=len(cluster_observations),
                   n_final_entities=len(group_entity_id), n_findings=len(resolution.findings),
                   seconds=round(time.time() - t_identity0, 1))
        _write_identity_resolution_artifact(
            dirs.tmp, cluster_observations, resolution, group_entity_id, entity_id_by_obs_index)

    # A cluster's label is intentionally stable before VLM naming. The VLM receives the roster's
    # location taxonomy and picks a canonical visible setting name, while cluster membership stays
    # purely visual and deterministic.
    location_options = [str(e["name"]) for e in roster if e["kind"] == "location"]
    # Seeded production treats location as frame-level prompt context, not an identity asset. The
    # scoring contract removes locations from all headline metrics, and full-frame clustering was
    # a major source of fragmented review cards. Proposal mode retains it for diagnostics.
    location_clusters = ([] if use_seeded_identity else cluster_scene_locations(
        [vector for _first, _last, _frame, vector in scene_observations],
        similarity_threshold=config.loc_scene_cluster_threshold,
    ))
    for cluster_number, member_indices in enumerate(location_clusters, start=1):
        cluster_name = f"location_cluster_{cluster_number:02d}"
        location_attrs = (
            {"roster_location_options": ", ".join(location_options)}
            if location_options else {}
        )
        for member_index in member_indices:
            first, last, scene_frame, scene_vec = scene_observations[member_index]
            loc, lrep, _lnew = reid_assign(
                registry, chunk_id=to_chunk(scene_frame), kind="location", name=cluster_name,
                description="", static_attributes=location_attrs, signature=None, weights=weights,
                crop_path=str(_frame_path(scene_frame)), bbox=[0, 0, 1000, 1000],
                frame_index=scene_frame, grounding_score=0.0, track_id=None,
                reid_threshold=config.reid_threshold, bbox_source="full_frame")
            # Full-frame vectors are not suitable for character identity, but retain every one for
            # location dispersion/review and later scene-cluster diagnostics.
            registry.embeddings[lrep.embedding_key] = scene_vec
            rep_spans.setdefault(loc.entity_id, []).append((lrep.representation_id, (first, last)))
    evlog.emit("location_clusters", n_clusters=len(location_clusters),
               threshold=config.loc_scene_cluster_threshold,
               sizes=[len(cluster) for cluster in location_clusters],
               mode="frame_context" if use_seeded_identity else "entity")
    evlog.emit("identity", n_entities=len(registry.entities),
               n_tracklet_spans=sum(len(v) for v in rep_spans.values()))

    # --- step 5a: per-crop QA (§3.3) -- drop mixed-class reps before presence/naming ---------
    if config.use_crop_classify and config.crop_classify_method == "prototype":
        flagged = audit_registry_crops(registry, margin=config.crop_classify_margin)
        if flagged:
            _prune_reps(registry, rep_spans, set(flagged))
            evlog.emit("crop_qa", n_flagged=len(flagged), method="prototype")

    # --- step 5b: deterministic presence / first appearance ---------------------------------
    entity_spans = {eid: [span for _rid, span in lst] for eid, lst in rep_spans.items()
                    if eid in registry.entities and lst}
    present_by_chunk, first_appearance = presence_for_chunks(chunks, entity_spans)
    for eid, cid in first_appearance.items():
        if eid in registry.entities:
            registry.entities[eid].first_chunk = cid
    # Q3: deterministic per-entity time metadata from presence spans (metadata; not scored).
    for eid, ent in registry.entities.items():
        tm = entity_time_metadata(entity_spans.get(eid, []), fps)
        ent.presence_spans = tm["presence_spans"]
        ent.first_frame = tm["first_frame"]; ent.first_seconds = tm["first_seconds"]
        ent.last_frame = tm["last_frame"]; ent.last_seconds = tm["last_seconds"]
        ent.screen_time_seconds = tm["screen_time_seconds"]
        ent.max_absence_frames = tm["max_absence_frames"]
        ent.max_absence_seconds = tm["max_absence_seconds"]

    # --- step 6: per-entity naming (once, checkpointed) + commit crops to the asset bank -----
    locked_seed_ids = set(roster_seed.by_id) if roster_seed and config.lock_seed_identity else set()
    _name_and_commit(
        registry, assets_dir, out, namer_vlm, config, evlog, resume=resume,
        locked_entity_ids=locked_seed_ids)

    # Re-slug entity_ids from the final VLM name so ids match authoritative display names.
    idmap = reslug_entities(registry, assets_dir, locked_ids=locked_seed_ids)
    if idmap:
        present_by_chunk = {cid: [idmap.get(e, e) for e in ids]
                            for cid, ids in present_by_chunk.items()}
        first_appearance = {idmap.get(e, e): cid for e, cid in first_appearance.items()}
        evlog.emit("entity_reslug", mapping=idmap)

    # Merge proposals only (never auto-applied): text + body cosine report under tmp/.
    if text_embed_fn is not None and config.use_text_embed and not use_seeded_identity:
        import json
        from vmem_bench.annotation.pipeline_track_first.entity_merge import propose_entity_merges
        proposals = propose_entity_merges(
            registry, text_embed_fn,
            text_threshold=config.merge_text_threshold,
            body_threshold=config.merge_body_threshold)
        dirs.merge_proposals.write_text(
            json.dumps(proposals, indent=2), encoding="utf-8")
        evlog.emit("merge_proposals", n=len(proposals))

    # Review-only evidence: a ranked cross-cue queue for identity decisions.  It is intentionally
    # generated before drafting, stored under tmp/, and never applied to the registry automatically.
    import json
    from vmem_bench.annotation.pipeline_track_first.identity_diagnostics import identity_candidates
    candidates = ([] if use_seeded_identity else identity_candidates(
        registry, text_embed_fn=(text_embed_fn if config.use_text_embed else None)))
    dirs.identity_candidates.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    evlog.emit("identity_candidates", n=len(candidates))

    # --- steps 7+8: per-chunk draft (checkpointed) + assemble + persist ----------------------
    from vmem_bench.common.schemas import ChunkAnnotation
    presence_history: dict[str, list[int]] = {}
    annotations = []
    qa_report: list[dict[str, Any]] = []
    prev_prompt = ""
    global_identity_findings = list(identity_resolution_findings)
    allowed_events_by_entity = ({
        entity.entity_id: set(entity.allowed_state_events)
        for entity in roster_seed.entities
    } if roster_seed is not None else None)
    for c in chunks:
        cid = int(c["chunk_id"])
        t0 = time.time()
        present_ids = [e for e in present_by_chunk.get(cid, []) if e in registry.entities]
        first_ids = {e for e in present_ids if first_appearance.get(e) == cid}
        cached = resume_mod.load_chunk(out, cid) if resume else None
        payload_findings: list[dict[str, str]] = []
        if cid == int(chunks[0]["chunk_id"]):
            payload_findings.extend(global_identity_findings)
        if cached is not None:
            # Keep the cached VLM prompt, but refresh entity-linked fields from the current
            # registry/presence. A prior run's reslug (or a failed resume reslug) can leave
            # stale present/forbidden ids in the chunk checkpoint.
            ann = ChunkAnnotation.from_dict(cached)
            ann.present = list(present_ids)
            ann.first_appearances = sorted(first_ids)
            ann.gold_instructions = gold_instructions_for(present_ids, first_ids)
            ann.forbidden = materialize_forbidden(registry, cid)
            has_event = any(ev.chunk_id == cid
                            for e in registry.entities.values() for ev in e.state_events)
            ann.scenario_tags = scenario_tags_for(
                cid, present_ids, first_ids, presence_history, registry, has_event)
            prompt = ann.prompt
        else:
            first, last = c["frame_span"]
            draft_ok = True
            try:
                payloads, present_findings = _present_payloads(
                    registry, present_ids, first_ids, cid,
                    per_entity_limit=config.draft_crops_per_entity,
                    total_limit=config.draft_max_entity_crops, crop_root=out)
                payload_findings.extend(present_findings)
                idx = sample_frame_indices(first, last + 1, max_samples=config.max_sampled_frames)
                frames = [_frame_path(i) for i in idx]
                draft = drafter_vlm.draft_chunk(frames, payloads, prev_prompt, [],
                                                frame_indices=list(idx))
            except Exception as exc:  # noqa: BLE001 (a hung/failing chunk must not abort the whole
                # run: emit an error, leave an empty draft, and DO NOT checkpoint it so --resume
                # retries just this chunk next time instead of the whole video).
                logger.exception("draft failed for chunk %d", cid)
                evlog.emit("chunk_error", chunk_id=cid, error=str(exc))
                draft, draft_ok = {"prompt": "", "state_events": []}, False
            prompt = str(draft.get("prompt", ""))
            if not prompt.strip():
                payload_findings.append({"code": "empty_prompt", "entity_id": ""})
            event_policy_active = bool(allowed_events_by_entity is None or any(
                allowed_events_by_entity.get(eid, set()) for eid in present_ids))
            kept_events, rejected_events = filter_state_events(
                list(draft.get("state_events", [])) if event_policy_active else [],
                allowed_by_entity=allowed_events_by_entity)
            for item in rejected_events:
                payload_findings.append({
                    "code": "state_event_rejected",
                    "entity_id": str(item.get("entity_id") or ""),
                    "reason": str(item.get("rejection_reason") or "invalid_state_event"),
                })
            if rejected_events:
                evlog.emit("state_events_filtered", chunk_id=cid, n=len(rejected_events),
                           blocking=bool(config.production_mode))
            # Prompt = VLM-drafted screenplay prose, verbatim. No canonical-entity suffix is
            # injected; prompt_completeness below only *measures* natural name coverage (it never
            # rewrites the prompt), so the present roster is never leaked into the SUT-facing text.
            state_events = state_events_from_draft(
                registry, cid, kept_events,
                fps=fps, frame_span=(int(first), int(last)))
            ann = build_chunk_annotation(
                chunk_id=cid, shot_span=c["shot_span"], frame_span=c["frame_span"], prompt=prompt,
                present_ids=present_ids, first_ids=first_ids, registry=registry,
                presence_history=presence_history, has_state_event=bool(state_events))
            # Q3: chunk time in seconds (metadata; not scored).
            ann.seconds_span = [round(first / fps, 2), round(last / fps, 2)]
            # Seeded production uses exact canonical-name coverage: deterministic and zero false
            # negatives. Proposal runs retain the semantic diagnostic for calibration.
            if draft_ok and use_seeded_identity:
                covered = {
                    eid: registry.entities[eid].name.casefold() in prompt.casefold()
                    for eid in present_ids
                }
                ann.prompt_completeness = {
                    "method": "canonical_name",
                    "scores": {eid: 1.0 if ok else 0.0 for eid, ok in covered.items()},
                    "flagged": [eid for eid, ok in covered.items() if not ok],
                }
                for eid in ann.prompt_completeness["flagged"]:
                    payload_findings.append({
                        "code": "prompt_missing_present_entity", "entity_id": str(eid)})
            elif draft_ok and text_embed_fn is not None and config.use_text_embed and present_ids:
                from vmem_bench.annotation.pipeline_track_first.text_match import prompt_completeness
                items = [(eid, f"{registry.entities[eid].name}. {registry.entities[eid].description}")
                         for eid in present_ids]
                ann.prompt_completeness = prompt_completeness(
                    items, prompt, text_embed_fn, threshold=config.prompt_completeness_threshold)
                for eid in ann.prompt_completeness.get("flagged", []):
                    payload_findings.append({"code": "prompt_missing_present_entity",
                                             "entity_id": str(eid)})
            if draft_ok:  # only a good draft is checkpointed; failed chunks are retried on resume
                resume_mod.save_chunk(out, cid, ann.to_dict())
        annotations.append(ann)
        for eid in present_ids:
            presence_history.setdefault(eid, []).append(cid)
        prev_prompt = prompt
        informational = {"state_event_filtered_reversible", "unknown_track_ignored"}
        blocking_findings = [
            finding for finding in payload_findings
            if str(finding.get("code") or "") not in informational
        ]
        qa_report.append({"chunk_id": cid, "rounds": 0,
                          "flagged": ((not use_seeded_identity and len(present_ids) == 0)
                                      or bool(blocking_findings)),
                          "accepted_branch": -1,
                          "failed_checks": [str(f.get("code") or "qa_finding")
                                            for f in blocking_findings],
                          "findings": payload_findings,
                          "seconds": round(time.time() - t0, 1)})
        evlog.emit("chunk_done", chunk_id=cid, n_present=len(present_ids),
                   n_first=len(first_ids), seconds=round(time.time() - t0, 1))

    summary = _persist(config, out, index, registry, annotations, qa_report)
    # VLM global identity adjudication (review-only recommendation, never mutates gold): finds
    # same-individual splits whose embedding/text signals collapsed. Runs after persist so it
    # reads the exact published registry; failure never fails the run.
    if hasattr(namer_vlm, "group_same_individuals") and not use_seeded_identity:
        try:
            from vmem_bench.annotation.pipeline_track_first.identity_adjudication import adjudicate_identities
            adjudication = adjudicate_identities(out, namer_vlm)
            evlog.emit("identity_adjudication",
                       n_groups=sum(1 for g in adjudication["groups"]
                                    if len(g["entity_ids"]) > 1))
        except Exception as exc:  # noqa: BLE001 — recommendation layer must never fail the run
            logger.exception("identity adjudication failed")
            evlog.emit("identity_adjudication_error", error=str(exc))
    if config.auto_review and not use_seeded_identity:
        try:
            from vmem_bench.annotation.pipeline_track_first.auto_review import run_auto_review
            report = run_auto_review(
                out,
                apply_safe=config.auto_apply_safe_merges,
                namer_vlm=namer_vlm,  # judge model: enables the three-vote auto-merge + audits
                text_embed_fn=(text_embed_fn if config.use_text_embed else None),
                auto_text=config.merge_auto_text_threshold,
                auto_body=config.merge_auto_body_threshold,
                merge_text_threshold=config.merge_text_threshold,
                merge_body_threshold=config.merge_body_threshold)
            evlog.emit("auto_review", **report["stats"])
            summary["auto_review"] = report["stats"]
        except Exception as exc:  # noqa: BLE001 — auto-review must never fail the run
            logger.exception("auto_review failed")
            evlog.emit("auto_review_error", error=str(exc))
    if use_seeded_identity:
        from vmem_bench.annotation.pipeline_track_first.review_queue import write_review_queue
        queue = write_review_queue(out)
        summary["review_queue"] = queue["summary"]
        evlog.emit("review_queue", **queue["summary"])
    # Scratch (decoded frames + every-detection candidate crops + clips) is only needed during the
    # run; gold references assets/ + gold/* only. Prune tmp/candidates+frames+clips (and any
    # legacy derived/ tree) so the published/stored run stays small.
    if getattr(config, "prune_scratch", False):
        freed = prune_scratch(out)
        evlog.emit("scratch_pruned", freed_mb=round(freed / 1e6, 1))
        summary["scratch_pruned_mb"] = round(freed / 1e6, 1)
    evlog.emit("run_done", **{k: v for k, v in summary.items() if k != "gold_dir"})
    return summary


def _write_seed_assignment_artifact(
    tmp_dir: Path, roster_seed: Any, seen_ids: set[str], findings: list[dict[str, Any]],
) -> None:
    """Persist the canonical assignment/reject trail for review and freeze diagnostics."""
    import json
    payload = {
        "version": 1,
        "mode": "seeded",
        "seed_path": str(roster_seed.source_path),
        "human_confirmed": bool(roster_seed.human_confirmed),
        "ignored_tracks": list(roster_seed.ignored_tracks),
        "canonical_entities": [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "kind": entity.kind,
                "identity_scope": entity.identity_scope,
                "seen": entity.entity_id in seen_ids,
            }
            for entity in roster_seed.entities
        ],
        "findings": findings,
    }
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "identity_resolution.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_identity_resolution_artifact(
    tmp_dir: Path, observations: Sequence[identity_resolution.TrackletObservation],
    resolution: identity_resolution.IdentityResolution, group_entity_id: dict[int, str],
    entity_id_by_observation: dict[int, str] | None = None,
) -> None:
    """Persist the cluster_vlm identity-resolution decision trail to ``tmp/identity_resolution.json``
    (review-only diagnostic, never re-read by the pipeline itself; mirrors ``identity_candidates.json``
    / ``identity_adjudication.json``). Lets ``review_queue.py`` resolve an observation index (as
    referenced by ``findings[*].members`` etc.) back to a name/entity_id without this module needing
    any registry/schema coupling."""
    import json
    payload = {
        "version": 1,
        "mode": "cluster_vlm",
        "n_observations": len(observations),
        "n_final_groups": len(resolution.final_groups),
        "group_entity_id": {str(gi): eid for gi, eid in group_entity_id.items()},
        "entity_id_by_observation": {str(i): eid
                                     for i, eid in (entity_id_by_observation or {}).items()},
        "observations": [
            {"index": obs.index, "name": obs.name, "kind": obs.kind,
             "identity_group": obs.identity_group, "roster_matched": obs.roster_matched}
            for obs in observations],
        "findings": resolution.findings,
    }
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "identity_resolution.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def prune_scratch(out: Path) -> int:
    """Delete disposable scratch after commit; return bytes freed.

    Removes tmp/candidates + tmp/frames + tmp/clips and the entire legacy ``derived/``
    tree — NEVER tmp/checkpoint or tmp/events.jsonl (resume/monitoring depend on them).
    Safe because gold's representation crop_paths were rewritten into assets/ by
    _name_and_commit, and gold/ + assets/ are the published artifacts. Idempotent."""
    import shutil
    out = Path(out)
    dirs = MovieDirs(out, write=True)
    freed = 0
    targets = [dirs.candidates, dirs.frames, dirs.clips, out / "derived"]
    for d in targets:
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    freed += f.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(d, ignore_errors=True)
    return freed


def _prune_reps(registry: Registry, rep_spans: dict[str, list[tuple[str, tuple[int, int]]]],
                flagged: set[str]) -> None:
    """Remove crop-QA-flagged representations from the registry + their body/face/class vectors +
    their presence spans; drop an entity that loses all representations. Mutates in place."""
    for eid in list(registry.entities):
        ent = registry.entities[eid]
        keep = [r for r in ent.representations if r.representation_id not in flagged]
        for r in ent.representations:
            if r.representation_id in flagged:
                registry.embeddings.pop(r.embedding_key, None)
                registry.face_embeddings.pop(r.embedding_key, None)
                registry.class_embeddings.pop(r.embedding_key, None)
        ent.representations = keep
        rep_spans[eid] = [(rid, span) for rid, span in rep_spans.get(eid, [])
                          if rid not in flagged]
        if not ent.representations:
            del registry.entities[eid]
            rep_spans.pop(eid, None)


def _rep_quality(rep) -> tuple[float, int, int, str]:
    """Stable crop-quality ordering: grounded, confident, non-placeholder crops first."""
    usable = int(bool(rep.crop_path))
    grounded = int(rep.bbox_source == "grounding_dino")
    return (float(rep.qa.get("grounding_score", 0.0)), grounded, usable,
            rep.representation_id)


def _diverse_reps(reps: Sequence, embeddings: dict[str, list[float]], limit: int) -> list:
    """Select high-quality current crops with greedy DINO diversity, deterministically.

    The best-quality crop anchors the set; each following crop maximizes its minimum visual
    distance from the selected set. Frame index is only a deterministic tie-breaker.
    """
    candidates = sorted((r for r in reps if r.crop_path), key=_rep_quality, reverse=True)
    if limit <= 0 or not candidates:
        return []
    chosen = [candidates.pop(0)]
    while candidates and len(chosen) < limit:
        def score(rep):
            vec = embeddings.get(rep.embedding_key)
            if vec is None:
                diversity = -1.0
            else:
                diversity = min(1.0 - cosine_similarity(vec, embeddings[s.embedding_key])
                                for s in chosen if s.embedding_key in embeddings) if any(
                                    s.embedding_key in embeddings for s in chosen) else -1.0
            return (diversity, _rep_quality(rep), int(rep.frame_index), rep.representation_id)
        best = max(candidates, key=score)
        chosen.append(best)
        candidates.remove(best)
    return chosen


def _present_payloads(registry: Registry, present_ids: list[str], first_ids: set[str],
                      chunk_id: int, *, per_entity_limit: int = 3,
                      total_limit: int = 12, crop_root: Path | None = None
                      ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build constrained drafter payloads from deterministic ``present`` evidence only.

    Current-chunk crops are preferred; historical crops are explicit continuity references only.
    The primary single-crop fields remain for older callers, while ``crops`` carries the selected
    multi-crop evidence for the track-first drafter.
    """
    payloads: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    remaining = max(0, total_limit)
    # First appearances and QA-risky entities receive scarce visual budget first.
    ordered = sorted(present_ids, key=lambda eid: (
        eid not in first_ids,
        not any(r.qa.get("flagged") for r in registry.entities[eid].representations
                if r.chunk_id == chunk_id), eid))
    for eid in ordered:
        entity = registry.entities[eid]
        current = [r for r in entity.representations if r.chunk_id == chunk_id and r.crop_path]
        wanted = min(max(0, per_entity_limit), remaining)
        selected = _diverse_reps(current, registry.embeddings, wanted)
        if eid in first_ids and not selected:
            findings.append({"code": "first_missing_current_crop", "entity_id": eid})
        # A continuity reference is only a clearly marked fallback, never a substitute for a
        # first-appearance crop.
        if not selected and eid not in first_ids and wanted:
            historical = [r for r in entity.representations if r.chunk_id < chunk_id and r.crop_path]
            selected = _diverse_reps(historical, registry.embeddings, min(1, wanted))
        if not selected:
            findings.append({"code": "present_missing_crop", "entity_id": eid})
        remaining -= len(selected)
        primary = selected[0] if selected else (
            _best_cover_rep(entity, registry.embeddings) or
            (entity.representations[0] if entity.representations else None))
        if primary is not None:
            p = present_payload(entity, primary, first_appearance=eid in first_ids)
        else:
            # Preserve the deterministic present contract even for a malformed/no-rep entity;
            # QA records the problem and the VLM cannot silently omit it from the roster.
            p = {"entity_id": entity.entity_id, "name": entity.name, "kind": entity.kind,
                 "description": entity.description, "crop_path": "", "representation_id": "",
                 "first_appearance": eid in first_ids, "bbox_source": "", "bbox": [],
                 "grounding_score": 0.0}
        p["prior_representations"] = [r.representation_id for r in entity.representations
                                      if r.chunk_id < chunk_id]
        p["identity_scope"] = entity.static_attributes.get("identity_scope", "")
        allowed = entity.static_attributes.get("allowed_state_events", "")
        p["allowed_state_events"] = [item for item in allowed.split(",") if item]
        def crop_path(rep) -> str:
            path = Path(rep.crop_path)
            return str(path if path.is_absolute() or crop_root is None else crop_root / path)
        p["crop_path"] = crop_path(primary) if primary is not None and primary.crop_path else ""
        p["crops"] = [{"representation_id": r.representation_id, "crop_path": crop_path(r),
                       "continuity_reference": r.chunk_id < chunk_id,
                       "frame_index": r.frame_index} for r in selected]
        payloads.append(p)
    # The ordering seen by the VLM follows deterministic present order, not budget priority.
    rank = {eid: i for i, eid in enumerate(present_ids)}
    payloads.sort(key=lambda p: rank[p["entity_id"]])
    return payloads, findings


def _name_and_commit(registry: Registry, assets_dir: Path, out: Path, namer_vlm,
                     config: AnnotationConfig, evlog: EventLog, *, resume: bool = False,
                     locked_entity_ids: set[str] | None = None) -> None:
    """Name each entity ONCE from its best crop(s) and copy committed crops into the asset bank.

    Identity is already fixed (tracking+re-ID); the VLM only produces the authoritative display
    name + pixel-grounded description. crop paths are rewritten to portable relative paths and each
    entity gets a cover (best grounded rep)."""
    import shutil
    n_ent = len(registry.entities)
    locked_entity_ids = set(locked_entity_ids or ())
    evlog.emit("naming_start", n_entities=n_ent, stage="naming")
    assigned_names: list[str] = []
    for _i, entity in enumerate(registry.entities.values()):
        edir = entity_asset_dir(assets_dir, entity.entity_id, entity.kind)
        edir.mkdir(parents=True, exist_ok=True)
        for rep in entity.representations:
            src = Path(rep.crop_path)
            if not src.is_absolute():
                src = out / rep.crop_path
            name = rep.representation_id.split("@")[-1].replace(".", "_")
            dst = edir / f"{name}.jpg"
            if not src.is_file():
                # Resume-after-prune: the scratch crop was pruned by the previous run, but the
                # asset bank may already hold this rep's copy (deterministic filename). Adopt it
                # instead of leaving a dead absolute path in gold (BBB v13: 7 dead crop links).
                if dst.is_file():
                    rep.crop_path = _rel(dst, out)
                continue
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            rep.crop_path = _rel(dst, out)
        cover = _best_cover_rep(entity, registry.embeddings)
        if cover is not None:
            cover_src = out / cover.crop_path
            if cover_src.is_file():
                shutil.copyfile(cover_src, edir / "cover.jpg")

        # Naming is a once-per-entity VLM call -> checkpoint it so --resume never re-names an entity
        # already named on a prior run (entity_id is deterministic from re-ID, so the cache applies).
        cached = resume_mod.load_name(out, entity.entity_id) if resume else None
        if entity.entity_id in locked_entity_ids:
            # Canonical seed name/description are human-owned source-of-truth fields. Persist the
            # cache for resume symmetry, but never let an old cache or VLM rename them.
            resume_mod.save_name(out, entity.entity_id, entity.name, entity.description)
        elif cached is not None:
            if cached.get("name"):
                entity.name = str(cached["name"])
            if cached.get("description"):
                entity.description = str(cached["description"])
        else:
            # Naming: show up to 3 best grounded crops.
            crops = [out / r.crop_path for r in entity.representations
                     if (out / r.crop_path).is_file()][:3]
            if crops:
                try:
                    named = namer_vlm.name_entity(crops, entity.kind, entity.static_attributes,
                                                  known_names=list(assigned_names))
                    if named.get("name"):
                        entity.name = str(named["name"])
                    if named.get("description"):
                        entity.description = str(named["description"])
                except TypeError:
                    # Older/fake namer without known_names support (tests, external roles).
                    named = namer_vlm.name_entity(crops, entity.kind, entity.static_attributes)
                    if named.get("name"):
                        entity.name = str(named["name"])
                    if named.get("description"):
                        entity.description = str(named["description"])
                except Exception as exc:  # noqa: BLE001 (naming failure must not abort the run)
                    logger.exception("naming failed for %s", entity.entity_id)
                    evlog.emit("name_error", entity_id=entity.entity_id, error=str(exc))
            resume_mod.save_name(out, entity.entity_id, entity.name, entity.description)
        if entity.name:
            assigned_names.append(entity.name)
        evlog.emit("naming_progress", done=_i + 1, n_entities=n_ent,
                   name=entity.name, stage="naming")
    evlog.emit("naming_done", n_entities=len(registry.entities))
