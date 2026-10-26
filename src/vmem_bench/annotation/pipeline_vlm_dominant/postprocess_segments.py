"""Video-free conversion: a web-MLLM ``vlm_output.json`` (prompt_v5 schema) -> the same frozen
gold contract (``common.schemas.EntityRegistry`` / ``ChunkAnnotations`` + a ``chunk_index.json``)
that ``pipeline_track_first`` / ``pipeline_vlm_dominant.postprocess`` produce, WITHOUT a source
video or shot-boundary detection.

Why a second postprocessor next to ``postprocess.py``:

``postprocess.build_gold`` consumes the *programmatic* VLM contract (``prompting.SCHEMA``:
``characters``/``props``/``locations`` with ``appearances`` + ``first_appearance_seconds`` +
a top-level ``shots`` array) and needs a real frame-based ``chunk_index`` (SBD on the video).
The web-MLLM prompt_v5 output we actually have in ``data/`` is a *different* shape: entities carry
model-assigned ``char_id``/``prop_id``/``loc_id`` + ``first_presence_seconds`` +
``last_presence_seconds``, and presence lives in ``screenplay.scenes[].visual_segments[]`` with
per-segment ``present_entity_ids``. There is no video and no SBD.

This module treats the VLM's own ``visual_segments`` as the chunk layout (1:1, seconds-based) and
derives every deterministic gold field the same way the track-first pipeline does, reusing
``pipeline_track_first.drafting`` helpers so gold stays lint-consistent
(``gold_instructions``/``forbidden``/``scenario_tags``). Design choices (pilot, video-free):

* **presence** = per-segment ``present_entity_ids`` (the fine-grained multi-interval truth; entity
  ``first_presence_seconds``/``last_presence_seconds`` are ignored for presence because they erase
  re-appearance gaps that MemRecall scores). Entity ids pass through as-is (trusted; no re-ID).
* **representations**: one synthesized rep per entity at its ``first_chunk`` with empty
  ``crop_path`` -- enough for the harness ObservationPacket to feed the SUT's memory
  (``scoring.runner.build_observation_packet`` emits an observation only for entities with a rep in
  that chunk; the SUT holds the asset from ``first_chunk`` onward). No pixels exist, so bbox is
  empty and ``bbox_source="vlm_fallback"``.
* **state events**: recorded (narrated, Avoidance-relevant metadata) but with empty ``deprecates``
  -- an honest consequence of "no visual grounding", identical to ``postprocess.build_gold``; the
  Avoidance metric stays film-level N/A and its weight is redistributed by the scorer.
* **entity time metadata** (``presence_spans`` + first/last/screen_time/max_absence) is populated in
  SECONDS (not frames -- there is no video); ``annotation_provenance.presence_time_unit="seconds"``
  records this. Not scored (SUT never sees it); drives human review + the temporal readout.

The output is written through ``MovieDirs`` into ``<movie>/gold/``. ``human_reviewed`` is set from
the ``trust_entities`` flag: the pilot trusts the VLM entity table and treats segment presence as
authoritative, so no per-entity human review gate is enforced here (segment-level auto-review and
crop-merge review are separate, later steps).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first.drafting import (
    build_chunk_annotation,
)
from vmem_bench.common.schemas import (
    SCHEMA_VERSION, ChunkAnnotations, Entity, EntityRegistry, Representation, StateEvent,
)

_GROUPS = (("characters", "character", "char_id"),
           ("props", "prop", "prop_id"),
           ("locations", "location", "loc_id"))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent [lo, hi] second intervals -> sorted disjoint list."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _flatten_segments(vlm: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten screenplay.scenes[].visual_segments[] into a time-ordered chunk list.

    Each entry is one benchmark chunk: its seconds window, the drafting prompt source (action +
    optional audio), the scene location id, and the model's present_entity_ids for that segment.
    """
    segs: list[dict[str, Any]] = []
    scenes = (vlm.get("screenplay") or {}).get("scenes") or []
    for scene in scenes:
        loc_id = str(scene.get("loc_id") or "").strip()
        for vs in scene.get("visual_segments") or []:
            start = float(vs.get("start_seconds", 0.0))
            end = float(vs.get("end_seconds", start))
            if end < start:
                start, end = end, start
            segs.append({
                "start": start, "end": end,
                "action": str(vs.get("action") or ""),
                "audio": str(vs.get("dialogue_or_audio") or ""),
                "present": [str(x) for x in (vs.get("present_entity_ids") or [])],
                "loc_id": loc_id,
            })
    segs.sort(key=lambda s: (s["start"], s["end"]))
    return segs


def _chunk_for_seconds(segs: list[dict[str, Any]], seconds: float) -> int:
    """Chunk id whose [start, end) window contains ``seconds`` (clamped to first/last)."""
    if not segs:
        return 0
    for cid, s in enumerate(segs):
        if s["start"] <= seconds < s["end"]:
            return cid
    return 0 if seconds < segs[0]["start"] else len(segs) - 1


def build_gold_from_segments(
    vlm: dict[str, Any], *, movie_id: str, model_name: str, trust_entities: bool = True,
) -> tuple[EntityRegistry, ChunkAnnotations, dict[str, Any]]:
    """Convert one web-MLLM ``vlm_output.json`` dict into ``(registry, annotations, chunk_index)``.

    ``trust_entities`` sets ``human_reviewed`` on both gold files: the pilot trusts the VLM entity
    table and its per-segment presence, so it freezes without a per-entity review pass. Set False to
    emit an unfrozen candidate (``human_reviewed=False``) for the normal review gate.
    """
    registry = Registry()
    first_presence: dict[str, float] = {}
    for group_key, kind, id_key in _GROUPS:
        for raw in vlm.get(group_key) or []:
            eid = str(raw.get(id_key) or "").strip()
            if not eid:
                raise ValueError(f"{group_key} entry missing {id_key}: {raw!r}")
            if eid in registry.entities:
                raise ValueError(f"duplicate entity id {eid!r}")
            registry.entities[eid] = Entity(
                entity_id=eid, kind=kind, name=str(raw.get("name") or "").strip(),
                description=str(raw.get("description") or ""), first_chunk=0,
                static_attributes={"identity_scope": str(raw.get("identity_scope") or "")})
            first_presence[eid] = float(raw.get("first_presence_seconds", 0.0))
    known = set(registry.entities)

    segs = _flatten_segments(vlm)
    if not segs:
        raise ValueError("vlm_output has no visual_segments; cannot build a chunk layout")

    # Per-chunk present set (segment present_entity_ids + scene location), filtered to known ids.
    present_by_chunk: dict[int, list[str]] = {}
    unknown_refs: set[str] = set()
    for cid, s in enumerate(segs):
        pres: list[str] = []
        for eid in s["present"]:
            if eid in known:
                if eid not in pres:
                    pres.append(eid)
            else:
                unknown_refs.add(eid)
        loc_id = s["loc_id"]
        if loc_id and loc_id in known and loc_id not in pres:
            pres.append(loc_id)
        present_by_chunk[cid] = pres

    # Safety net: an entity the VLM listed but never tagged in any segment is attached to the chunk
    # containing its self-reported first_presence_seconds, so it is observed at least once.
    seen = {eid for pres in present_by_chunk.values() for eid in pres}
    for eid in registry.entities:
        if eid in seen:
            continue
        cid = _chunk_for_seconds(segs, first_presence.get(eid, 0.0))
        present_by_chunk[cid].append(eid)

    # first_chunk = earliest chunk in which the entity is present.
    first_chunk: dict[str, int] = {}
    for cid in sorted(present_by_chunk):
        for eid in present_by_chunk[cid]:
            first_chunk.setdefault(eid, cid)
    for eid, ent in registry.entities.items():
        ent.first_chunk = first_chunk.get(eid, 0)

    # One synthesized representation per entity at its first_chunk (empty crop; no pixels).
    for eid, ent in registry.entities.items():
        fc = ent.first_chunk
        ent.representations = [Representation(
            representation_id=f"{eid}@c{fc:03d}", chunk_id=fc, crop_path="",
            bbox=[], bbox_source="vlm_fallback", frame_index=-1, embedding_key="")]

    # Entity time metadata in SECONDS from the segments the entity appears in.
    chunks_of: dict[str, list[int]] = {eid: [] for eid in registry.entities}
    for cid in sorted(present_by_chunk):
        for eid in present_by_chunk[cid]:
            chunks_of[eid].append(cid)
    for eid, ent in registry.entities.items():
        spans = _merge_intervals([(segs[c]["start"], segs[c]["end"]) for c in chunks_of[eid]])
        if not spans:
            continue
        ent.presence_spans = [[int(round(lo)), int(round(hi))] for lo, hi in spans]
        ent.first_seconds = round(spans[0][0], 2)
        ent.last_seconds = round(spans[-1][1], 2)
        ent.screen_time_seconds = round(sum(hi - lo for lo, hi in spans), 2)
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        ent.max_absence_seconds = round(max(gaps), 2) if gaps else 0.0

    # State events -> chunk by timestamp; deprecates empty (no crops to invalidate).
    event_chunks: set[int] = set()
    for group_key, _kind, id_key in _GROUPS:
        for raw in vlm.get(group_key) or []:
            eid = str(raw.get(id_key) or "").strip()
            ent = registry.entities.get(eid)
            if ent is None:
                continue
            for i, sc in enumerate(raw.get("state_changes") or []):
                secs = float(sc.get("seconds", 0.0))
                cid = _chunk_for_seconds(segs, secs)
                kind_str = str(sc.get("state_change_kind") or "").strip()
                raw_desc = str(sc.get("description") or "")
                desc = f"{kind_str}: {raw_desc}" if kind_str else raw_desc
                ent.state_events.append(StateEvent(
                    event_id=f"evt_c{cid:03d}_{eid}" + (f".{i}" if i else ""),
                    chunk_id=cid, description=desc, deprecates=[], seconds=secs))
                event_chunks.add(cid)

    # Chunk annotations (deterministic gold fields reuse the track-first drafting helpers).
    presence_history: dict[str, list[int]] = {eid: [] for eid in registry.entities}
    chunk_annos = []
    chunk_index_chunks: list[dict[str, Any]] = []
    for cid in sorted(present_by_chunk):
        s = segs[cid]
        pres = present_by_chunk[cid]
        first_ids = {eid for eid in pres if registry.entities[eid].first_chunk == cid}
        # Gold prompt = the S4 human-reviewed screenplay action (+ audio), verbatim.
        # No canonical-entity suffix is injected: the prompt must name entities only as far as the
        # reviewed prose naturally does (name-anchoring is the SUT's job to exploit, not the bench's
        # to leak the present roster). See running_eval.md §0 iron rule 3.
        prompt = s["action"]
        if s["audio"]:
            prompt = f"{prompt} ({s['audio']})".strip()
        anno = build_chunk_annotation(
            chunk_id=cid, shot_span=[], frame_span=[], prompt=prompt, present_ids=pres,
            first_ids=first_ids, registry=registry, presence_history=presence_history,
            has_state_event=(cid in event_chunks))
        anno.seconds_span = [round(s["start"], 2), round(s["end"], 2)]
        chunk_annos.append(anno)
        for eid in pres:
            presence_history[eid].append(cid)
        chunk_index_chunks.append({"chunk_id": cid, "shot_span": [], "frame_span": [],
                                   "seconds_span": [round(s["start"], 2), round(s["end"], 2)]})

    layout_hash = hashlib.sha256(json.dumps(
        {"chunks": [c["seconds_span"] for c in chunk_index_chunks],
         "entities": sorted(registry.entities)},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    provenance = {
        "pipeline": "vlm_dominant_video_free", "model": model_name, "layout_hash": layout_hash,
        "chunk_source": "visual_segments", "presence_time_unit": "seconds",
        "review": "trust_mode_pilot" if trust_entities else "candidate",
        "unknown_present_refs": sorted(unknown_refs),
    }
    registry_out = EntityRegistry(
        movie_id=movie_id, entities=list(registry.entities.values()),
        human_reviewed=trust_entities, annotation_provenance=provenance)
    annotations_out = ChunkAnnotations(
        movie_id=movie_id, chunks=chunk_annos, human_reviewed=trust_entities)
    chunk_index = {
        "schema_version": SCHEMA_VERSION, "movie_id": movie_id, "layout_hash": layout_hash,
        "fps": None, "time_unit": "seconds", "source": "visual_segments",
        "chunks": chunk_index_chunks,
    }
    return registry_out, annotations_out, chunk_index
