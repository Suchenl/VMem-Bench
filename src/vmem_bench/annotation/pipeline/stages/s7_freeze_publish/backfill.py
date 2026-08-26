"""Backfill deterministic scoring fields into already-frozen gold.

S7 (:mod:`build_gold`) writes ``gold_instructions`` at freeze time, so any gold
frozen with the current builder already activates Fidelity. Gold frozen *before*
that field was added shipped without it, leaving ``G=∅`` on every chunk and
Fidelity film-level N/A. ``gold_instructions`` is a pure set operation over each
chunk's ``present`` / ``first_appearances`` (returning entities require
``continuity``; first appearances require ``introduce``) — identical to the
inline logic in ``build_gold`` — so it can be recomputed deterministically for
any frozen gold **without re-running S1-S7** and **without touching
``layout_hash``** (which hashes only ``(chunk_id, seconds_span)``).

The same freeze-layer contract applies to ``state_events`` (ground truth for
**Avoidance** D4 and the deprecation check inside **VisualFidelity**). The VLM's
``vlm_output.json`` already carries per-entity ``state_changes`` with a
structured ``state_change_kind`` in the finite lifecycle ontology; the earlier
crop-based builder dropped them (``state_events=[]``), so Avoidance stayed
film-level N/A even though the frozen gold has real crops to deprecate. We
materialize them deterministically — no model call, no re-run:

* each ``state_change`` becomes a ``StateEvent`` whose ``chunk_id`` is the chunk
  whose ``seconds_span`` contains ``state_changes[i].seconds``;
* the finite-ontology / reversible gate is
  :func:`pipeline_track_first.drafting.filter_state_events` (the same
  deterministic anti-hallucination filter the annotation pipeline uses). The
  explicit ``state_change_kind`` is passed as ``event_type`` and the English
  kind is prefixed onto the (possibly non-English) description so the filter's
  English regex can corroborate or reject it;
* ``deprecates`` = every representation of that entity acquired at or before the
  event chunk (the default "the entity's prior look is now superseded"), scoped
  to the **real frozen crop ids** — which is exactly what makes Avoidance and
  VisualFidelity's deprecation branch fire.

Idempotent: by default a movie whose registry already carries ``state_events``
is left untouched; pass ``overwrite=True`` to recompute. Neither backfill
touches ``layout_hash``, crops, entity ids, or ``seconds_span``.

This lives under ``s7_freeze_publish`` because it is the freeze layer's own
deterministic-field contract applied retroactively — not a new pipeline stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline_track_first.drafting import filter_state_events

# vlm_output.json entity groups -> (list key, id key). Locations are materialized too for
# completeness but never affect Avoidance (the scorer skips ``kind == "location"``).
_VLM_GROUPS = (("characters", "char_id"), ("props", "prop_id"), ("locations", "loc_id"))


def gold_instructions_for_chunk(present: list[str], first_appearances: list[str]) -> list[dict[str, str]]:
    """Deterministic per-chunk gold_instructions (mirrors ``build_gold``)."""
    first = set(first_appearances)
    return [
        {
            "entity_id": entity_id,
            "requirement": "introduce" if entity_id in first else "continuity",
        }
        for entity_id in present
    ]


def _backfill_chunks(chunks: list[dict[str, Any]], *, overwrite: bool) -> int:
    """Add gold_instructions to chunks in place; return number of chunks changed."""
    changed = 0
    for chunk in chunks:
        if not overwrite and chunk.get("gold_instructions"):
            continue
        computed = gold_instructions_for_chunk(
            [str(x) for x in chunk.get("present", [])],
            [str(x) for x in chunk.get("first_appearances", [])],
        )
        if chunk.get("gold_instructions") != computed:
            chunk["gold_instructions"] = computed
            changed += 1
    return changed


def backfill_gold_instructions(movie_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Backfill gold_instructions into a frozen movie's gold in place.

    Updates both ``gold/chunk_annotations.json`` (read by the scoring harness) and
    ``gold/chunk_index.json`` (which carries the same chunk rows) so the two stay
    consistent. Leaves every other field, ``layout_hash``, and crops untouched.
    Returns a summary dict of per-file chunk-change counts.
    """
    gold = Path(movie_dir) / "gold"
    summary: dict[str, Any] = {"movie_dir": str(movie_dir), "files": {}}
    for name in ("chunk_annotations.json", "chunk_index.json"):
        path = gold / name
        if not path.is_file():
            summary["files"][name] = "missing"
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            summary["files"][name] = "no_chunks"
            continue
        changed = _backfill_chunks(chunks, overwrite=overwrite)
        if changed:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        summary["files"][name] = {"chunks_changed": changed, "n_chunks": len(chunks)}
    return summary


# --------------------------------------------------------------------------------------------------
# state_events materialization (Avoidance / VisualFidelity deprecation)
# --------------------------------------------------------------------------------------------------


def _chunk_for_seconds(chunks: list[dict[str, Any]], seconds: float) -> int:
    """Chunk id whose ``[start, end)`` seconds window contains ``seconds`` (clamped to ends).

    Mirrors ``pipeline_vlm_dominant.postprocess_segments._chunk_for_seconds`` so a backfilled
    event lands in the same chunk the pipeline would have assigned.
    """
    spans: list[tuple[int, float, float]] = []
    for c in chunks:
        span = c.get("seconds_span") or []
        if len(span) < 2:
            continue
        spans.append((int(c["chunk_id"]), float(span[0]), float(span[1])))
    if not spans:
        return 0
    for cid, start, end in spans:
        if start <= seconds < end:
            return cid
    return spans[0][0] if seconds < spans[0][1] else spans[-1][0]


def materialize_state_events(
    vlm: dict[str, Any],
    entities: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Deterministically derive ``state_events`` from VLM ``state_changes``.

    Returns ``(events_by_entity, rejected)`` where ``events_by_entity`` maps entity_id -> a list of
    serialized ``StateEvent`` dicts (``event_id`` / ``chunk_id`` / ``description`` / ``deprecates``
    / ``seconds``), and ``rejected`` records each ``state_change`` the finite-ontology / reversible
    gate dropped (with its ``rejection_reason``) for reviewer visibility.

    Never mutates its inputs; performs no model call.
    """
    reps_by_entity: dict[str, list[dict[str, Any]]] = {
        str(e.get("entity_id")): list(e.get("representations") or []) for e in entities
    }
    known = set(reps_by_entity)

    # Flatten every VLM state_change into a filter-shaped draft, tagged with its entity + seconds.
    drafted: list[dict[str, Any]] = []
    for group_key, id_key in _VLM_GROUPS:
        for raw in vlm.get(group_key) or []:
            eid = str(raw.get(id_key) or "").strip()
            if eid not in known:
                continue
            for sc in raw.get("state_changes") or []:
                kind = str(sc.get("state_change_kind") or "").strip()
                desc = str(sc.get("description") or "").strip()
                # Prefix the English ontology kind so the filter's English regex can corroborate a
                # (possibly Chinese) description; the explicit event_type stays authoritative.
                prefixed = f"{kind}: {desc}" if kind and desc else (kind or desc)
                drafted.append({
                    "entity_id": eid,
                    "event_type": kind,
                    "description": prefixed,
                    "seconds": float(sc.get("seconds", 0.0)),
                })

    kept, rejected = filter_state_events(drafted)

    events_by_entity: dict[str, list[dict[str, Any]]] = {}
    for item in kept:
        eid = str(item.get("entity_id"))
        seconds = float(item.get("seconds", 0.0))
        cid = _chunk_for_seconds(chunks, seconds)
        # Default deprecation: every prior/current representation of this entity is superseded once
        # the irreversible change happens. Scoped to the real frozen crop ids.
        deprecates = [
            str(r.get("representation_id"))
            for r in reps_by_entity.get(eid, [])
            if int(r.get("chunk_id", -1)) <= cid
        ]
        bucket = events_by_entity.setdefault(eid, [])
        suffix = f".{len(bucket)}" if bucket else ""
        bucket.append({
            "event_id": f"evt_c{cid:03d}_{eid}{suffix}",
            "chunk_id": cid,
            "description": str(item.get("description", "")),
            "deprecates": deprecates,
            "frame_index": None,
            "seconds": seconds,
        })
    return events_by_entity, rejected


def forbidden_for_chunk(
    events_by_entity: dict[str, list[dict[str, Any]]], chunk_id: int
) -> list[dict[str, str]]:
    """F_active(t): reps deprecated by events with ``event.chunk_id < chunk_id`` (mirrors
    ``drafting.materialize_forbidden``).

    Shared with :mod:`build_gold` so freeze-time and backfill produce byte-identical forbidden
    tables from the same ``events_by_entity`` mapping.
    """
    forbidden: list[dict[str, str]] = []
    for events in events_by_entity.values():
        for ev in events:
            if int(ev["chunk_id"]) < chunk_id:
                forbidden += [{"representation_id": rid, "reason": ev["event_id"]}
                              for rid in ev["deprecates"]]
    return forbidden


def backfill_state_events(movie_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Materialize ``state_events`` into a frozen movie's gold in place (no re-run, no model).

    Writes into ``gold/entity_registry.json`` (the scoring-authoritative source read by
    ``score_run`` / ``build_observation_packet``) and keeps two derived, deterministic gold fields
    consistent: the per-chunk ``forbidden`` tables in ``gold/chunk_annotations.json`` and the
    ``state_events`` blocks in ``gold/observations.jsonl``. ``layout_hash``, crops, entity ids, and
    ``seconds_span`` are never touched. Returns a summary including the rejected state_changes.
    """
    movie_dir = Path(movie_dir)
    gold = movie_dir / "gold"
    summary: dict[str, Any] = {"movie_dir": str(movie_dir), "events": {}, "rejected": []}

    registry_path = gold / "entity_registry.json"
    vlm_path = movie_dir / "vlm_output.json"
    index_path = gold / "chunk_index.json"
    if not registry_path.is_file() or not vlm_path.is_file():
        summary["status"] = "missing_inputs"
        return summary

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entities = registry.get("entities") or []
    if not overwrite and any(e.get("state_events") for e in entities):
        summary["status"] = "already_has_state_events (use overwrite=True to recompute)"
        return summary

    vlm = json.loads(vlm_path.read_text(encoding="utf-8"))
    chunks_for_time = (
        json.loads(index_path.read_text(encoding="utf-8")).get("chunks")
        if index_path.is_file() else None
    ) or []

    events_by_entity, rejected = materialize_state_events(vlm, entities, chunks_for_time)
    summary["rejected"] = rejected

    # 1) entity_registry.json — the scoring-authoritative source.
    n_events = 0
    for entity in entities:
        eid = str(entity.get("entity_id"))
        events = events_by_entity.get(eid, [])
        entity["state_events"] = events
        n_events += len(events)
        if events:
            summary["events"][eid] = [
                {"chunk_id": ev["chunk_id"], "kind": ev["description"].split(":", 1)[0],
                 "n_deprecates": len(ev["deprecates"])}
                for ev in events
            ]
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["n_events"] = n_events
    summary["n_entities_with_events"] = len(summary["events"])

    # 2) chunk_annotations.json — derived forbidden tables + state-change scenario tag.
    anno_path = gold / "chunk_annotations.json"
    if anno_path.is_file():
        anno = json.loads(anno_path.read_text(encoding="utf-8"))
        event_chunks = {int(ev["chunk_id"]) for evs in events_by_entity.values() for ev in evs}
        for chunk in anno.get("chunks") or []:
            cid = int(chunk["chunk_id"])
            chunk["forbidden"] = forbidden_for_chunk(events_by_entity, cid)
            tags = [t for t in (chunk.get("scenario_tags") or []) if t != "none"]
            if cid in event_chunks and "state-change" not in tags:
                tags.append("state-change")
            chunk["scenario_tags"] = sorted(tags) or ["none"]
        anno_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["chunk_annotations_updated"] = True

    # 3) observations.jsonl — derived per-chunk state_events (not scoring-authoritative, kept in
    #    sync for offline consumers). Each line is one ObservationPacket dict.
    obs_path = gold / "observations.jsonl"
    if obs_path.is_file():
        by_chunk: dict[int, list[dict[str, Any]]] = {}
        for evs in events_by_entity.values():
            for ev in evs:
                by_chunk.setdefault(int(ev["chunk_id"]), []).append(ev)
        lines: list[str] = []
        for line in obs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            packet = json.loads(line)
            packet["state_events"] = by_chunk.get(int(packet.get("chunk_id", -1)), [])
            lines.append(json.dumps(packet, ensure_ascii=False))
        obs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["observations_updated"] = True

    return summary


def backfill_all(movie_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Run both deterministic freeze-layer backfills (gold_instructions + state_events)."""
    return {
        "gold_instructions": backfill_gold_instructions(movie_dir, overwrite=overwrite),
        "state_events": backfill_state_events(movie_dir, overwrite=overwrite),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--movie-dir", type=Path, required=True,
                    help="frozen movie dir containing gold/ (e.g. data/.../big_buck_bunny)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if the field is already present")
    ap.add_argument("--only", choices=("instructions", "state-events", "all"), default="all",
                    help="which deterministic field(s) to backfill (default: all)")
    args = ap.parse_args(argv)
    if args.only == "instructions":
        summary = backfill_gold_instructions(args.movie_dir, overwrite=args.overwrite)
    elif args.only == "state-events":
        summary = backfill_state_events(args.movie_dir, overwrite=args.overwrite)
    else:
        summary = backfill_all(args.movie_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
