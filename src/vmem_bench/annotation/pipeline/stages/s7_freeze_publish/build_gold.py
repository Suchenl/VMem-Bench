"""Materialize the new publishable crop-only gold layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.backfill import (
    forbidden_for_chunk,
    materialize_state_events,
)
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.gates import (
    unresolved_s3_blockers,
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _roster(annotation: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for item in annotation.get(group) or []:
            entity_id = str(item.get(id_key) or "")
            if entity_id:
                output[entity_id] = {
                    "entity_id": entity_id,
                    "kind": kind,
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or ""),
                }
    return output


def _segments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        rows.extend(scene.get("visual_segments") or [])
    return sorted(rows, key=lambda row: (float(row["start_seconds"]), str(row["segment_id"])))


def _scenario_tags(
    *,
    chunk_id: int,
    present: list[str],
    first: list[str],
    roster: dict[str, dict[str, str]],
    presence_history: dict[str, list[int]],
) -> list[str]:
    """Derive replay tags that require no additional model judgement.

    State-change tags are NOT added here (they must never be guessed from entity presence alone).
    They are added after the loop from the state events deterministically materialized out of the
    reviewed VLM ``state_changes`` (finite ontology + reversible gate), not from presence.
    """
    tags: set[str] = set()
    first_ids = set(first)
    for entity_id in present:
        if entity_id in first_ids:
            continue
        history = [cid for cid in presence_history.get(entity_id, []) if cid < chunk_id]
        if history and max(history) < chunk_id - 1:
            kind = roster[entity_id]["kind"]
            tags.add("scene-return" if kind == "location" else "re-appearance")
    return sorted(tags) or ["none"]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_crop_source(movie_dir: Path, crop_path: str) -> Path | None:
    """Resolve absolute or MemStrata-relative crop paths from S6 accepted rows."""
    raw = Path(str(crop_path))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(movie_dir / raw)
        # e.g. data/<dataset>/<movie>/tmp/... relative to benchmarks/MemStrata
        if len(movie_dir.parts) >= 3:
            candidates.append(movie_dir.parents[2] / raw)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _crop_leaf(*, chunk_id: int, source: Path, crop_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in crop_id)
    suffix = source.suffix.lower() if source.suffix else ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    return f"c{chunk_id:05d}_{safe}{suffix}"


def build_gold(
    *,
    movie_dir: Path,
    annotation: dict[str, Any],
    accepted_crops: list[dict[str, Any]],
    automation_smoke: bool = False,
) -> Path:
    """Write `gold/` from S3 revised annotation and S6 accepted crop proposals."""
    if not automation_smoke:
        unresolved = unresolved_s3_blockers(movie_dir / "tmp" / "pipeline")
        if unresolved:
            preview = ", ".join(
                f"{item['segment_id']}:{item['verdict']}" for item in unresolved[:8]
            )
            raise ValueError(
                f"cannot build human-reviewed gold with unresolved S3 blockers "
                f"({len(unresolved)}; {preview})"
            )
    gold = movie_dir / "gold"
    if gold.exists():
        shutil.rmtree(gold)
    crops_root = gold / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    roster = _roster(annotation)
    # Keep *all* accepted crops per (chunk, entity). A dict overwrite used to
    # silently drop human_add / multi-rep keeps for the same slot.
    by_slot: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in accepted_crops:
        if not item.get("crop_path"):
            continue
        by_slot[(int(item["chunk_id"]), str(item["entity_id"]))].append(item)
    segments = _segments(annotation)
    prompts: list[dict[str, Any]] = []
    score_contexts: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    crop_entries: list[dict[str, Any]] = []
    registry: dict[str, dict[str, Any]] = {
        entity_id: {
            **entity,
            "first_chunk": None,
            "reps": [],
            "representations": [],
            "state_events": [],
            "presence_spans": [],
        }
        for entity_id, entity in roster.items()
    }
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    presence_history: dict[str, list[int]] = defaultdict(list)
    materialized: set[str] = set()

    def _materialize(proposal: dict[str, Any], *, chunk_id: int, entity_id: str) -> str | None:
        entity = roster.get(entity_id)
        if entity is None:
            return None
        crop_id = str(proposal.get("representation_id") or f"{entity_id}@c{chunk_id:05d}")
        if crop_id in materialized:
            return crop_id
        source = _resolve_crop_source(movie_dir, str(proposal["crop_path"]))
        if source is None:
            return None
        leaf = _crop_leaf(chunk_id=chunk_id, source=source, crop_id=crop_id)
        relative = Path("crops") / f"{entity['kind']}s" / entity_id / leaf
        destination = gold / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        attrs = (
            proposal.get("crop_attributes")
            if isinstance(proposal.get("crop_attributes"), dict)
            else {}
        )
        state = str(attrs.get("state_angle") or proposal.get("state") or "default")
        entry = {
            "crop_id": crop_id,
            "representation_id": crop_id,
            "crop_path": str(relative),
            "entity_id": entity_id,
            "kind": entity["kind"],
            "chunk_id": chunk_id,
            "frame_index": int(proposal.get("frame_index", -1)),
            "bbox_norm": proposal.get("bbox_norm", []),
            "file_hash": _sha256(destination),
            "task_kind": proposal.get("task_kind", "acquire"),
            "bind_source_chunk_id": proposal.get("bind_source_chunk_id"),
            "crop_attributes": attrs or None,
        }
        crop_entries.append(entry)
        rep = {
            "representation_id": crop_id,
            "chunk_id": chunk_id,
            "crop_path": str(relative),
            "bbox": proposal.get("bbox_norm", []),
            "bbox_source": proposal.get("bbox_source") or "s5_vlm_sam3",
            "frame_index": int(proposal.get("frame_index", -1)),
            "embedding_key": "",
            "state": state,
            "task_kind": proposal.get("task_kind", "acquire"),
            "bind_source_chunk_id": proposal.get("bind_source_chunk_id"),
            "crop_attributes": attrs or None,
            "qa": proposal.get("qa", {}),
        }
        # Keep ``reps`` as a legacy alias; scoring / Entity.from_dict read
        # canonical ``representations`` (see schemas.Entity).
        registry[entity_id]["reps"].append(rep)
        registry[entity_id]["representations"].append(rep)
        materialized.add(crop_id)
        return crop_id

    for chunk_id, segment in enumerate(segments):
        present = [str(item) for item in segment.get("present_entity_ids") or [] if str(item) in roster]
        action = str(segment.get("action") or "")
        audio = str(segment.get("dialogue_or_audio") or "")
        prompt = f"{action} ({audio})".strip() if audio else action
        prompts.append({"schema_version": "3.0.0", "chunk_id": chunk_id, "prompt": prompt})
        targets: list[dict[str, Any]] = []
        obs: list[dict[str, Any]] = []
        first = [entity_id for entity_id in present if entity_id not in seen]
        for entity_id in first:
            registry[entity_id]["first_chunk"] = chunk_id
        for entity_id in present:
            proposals = by_slot.get((chunk_id, entity_id)) or []
            if not proposals:
                continue
            entity = roster[entity_id]
            crop_ids: list[str] = []
            for proposal in proposals:
                crop_id = _materialize(proposal, chunk_id=chunk_id, entity_id=entity_id)
                if crop_id:
                    crop_ids.append(crop_id)
                    obs.append(
                        {
                            "entity_id": entity_id,
                            "kind": entity["kind"],
                            "name": entity["name"],
                            "representation_id": crop_id,
                            "crop_path": next(
                                (
                                    row["crop_path"]
                                    for row in crop_entries
                                    if row["crop_id"] == crop_id
                                ),
                                "",
                            ),
                            "description": entity["description"] if entity_id in first else "",
                        }
                    )
            if crop_ids:
                targets.append(
                    {
                        "slot_id": entity_id,
                        "entity_id": entity_id,
                        "kind": entity["kind"],
                        "target_crop_ids": crop_ids,
                        "applicable": entity_id in seen,
                        "first_appearance": entity_id in first,
                    }
                )
        score_contexts.append({"chunk_id": chunk_id, "targets": targets})
        observations.append(
            {
                "schema_version": "3.0.0",
                "chunk_id": chunk_id,
                "chunk_video": "",
                "observations": obs,
                "state_events": [],
            }
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "shot_span": [],
                "frame_span": [],
                "seconds_span": [float(segment["start_seconds"]), float(segment["end_seconds"])],
                "prompt": prompt,
                "present": present,
                "first_appearances": first,
                # Fidelity is deterministic once present/first are frozen:
                # returning entities require continuity; first appearances
                # require introduce.  Do not leave this contract field empty.
                "gold_instructions": [
                    {
                        "entity_id": entity_id,
                        "requirement": "introduce" if entity_id in first else "continuity",
                    }
                    for entity_id in present
                ],
                # Seeded empty; the deterministic forbidden table is filled after all reps are
                # materialized, from the state events derived from the reviewed VLM state_changes
                # (see the state-event materialization block below).
                "forbidden": [],
                "scenario_tags": _scenario_tags(
                    chunk_id=chunk_id,
                    present=present,
                    first=first,
                    roster=roster,
                    presence_history=presence_history,
                ),
            }
        )
        seen.update(present)
        for entity_id in present:
            presence_history[entity_id].append(chunk_id)

    # Defensive: any accepted crop not attached via presence still enters gold.
    for proposal in accepted_crops:
        if not proposal.get("crop_path"):
            continue
        entity_id = str(proposal.get("entity_id") or "")
        chunk_id = int(proposal.get("chunk_id", -1))
        crop_id = str(proposal.get("representation_id") or f"{entity_id}@c{chunk_id:05d}")
        if crop_id in materialized or entity_id not in roster:
            continue
        _materialize(proposal, chunk_id=max(chunk_id, 0), entity_id=entity_id)

    # State events: deterministically materialize the reviewed VLM ``state_changes`` into the
    # finite lifecycle ontology — same gate (``drafting.filter_state_events``), seconds->chunk
    # mapping, and ``deprecates`` scoping the freeze-layer backfill uses (one canonical
    # implementation). This activates Avoidance (D4) and VisualFidelity's deprecation branch at
    # freeze time. Zero model calls: the explicit ``state_change_kind`` is authoritative and only
    # reversible/out-of-ontology events are dropped. Runs after all reps are materialized so
    # ``deprecates`` sees the entity's full crop history; ``layout_hash`` is unaffected (it hashes
    # only ``(chunk_id, seconds_span)``).
    events_by_entity, _rejected = materialize_state_events(
        annotation, list(registry.values()), chunks
    )
    event_chunks = {ev["chunk_id"] for events in events_by_entity.values() for ev in events}
    for entity_id, entity in registry.items():
        entity["state_events"] = events_by_entity.get(entity_id, [])
    for chunk in chunks:
        cid = int(chunk["chunk_id"])
        chunk["forbidden"] = forbidden_for_chunk(events_by_entity, cid)
        if cid in event_chunks:
            tags = [tag for tag in chunk["scenario_tags"] if tag != "none"]
            if "state-change" not in tags:
                tags.append("state-change")
            chunk["scenario_tags"] = sorted(tags)
    for packet in observations:
        cid = int(packet["chunk_id"])
        packet["state_events"] = [
            ev for events in events_by_entity.values() for ev in events if ev["chunk_id"] == cid
        ]

    layout_hash = hashlib.sha256(
        json.dumps(
            [(chunk["chunk_id"], chunk["seconds_span"]) for chunk in chunks],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    for entity in registry.values():
        if entity["first_chunk"] is None:
            entity["first_chunk"] = 0
    reviewed = not automation_smoke
    manifest = {
        "schema_version": "3.0.0",
        "movie_id": movie_dir.name,
        "human_reviewed": reviewed,
        "automation_smoke_only": automation_smoke,
        "layout_hash": layout_hash,
        "pipeline": "annotation/pipeline/s1-s7",
    }
    (gold / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(gold / "prompts.jsonl", prompts)
    _write_jsonl(gold / "score_context.jsonl", score_contexts)
    _write_jsonl(gold / "observations.jsonl", observations)
    (gold / "crop_index.json").write_text(
        json.dumps({"schema_version": "3.0.0", "crops": crop_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (gold / "chunk_index.json").write_text(
        json.dumps(
            {"schema_version": "3.0.0", "layout_hash": layout_hash, "chunks": chunks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (gold / "entity_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0.0",
                "movie_id": movie_dir.name,
                "human_reviewed": reviewed,
                "entities": list(registry.values()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (gold / "chunk_annotations.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0.0",
                "movie_id": movie_dir.name,
                "human_reviewed": reviewed,
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-dir", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--accepted-crops", type=Path, required=True)
    parser.add_argument("--automation-smoke", action="store_true")
    args = parser.parse_args()
    build_gold(
        movie_dir=args.movie_dir,
        annotation=json.loads(args.annotation.read_text(encoding="utf-8")),
        accepted_crops=json.loads(args.accepted_crops.read_text(encoding="utf-8")),
        automation_smoke=args.automation_smoke,
    )


if __name__ == "__main__":
    main()
