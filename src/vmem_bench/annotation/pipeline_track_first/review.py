"""Human review artifacts (workflow step 9): static review.html, patch application, freeze.

The page is a single self-contained HTML file: flagged items + a random spot-check sample
are pre-expanded; the reviewer toggles/edit fields and exports review_patch.json in the
browser (no server). ``apply_patch`` folds the patch back and ``freeze`` seals the gold.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
import tempfile
import random
from pathlib import Path
from typing import Any

from vmem_bench.common.gold_lint import lint_annotations
from vmem_bench.common.paths import (
    MovieDirs, asset_crop_relpath, entity_asset_dir, is_entity_asset_path, movie_root_from)
from vmem_bench.common.schemas import ChunkAnnotations, EntityRegistry, GoldInstruction

logger = logging.getLogger(__name__)

_DISPOSITION_ACTIONS = {"merged", "kept_distinct", "dropped"}


def _load_dispositions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("review dispositions must be an object keyed by entity_id")
    return _validate_dispositions(raw)


def _validate_dispositions(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("dispositions must be an object keyed by entity_id")
    checked: dict[str, dict[str, str]] = {}
    for entity_id, item in raw.items():
        if not isinstance(entity_id, str) or not entity_id or not isinstance(item, dict):
            raise ValueError("each disposition requires a non-empty entity id and object value")
        action = item.get("action")
        reason = item.get("reason")
        if action not in _DISPOSITION_ACTIONS:
            raise ValueError(f"invalid disposition action for {entity_id}: {action!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"disposition reason must be non-empty for {entity_id}")
        entry = {"action": action, "reason": reason.strip()}
        if item.get("patch") is not None:
            if not isinstance(item["patch"], str):
                raise ValueError(f"disposition patch must be a string for {entity_id}")
            entry["patch"] = item["patch"]
        checked[entity_id] = entry
    return checked


def _write_dispositions_atomic(path: Path, dispositions: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dispositions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _apply_state_event_reviews(registry, chunks, dirs: MovieDirs, raw: Any) -> None:
    """Validate/apply human state decisions and append immutable raw→human audit pairs."""
    if not raw:
        return
    if not isinstance(raw, dict):
        raise ValueError("state_event_reviews must be an object keyed by event_id")
    events = {event.event_id: (entity, event) for entity in registry.entities for event in entity.state_events}
    decisions = (json.loads(dirs.state_event_dispositions.read_text(encoding="utf-8"))
                 if dirs.state_event_dispositions.is_file() else {})
    pairs: list[dict[str, Any]] = []
    for event_id, decision in raw.items():
        if event_id not in events or not isinstance(decision, dict):
            raise ValueError(f"unknown or invalid state-event review: {event_id}")
        action, reason = decision.get("action"), str(decision.get("reason") or "").strip()
        if action not in {"confirmed", "rejected", "edited"} or not reason:
            raise ValueError(f"state-event review {event_id} requires valid action and non-empty reason")
        entity, event = events[event_id]
        before = event.to_dict()
        if action == "edited":
            edit = decision.get("patch")
            if not isinstance(edit, dict): raise ValueError(f"edited state event {event_id} requires patch")
            chunk_id = int(edit.get("chunk_id", event.chunk_id))
            valid_chunks = {chunk.chunk_id for chunk in chunks.chunks}
            if chunk_id not in valid_chunks: raise ValueError(f"state event {event_id} references unknown chunk")
            deprecates = [str(x) for x in edit.get("deprecates", event.deprecates)]
            owned = {rep.representation_id for rep in entity.representations if rep.chunk_id <= chunk_id}
            if not set(deprecates).issubset(owned): raise ValueError(f"state event {event_id} deprecates invalid representation")
            event.chunk_id, event.deprecates = chunk_id, deprecates
            event.description = str(edit.get("description", event.description))
            event.frame_index = edit.get("frame_index", event.frame_index)
        elif action == "rejected":
            entity.state_events = [candidate for candidate in entity.state_events if candidate.event_id != event_id]
        decisions[event_id] = {"action": action, "reason": reason}
        pairs.append({"event_id": event_id, "entity_id": entity.entity_id, "raw": before,
                      "decision": decision, "result": None if action == "rejected" else event.to_dict()})
    _write_dispositions_atomic(dirs.state_event_dispositions, decisions)
    if pairs:
        dirs.state_event_pairs.parent.mkdir(parents=True, exist_ok=True)
        with dirs.state_event_pairs.open("a", encoding="utf-8") as handle:
            for pair in pairs: handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _load(gold_dir: Path) -> tuple[EntityRegistry, ChunkAnnotations]:
    registry = EntityRegistry.from_dict(
        json.loads((gold_dir / "entity_registry.json").read_text(encoding="utf-8")))
    chunks = ChunkAnnotations.from_dict(
        json.loads((gold_dir / "chunk_annotations.json").read_text(encoding="utf-8")))
    return registry, chunks


def _chunk_review_risk(c, entities_by_id: dict) -> float:
    """Heuristic review priority: state-changing / multi-instance / re-appearance chunks and
    chunks with low-grounding-score or vlm_fallback crops are more likely to hide annotation
    errors, so they get higher spot-check weight (Pitfall_Notes)."""
    risk = 0.5  # floor so every chunk has a nonzero chance
    for tag in c.scenario_tags:
        if tag in ("state-change", "multi-instance", "re-appearance"):
            risk += 2.0
    for eid in c.present:
        e = entities_by_id.get(eid)
        if e is None:
            continue
        for r in e.representations:
            if r.chunk_id != c.chunk_id:
                continue
            if r.bbox_source == "vlm_fallback":
                risk += 1.0
            gs = float(r.qa.get("grounding_score", 0.0))
            if r.bbox_source == "grounding_dino" and gs < 0.5:
                risk += 1.0
    return risk


def _risk_weighted_spot(chunks, registry, flagged_chunks, spot_check, rng) -> set[int]:
    """Risk-weighted sampling without replacement (priority to error-prone chunks). ponytail:
    O(n*k) manual weighted draw; ceiling is fine because n_chunks and spot_check are small.
    Upgrade path: numpy.random.choice with p= and replace=False once numpy is a hard dep here."""
    ents = {e.entity_id: e for e in registry.entities}
    pool = [(c.chunk_id, _chunk_review_risk(c, ents))
            for c in chunks.chunks if c.chunk_id not in flagged_chunks]
    spot: set[int] = set()
    while pool and len(spot) < spot_check:
        pick = rng.random() * sum(w for _, w in pool)
        acc = 0.0
        for i, (cid, w) in enumerate(pool):
            acc += w
            if acc >= pick:
                spot.add(cid)
                pool.pop(i)
                break
        else:  # float rounding never reached pick -> take the last
            spot.add(pool[-1][0])
            pool.pop()
    return spot


def _load_embeddings(gold_dir: Path) -> dict[str, list[float]]:
    emb_path = Path(gold_dir) / "embeddings.safetensors"
    if not emb_path.is_file():
        return {}
    from safetensors.numpy import load_file
    return {k: list(map(float, v)) for k, v in load_file(str(emb_path)).items()}


def generate_review_html(out_dir: Path, *, spot_check: int = 10,
                         seed: int | None = None,
                         machine_queue: dict[str, dict] | None = None) -> Path:
    out_dir = Path(out_dir)
    registry, chunks = _load(MovieDirs(out_dir).gold)
    qa_path = MovieDirs(out_dir).qa_report
    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else []
    flagged_chunks = {q["chunk_id"] for q in qa if q.get("flagged")}
    # seed=None => system-entropy random so different reviewers/runs spot-check different chunks
    # (Pitfall_Notes: a fixed seed=0 made the spot check deterministic across runs/reviewers).
    # Pass an int only when you need to reproduce a specific review page.
    rng = random.Random(seed)
    spot = _risk_weighted_spot(chunks, registry, flagged_chunks, spot_check, rng)

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(out_dir.resolve()))
        except ValueError:
            return p

    entities = list(registry.entities)
    if machine_queue is not None:
        # Score desc; unscored last; stable among ties / unscored.
        def _sort_key(e, idx: int) -> tuple:
            info = machine_queue.get(e.entity_id)
            if info is None:
                return (1, 0.0, idx)
            return (0, -float(info.get("score", 0.0)), idx)
        entities = [e for _, e in sorted(
            ((i, e) for i, e in enumerate(entities)),
            key=lambda t: _sort_key(t[1], t[0]))]

    rows = []
    for e in entities:
        qa_flagged = any(r.qa.get("flagged") for r in e.representations)
        mq = (machine_queue or {}).get(e.entity_id) if machine_queue is not None else None
        machine_flagged = mq is not None and float(mq.get("score", 0.0)) >= 1.0
        flagged = machine_flagged if machine_queue is not None else qa_flagged
        machine_span = ""
        if mq is not None:
            reason_txt = html.escape("; ".join(str(r) for r in mq.get("reasons", [])))
            machine_span = (f'<br><span class="machine">score={float(mq.get("score", 0.0)):.1f}'
                            f'{(": " + reason_txt) if reason_txt else ""}</span>')
        crops = "".join(
            f'<figure><img src="{rel(r.crop_path)}" loading="lazy">'
            f'<figcaption>c{r.chunk_id:03d} · {r.bbox_source}'
            f'{" ⚠" if r.qa.get("flagged") else ""}</figcaption></figure>'
            for r in e.representations)
        rows.append(f"""
<tr class="{'flagged' if flagged else ''}" data-eid="{e.entity_id}">
 <td><code>{e.entity_id}</code><br><span class="kind">{e.kind}</span>{machine_span}</td>
 <td><input class="name" value="{html.escape(e.name, quote=True)}">
     <p class="desc">{html.escape(e.description)}</p>
     <label><input type="checkbox" class="drop"> drop</label>
     <input class="merge" placeholder="merge into entity_id"></td>
 <td class="crops">{crops}</td>
</tr>""")

    chunk_rows = []
    for c in chunks.chunks:
        cls = "flagged" if c.chunk_id in flagged_chunks else ("spot" if c.chunk_id in spot else "")
        chunk_rows.append(f"""
<tr class="{cls}" data-cid="{c.chunk_id}">
 <td>c{c.chunk_id:03d}<br><span class="kind">{' '.join(c.scenario_tags)}</span></td>
 <td><textarea class="prompt">{html.escape(c.prompt)}</textarea></td>
 <td>{', '.join(c.present)}<br>first: {', '.join(c.first_appearances) or '—'}
     <br>forbidden: {len(c.forbidden)}</td>
</tr>""")

    machine_note = ("机器审核已排序,红底为机器判定必审项,其余默认机器通过。"
                    if machine_queue is not None else "")
    page = f"""<!doctype html><meta charset="utf-8"><title>MemStrata-Bench review · {registry.movie_id}</title>
<style>
body{{font-family:system-ui;margin:20px;background:#fafafa}} table{{border-collapse:collapse;width:100%;margin-bottom:2em}}
td{{border:1px solid #ddd;padding:6px;vertical-align:top;background:#fff}} tr.flagged td{{background:#fff3f3}}
tr.spot td{{background:#f3f7ff}} .crops{{display:flex;flex-wrap:wrap;gap:4px;max-width:60vw}}
figure{{margin:0;text-align:center;font-size:10px}} img{{max-height:96px;border-radius:4px}}
textarea{{width:100%;min-height:60px}} input.name{{font-weight:600}} .kind{{color:#888;font-size:11px}}
button{{position:fixed;top:12px;right:12px;padding:10px 18px;font-size:14px}}
</style>
<h1>{registry.movie_id} · annotation review</h1>
<p>红底 = 自检未通过（必审）；蓝底 = 随机抽检。改名/勾 drop/填 merge/改 prompt 后点 Export。{machine_note}</p>
<button onclick="exportPatch()">Export review_patch.json</button>
<h2>Entities ({len(registry.entities)})</h2>
<table>{''.join(rows)}</table>
<h2>Chunks ({len(chunks.chunks)})</h2>
<table>{''.join(chunk_rows)}</table>
<script>
function exportPatch() {{
  const patch = {{schema_version:"2.0.0", merges:[], splits:[], renames:{{}}, drops:[], field_edits:[]}};
  document.querySelectorAll("tr[data-eid]").forEach(tr => {{
    const eid = tr.dataset.eid;
    if (tr.querySelector(".drop").checked) patch.drops.push(eid);
    const merge = tr.querySelector(".merge").value.trim();
    if (merge) patch.merges.push([merge, eid]);
    const name = tr.querySelector(".name");
    if (name.value !== name.defaultValue) patch.renames[eid] = name.value;
  }});
  document.querySelectorAll("tr[data-cid]").forEach(tr => {{
    const ta = tr.querySelector(".prompt");
    if (ta.value !== ta.defaultValue)
      patch.field_edits.push({{path:`chunks[${{tr.dataset.cid}}].prompt`, value:ta.value}});
  }});
  const blob = new Blob([JSON.stringify(patch, null, 2)], {{type:"application/json"}});
  const a = Object.assign(document.createElement("a"),
    {{href:URL.createObjectURL(blob), download:"review_patch.json"}});
  a.click();
}}
</script>"""
    path = out_dir / "review.html"
    path.write_text(page, encoding="utf-8")
    return path


def apply_patch(gold_dir: Path, patch_path: Path) -> None:
    """Fold a review patch back into the gold drafts (before freezing).

    ``gold_dir`` may be either the ``gold/`` directory itself or the movie root
    (if the basename is not "gold", it is treated as the movie root)."""
    gold_dir = MovieDirs(movie_root_from(gold_dir)).gold
    movie_dir = movie_root_from(gold_dir)
    dirs = MovieDirs(movie_dir, write=True)
    registry, chunks = _load(gold_dir)
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))
    new_dispositions = _validate_dispositions(patch["dispositions"]) if "dispositions" in patch else {}
    # A merge/drop is a human decision, not merely a JSON operation. Require its matching record
    # when the patch opts into dispositions; legacy patches remain readable as promised.
    for _target, source in patch.get("merges", []):
        if new_dispositions and new_dispositions.get(source, {}).get("action") != "merged":
            raise ValueError(f"merge of {source} requires disposition action='merged'")
    for source in patch.get("drops", []):
        if new_dispositions and new_dispositions.get(source, {}).get("action") != "dropped":
            raise ValueError(f"drop of {source} requires disposition action='dropped'")
    entities = {e.entity_id: e for e in registry.entities}
    id_map: dict[str, str] = {}

    for target_id, source_id in patch.get("merges", []):
        source, target = entities.get(source_id), entities.get(target_id)
        if source is None or target is None:
            logger.warning("merge skipped, unknown id: %s <- %s", target_id, source_id)
            continue
        target.representations += source.representations
        target.state_events += source.state_events
        target.first_chunk = min(target.first_chunk, source.first_chunk)
        # Keep crops under the surviving entity's canonical kind directory so the
        # portable crop_path remains valid after a merge.
        movie_root = movie_root_from(gold_dir)
        for rep in source.representations:
            if is_entity_asset_path(rep.crop_path, source_id, source.kind):
                leaf = Path(rep.crop_path).name
                src_file = movie_root / rep.crop_path
                dst_file = entity_asset_dir(movie_root / "assets", target_id, target.kind) / leaf
                if src_file.is_file():
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    if src_file.resolve() != dst_file.resolve():
                        import shutil
                        shutil.copyfile(src_file, dst_file)
                rep.crop_path = asset_crop_relpath(target_id, target.kind, leaf)
            if rep.representation_id.startswith(source_id + "@"):
                new_rid = target_id + rep.representation_id[len(source_id):]
                rep.representation_id = new_rid
                rep.embedding_key = new_rid
        del entities[source_id]
        id_map[source_id] = target_id

    for source_id in patch.get("drops", []):
        if entities.pop(source_id, None) is not None:
            id_map[source_id] = ""

    for eid, new_name in patch.get("renames", {}).items():
        if eid in entities:
            entities[eid].name = str(new_name)

    if patch.get("splits"):
        # ponytail: automated split needs re-consolidation; flag for manual JSON edit instead.
        logger.warning("splits are not automated; edit gold JSON manually for: %s", patch["splits"])

    for c in chunks.chunks:
        c.present = sorted({id_map.get(e, e) for e in c.present} - {""})
        firsts = {e for e in c.present if entities[e].first_chunk == c.chunk_id}
        c.first_appearances = sorted(firsts)
        c.gold_instructions = [GoldInstruction(entity_id=e,
                               requirement="introduce" if e in firsts else "continuity")
                               for e in c.present]
        kept_reps = {r.representation_id for e in entities.values() for r in e.representations}
        c.forbidden = [f for f in c.forbidden if f.representation_id in kept_reps]

    # Recompute scenario_tags: merges/drops/renames change present sets and presence history, so
    # stale tags (e.g. multi-instance on a now-merged pair, re-appearance on a dropped entity)
    # must not survive into frozen gold (Pitfall_Notes). scenario_tags_for needs a Registry-style
    # view (entities dict + embeddings), so rebuild one from the gold sidecar.
    from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
    from vmem_bench.annotation.pipeline_track_first.drafting import scenario_tags_for
    presence_history: dict[str, list[int]] = {}
    for c in sorted(chunks.chunks, key=lambda x: x.chunk_id):
        for eid in c.present:
            presence_history.setdefault(eid, []).append(c.chunk_id)
    reg = Registry()
    reg.entities = dict(entities)
    reg.embeddings = _load_embeddings(gold_dir)
    for c in chunks.chunks:
        has_event = any(ev.chunk_id == c.chunk_id
                        for e in entities.values() for ev in e.state_events)
        c.scenario_tags = scenario_tags_for(c.chunk_id, c.present, set(c.first_appearances),
                                             presence_history, reg, has_event)

    for edit in patch.get("field_edits", []):
        path, value = str(edit["path"]), edit["value"]
        if path.startswith("chunks[") and path.endswith("].prompt"):
            cid = int(path[len("chunks["):-len("].prompt")])
            for c in chunks.chunks:
                if c.chunk_id == cid:
                    c.prompt = str(value)
        elif path.startswith("entities[") and path.endswith("].description"):
            # Repair of a bad first-appearance description (consolidation now keeps the first
            # non-empty description instead of last-writer-wins; human review edits it here).
            eid = path[len("entities["):-len("].description")]
            if eid in entities:
                entities[eid].description = str(value)
            else:
                logger.warning("field_edit unknown entity: %s", eid)
        else:
            logger.warning("unsupported field_edit path: %s", path)

    registry.entities = list(entities.values())
    _apply_state_event_reviews(registry, chunks, dirs, patch.get("state_event_reviews"))
    # State decisions change Avoidance ground truth; never leave derived forbidden/tags stale.
    from vmem_bench.annotation.pipeline_track_first.drafting import materialize_forbidden
    for chunk in chunks.chunks:
        chunk.forbidden = materialize_forbidden(reg, chunk.chunk_id)

    registry.entities = list(entities.values())
    registry.annotation_provenance["review_patch_applied"] = Path(patch_path).name
    _save(gold_dir, registry, chunks)
    if new_dispositions:
        existing = _load_dispositions(dirs.review_dispositions)
        existing.update(new_dispositions)
        # Persist only after the gold patch succeeds. This process artifact stays under tmp/.
        _write_dispositions_atomic(dirs.review_dispositions, existing)


def preview_patch(movie_dir: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a patch only to a temporary movie copy and return strict-lint blockers.

    The preview deliberately uses the production patch function so reviewers see the same derived
    fields as Apply, while the real movie root and its asset bank remain untouched.
    """
    movie_dir = movie_root_from(Path(movie_dir))
    with tempfile.TemporaryDirectory(prefix="memstrata-review-preview-") as temp:
        root = Path(temp) / "movie"
        shutil.copytree(movie_dir / "gold", root / "gold")
        if (movie_dir / "tmp").is_dir():
            shutil.copytree(movie_dir / "tmp", root / "tmp")
        if (movie_dir / "build").is_dir():
            shutil.copytree(movie_dir / "build", root / "build")
        patch_path = root / "tmp" / "preview_patch.json"
        _write_dispositions_atomic(patch_path, patch)
        try:
            apply_patch(root, patch_path)
            registry, chunks = _load(root / "gold")
            layout_path = MovieDirs(root).chunk_index
            layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else None
            qa_path = MovieDirs(root).qa_report
            qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else []
            violations = lint_annotations(registry, chunks, layout=layout, qa_report=qa, strict_review=True)
            errors = [violation.to_dict() for violation in violations if violation.severity == "error"]
            return {"ok": not errors, "errors": errors,
                    "summary": {"n_errors": len(errors), "n_chunks": len(chunks.chunks),
                                "n_entities": len(registry.entities)}}
        except Exception as exc:  # noqa: BLE001 - preview returns validation failures to the UI
            return {"ok": False, "errors": [{"code": "patch_preview_error", "message": str(exc)}],
                    "summary": {"n_errors": 1}}


def freeze(gold_dir: Path) -> None:
    """Mark gold as human-reviewed (harness refuses unfrozen gold).

    ``gold_dir`` may be either the ``gold/`` directory itself or the movie root
    (if the basename is not "gold", it is treated as the movie root)."""
    movie_dir = movie_root_from(gold_dir)
    dirs = MovieDirs(movie_dir)
    gold_dir = dirs.gold
    registry, chunks = _load(gold_dir)
    layout_path = dirs.chunk_index
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else None
    qa_path = dirs.qa_report
    qa_report = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else []
    if dirs.auto_review_json.is_file():
        auto_review = json.loads(dirs.auto_review_json.read_text(encoding="utf-8"))
        required = [str(eid) for eid in auto_review.get("must_review", [])]
        dispositions = _load_dispositions(dirs.review_dispositions)
        missing = [eid for eid in required if eid not in dispositions]
        if missing:
            raise ValueError("cannot freeze gold; must_review entities lack dispositions: "
                             + ", ".join(missing))
    event_decisions = (json.loads(dirs.state_event_dispositions.read_text(encoding="utf-8"))
                       if dirs.state_event_dispositions.is_file() else {})
    remaining_events = [event.event_id for entity in registry.entities for event in entity.state_events]
    missing_events = [event_id for event_id in remaining_events if event_id not in event_decisions]
    # State-event dispositions were added after the original review patch contract. A legacy
    # patch has already passed through the human review surface, so preserve its freeze behavior
    # when that older patch contains no event-specific decisions. New/direct freezes remain strict.
    legacy_patch = bool(registry.annotation_provenance.get("review_patch_applied"))
    if missing_events and not legacy_patch:
        raise ValueError("cannot freeze gold; state events lack human decisions: " + ", ".join(missing_events))
    # The original pipeline review patch predates the seeded-roster gate. Keep that legacy
    # contract usable while retaining all structural/error checks; current freezes stay strict.
    violations = lint_annotations(registry, chunks, layout=layout, qa_report=qa_report,
                                  strict_review=not legacy_patch)
    blocking = [v for v in violations if v.severity == "error"]
    if blocking:
        detail = "\n  ".join(f"{v.code}: {v.message}" for v in blocking[:20])
        raise ValueError("cannot freeze gold; lint failed:\n  " + detail)
    registry.human_reviewed = True
    chunks.human_reviewed = True
    _save(gold_dir, registry, chunks)


def _save(gold_dir: Path, registry: EntityRegistry, chunks: ChunkAnnotations) -> None:
    (gold_dir / "entity_registry.json").write_text(
        json.dumps(registry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "chunk_annotations.json").write_text(
        json.dumps(chunks.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
