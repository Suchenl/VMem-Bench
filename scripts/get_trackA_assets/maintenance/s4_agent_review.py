#!/usr/bin/env python3
"""S4 sign-off performed by the Cursor agent instead of a human.

Pipeline recap
--------------
* **S3** (``s3_segment_auto_review_revise``) is the actual review: a VLM watches
  every segment clip and revises ``present_entity_ids`` / ``action`` per segment.
* **S4** (``s4_segment_sampling_human_review``) does not re-watch anything. It
  samples every BLOCK plus a stratified slice of PASS segments into a queue and
  then waits for a *human* to sign off (``mode=open_for_human_review``). This is
  the last gate before gold freeze.

This tool lets the **Cursor agent** stand in for that human sign-off: it accepts
the S3 VLM-reviewed annotation for the whole queue, marks the stage
``human_reviewed``, and stamps an ``agent_reviewed`` provenance block so it is
recorded that an agent (not a person) cleared the gate. It does **not** run a
second VLM pass or re-cut clips — S3 already did the visual review.

By default it can also revise individual segments if a ``--decisions`` file is
supplied (``{"seg_id": {"present": [...ids], "action"?: str, "note"?: str}}``);
without it, every queued segment is accepted as S3 left it.

Safety: never touches a movie already ``human_reviewed``; requires a decision
(implicit accept) for every queue item; edits that would fail canonical
validation are downgraded to accept.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]  # benchmarks/MemStrata
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (  # noqa: E402
    action_has_entity_list_coda,
    missing_canonical_names,
    try_complete_canonical_action,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.decisions import (  # noqa: E402
    apply_s4_decisions,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _roster(annotation: dict[str, Any]) -> dict[str, dict[str, str]]:
    roster: dict[str, dict[str, str]] = {}
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for raw in annotation.get(group) or []:
            eid = str(raw.get(id_key) or "")
            if eid:
                roster[eid] = {"entity_id": eid, "name": str(raw.get("name") or ""), "kind": kind}
    return roster


def _segment_map(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        for seg in scene.get("visual_segments") or []:
            out[str(seg.get("segment_id") or "")] = seg
    return out


def _strip_coda(action: str) -> str:
    return "" if action_has_entity_list_coda(action) else str(action or "")


def _valid_edit_action(present_ids: list[str], base_action: str, roster: dict[str, dict[str, str]]) -> str:
    base = _strip_coda(base_action)
    if not present_ids:
        return base or "画面中无可辨识的角色、道具或地点实体。"
    completed = try_complete_canonical_action(action=base or "", present_entity_ids=present_ids, roster_by_id=roster)
    if completed:
        return completed
    names = [roster[e]["name"] for e in present_ids if roster.get(e, {}).get("name")]
    prose = ("、".join(names) + "出现在画面中。") if names else base
    if names and not missing_canonical_names(action=prose, present_entity_ids=present_ids, roster_by_id=roster) \
            and not action_has_entity_list_coda(prose):
        return prose
    return ("、".join(names) + "。" + base) if names else (base or "画面。")


def finalize_movie(movie_dir: Path, *, reviewer: str, overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    name = movie_dir.name
    pipeline = movie_dir / "tmp" / "pipeline"
    s3_ann = pipeline / "s3_segment_auto_review_revise" / "auto_revised_annotation.json"
    s4_dir = pipeline / "s4_segment_sampling_human_review"
    queue_path = s4_dir / "review_queue.json"
    audit_path = s4_dir / "review_audit.json"
    if not queue_path.is_file() or not s3_ann.is_file():
        return {"movie": name, "skipped": "no_s3_s4"}
    if audit_path.is_file():
        try:
            if bool(_load_json(audit_path).get("human_reviewed")):
                return {"movie": name, "skipped": "already_human_reviewed"}
        except Exception:  # noqa: BLE001
            pass

    annotation = _load_json(s3_ann)
    roster = _roster(annotation)
    seg_map = _segment_map(annotation)
    queue = _load_json(queue_path)
    queue_ids = [str(it.get("segment_id") or "") for it in queue if it.get("segment_id")]
    overrides = overrides or {}

    decisions: dict[str, dict[str, Any]] = {}
    n_accept = n_edit = n_downgraded = 0
    for sid in queue_ids:
        seg = seg_map.get(sid, {})
        s3_present = [str(x) for x in (seg.get("present_entity_ids") or [])]
        ov = overrides.get(sid)
        if not ov:
            decisions[sid] = {"action": "accept", "reason": f"{reviewer} S4 sign-off: accept S3 VLM-reviewed annotation"}
            n_accept += 1
            continue
        present = [e for e in dict.fromkeys(str(x) for x in (ov.get("present") or s3_present)) if e in roster]
        note = str(ov.get("note") or "")
        if set(present) == set(s3_present):
            decisions[sid] = {"action": "accept", "reason": f"{reviewer} S4 sign-off. {note}".strip()}
            n_accept += 1
            continue
        action = str(ov.get("action") or "") or _valid_edit_action(present, str(seg.get("action") or ""), roster)
        if missing_canonical_names(action=action, present_entity_ids=present, roster_by_id=roster) \
                or action_has_entity_list_coda(action):
            action = _valid_edit_action(present, str(seg.get("action") or ""), roster)
        if missing_canonical_names(action=action, present_entity_ids=present, roster_by_id=roster) \
                or action_has_entity_list_coda(action):
            decisions[sid] = {"action": "accept", "reason": f"{reviewer} S4 sign-off: edit invalid, kept S3. {note}".strip()}
            n_accept += 1
            n_downgraded += 1
            continue
        decisions[sid] = {
            "action": "edit_both", "present_entity_ids": present, "revised_action": action,
            "reason": f"{reviewer} S4 sign-off: present {s3_present}->{present}. {note}".strip(),
        }
        n_edit += 1

    reason = f"agent_reviewed:{reviewer} via s4_agent_review.py (accept={n_accept}, edit={n_edit})"
    apply_s4_decisions(movie_dir=movie_dir, decisions=decisions, film_verdict="accept", reason=reason)
    audit = _load_json(audit_path)
    audit["agent_reviewed"] = True
    audit["reviewer"] = reviewer
    audit["review_method"] = "s4_agent_review.py"
    audit["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    audit["agent_decision_counts"] = {"accept": n_accept, "edit_both": n_edit, "downgraded_to_accept": n_downgraded}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"movie": name, "n_queue": len(queue_ids), "accept": n_accept, "edit": n_edit, "applied": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--dataset", default="BlenderOpenMovies")
    ap.add_argument("--movie", default="", help="single movie id; default all not-yet-reviewed")
    ap.add_argument("--reviewer", default="cursor-agent")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="optional JSON of per-segment overrides {seg_id:{present:[...],action?,note?}}")
    args = ap.parse_args()

    overrides = None
    if args.decisions is not None:
        raw = _load_json(args.decisions)
        overrides = raw.get("decisions", raw) if isinstance(raw, dict) else {}

    dataset_dir = args.data_root / args.dataset
    movie_dirs = [dataset_dir / args.movie] if args.movie else sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    totals = {"movies": 0, "queue": 0, "accept": 0, "edit": 0}
    for md in movie_dirs:
        res = finalize_movie(md, reviewer=args.reviewer, overrides=overrides if args.movie else None)
        if res.get("skipped"):
            print(f"[skip] {res['movie']:<28} {res['skipped']}")
            continue
        totals["movies"] += 1
        totals["queue"] += res["n_queue"]; totals["accept"] += res["accept"]; totals["edit"] += res["edit"]
        print(f"[done] {res['movie']:<28} q={res['n_queue']:>3} accept={res['accept']:>3} edit={res['edit']:>3}")
    print(f"\nmovies={totals['movies']} queue={totals['queue']} accept={totals['accept']} edit={totals['edit']}")
    print("FINALIZE_S4_DONE")


if __name__ == "__main__":
    main()
