#!/usr/bin/env python3
"""Re-adjudicate S3 verdicts and rebuild the S4 queue from stored VLM outputs.

Why this exists
---------------
The S3 canonical-name gate used to reject an otherwise-good VLM rewrite when a
paraphrase dropped an already-present location/generic mention (e.g. shortened
"郊区街道" to "街道") while adding a newly confirmed character. Every such
segment became a deterministic ``action_missing_canonical_name`` BLOCK, and S4
queues *every* BLOCK, flooding the human review surface.

The fix re-applies the deterministic canonicalizer (``rewrite_action_canonical_
mentions``) inside ``_repair_action_with_trusted_names`` and ``_apply_canonical_
name_gate``. This tool propagates that fix to movies whose S3 already ran, using
only the VLM outputs already stored in ``segment_audit.jsonl`` — no GPU / VLM
calls. It reuses the exact production adjudication functions, so the result
matches what a fresh S3 run would produce for the recovered segments.

Safety
------
* Dry-run by default; pass ``--apply`` to write.
* Never touches a movie whose S4 audit is already ``human_reviewed`` (that would
  clobber human decisions).
* Only rewrites ``revised_action`` for segments where the deterministic
  canonicalizer *fully* clears the coverage gap; genuine gaps (a present
  character/prop never named in the action) stay BLOCK and reach human review.
* Writes a ``.bak_readjudicate`` copy of each mutated file the first time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path
from typing import Any

# Must mirror orchestrator.py (the single source of truth for S4 sampling).
# Imported by value rather than symbol to avoid pulling in the S5 numpy stack.
_S4_SAMPLE_MINIMUM = 3
_S4_SAMPLE_RATE = 0.01

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.canonical_names import (  # noqa: E402
    action_has_entity_list_coda,
    try_complete_canonical_action,
)
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.vlm_auto_review import (  # noqa: E402
    SegmentReview,
    _apply_canonical_name_gate,
    _apply_typed_verdict,
)
from vmem_bench.annotation.pipeline.stages.s4_segment_sampling_human_review.sampling import (  # noqa: E402
    build_sample,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _roster(annotation: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    roster: list[dict[str, str]] = []
    for group, id_key, kind in (
        ("characters", "char_id", "character"),
        ("props", "prop_id", "prop"),
        ("locations", "loc_id", "location"),
    ):
        for raw in annotation.get(group) or []:
            eid = str(raw.get(id_key) or "")
            if eid:
                roster.append({"entity_id": eid, "name": str(raw.get("name") or ""), "kind": kind})
    return roster, {r["entity_id"]: r for r in roster}


def _segment_map(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scene in (annotation.get("screenplay") or {}).get("scenes") or []:
        for seg in scene.get("visual_segments") or []:
            out[str(seg.get("segment_id") or "")] = seg
    return out


def _best_pre_gate_action(audit_row: dict[str, Any], present: list[str], roster_by_id: dict[str, dict[str, str]]) -> str:
    """Reconstruct the action that the *fixed* repair+gate would start from.

    The fixed ``_repair_action_with_trusted_names`` adopts a repair candidate
    once the deterministic canonicalizer fully clears it; otherwise the gate
    canonicalizes the raw VLM review output. Mirror that preference order here.
    """
    rounds = audit_row.get("rounds") or []
    raw = (rounds[-1].get("raw") if rounds else audit_row.get("raw")) or {}
    candidate = str((raw.get("text_action_repair") or {}).get("candidate") or "")
    if candidate and not action_has_entity_list_coda(candidate):
        completed = try_complete_canonical_action(
            action=candidate, present_entity_ids=present, roster_by_id=roster_by_id
        )
        if completed:
            return completed
    return str(raw.get("revised_action") or audit_row.get("revised_action") or "")


def readjudicate_movie(movie_dir: Path, *, apply: bool) -> dict[str, Any]:
    pipeline = movie_dir / "tmp" / "pipeline"
    s3_dir = pipeline / "s3_segment_auto_review_revise"
    s4_dir = pipeline / "s4_segment_sampling_human_review"
    ann_path = s3_dir / "auto_revised_annotation.json"
    audit_path = s3_dir / "segment_audit.jsonl"
    if not ann_path.is_file() or not audit_path.is_file():
        return {"movie": movie_dir.name, "skipped": "no_s3_output"}

    s4_audit = s4_dir / "review_audit.json"
    if s4_audit.is_file():
        try:
            if bool(_load_json(s4_audit).get("human_reviewed")):
                return {"movie": movie_dir.name, "skipped": "s4_human_reviewed"}
        except Exception:  # noqa: BLE001
            pass

    annotation = _load_json(ann_path)
    roster, roster_by_id = _roster(annotation)
    seg_map = _segment_map(annotation)

    audit_rows: list[dict[str, Any]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            audit_rows.append(json.loads(line))

    before = {}
    after = {}
    changed_actions = 0
    new_rows: list[dict[str, Any]] = []
    for row in audit_rows:
        sid = str(row.get("segment_id") or "")
        before[row.get("verdict")] = before.get(row.get("verdict"), 0) + 1
        present = [str(x) for x in (row.get("revised_present") or [])]
        rounds = row.get("rounds") or []
        raw = (rounds[-1].get("raw") if rounds else row.get("raw")) or {}
        pre_action = _best_pre_gate_action(row, present, roster_by_id)
        review = SegmentReview(
            segment_id=sid,
            revised_present=present,
            revised_action=pre_action,
            confidence=str(row.get("confidence") or "low"),
            risk_reasons=[str(x) for x in (raw.get("risk_reasons") or [])],
            raw={k: v for k, v in raw.items() if k not in ("verdict", "findings", "recommended_action")},
            accepted=bool(raw.get("accepted")),
        )
        review = _apply_canonical_name_gate(
            review, roster=roster, previous_action=str(row.get("revised_action") or "")
        )
        if not review.accepted:
            review.risk_reasons = list(dict.fromkeys([*review.risk_reasons, "max_review_rounds_exhausted"]))
        review = _apply_typed_verdict(review)

        merged = dict(row)
        if review.revised_action and review.revised_action != row.get("revised_action"):
            changed_actions += 1
        merged["revised_action"] = review.revised_action
        merged["verdict"] = review.verdict
        merged["findings"] = review.findings
        merged["recommended_action"] = review.recommended_action
        merged["risk_reasons"] = review.risk_reasons
        merged["accepted"] = review.accepted
        new_rows.append(merged)
        after[review.verdict] = after.get(review.verdict, 0) + 1

        seg = seg_map.get(sid)
        if seg is not None and review.revised_action:
            seg["action"] = review.revised_action
            seg["present_entity_ids"] = list(review.revised_present)

    sample_seed = zlib.adler32(movie_dir.name.encode("utf-8")) & 0x7FFFFFFF
    queue = build_sample(new_rows, minimum=_S4_SAMPLE_MINIMUM, rate=_S4_SAMPLE_RATE, seed=sample_seed)
    result = {
        "movie": movie_dir.name,
        "verdicts_before": before,
        "verdicts_after": after,
        "actions_rewritten": changed_actions,
        "old_queue": len(_load_json(s4_dir / "review_queue.json")) if (s4_dir / "review_queue.json").is_file() else None,
        "new_queue": len(queue),
        "applied": False,
    }

    if apply:
        def _backup(path: Path) -> None:
            bak = path.with_suffix(path.suffix + ".bak_readjudicate")
            if path.is_file() and not bak.is_file():
                shutil.copy2(path, bak)

        _backup(ann_path)
        _backup(audit_path)
        s4_dir.mkdir(parents=True, exist_ok=True)
        queue_path = s4_dir / "review_queue.json"
        _backup(queue_path)

        ann_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        with audit_path.open("w", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
        result["applied"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--dataset", default="LSMDC")
    parser.add_argument("--movie", default="", help="Single movie id; default: all movies in dataset")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = parser.parse_args()

    dataset_dir = args.data_root / args.dataset
    if args.movie:
        movie_dirs = [dataset_dir / args.movie]
    else:
        movie_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    totals = {"old_queue": 0, "new_queue": 0, "actions_rewritten": 0, "movies_changed": 0}
    for movie_dir in movie_dirs:
        res = readjudicate_movie(movie_dir, apply=args.apply)
        if res.get("skipped"):
            continue
        old_q = res.get("old_queue") or 0
        new_q = res.get("new_queue") or 0
        if old_q != new_q or res.get("actions_rewritten"):
            totals["movies_changed"] += 1
            totals["old_queue"] += old_q
            totals["new_queue"] += new_q
            totals["actions_rewritten"] += res.get("actions_rewritten") or 0
            print(
                f"{res['movie']:<40} queue {old_q:>4} -> {new_q:<4} "
                f"| actions rewritten {res.get('actions_rewritten'):>4} "
                f"| verdicts {res['verdicts_before']} -> {res['verdicts_after']}"
            )
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(
        f"\n[{mode}] movies changed={totals['movies_changed']} "
        f"S4 queue {totals['old_queue']} -> {totals['new_queue']} "
        f"actions rewritten={totals['actions_rewritten']}"
    )


if __name__ == "__main__":
    main()
