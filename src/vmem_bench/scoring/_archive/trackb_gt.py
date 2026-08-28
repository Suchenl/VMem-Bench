"""[RETIRED 2026-07 — do not use] MoVE-Bench Track B — ground-truth exporter.

Superseded by the hand-authored hard-case pipeline:
  * author:  assets/trackB/en/gt_source/<story>.json
  * compile: assets/trackB/complete_gt.py  -> assets/trackB/en/gt/<story>.json (gt_version 2.0)
  * prompts: assets/trackB/get_sut_prompts.py -> assets/trackB/en/sut_prompts/<story>_*.json
The current judge (``end2end_coverage.py``) reads the 2.0 per-segment GT, NOT this output.
Kept only for provenance. Original docstring follows.

MoVE-Bench Track B — ground-truth exporter (bench-side, SUT-independent).

Turns a ``production_screenplay`` JSON (+ a hand-authored hard-case sidecar) into the
frozen per-shot GT that the Track B end-to-end judge (``end2end_coverage.py``) scores
against. This module deliberately **does not import** ``memstrata.*`` (bench<->SUT zero
-import rule, benchmarks/MemStrata/AGENTS.md §2): it reads the raw screenplay JSON so the
GT can also score *other* end-to-end pipelines, not just MemStrata.

Derivation mirrors the SUT loop's own reading of the screenplay (``memstrata.adapters.
screenplay.iter_shots``) so GT and the SUT are talking about the same entities:
  present_required = active_characters ∪ planned_assets(op ∈ {preserve, transform})
  forbidden        = planned_assets(op ∈ {avoid, deprecate})
Everything the ``operation`` field cannot express — the *new-state text* after a
transform (E3 "weathered" is only in continuity_requirements prose!), which pairs are
look-alikes, and any present-set completeness fixes — comes from the sidecar
``<story>.overrides.json`` and is propagated with the rules below.

Sidecar schema (all keys optional)::

    {"story_id": "...",
     "lookalike_pairs": [{"pair": ["E2","E6"], "features": {"E2": "...", "E6": "..."}, "note": "..."}],
     "state_changes":   [{"eid": "E4", "from_shot": "shot_0010", "label": "cracked", "desc": "..."}],
     "present_add":  {"shot_0003": ["E1"]},          # present_required completeness fixes
     "allowed_add":  {"shot_0003": ["E7"]},          # present_allowed (not scored for recall/precision-violation)
     "decoy_force":  {"shot_0005": ["E6"]}}          # force a hard absent entity into the blinded roster

State propagation is **sticky and monotone**: a state_change at ``from_shot`` attaches to
every later shot where the entity is in ``present_required`` (0001's E4 cracks at shot_0010
and must stay cracked in 0011/0014/0015). Multiple changes for one entity keep the latest
whose ``from_shot`` index ≤ current shot (supports 完好→碎裂→拼合).

CLI::

    python -m vmem_bench.scoring.trackb_gt \
        --screenplay <memstrata-repo>/production/screenplay/products/cn/0001_lighthouse_keeper.json \
        --out        data/MoVE-Bench/trackB/gt/0001_lighthouse_keeper.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

_ENTITY_TAG = re.compile(r"\s*[（(]\s*(E\d+)\s*[）)]")  # full-width or ASCII parens
_KIND_BY_TYPE = {"character": "character", "object": "prop", "prop": "prop", "location": "location"}
_PRESERVE = {"preserve", "transform"}
_FORBID = {"avoid", "deprecate"}

# blinded-roster decoy sampling: how many clearly-absent entities to mix in per shot
DECOY_MIN, DECOY_MAX = 3, 5


# --------------------------------------------------------------------------- helpers
def _entities(sp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """entity_id -> {name, kind, appearance}. Prefer main_entities (richer appearance)."""
    out: dict[str, dict[str, Any]] = {}
    for e in sp.get("main_entities", []) or sp.get("cast_and_assets", []):
        eid = str(e["entity_id"])
        out[eid] = {
            "name": str(e.get("name", eid)),
            "kind": _KIND_BY_TYPE.get(str(e.get("entity_type", "")).lower(), "character"),
            "appearance": str(e.get("appearance", e.get("creative_description", ""))),
        }
    return out


def _sub_tags(text: str, describe) -> str:
    """Replace every ``(E#)`` tag in ``text`` via ``describe(eid) -> str``.

    ``describe`` returns the *already-parenthesized* replacement (e.g. ``（外观…）``) or ``""``
    to drop the tag entirely. The canonical entity name is already written into the screenplay
    prose, so a continuity mention needs no substitution at all — we only inline an appearance or
    a state phrase at the shot where it is actually needed (see ``render_prompts``). This is the
    ONLY place prompts are shaped; no SUT/eval may post-process them (Screenplay design §11).
    """
    def _sub(m: "re.Match[str]") -> str:
        return describe(m.group(1))
    return _ENTITY_TAG.sub(_sub, text).strip()


def _make_describer(ents: dict[str, dict[str, Any]], described: set[str],
                    starts: dict[str, str], state_done: set[str]):
    """Build the per-shot ``describe(eid)`` closure used by :func:`_sub_tags`.

    * first textual mention of an entity  -> inline its ``appearance`` (+ ``initial_state``);
    * a shot where a state change takes effect -> inline the new-state text once;
    * every other mention -> drop the tag (bare name; continuity is carried by memory).
    """
    def describe(eid: str) -> str:
        e = ents.get(eid)
        if e is None:
            return ""
        if eid not in described:
            described.add(eid)
            appr = str(e.get("appearance", "")).strip().rstrip("。.")
            init = str(e.get("initial_state", "")).strip().rstrip("。.")
            if eid in starts and starts.get(eid):  # first appearance IS the change shot
                state_done.add(eid)
                parts = [p for p in (appr, starts[eid].strip().rstrip("。.")) if p]
            else:
                parts = [p for p in (appr, init) if p]
            desc = "，".join(parts)
            return f"（{desc}）" if desc else ""
        if eid in starts and eid not in state_done and starts.get(eid):
            state_done.add(eid)
            return f"（{starts[eid].strip().rstrip('。.')}）"
        return ""
    return describe


def _shot_present(shot: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(present_required, forbidden) exactly as the SUT's iter_shots reads planned_assets."""
    required: set[str] = {str(e) for e in shot.get("active_characters", [])}
    forbidden: set[str] = set()
    for pa in shot.get("planned_assets", []):
        pid = str(pa.get("planned_asset_id", ""))
        op = str(pa.get("operation", "preserve")).lower()
        if op in _FORBID:
            forbidden.add(pid)
        elif pa.get("required") or op in _PRESERVE:
            required.add(pid)
    return required, forbidden


def _seeded_sample(candidates: list[str], k: int, salt: str) -> list[str]:
    """Deterministic decoy sample: reproducible across machines (hashlib seed, not hash())."""
    if not candidates or k <= 0:
        return []
    seed = int(hashlib.md5(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pool = sorted(candidates)
    rng.shuffle(pool)
    return sorted(pool[:min(k, len(pool))])


# --------------------------------------------------------------------------- build
def build_gt(sp: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = overrides or {}
    story_id = str(sp.get("story_id", "unknown"))
    ents = _entities(sp)
    names = {eid: e["name"] for eid, e in ents.items()}
    all_ids = set(ents)
    shots = sp.get("production_screenplay", {}).get("shots", [])

    # sidecar
    lookalike_pairs = overrides.get("lookalike_pairs", [])
    state_changes = overrides.get("state_changes", [])
    present_add = overrides.get("present_add", {})
    allowed_add = overrides.get("allowed_add", {})
    decoy_force = overrides.get("decoy_force", {})
    shot_index = {str(s.get("shot_id", f"shot_{i:04d}")): i for i, s in enumerate(shots)}

    warnings: list[str] = []
    # transform shots that have no sidecar state text -> flag (state GT would be incomplete)
    sidecar_change_shots = {(sc["eid"], shot_index.get(sc["from_shot"])) for sc in state_changes}

    seen: set[str] = set()          # entities that have appeared in present_required so far
    last_seen: dict[str, int] = {}  # eid -> last chunk index it was present_required
    out_shots: list[dict[str, Any]] = []

    for i, shot in enumerate(shots):
        shot_id = str(shot.get("shot_id", f"shot_{i:04d}"))
        req, forb = _shot_present(shot)
        req |= {str(e) for e in present_add.get(shot_id, [])}
        allowed = ({str(e) for e in allowed_add.get(shot_id, [])} - req) - forb

        # flag machine transforms with no sidecar text
        for pa in shot.get("planned_assets", []):
            if str(pa.get("operation", "")).lower() == "transform":
                eid = str(pa.get("planned_asset_id", ""))
                if (eid, i) not in sidecar_change_shots:
                    warnings.append(f"{shot_id}: planned transform of {eid} has no sidecar state_change text")

        first = {e for e in req if e not in seen}
        continuity = sorted(req - first)

        # gap: chunks since each present_required entity was last present
        gap = {e: (i - last_seen[e] - 1) for e in req if e in last_seen and (i - last_seen[e] - 1) > 0}

        # decoys: clearly-absent entities, deterministic sample + forced hard decoys
        cand = sorted(all_ids - req - allowed - forb)
        decoys = set(_seeded_sample(cand, DECOY_MAX, f"{story_id}:{shot_id}"))
        decoys |= {str(e) for e in decoy_force.get(shot_id, []) if e in all_ids and e not in req and e not in allowed and e not in forb}
        if len(decoys) < min(DECOY_MIN, len(cand)):
            decoys |= set(cand[: DECOY_MIN])
        decoys = sorted(decoys)

        # sticky state_expected: latest change whose from_shot index <= i, for entities present
        state_expected: dict[str, dict[str, Any]] = {}
        for sc in state_changes:
            fi = shot_index.get(sc["from_shot"])
            if fi is not None and fi <= i and sc["eid"] in req:
                prev = state_expected.get(sc["eid"])
                if prev is None or shot_index.get(prev["from_shot"], -1) <= fi:
                    state_expected[sc["eid"]] = {"label": sc["label"], "desc": sc.get("desc", ""),
                                                 "from_shot": sc["from_shot"]}

        # look-alike pairs active this shot (>=1 member in present_required)
        lookalike_active = []
        for lp in lookalike_pairs:
            members = [str(x) for x in lp.get("pair", [])]
            present_members = [m for m in members if m in req]
            if present_members:
                lookalike_active.append({"pair": members, "features": lp.get("features", {}),
                                         "present_members": present_members, "note": lp.get("note", "")})

        out_shots.append({
            "chunk_id": i, "shot_id": shot_id, "scene_id": str(shot.get("scene_id", "")),
            "transition": str(shot.get("transition", "cut")).lower(),
            "duration_sec": float(shot.get("duration_sec", 5.0)),
            "present_required": sorted(req), "present_allowed": sorted(allowed),
            "forbidden": sorted(forb), "first_appearances": sorted(first),
            "continuity": continuity, "decoys": decoys, "gap": gap,
            "state_expected": state_expected, "lookalike_active": lookalike_active,
        })
        seen |= req
        for e in req:
            last_seen[e] = i

    kinds = {eid: e["kind"] for eid, e in ents.items()}
    summary = {
        "n_shots": len(out_shots),
        "n_entities": len(ents),
        "kind_counts": {k: sum(1 for v in kinds.values() if v == k) for k in ("character", "prop", "location")},
        "n_forbidden_opportunities": sum(len(s["forbidden"]) for s in out_shots),
        "n_state_shots": sum(1 for s in out_shots if s["state_expected"]),
        "n_lookalike_shots": sum(1 for s in out_shots if s["lookalike_active"]),
        "n_long_gap_ge5": sum(1 for s in out_shots for g in s["gap"].values() if g >= 5),
    }
    return {
        "story_id": story_id, "gt_version": "trackB-gt-0.2",
        "entities": {eid: {**ents[eid]} for eid in ents},
        "lookalike_pairs": lookalike_pairs,
        "shots": out_shots, "summary": summary, "warnings": warnings,
    }


def render_prompts(sp: dict[str, Any], overrides: dict[str, Any] | None = None,
                   register: str = "name_anchored") -> dict[str, Any]:
    """Freeze the per-shot prose prompt + a content SHA — the exact text fed to EVERY system.

    The frozen prompt is authored here once and consumed verbatim; no SUT or eval step may
    rewrite it (Screenplay design_principles §11). Tag handling follows §12: an entity's first
    textual mention carries its appearance (there is no memory yet), a state-change shot carries
    the new state, and continuity mentions are the bare name (identity restored from memory).
    """
    overrides = overrides or {}
    # Richer per-entity map (appearance + initial_state), independent of the GT ``_entities``.
    ents: dict[str, dict[str, Any]] = {}
    for e in sp.get("main_entities", []) or sp.get("cast_and_assets", []):
        ents[str(e["entity_id"])] = {
            "appearance": str(e.get("appearance", e.get("creative_description", ""))),
            "initial_state": str(e.get("initial_state", "")),
        }
    shots = sp.get("production_screenplay", {}).get("shots", [])
    shot_index = {str(s.get("shot_id", f"shot_{i:04d}")): i for i, s in enumerate(shots)}
    # state changes that START at each shot index (from the hand-authored hard-case sidecar)
    change_start: dict[int, dict[str, str]] = {}
    for sc in overrides.get("state_changes", []):
        fi = shot_index.get(sc["from_shot"])
        if fi is not None:
            change_start.setdefault(fi, {})[str(sc["eid"])] = str(sc.get("desc", ""))

    described: set[str] = set()  # entities whose appearance was already established (story-wide)
    rows = []
    for i, shot in enumerate(shots):
        starts = change_start.get(i, {})
        state_done: set[str] = set()
        describe = _make_describer(ents, described, starts, state_done)
        actions = shot.get("visual_track", {}).get("actions", [])
        prompt = " ".join(_sub_tags(a, describe) for a in actions).strip()
        rows.append({"chunk_id": i, "shot_id": str(shot.get("shot_id", f"shot_{i:04d}")), "prompt": prompt})
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return {"story_id": str(sp.get("story_id", "unknown")), "register": register,
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "prompts": rows}


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MoVE-Bench Track B GT exporter")
    ap.add_argument("--screenplay", required=True, type=Path)
    ap.add_argument("--overrides", type=Path, default=None,
                    help="hand-authored hard-case sidecar (default: <screenplay>.overrides.json)")
    ap.add_argument("--out", type=Path, required=True, help="output GT json")
    ap.add_argument("--prompts-out", type=Path, default=None, help="also freeze name_anchored prompts here")
    a = ap.parse_args(argv)

    sp = json.loads(a.screenplay.read_text(encoding="utf-8"))
    ov_path = a.overrides or a.screenplay.with_suffix(".overrides.json")
    overrides = json.loads(ov_path.read_text(encoding="utf-8")) if ov_path.is_file() else {}
    if not ov_path.is_file():
        print(f"[trackb-gt] WARNING: no sidecar at {ov_path} — state/lookalike GT will be empty")

    gt = build_gt(sp, overrides)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[trackb-gt] {gt['story_id']}: {json.dumps(gt['summary'], ensure_ascii=False)}")
    for w in gt["warnings"]:
        print(f"[trackb-gt] WARN {w}")
    if a.prompts_out:
        pr = render_prompts(sp, overrides, "name_anchored")
        a.prompts_out.parent.mkdir(parents=True, exist_ok=True)
        a.prompts_out.write_text(json.dumps(pr, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[trackb-gt] prompts sha256={pr['sha256'][:12]} -> {a.prompts_out}")
    print(f"[trackb-gt] GT -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
