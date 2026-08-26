#!/usr/bin/env python3
"""Complete a hand-authored Track B timeline source into a full per-segment GT.

The human authors only a partial skeleton (present + semantic events); this
script *completes* it by deriving every memory label (op/gap/probe/forbidden)
that would be error-prone to hand-write at 50-200 segments.

Author layer (gt_source/<story>.json): a human writes only
  * entities + state machines + lookalike pairs, and
  * a list of scenes, each carrying `present` (who is on screen), hand-written
    5s `actions` (one prose line == one segment), optional `lookalike_present`,
    and semantic `events` (state_change / remove).

This builder derives everything a human would otherwise hand-label and get
wrong at 50-200 segments:
  * memory_op per (segment, entity): introduce / recall / recall_after_gap /
    transform / persist / forbid,
  * gap (segments since last appearance) and last_seen,
  * memory_probes per segment,
  * forbidden roster entries (permanent removals + absent-lookalike twins +
    auto-detected reference_indirect where a removed entity's name still
    appears in the prose),
  * a manifest with probe/op counts, gap histogram, longest gap, etc.

It ONLY propagates and classifies; it never invents semantics. The human must
author the state changes and removals. The builder then ENFORCES the memory
rules and fails loudly on violations:
  * every forbid must be grounded by a prior removal event,
  * every persist must follow a prior transform,
  * a removed entity must never re-enter `present`,
  * state_change targets must exist in the entity's state machine.

Design decisions (documented, not "truth"):
  * gap_long_threshold: gap (segments absent) >= threshold => recall_after_gap.
  * avoidance_probe_window: mark deprecation_avoidance probe only within this
    many segments after a removal (plus always at reference_indirect segments),
    so the manifest highlights avoidance tests instead of tagging every tail
    segment.

Usage:
    python complete_gt.py                     # all gt_source/*.json -> gt/
    python complete_gt.py --story 0001_lighthouse_keeper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # trackB root (scripts/ lives one level down)


def _locale_root() -> Path:
    if (HERE / "gt_source").is_dir() or (HERE / "gt").is_dir():
        return HERE
    for loc in (HERE / "en", HERE / "zh"):
        if (loc / "gt_source").is_dir() or (loc / "gt").is_dir():
            return loc
    return HERE


_PACK = _locale_root()
SRC_DIR = _PACK / "gt_source"
OUT_DIR = _PACK / "gt"

DEFAULT_GAP_LONG = 30
DEFAULT_AVOID_WINDOW = 4


class BuildError(Exception):
    pass


def _segid(idx: int) -> str:
    return f"seg_{idx + 1:03d}"


def _present_pairs(present: list) -> list[tuple[str, dict[str, Any]]]:
    """A `present` item is either an eid string or an object with `eid` plus
    per-(segment,entity) metadata: `confusable_with` (false-friend / C),
    `count` (quantity memory / E), `anchor` (temporal/intent recall / A)."""
    out: list[tuple[str, dict[str, Any]]] = []
    for it in present:
        if isinstance(it, str):
            out.append((it, {}))
        else:
            out.append((it["eid"], {k: v for k, v in it.items() if k != "eid"}))
    return out


def _flatten(src: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn scenes into a flat ordered segment list (no memory labels yet).

    A scene's `actions` item may be either a plain string (uses scene-level
    `present` / `lookalike_present`) or an object
        {"action": str, "present"?: [...], "events"?: [...], "lookalike_present"?: {...}}
    for per-segment control (mid-scene intercuts, deaths, state changes).
    Scene-level `events` with an integer `at` (local index) are still supported
    and merged with any per-item events.
    """
    segs: list[dict[str, Any]] = []
    for scene in src.get("scenes", []):
        actions = scene.get("actions", [])
        if not actions:
            raise BuildError(f"scene {scene.get('id')} has no actions")
        present_default = list(scene.get("present", []))
        scene_lp = scene.get("lookalike_present")
        n = len(actions)
        ev_by_local: dict[int, list[dict[str, Any]]] = {}
        for ev in scene.get("events", []):
            at = int(ev.get("at", 0))
            if not (0 <= at < n):
                raise BuildError(
                    f"scene {scene['id']} event at={at} out of range (0..{n - 1})")
            ev_by_local.setdefault(at, []).append(ev)
        for local, item in enumerate(actions):
            if isinstance(item, str):
                action, present = item, present_default
                item_events: list[dict[str, Any]] = []
                lp = scene_lp
            else:
                action = item["action"]
                present = list(item.get("present", present_default))
                item_events = list(item.get("events", []))
                lp = item.get("lookalike_present", scene_lp)
            segs.append({
                "scene_id": scene["id"],
                "action": str(action).strip(),
                "present": present,
                "transition": "cut" if local == 0 else "continue",
                "events": ev_by_local.get(local, []) + item_events,
                "lookalike_present": lp,
            })
    return segs


def build(src: dict[str, Any]) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = src.get("entities", {})
    state_machines: dict[str, list[str]] = src.get("state_machines", {})
    lookalike_pairs: list[dict[str, Any]] = src.get("lookalike_pairs", [])
    seg_sec = float(src.get("segment_sec", 5.0))
    gap_long = int(src.get("gap_long_threshold", DEFAULT_GAP_LONG))
    avoid_window = int(src.get("avoidance_probe_window", DEFAULT_AVOID_WINDOW))

    pair_lookup = {tuple(p["pair"]): p for p in lookalike_pairs}
    errors: list[str] = []
    warnings: list[str] = []

    def initial_state(eid: str) -> str | None:
        e = entities.get(eid, {})
        if e.get("initial_state"):
            return e["initial_state"]
        if eid in state_machines:
            return state_machines[eid][0]
        return None

    def appearance_of(eid: str, label: str | None = None) -> str:
        """Resolve the appearance for an entity in a given state. Stateful entities
        carry per-state descriptions under ``states``; stateless ones a single
        ``appearance``. One state == one description; never concatenated."""
        e = entities.get(eid, {})
        states = e.get("states")
        if states:
            if label is None:
                label = initial_state(eid)
            return states.get(label, "")
        return e.get("appearance", "")

    flat = _flatten(src)

    # first pass: validate a removed entity never re-enters `present`
    removed_after: dict[str, int] = {}
    for i, seg in enumerate(flat):
        for ev in seg["events"]:
            if ev["type"] == "remove":
                removed_after.setdefault(ev["eid"], i)
    for i, seg in enumerate(flat):
        for eid, _ in _present_pairs(seg["present"]):
            if eid in removed_after and i > removed_after[eid]:
                errors.append(
                    f"{_segid(i)}: removed entity {eid} re-enters present "
                    f"(removed at {_segid(removed_after[eid])})")

    last_seen: dict[str, int] = {}          # eid -> global idx last present
    current_state: dict[str, str] = {}      # eid -> latest state label
    removal: dict[str, dict[str, Any]] = {}  # eid -> {idx, reason, seg_id}

    out_segments: list[dict[str, Any]] = []
    op_counts: dict[str, int] = {}
    probe_counts: dict[str, int] = {}
    gaps_recorded: list[dict[str, Any]] = []

    for i, seg in enumerate(flat):
        seg_id = _segid(i)
        action = seg["action"]
        present_pairs = _present_pairs(seg["present"])
        present_eids = [e for e, _ in present_pairs]
        probes: set[str] = set()

        ev_state: dict[str, str] = {}
        ev_remove: dict[str, dict[str, Any]] = {}
        for ev in seg["events"]:
            if ev["type"] == "state_change":
                ev_state[ev["eid"]] = ev["to"]
            elif ev["type"] == "remove":
                ev_remove[ev["eid"]] = ev
            else:
                errors.append(f"{seg_id}: unknown event type {ev['type']!r}")

        # --- cast (present entities) ---
        cast: list[dict[str, Any]] = []
        for eid, extras in present_pairs:
            if eid not in entities:
                errors.append(f"{seg_id}: present entity {eid} not in registry")
                continue
            entry: dict[str, Any] = {"eid": eid, "op": None}
            st_override = extras.get("state")  # flashback / temporal-to-past-state
            if st_override is not None and eid in state_machines and st_override not in state_machines[eid]:
                errors.append(
                    f"{seg_id}: flashback state '{st_override}' not in "
                    f"state_machine[{eid}]={state_machines[eid]}")
            transforms_here = None
            if eid in ev_state:
                transforms_here = ev_state[eid]
            elif eid in ev_remove and ev_remove[eid].get("shown", True):
                transforms_here = ev_remove[eid].get("to")

            if transforms_here is not None and st_override is None:
                if eid in state_machines and transforms_here not in state_machines[eid]:
                    errors.append(
                        f"{seg_id}: state '{transforms_here}' not in "
                        f"state_machine[{eid}]={state_machines[eid]}")
                entry["op"] = "transform"
                entry["state"] = {"label": transforms_here,
                                  "appearance": appearance_of(eid, transforms_here)}
                current_state[eid] = transforms_here
                probes.add("state_change")
                # a transform after a gap is ALSO a re-appearance: record the gap so
                # it feeds the decay curve and counts toward long_gap_reappearance.
                if eid in last_seen:
                    gap = i - last_seen[eid] - 1
                    entry["gap"] = gap
                    entry["last_seen"] = _segid(last_seen[eid])
                    if gap >= gap_long:
                        probes.add("long_gap_reappearance")
                    gaps_recorded.append({"eid": eid, "gap": gap, "at": seg_id})
            elif eid not in last_seen:
                entry["op"] = "introduce"
                probes.add("first_appearance")
            else:
                gap = i - last_seen[eid] - 1
                changed = eid in current_state and current_state[eid] != initial_state(eid)
                if st_override is not None:
                    # flashback: show a PAST state without reverting the timeline
                    entry["op"] = "recall_after_gap" if gap >= gap_long else "recall"
                    entry["state"] = {"label": st_override,
                                      "appearance": appearance_of(eid, st_override),
                                      "flashback": True}
                    entry["gap"] = gap
                    entry["last_seen"] = _segid(last_seen[eid])
                    probes.add("long_gap_reappearance" if gap >= gap_long else "continuity")
                elif changed:
                    entry["op"] = "persist"
                    entry["state"] = {"label": current_state[eid],
                                      "appearance": appearance_of(eid, current_state[eid])}
                    entry["gap"] = gap
                    entry["last_seen"] = _segid(last_seen[eid])
                    probes.add("persist_state")
                    if gap >= gap_long:
                        probes.add("long_gap_reappearance")
                elif gap >= gap_long:
                    entry["op"] = "recall_after_gap"
                    entry["gap"] = gap
                    entry["last_seen"] = _segid(last_seen[eid])
                    probes.add("long_gap_reappearance")
                else:
                    entry["op"] = "recall"
                    entry["gap"] = gap
                    entry["last_seen"] = _segid(last_seen[eid])
                    probes.add("continuity")
                if entry.get("gap") is not None:
                    gaps_recorded.append({"eid": eid, "gap": entry["gap"], "at": seg_id})

            # per-(segment,entity) authored metadata -> extra probes
            cw = extras.get("confusable_with")
            if cw is not None:
                entry["confusable_with"] = cw
                if cw not in entities:
                    errors.append(f"{seg_id}: {eid} confusable_with unknown {cw}")
                if entry["op"] == "introduce":
                    probes.add("false_friend")
            cnt = extras.get("count")
            if cnt is not None:
                entry["count"] = cnt
                if entry["op"] in ("recall", "recall_after_gap", "persist"):
                    probes.add("count_memory")
            anch = extras.get("anchor")
            if anch is not None:
                entry["anchor"] = anch
                if anch.get("type") == "temporal":
                    probes.add("temporal_reference")
                    rt = anch.get("resolves_to")
                    if rt and rt not in entities:
                        errors.append(f"{seg_id}: {eid} anchor.resolves_to unknown {rt}")
                    nm = entities.get(eid, {}).get("name", "")
                    if nm and nm in action:
                        warnings.append(
                            f"{seg_id}: temporal-anchored {eid} name『{nm}』appears in "
                            "action; a temporal reference should be name-free")

            cast.append(entry)
            op_counts[entry["op"]] = op_counts.get(entry["op"], 0) + 1
            last_seen[eid] = i

        # apply removals AFTER counting presence in this segment
        for eid, ev in ev_remove.items():
            if eid not in entities:
                errors.append(f"{seg_id}: remove of unknown entity {eid}")
                continue
            if ev.get("shown", True) and eid not in present_eids:
                errors.append(
                    f"{seg_id}: shown removal of {eid} but it is not in present")
            removal[eid] = {
                "idx": i,
                "reason": ev.get("reason", "removed"),
                "seg_id": seg_id,
            }

        # --- forbidden roster ---
        forbidden: list[dict[str, Any]] = []
        for eid, info in removal.items():
            if i <= info["idx"]:
                continue  # present/being-removed this segment, not yet forbidden
            fentry = {
                "eid": eid,
                "reason": info["reason"],
                "grounded_by": info["seg_id"],
            }
            name = entities.get(eid, {}).get("name", "")
            if name and name in action:
                fentry["reference_indirect"] = True
                probes.add("reference_indirect")
                probes.add("deprecation_avoidance")
            elif i - info["idx"] <= avoid_window:
                probes.add("deprecation_avoidance")
            forbidden.append(fentry)

        # --- lookalike (co-occurrence or absent twin) ---
        lookalike_active: list[dict[str, Any]] = []
        lp = seg.get("lookalike_present")
        if lp:
            pair = tuple(lp["pair"])
            if pair not in pair_lookup:
                errors.append(f"{seg_id}: lookalike pair {pair} not declared")
            members = lp.get("members", [])
            for m in members:
                if m not in pair:
                    errors.append(f"{seg_id}: lookalike member {m} not in pair {pair}")
            probes.add("lookalike_disambiguation")
            lookalike_active.append({
                "pair": list(pair),
                "features": pair_lookup.get(pair, {}).get("features", {}),
                "present_members": members,
            })
            for m in pair:
                if m not in members:
                    forbidden.append({
                        "eid": m,
                        "reason": "lookalike_absent",
                        "grounded_by": f"lookalike:{'/'.join(pair)}",
                    })

        for p in probes:
            probe_counts[p] = probe_counts.get(p, 0) + 1

        out_seg: dict[str, Any] = {
            "segment_id": seg_id,
            "scene_id": seg["scene_id"],
            "duration_sec": seg_sec,
            "transition": seg["transition"],
            "action": action,
            "memory_probes": sorted(probes),
            "cast": cast,
            "forbidden": forbidden,
        }
        if lookalike_active:
            out_seg["lookalike_active"] = lookalike_active
        out_segments.append(out_seg)

    # --- validation: persist must follow a transform (by construction, but double check) ---
    seen_transform: set[str] = set()
    for seg in out_segments:
        for c in seg["cast"]:
            if c["op"] == "transform":
                seen_transform.add(c["eid"])
            if c["op"] == "persist" and c["eid"] not in seen_transform:
                errors.append(
                    f"{seg['segment_id']}: persist of {c['eid']} without prior transform")

    if errors:
        raise BuildError("memory-rule validation failed:\n  - " + "\n  - ".join(errors))

    # --- manifest ---
    gap_hist = {"1-4": 0, "5-29": 0, ">=30": 0}
    for g in gaps_recorded:
        v = g["gap"]
        if v <= 4:
            gap_hist["1-4"] += 1
        elif v < 30:
            gap_hist["5-29"] += 1
        else:
            gap_hist[">=30"] += 1
    longest = max(gaps_recorded, key=lambda x: x["gap"], default=None)

    # GT-facing entities: a stateful entity's appearance lives ONLY per-state
    # (one state, one look). We keep `states` + `initial_state` and DO NOT mirror
    # a top-level `appearance`; the scorer/prompt builder resolve a base look from
    # states[initial_state] when they need one (decoys/forbidden/first sight).
    gt_entities: dict[str, dict[str, Any]] = {}
    for eid, e in entities.items():
        ge = dict(e)
        if e.get("states"):
            ge.pop("appearance", None)
            ge["initial_state"] = initial_state(eid)
        gt_entities[eid] = ge

    kind_counts: dict[str, int] = {}
    for e in entities.values():
        kind_counts[e.get("kind", "?")] = kind_counts.get(e.get("kind", "?"), 0) + 1

    # --- balance check over HARD probes (continuity/persist/first_appearance are
    # abundant controls, not balanced). Each hard capability needs enough
    # opportunities for its per-capability score to be statistically stable. ---
    HARD_PROBES = ["long_gap_reappearance", "state_change", "lookalike_disambiguation",
                   "reference_indirect", "deprecation_avoidance", "count_memory",
                   "false_friend", "temporal_reference"]
    default_target = int(src.get("probe_target_default", 4))
    probe_targets = src.get("probe_targets", {})
    balance: dict[str, dict[str, Any]] = {}
    for p in HARD_PROBES:
        tgt = int(probe_targets.get(p, default_target))
        got = probe_counts.get(p, 0)
        balance[p] = {"count": got, "target": tgt, "ok": got >= tgt}
        if got < tgt:
            warnings.append(f"balance: probe '{p}' underrepresented ({got}<{tgt})")

    gt = {
        "story_id": src.get("story_id"),
        "gt_version": "trackB-gt-2.0",
        "title": src.get("title"),
        "premise": src.get("premise"),
        "built_from": f"gt_source/{src.get('story_id')}.json",
        "params": {
            "segment_sec": seg_sec,
            "gap_long_threshold": gap_long,
            "avoidance_probe_window": avoid_window,
        },
        "entities": gt_entities,
        "state_machines": state_machines,
        "lookalike_pairs": lookalike_pairs,
        "segments": out_segments,
        "summary": {
            "n_segments": len(out_segments),
            "n_scenes": len(src.get("scenes", [])),
            "n_entities": len(entities),
            "kind_counts": kind_counts,
            "n_state_threads": len(state_machines),
            "n_removals": len(removal),
            "op_counts": op_counts,
            "probe_counts": probe_counts,
            "hard_probe_balance": balance,
            "gap_histogram": gap_hist,
            "longest_gap": longest,
        },
        "warnings": warnings,
    }
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--story", help="story_id (basename); default: all in gt_source/")
    ap.add_argument("--src-dir", default=str(SRC_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    src_dir, out_dir = Path(a.src_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.story:
        src_files = [src_dir / f"{a.story}.json"]
    else:
        src_files = sorted(src_dir.glob("*.json"))
    if not src_files:
        raise SystemExit(f"no source files found in {src_dir}")

    rc = 0
    for src_path in src_files:
        src = json.loads(src_path.read_text(encoding="utf-8"))
        try:
            gt = build(src)
        except BuildError as e:
            print(f"[FAIL] {src_path.name}: {e}")
            rc = 1
            continue
        out_path = out_dir / f"{gt['story_id']}.json"
        out_path.write_text(
            json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        s = gt["summary"]
        lg = s["longest_gap"]
        lg_str = f"{lg['gap']}@{lg['at']}" if lg else "-"
        print(f"[ok] {src_path.name} -> {out_path.relative_to(HERE)}  "
              f"{s['n_segments']} segs, {s['n_entities']} ents, "
              f"longest_gap={lg_str}")
        print(f"     probes={s['probe_counts']}")
        print(f"     ops={s['op_counts']} gap_hist={s['gap_histogram']}")
        bw = [w.split("balance: ", 1)[1] for w in gt["warnings"] if w.startswith("balance:")]
        if bw:
            print(f"     ! balance underrepresented: {'; '.join(bw)}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
