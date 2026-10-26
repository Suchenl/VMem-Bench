#!/usr/bin/env python3
"""Derive SUT-facing prompt streams from hand-authored Track B GT.

Data flow (Track B):

    gt/<story>.json  --(this script, deterministic rules)-->  sut_prompts/<story>_<register>.json

The GT is the single source of truth: hand-authored hard cases that probe the
memory of a causal long-video generator. This script turns each GT into the
*only* thing a System Under Test ever reads: an ordered stream of segment
prompts, with NO labels, NO memory_op, and NO appearance bank. The SUT must
remember appearances/states on its own -- that is the test.

Register rules
--------------
name_anchored (the only register implemented so far):
  * op == "introduce": append the entity's initial-state appearance in
    parentheses right after the first occurrence of its name in the action.
  * op == "transform": append the NEW state's appearance (the current state's
    own description) right after the name -- one state, one description, never
    all states concatenated. This is where the SUT learns the changed look.
  * op in {recall, recall_after_gap, persist}: leave the bare name -- the SUT
    must recall the appearance/state from memory (that is the test). Flashback
    (temporal-to-past-state) segments are name-free by design, so nothing is
    injected there either.
  * forbidden entities: never injected. If a forbidden entity's name happens to
    appear in the action (e.g. engraved on a plaque -> reference_indirect), it
    is left verbatim and gets NO appearance -- the SUT must remember it is gone.

Usage
-----
    python get_sut_prompts.py                     # all gt/*.json -> sut_prompts/
    python get_sut_prompts.py --story 0001_lighthouse_keeper
    python get_sut_prompts.py --register name_anchored
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # trackB root (scripts/ lives one level down)


def _locale_root() -> Path:
    if (HERE / "gt").is_dir() or (HERE / "sut_prompts").is_dir():
        return HERE
    for loc in (HERE / "en", HERE / "zh"):
        if (loc / "gt").is_dir() or (loc / "sut_prompts").is_dir():
            return loc
    return HERE


_PACK = _locale_root()
GT_DIR = _PACK / "gt"
OUT_DIR = _PACK / "sut_prompts"

# ops that reveal an appearance in-prompt: first sight and state changes
_REVEAL_OPS = {"introduce", "transform"}


def _reveal_desc(member: dict[str, Any], ent: dict[str, Any]) -> str:
    """The appearance text to splice in for a revealing member.

    introduce -> entity's (initial-state) appearance;
    transform -> the NEW state's own appearance carried on the cast entry.
    """
    if member.get("op") == "transform":
        desc = str(member.get("state", {}).get("appearance", "")).strip()
    else:
        states = ent.get("states")
        if states:  # stateful entity: reveal its initial-state look on first sight
            desc = str(states.get(ent.get("initial_state"), next(iter(states.values()), ""))).strip()
        else:
            desc = str(ent.get("appearance", "")).strip()
    return desc.rstrip("。.")


def render_segment(seg: dict[str, Any], entities: dict[str, dict[str, Any]],
                   register: str) -> tuple[str, list[str]]:
    """Return (prompt_text, warnings) for one segment under `register`.

    Appearance injection is computed on the ORIGINAL action in a single pass and
    spliced right-to-left, so a name that only appears *inside* another entity's
    injected appearance text is never matched (fixes nested-parenthesis bug).
    Names are matched longest-first to avoid substring collisions
    (e.g. 海燕 vs 海燕号).
    """
    if register != "name_anchored":
        raise NotImplementedError(f"register {register!r} not implemented yet")

    action = str(seg.get("action", "")).strip()
    warnings: list[str] = []
    forbidden = {f["eid"] for f in seg.get("forbidden", [])}

    reveals: list[tuple[str, str, str]] = []  # (name, eid, desc), longest name first
    for member in seg.get("cast", []):
        eid = member["eid"]
        if eid in forbidden:
            warnings.append(f"{seg['segment_id']}: {eid} is both cast and forbidden")
            continue
        if member.get("op") not in _REVEAL_OPS:
            continue
        ent = entities.get(eid)
        if ent is None:
            warnings.append(f"{seg['segment_id']}: unknown entity {eid}")
            continue
        reveals.append((ent["name"], eid, _reveal_desc(member, ent)))
    reveals.sort(key=lambda r: len(r[0]), reverse=True)

    # choose one non-overlapping insertion point per entity, on the original string
    inserts: list[tuple[int, str]] = []  # (position, text)
    taken: list[tuple[int, int]] = []    # occupied [start,end) name spans
    for name, eid, desc in reveals:
        placed = False
        start = 0
        while True:
            idx = action.find(name, start)
            if idx < 0:
                break
            end = idx + len(name)
            if not any(s < end and idx < e for s, e in taken):
                taken.append((idx, end))
                inserts.append((end, f"（{desc}）"))
                placed = True
                break
            start = idx + 1
        if not placed:
            warnings.append(
                f"{seg['segment_id']}: introduce entity {eid}"
                f"（{name}）name not found in action; appearance not revealed")

    for pos, text in sorted(inserts, key=lambda x: x[0], reverse=True):
        action = f"{action[:pos]}{text}{action[pos:]}"
    return action, warnings


def render_story(gt: dict[str, Any], register: str) -> dict[str, Any]:
    entities = gt.get("entities", {})
    segments_out: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for seg in gt.get("segments", []):
        prompt, warns = render_segment(seg, entities, register)
        all_warnings.extend(warns)
        segments_out.append({
            "segment_id": seg["segment_id"],
            "duration_sec": seg.get("duration_sec"),
            "transition": seg.get("transition"),
            "prompt": prompt,
        })
    return {
        "story_id": gt.get("story_id"),
        "title": gt.get("title"),
        "register": register,
        "generated_from": f"gt/{gt.get('story_id')}.json",
        "note": "SUT-facing prompt stream. Read segments in order; no labels are provided by design.",
        "n_segments": len(segments_out),
        "segments": segments_out,
        "warnings": all_warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--story", help="story_id (basename without .json); default: all in gt/")
    ap.add_argument("--register", default="name_anchored", choices=["name_anchored"])
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    gt_dir, out_dir = Path(a.gt_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.story:
        gt_files = [gt_dir / f"{a.story}.json"]
    else:
        gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise SystemExit(f"no GT files found in {gt_dir}")

    for gt_path in gt_files:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        out = render_story(gt, a.register)
        out_path = out_dir / f"{out['story_id']}_{a.register}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tag = f" [{len(out['warnings'])} warning(s)]" if out["warnings"] else ""
        print(f"{gt_path.name} -> {out_path.relative_to(HERE)}  "
              f"({out['n_segments']} segments){tag}")
        for w in out["warnings"]:
            print(f"    ! {w}")


if __name__ == "__main__":
    main()
