#!/usr/bin/env python3
"""Supervisor gate for the English Track B gt_source translation.

Compares en/gt_source/<story>.json against zh/gt_source/<story>.json and refuses
anything that would break the deterministic build (complete_gt -> get_sut_prompts)
or drift from the frozen Chinese structure.

Checks (all must pass):
  1. JSON parses.
  2. Structural skeleton identical to zh: same eids, same kinds, same scene ids
     and order, same action-item shape, same numeric params, same enum/identifier
     leaves (kind / eid / count / confusable_with / type / reason / shown /
     resolves_to / pair / anchor.type).
  3. State-name referential integrity WITHIN en (state names may be remapped to
     English, but must be remapped CONSISTENTLY):
       set(entities[e].states.keys()) == set(state_machines[e])   (order too)
       every events[].to      in that entity's states
       every present[].state  in that entity's states
  4. Full English: ZERO CJK characters anywhere in the file.
  5. Every translatable field is non-empty.
  6. Deterministic build succeeds: complete_gt + get_sut_prompts run without
     exception; no "name not found in action" warning (else the SUT prompt would
     silently lose an appearance reveal -> the English name must appear verbatim
     in its introduce/transform action).

Exit code 0 = PASS, 1 = FAIL (errors printed).

Usage:
    python validate_en_source.py --story 0003_desert_archaeologist
    python validate_en_source.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve().parent  # scripts/ (where the build scripts live)
HERE = SELF.parent  # trackB root (holds zh/ en/ and is the build scripts' data anchor)
ZH_DIR = HERE / "zh" / "gt_source"
EN_DIR = HERE / "en" / "gt_source"
# Broad CJK (incl. fullwidth punctuation) for source translatable-field checks.
CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
# Han ideographs only, for the BUILT SUT prompt (get_sut_prompts injects fullwidth
# （）around appearances; those are punctuation, not untranslated Chinese text).
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# leaf field names whose VALUE is an identifier/number/enum -> must equal zh
PROTECTED = {
    "story_id", "kind", "eid", "count", "confusable_with", "type", "reason",
    "shown", "resolves_to", "segment_sec", "gap_long_threshold",
    "avoidance_probe_window", "probe_target_default", "id",
}


def has_cjk(s) -> bool:
    return bool(CJK.search(str(s)))


class V:
    def __init__(self):
        self.errs: list[str] = []
        self.warns: list[str] = []

    def err(self, msg: str):
        self.errs.append(msg)

    def eq(self, a, b, where: str):
        if a != b:
            self.err(f"{where}: identifier changed {b!r} -> {a!r} (must stay equal to zh)")

    def translatable(self, val, where: str):
        if not isinstance(val, str) or not val.strip():
            self.err(f"{where}: empty/invalid translatable text")
        elif has_cjk(val):
            self.err(f"{where}: still contains CJK -> not translated")


def check_entities(zh, en, v: V):
    ze, ee = zh.get("entities", {}), en.get("entities", {})
    if set(ze) != set(ee):
        v.err(f"entities: eid set changed {sorted(set(ze)^set(ee))}")
        return
    for eid in ze:
        z, e = ze[eid], ee[eid]
        v.eq(e.get("kind"), z.get("kind"), f"entities.{eid}.kind")
        v.translatable(e.get("name", ""), f"entities.{eid}.name")
        if "states" in z:
            if "states" not in e or not isinstance(e["states"], dict):
                v.err(f"entities.{eid}.states: missing/invalid")
                continue
            if len(e["states"]) != len(z["states"]):
                v.err(f"entities.{eid}.states: key count {len(e['states'])} != zh {len(z['states'])}")
            for k, val in e["states"].items():
                if has_cjk(k):
                    v.err(f"entities.{eid}.states key {k!r} still CJK")
                v.translatable(val, f"entities.{eid}.states[{k}]")
        if "appearance" in z:
            v.translatable(e.get("appearance", ""), f"entities.{eid}.appearance")


def check_state_machines(zh, en, v: V):
    zsm, esm = zh.get("state_machines", {}), en.get("state_machines", {})
    if set(zsm) != set(esm):
        v.err(f"state_machines: eid set changed {sorted(set(zsm)^set(esm))}")
        return
    ee = en.get("entities", {})
    for eid, zlist in zsm.items():
        elist = esm[eid]
        if len(elist) != len(zlist):
            v.err(f"state_machines.{eid}: len {len(elist)} != zh {len(zlist)}")
        skeys = list(ee.get(eid, {}).get("states", {}).keys())
        if elist != skeys:
            v.err(f"state_machines.{eid}: {elist} != entity states order {skeys} "
                  f"(state names must be remapped consistently)")


def _entity_states(en, eid) -> set:
    return set(en.get("entities", {}).get(eid, {}).get("states", {}).keys())


def check_lookalikes(zh, en, v: V):
    zl, el = zh.get("lookalike_pairs", []), en.get("lookalike_pairs", [])
    if len(zl) != len(el):
        v.err(f"lookalike_pairs: count {len(el)} != zh {len(zl)}")
        return
    for i, (z, e) in enumerate(zip(zl, el)):
        v.eq(e.get("pair"), z.get("pair"), f"lookalike_pairs[{i}].pair")
        zf, ef = z.get("features", {}), e.get("features", {})
        if set(ef) != set(zf):
            v.err(f"lookalike_pairs[{i}].features: eid keys changed")
        for k, val in ef.items():
            v.translatable(val, f"lookalike_pairs[{i}].features[{k}]")
        if "note" in z:
            v.translatable(e.get("note", ""), f"lookalike_pairs[{i}].note")


def check_present(zpres, epres, eid_states, where, v: V):
    if not isinstance(zpres, list) or not isinstance(epres, list) or len(zpres) != len(epres):
        v.err(f"{where}: present shape mismatch")
        return
    for i, (z, e) in enumerate(zip(zpres, epres)):
        w = f"{where}[{i}]"
        if isinstance(z, str):
            v.eq(e, z, f"{w} eid")
        elif isinstance(z, dict):
            if not isinstance(e, dict):
                v.err(f"{w}: expected dict")
                continue
            eid = z.get("eid")
            v.eq(e.get("eid"), eid, f"{w}.eid")
            for pk in ("count", "confusable_with"):
                if pk in z:
                    v.eq(e.get(pk), z.get(pk), f"{w}.{pk}")
            if "state" in z:
                st = e.get("state")
                if st not in eid_states.get(eid, set()):
                    v.err(f"{w}.state {st!r} not in entity {eid} states")
            if "anchor" in z:
                za, ea = z["anchor"], e.get("anchor", {})
                v.eq(ea.get("type"), za.get("type"), f"{w}.anchor.type")
                v.eq(ea.get("resolves_to"), za.get("resolves_to"), f"{w}.anchor.resolves_to")
                v.translatable(ea.get("phrase", ""), f"{w}.anchor.phrase")


def check_scenes(zh, en, v: V):
    zs, es = zh.get("scenes", []), en.get("scenes", [])
    if len(zs) != len(es):
        v.err(f"scenes: count {len(es)} != zh {len(zs)}")
        return
    eid_states = {eid: _entity_states(en, eid) for eid in en.get("entities", {})}
    for si, (z, e) in enumerate(zip(zs, es)):
        w = f"scenes[{si}]"
        v.eq(e.get("id"), z.get("id"), f"{w}.id")
        v.translatable(e.get("setting", ""), f"{w}.setting")
        if "present" in z:
            check_present(z["present"], e.get("present", []), eid_states, f"{w}.present", v)
        za, ea = z.get("actions", []), e.get("actions", [])
        if len(za) != len(ea):
            v.err(f"{w}.actions: count {len(ea)} != zh {len(za)}")
            continue
        for ai, (zi, ei) in enumerate(zip(za, ea)):
            aw = f"{w}.actions[{ai}]"
            if isinstance(zi, str):
                v.translatable(ei if isinstance(ei, str) else "", f"{aw}")
            elif isinstance(zi, dict):
                if not isinstance(ei, dict):
                    v.err(f"{aw}: expected dict")
                    continue
                v.translatable(ei.get("action", ""), f"{aw}.action")
                if "present" in zi:
                    check_present(zi["present"], ei.get("present", []), eid_states, f"{aw}.present", v)
                for evi, ev in enumerate(zi.get("events", [])):
                    eev = ei.get("events", [])[evi] if evi < len(ei.get("events", [])) else {}
                    v.eq(eev.get("type"), ev.get("type"), f"{aw}.events[{evi}].type")
                    v.eq(eev.get("eid"), ev.get("eid"), f"{aw}.events[{evi}].eid")
                    if "to" in ev:
                        to = eev.get("to")
                        if to not in eid_states.get(ev.get("eid"), set()):
                            v.err(f"{aw}.events[{evi}].to {to!r} not in entity {ev.get('eid')} states")
                    for pk in ("reason", "shown"):
                        if pk in ev:
                            v.eq(eev.get(pk), ev.get(pk), f"{aw}.events[{evi}].{pk}")


def check_build(story: str, v: V):
    # temp dir MUST live under HERE: the build scripts print out_path.relative_to(HERE).
    with tempfile.TemporaryDirectory(dir=HERE, prefix="_valtmp_") as td:
        td = Path(td)
        gt_out, sut_out = td / "gt", td / "sut"
        r1 = subprocess.run(
            [sys.executable, str(SELF / "complete_gt.py"), "--story", story,
             "--src-dir", str(EN_DIR), "--out-dir", str(gt_out)],
            capture_output=True, text=True)
        if r1.returncode != 0:
            v.err(f"complete_gt failed: {r1.stderr.strip()[:400]}")
            return
        r2 = subprocess.run(
            [sys.executable, str(SELF / "get_sut_prompts.py"), "--story", story,
             "--gt-dir", str(gt_out), "--out-dir", str(sut_out)],
            capture_output=True, text=True)
        if r2.returncode != 0:
            v.err(f"get_sut_prompts failed: {r2.stderr.strip()[:400]}")
            return
        for line in (r1.stdout + r2.stdout).splitlines():
            if "not found in action" in line:
                v.err(f"BUILD WARN (name not in action): {line.strip()}")
        # confirm the built SUT prompts are English + CJK-free
        sf = sut_out / f"{story}_name_anchored.json"
        if sf.exists():
            d = json.loads(sf.read_text(encoding="utf-8"))
            for seg in d.get("segments", []):
                if HAN.search(str(seg.get("prompt", ""))):
                    v.err(f"built SUT prompt {seg['segment_id']} still has Han CJK")
                    break


def validate(story: str) -> V:
    v = V()
    zp, ep = ZH_DIR / f"{story}.json", EN_DIR / f"{story}.json"
    if not ep.exists():
        v.err(f"missing {ep}")
        return v
    try:
        zh = json.loads(zp.read_text(encoding="utf-8"))
        en = json.loads(ep.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        v.err(f"JSON parse error: {exc}")
        return v
    v.eq(en.get("story_id"), zh.get("story_id"), "story_id")
    for p in ("segment_sec", "gap_long_threshold", "avoidance_probe_window", "probe_target_default"):
        v.eq(en.get(p), zh.get(p), p)
    v.translatable(en.get("title", ""), "title")
    if "premise" in zh:
        v.translatable(en.get("premise", ""), "premise")
    check_entities(zh, en, v)
    check_state_machines(zh, en, v)
    check_lookalikes(zh, en, v)
    check_scenes(zh, en, v)
    # global full-English guarantee (catches anything the targeted checks missed,
    # e.g. _comment or stray keys). state keys are covered above.
    leftover = sorted({m.group(0) for m in CJK.finditer(json.dumps(en, ensure_ascii=False))})
    if leftover:
        v.err(f"file still contains CJK chars: {''.join(leftover)[:60]}")
    if not v.errs:
        check_build(story, v)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--story")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        stories = [p.stem for p in sorted(ZH_DIR.glob("*.json"))]
    elif a.story:
        stories = [a.story]
    else:
        raise SystemExit("need --story or --all")
    n_fail = 0
    for s in stories:
        v = validate(s)
        if v.errs:
            n_fail += 1
            print(f"FAIL {s}  ({len(v.errs)} error(s))")
            for e in v.errs[:25]:
                print(f"   - {e}")
        else:
            print(f"PASS {s}")
    print(f"\n{len(stories)-n_fail}/{len(stories)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
