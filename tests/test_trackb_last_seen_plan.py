"""Track-B last-seen reference plan: causal, id-only, fixed ordering."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.testbed.controls import build_last_seen_plan

ROOT = Path(__file__).resolve().parents[1]


def _gold() -> dict:
    return {
        "entities": {
            "E1": {"name": "Elias", "kind": "character"},
            "E2": {"name": "Mara", "kind": "character"},
            "E3": {"name": "lens", "kind": "prop"},
            "E4": {"name": "lantern room", "kind": "location"},
        },
        "segments": [
            {"segment_id": "seg_001", "cast": [{"eid": "E1", "op": "introduce"}, {"eid": "E4", "op": "introduce"}]},
            {"segment_id": "seg_002", "cast": [{"eid": "E2", "op": "introduce"}, {"eid": "E3", "op": "introduce"}]},
            {"segment_id": "seg_003", "cast": [
                {"eid": "E3", "op": "persist", "gap": 0, "last_seen": "seg_002"},
                {"eid": "E1", "op": "recall", "gap": 1, "last_seen": "seg_001"},
                {"eid": "E2", "op": "recall", "gap": 0, "last_seen": "seg_002"},
                {"eid": "E4", "op": "recall", "gap": 1, "last_seen": "seg_001"},
            ]},
        ],
    }


def test_plan_orders_by_gap_then_eid_and_skips_locations() -> None:
    plan, audit = build_last_seen_plan(_gold(), ["seg_001", "seg_002", "seg_003"])
    assert plan["plan"] == {"0": [], "1": [], "2": [0, 1, 1]}
    assert [r["eid"] for r in audit["entities"]["2"]] == ["E1", "E2", "E3"]


def test_real_gold_plans_are_causal_and_id_only() -> None:
    gt = ROOT / "assets/trackB/en/gt"
    prompts = ROOT / "assets/trackB/en/sut_prompts"
    n_stories = 0
    for i in range(1, 31):
        path = next(gt.glob(f"{i:04d}_*.json"))
        gold = json.loads(path.read_text(encoding="utf-8"))
        stream = json.loads((prompts / f"{path.stem}_name_anchored.json").read_text(encoding="utf-8"))
        ids = [s["segment_id"] for s in stream["segments"]]
        plan, _ = build_last_seen_plan(gold, ids)
        assert set(plan) == {"rule", "plan"}
        assert sorted(int(k) for k in plan["plan"]) == list(range(len(ids)))
        for shot, sources in plan["plan"].items():
            assert all(type(s) is int and 0 <= s < int(shot) for s in sources)
        blob = json.dumps(plan["plan"])
        for ent in gold["entities"].values():
            assert ent["name"] not in blob
        n_stories += 1
    assert n_stories == 30
