"""trackB-identity-1.0: unit building, clip mapping, judge I/O, aggregation, bootstrap determinism."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from vmem_bench.scoring import identity_consistency as ic


def _gold() -> dict:
    ents = {
        "E1": {"name": "Ana", "kind": "character", "states": {"young": "a girl", "old": "an old woman"},
               "initial_state": "young"},
        "E2": {"name": "lamp", "kind": "prop", "appearance": "a brass lamp"},
        "E3": {"name": "hall", "kind": "location", "appearance": "a stone hall"},
    }
    cast = {
        1: [("E1", "introduce", None, None), ("E2", "introduce", None, None), ("E3", "introduce", None, None)],
        2: [("E1", "persist", 0, 1), ("E3", "recall", 0, 1)],
        3: [("E2", "recall", 1, 1)],
        4: [("E1", "recall", 1, 2), ("E2", "recall", 0, 3), ("E3", "recall", 1, 2)],
    }
    segs = []
    for n in range(1, 40):
        c = []
        for eid, op, gap, last in cast.get(n, []):
            e = {"eid": eid, "op": op}
            if gap is not None:
                e.update(gap=gap, last_seen=f"seg_{last:03d}")
            c.append(e)
        segs.append({"segment_id": f"seg_{n:03d}", "cast": c})
    segs[38]["cast"] = [{"eid": "E1", "op": "transform", "gap": 34, "last_seen": "seg_004",
                         "state": {"label": "old", "appearance": "an old woman"}}]
    return {"s1": {"entities": ents, "segments": segs}}


def _cfg(**over) -> dict:
    cfg = ic.load_config()
    cfg.update(over)
    return cfg


def test_units_strata_and_previous_appearance() -> None:
    units, audit = ic.build_units(_gold(), _cfg(gap0_sample_rate=1.0))
    got = {(u["cur_gt"], u["eid"]): (u["stratum"], u["prev_gt"], u["gap_bin"]) for u in units}
    assert got == {(2, "E1"): ("gap0", 1, "0"), (3, "E2"): ("main", 1, "1-4"),
                   (4, "E1"): ("main", 2, "1-4"), (4, "E2"): ("gap0", 3, "0"),
                   (39, "E1"): ("transform_ge30", 4, ">=30")}
    tr = next(u for u in units if u["stratum"] == "transform_ge30")
    assert (tr["state_A"], tr["state_B"], tr["desc_A"], tr["desc_B"]) == ("young", "old", "a girl", "an old woman")
    assert audit == {}
    assert all(u["kind"] != "location" for u in units)
    loc, _ = ic.build_units(_gold(), _cfg(gap0_sample_rate=0.0), kinds=["location"])
    assert [(u["cur_gt"], u["stratum"]) for u in loc] == [(4, "main")]


def test_gap0_sampling_is_seeded() -> None:
    a, _ = ic.build_units(_gold(), _cfg(gap0_sample_rate=0.5), rng=random.Random(7))
    b, _ = ic.build_units(_gold(), _cfg(gap0_sample_rate=0.5), rng=random.Random(7))
    assert ic.jsonl(a) == ic.jsonl(b)
    none, _ = ic.build_units(_gold(), _cfg(gap0_sample_rate=0.0))
    assert all(u["stratum"] != "gap0" for u in none)


def test_controls_positive_same_clip_negative_same_kind() -> None:
    gold = _gold()
    gold["s1"]["entities"]["E4"] = {"name": "Bo", "kind": "character", "appearance": "a boy"}
    units, _ = ic.build_units(gold, _cfg(gap0_sample_rate=1.0))
    presence = {"s1": {2: {"E1"}, 3: {"E2"}, 4: {"E1", "E2"}, 5: {"E4"}}}
    ctl = ic.build_controls(units, gold, presence, _cfg(n_controls={"ctl_pos": 10, "ctl_neg": 10}))
    pos = [c for c in ctl if c["stratum"] == "ctl_pos"]
    neg = [c for c in ctl if c["stratum"] == "ctl_neg"]
    assert pos and all(c["prev_gt"] == c["cur_gt"] and (c["variant_A"], c["variant_B"]) == ("posA", "posB") for c in pos)
    assert {(c["eid"], c["eid_B"]) for c in neg} <= {("E1", "E4"), ("E4", "E1")}
    for c in neg:
        assert c["eid"] not in presence["s1"][c["cur_gt"]]


def test_clip_layout_offsets(tmp_path: Path) -> None:
    (tmp_path / "a" / "s1").mkdir(parents=True)
    (tmp_path / "a" / "s1" / "seg_000.mp4").write_bytes(b"x")
    conds = {"_doc": "x",
             "fixed": {"clip_template": "{root}/b/{story}/seg_{idx:03d}.mp4", "file_index_offset": -1},
             "auto": {"clip_template": "{root}/a/{story}/seg_{idx:03d}.mp4", "file_index_offset": "auto"},
             "auto0": {"clip_template": "{root}/c/{story}/seg_{idx:03d}.mp4", "file_index_offset": "auto"}}
    lay = ic.ClipLayout(conds, {"root": str(tmp_path)})
    assert lay.clip("fixed", "s1", 5) == tmp_path / "b/s1/seg_004.mp4"
    assert lay.clip("auto", "s1", 5) == tmp_path / "a/s1/seg_004.mp4"
    assert lay.clip("auto0", "s1", 5) == tmp_path / "c/s1/seg_005.mp4"
    assert "_doc" not in lay.conditions


def test_parse_and_prompt() -> None:
    assert ic.parse_answer('x {"visible_A": "yes", "visible_B": false, "same": null} y') == \
        {"visible_A": True, "visible_B": False, "same": None}
    assert ic.parse_answer('{"visible_A": "maybe", "visible_B": true, "same": "UNSURE"}')["visible_A"] == "unsure"
    assert ic.parse_answer("no json") is None
    u = {"name": "Ana", "kind": "character", "desc_A": "a girl", "desc_B": "an old woman"}
    assert ic.STATE_NOTE in ic.build_prompt(u)
    assert ic.STATE_NOTE not in ic.build_prompt({**u, "desc_B": "a girl"})


def _write_judgments(out: Path, cond: str, units: list[dict], answer_fn) -> None:
    for u in units:
        t = {"cond": cond, "unit": u}
        p = ic.result_path(out, t)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"task": {"uid": u["uid"]}, "answer": answer_fn(u), "parse_error": False}))


def _units(n_stories: int = 6) -> list[dict]:
    return [{"uid": f"s{i}|{k:03d}|E1", "story": f"s{i}", "stratum": "main", "kind": "character",
             "gap_bin": "1-4", "eid": "E1"} for i in range(n_stories) for k in range(4)]


def test_aggregate_counts_bootstrap_and_determinism(tmp_path: Path) -> None:
    units = _units()
    yes = {"visible_A": True, "visible_B": True, "same": True}
    _write_judgments(tmp_path, "no_memory", units, lambda u: {"visible_A": True, "visible_B": False, "same": None})
    _write_judgments(tmp_path, "mem", units, lambda u: yes if u["uid"].endswith(("000|E1", "001|E1")) else None)
    for name in ("ctl_pos", "ctl_neg"):
        _write_judgments(tmp_path, name, units[:2], lambda u: yes)
    cfg = _cfg(bootstrap={"n_boot": 200, "seed": 5})
    stories = sorted({u["story"] for u in units})
    r1 = ic.aggregate(tmp_path, units, ["no_memory", "mem"], stories, cfg)
    r2 = ic.aggregate(tmp_path, units, ["no_memory", "mem"], stories, cfg)
    assert r1 == r2
    cell = r1["cells"]["main|all"]
    assert cell["no_memory"]["consistent_rate"] == 0.0
    assert cell["mem"]["counts"]["parse"] == 12 and cell["mem"]["consistent_rate"] == 0.5
    d = cell["mem"]["diff_vs_no_memory"]["consistent_rate"]
    assert d["diff"] == 0.5 and d["lo"] == d["hi"] == 0.5  # every story has the same 2/4 split
    assert r1["controls"]["summary"]["pos_same_given_both"] == 1.0


def test_bootstrap_draws_are_paired_and_seeded() -> None:
    a = ic.bootstrap_draws(["b", "a", "c"], 10, 1)
    assert a == ic.bootstrap_draws(["c", "a", "b"], 10, 1)
    assert a != ic.bootstrap_draws(["a", "b", "c"], 10, 2)
    assert ic.percentile_ci(list(range(100)), 100) == (2, 96)


def test_run_judge_is_resume_safe(tmp_path: Path) -> None:
    units = _units(1)
    tasks = ic.judge_tasks(units, [], ["c1"], "c1")
    calls = []

    def fake(ep, model, t, out, jcfg):
        calls.append(t["unit"]["uid"])
        return {"task": {"uid": t["unit"]["uid"]}, "answer": ic.parse_answer('{"visible_A":1,"visible_B":1,"same":1}'),
                "parse_error": False}

    s1 = ic.run_judge(tasks, tmp_path, ["http://x/v1"], {"model": "m"}, per_endpoint=2, call=fake)
    s2 = ic.run_judge(tasks, tmp_path, ["http://x/v1"], {"model": "m"}, per_endpoint=2, call=fake)
    assert (s1["done"], s2["todo"], len(calls)) == (4, 0, 4)


def test_default_config_is_versioned() -> None:
    cfg = ic.load_config()
    assert cfg["metric_version"] == ic.METRIC_VERSION == "trackB-identity-1.0"
    assert cfg["bootstrap"]["seed"] == 20260925 and cfg["judge"]["temperature"] == 0.0


@pytest.mark.parametrize("gap,b", [(0, "0"), (1, "1-4"), (4, "1-4"), (5, "5-29"), (29, "5-29"), (30, ">=30")])
def test_gap_bins(gap: int, b: str) -> None:
    assert ic.gap_bin(gap) == b
