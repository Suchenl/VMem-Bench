#!/usr/bin/env python3
"""Paired, story-clustered bootstrap CIs for 'retrieval memory minus no memory' on Track-B.

    python3 scripts/memory_gain_bootstrap.py

Reads per-segment tallies (score.json, per_segment.acc) of the shared-generator testbed run through the local
results/ index of the VMem-Bench repository (results/trackB/testbed_rerun_20260927: seed 2026, 30 stories,
six cells), pools counts per story, resamples the 30 stories with replacement (same draw for every condition,
so differences are paired), and reports point estimates, 95% CIs of each difference, and per-condition
parse-error counts. Writes tables/experiments/memory_gain_bootstrap.json (retrieval minus no memory) and
tables/experiments/last_seen_bootstrap.json (last-seen control; key "oracle"). Deterministic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[3]
TB = REPO / "results" / "trackB"
RUN = TB / "testbed_rerun_20260927"
T = PAPER / "tables" / "experiments"
OUT = T / "memory_gain_bootstrap.json"
OUT_LAST_SEEN = T / "last_seen_bootstrap.json"
N_BOOT, SEED = 2000, 20260925


def _cell(cell: str):
    return lambda s: RUN / "scores" / cell / s / "score.json"


CONDS = {
    "retr_frame_text": _cell("frame_text_b3__s2026"),
    "retr_seg_uniform": _cell("seg_uniform_b3__s2026"),
    "retr_seg_dinokey": _cell("seg_dinokey_b3__s2026"),
    "retr_seg_framererank": _cell("seg_framererank_b3__s2026"),
    "no_memory": _cell("none_b0__s2026"),
}
LAST_SEEN = _cell("last_seen_b3__s2026")


def progress(cell: str, story: str) -> dict:
    """Generation log of one story: per segment its gold id and the references it was given."""
    d = RUN / "gen" / cell.split("__")[0] / story / "name_anchored" / RUN.name / "seed_2026" / "progress.json"
    return json.loads(d.read_text())


def stories() -> list[str]:
    return sorted(p.name for p in (RUN / "scores" / "none_b0__s2026").iterdir() if p.is_dir() and p.name[:4].isdigit())


# metric -> (numerator keys, denominator keys, kind); kind "avoid" means 1 - viol/opp
METRICS = {
    "recall": (["recall.character.hit", "recall.prop.hit"], ["recall.character.tot", "recall.prop.tot"], "rate"),
    "precision": (["prec.hit"], ["prec.tot"], "rate"),
    "avoid_ok": (["avoid.viol"], ["avoid.opp"], "avoid"),
    "gap_1_4": (["gap.1-4.hit"], ["gap.1-4.tot"], "rate"),
    "gap_5_29": (["gap.5-29.hit"], ["gap.5-29.tot"], "rate"),
    "gap_ge30": (["gap.>=30.hit"], ["gap.>=30.tot"], "rate"),
    "persist": (["cap.persist_state.ok"], ["cap.persist_state.tot"], "rate"),
    "count": (["cap.count.exact"], ["cap.count.tot"], "rate"),
    "lookalike": (["cap.lookalike.correct"], ["cap.lookalike.tot"], "rate"),
    "temporal": (["cap.temporal.hit"], ["cap.temporal.tot"], "rate"),
    "false_friend": (["cap.false_friend.correct_id"], ["cap.false_friend.tot_id"], "rate"),
    "state_change": (["cap.state_change.ok"], ["cap.state_change.tot"], "rate"),
    "deprecation": (["cap.deprecation.viol"], ["cap.deprecation.opp"], "avoid"),
    "indirect": (["cap.reference_indirect.viol"], ["cap.reference_indirect.opp"], "avoid"),
}


def load(path, stories: list[str]) -> tuple[dict, int]:
    per_story, n_parse = {}, 0
    for s in stories:
        data = json.loads(path(s).read_text())
        tot: dict[str, int] = {}
        for seg in data["per_segment"]:
            n_parse += bool(seg.get("parse_error"))
            for k, v in (seg.get("acc") or {}).items():
                tot[k] = tot.get(k, 0) + int(v)
        per_story[s] = tot
    return per_story, n_parse


def value(tallies: list[dict], metric: str) -> float | None:
    nk, dk, kind = METRICS[metric]
    num = sum(t.get(k, 0) for t in tallies for k in nk)
    den = sum(t.get(k, 0) for t in tallies for k in dk)
    if not den:
        return None
    r = num / den
    return 1 - r if kind == "avoid" else r


def ci(data: dict, a: str, b: str, metric: str, stories: list[str], draws: list[list[str]]) -> dict:
    base = value([data[a][s] for s in stories], metric) - value([data[b][s] for s in stories], metric)
    ds = sorted(value([data[a][s] for s in d], metric) - value([data[b][s] for s in d], metric) for d in draws)
    return {"diff": round(base, 4), "lo": round(ds[int(0.025 * N_BOOT)], 4), "hi": round(ds[int(0.975 * N_BOOT) - 1], 4)}


def main() -> int:
    ss = stories()
    conds = {**CONDS, "oracle": LAST_SEEN}
    data, parse = {}, {}
    for c, path in conds.items():
        data[c], parse[c] = load(path, ss)
    rng = random.Random(SEED)
    draws = [[rng.choice(ss) for _ in ss] for _ in range(N_BOOT)]
    # 6 decimals so that tables rounding to 3 do not double-round
    point = {c: {m: round(value([data[c][s] for s in ss], m), 6) for m in METRICS} for c in conds}
    retr = [c for c in CONDS if c.startswith("retr_")]
    head = {"n_stories": len(ss), "n_boot": N_BOOT, "seed": SEED}
    out = {**head, "parse_errors": {c: parse[c] for c in CONDS}, "point": {c: point[c] for c in CONDS},
           "diff_vs_no_memory": {c: {m: ci(data, c, "no_memory", m, ss, draws) for m in METRICS} for c in retr}}
    last = {**head, "parse_errors": parse, "point": point,
            "oracle_minus_no_memory": {m: ci(data, "oracle", "no_memory", m, ss, draws) for m in METRICS},
            "retrieval_minus_oracle": {c: {m: ci(data, c, "oracle", m, ss, draws) for m in METRICS} for c in retr}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    OUT_LAST_SEEN.write_text(json.dumps(last, indent=1) + "\n")
    print(f"wrote {OUT} and {OUT_LAST_SEEN.name}")
    print("parse errors:", parse)
    for m in METRICS:
        row = " ".join(f"{d['diff']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]"
                       for d in (out["diff_vs_no_memory"][c][m] for c in retr))
        o = last["oracle_minus_no_memory"][m]
        print(f"{m:13s} nomem={point['no_memory'][m]:.3f} | {row} | last-seen {o['diff']:+.3f}[{o['lo']:+.3f},{o['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
