#!/usr/bin/env python3
"""Paper-ready Track-B identity tables from the scorer outputs (deterministic, no recomputation).

    python3 scripts/trackb_identity_tables.py --vlm <identity aggregate json> --embed <embed aggregate json>

Reads the JSON written by ``vmem_bench.scoring.identity_consistency aggregate`` (trackB-identity-1.0)
and ``vmem_bench.scoring.embedding_consistency aggregate`` (trackB-embed-1.0) and writes compact
tables/experiments/trackb_identity_vlm.json and trackb_identity_embed.json: point estimates per
condition and cell, differences vs no memory with 95% story-clustered paired bootstrap CIs, strata
sizes, control calibration, and provenance (VMem-Bench commit, models, thresholds).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[3]
OUT_DIR = PAPER / "tables" / "experiments"
CONDS = ["no_memory", "text_frame", "seg_uniform", "seg_dinokey", "seg_frame_clean", "retr_oracle"]
LABELS = {"no_memory": "no memory", "text_frame": "text-frame", "seg_uniform": "seg-uniform",
          "seg_dinokey": "seg-DINO-key", "seg_frame_clean": "seg-frame", "retr_oracle": "last-seen oracle"}
# testbed-rerun cell name -> paper condition key
RERUN = {"none_b0__s2026": "no_memory", "frame_text_b3__s2026": "text_frame", "seg_uniform_b3__s2026": "seg_uniform",
         "seg_dinokey_b3__s2026": "seg_dinokey", "seg_framererank_b3__s2026": "seg_frame_clean",
         "last_seen_b3__s2026": "retr_oracle"}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _compact(cells: dict, metrics: list[str], extra: list[str]) -> dict:
    out = {}
    for cell, row in sorted(cells.items()):
        row = {RERUN.get(c, c): r for c, r in row.items()}
        out[cell] = {"n": row["no_memory"]["counts"]["n"], "conditions": {}}
        for c in CONDS:
            r = row[c]
            e = {m: r.get(m) for m in metrics + extra}
            d = next((v for k, v in r.items() if k.startswith("diff_vs_")), None)
            if d:
                e["diff_vs_no_memory"] = {m: [d[m]["diff"], d[m]["lo"], d[m]["hi"]] for m in metrics if m in d}
            out[cell]["conditions"][c] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vlm", required=True, type=Path)
    ap.add_argument("--embed", required=True, type=Path)
    ap.add_argument("--retest-from", type=Path, help="earlier trackb_identity_vlm.json whose judge test-retest to "
                    "reuse when the --vlm run has no repeat judgments (the retest measures the judge, not a condition)")
    a = ap.parse_args()
    commit = _commit()
    v = json.loads(a.vlm.read_text())
    if not v["test_retest"]:
        if not a.retest_from:
            ap.error("--vlm has no test-retest; pass --retest-from")
        v["test_retest"] = json.loads(a.retest_from.read_text())["test_retest"]
    vlm = {
        "metric_version": v["metric_version"], "vmem_bench_commit": commit, "labels": LABELS,
        "judge": v["judge"], "bootstrap": v["bootstrap"],
        "units": {k: v["units_manifest"][k] for k in ("n_units", "strata", "n_controls", "units_sha256", "controls_sha256")},
        "controls": v["controls"], "test_retest": v["test_retest"],
        "diff_format": "[diff, ci_lo, ci_hi]",
        "cells": _compact(v["cells"], ["consistent_rate", "identity_given_both", "visible_B_rate"],
                          ["visible_A_rate", "both_visible_rate", "unsure_visible_B_rate"]),
    }
    e = json.loads(a.embed.read_text())
    cfg = e["config"]
    emb = {
        "metric_version": e["metric_version"], "vmem_bench_commit": commit, "labels": LABELS,
        "models": {"grounding": cfg["grounding"], "appearance": cfg["appearance"], "face": cfg["face"],
                   "location": cfg["location"]},
        "similarity": cfg["similarity"], "thresholds": cfg["thresholds"], "threshold_rule": cfg["calibration"],
        "bootstrap": cfg["bootstrap"],
        "units": {k: e["units_manifest"][k] for k in ("n_units", "strata", "n_controls", "units_sha256", "controls_sha256")},
        "calibration": e["calibration"], "diff_format": "[diff, ci_lo, ci_hi]",
        "modalities": {m: {"threshold": r["threshold"],
                           "cells": _compact(r["cells"], ["consistent_rate", "mean_similarity", "consistent_given_both"],
                                             ["detected_A_rate", "detected_B_rate", "both_detected_rate"])}
                       for m, r in e["modalities"].items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in (("trackb_identity_vlm.json", vlm), ("trackb_identity_embed.json", emb)):
        (OUT_DIR / name).write_text(json.dumps(obj, indent=1) + "\n")
        print(f"wrote {OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
