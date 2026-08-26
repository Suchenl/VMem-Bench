#!/usr/bin/env python3
"""Aggregate the formal two-movie x name_anchored Track A run into one table.

Scans each movie's benchmark_run/_visual_score/<runname>/score.json and emits
results.md (+ results.json) with one row per (movie, system, mode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH = Path(".")
MOVIES = [
    ("big_buck_bunny", BENCH / "assets/trackA/BlenderOpenMovies/big_buck_bunny"),
    ("0022_Reservoir_Dogs", BENCH / "assets/trackA/LSMDC/0022_Reservoir_Dogs"),
]
SYSTEMS = [
    "memstrata",
    "longlive_rag",
    "memflow",
    "memflow_sma",
    "iamflow",
    "retrieval_frame_text_ablation",
    "retrieval_seg_uniform_ablation",
    "retrieval_seg_dinokey_ablation",
    "retrieval_seg_framererank_ablation",
]
MODES = [("name_anchored", "")]


def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for movie_name, movie_dir in MOVIES:
        for sys in SYSTEMS:
            for mode, suffix in MODES:
                runname = sys + suffix
                sp = (
                    BENCH / "outputs/evaluation/trackA"
                    / runname / movie_dir.parent.name / movie_dir.name
                    / "_visual_score" / runname / "score.json"
                )
                if not sp.is_file():
                    sp = movie_dir / "benchmark_run/_visual_score" / runname / "score.json"
                if not sp.is_file():
                    rows.append({"movie": movie_name, "system": sys, "mode": mode,
                                 "status": "MISSING"})
                    continue
                try:
                    raw = json.loads(sp.read_text(encoding="utf-8"))
                    s = raw.get("summary", raw)  # score.json nests metrics under "summary"
                except Exception as e:  # noqa: BLE001
                    rows.append({"movie": movie_name, "system": sys, "mode": mode,
                                 "status": f"BAD_JSON:{e}"})
                    continue
                # Headline = duration-weighted (*_wmean, chunks are NOT equal length);
                # fall back to equal-weight *_mean for pre-weighting runs.
                def pick(w, m):
                    v = s.get(w)
                    return v if v is not None else s.get(m)
                rows.append({
                    "movie": movie_name, "system": sys, "mode": mode, "status": "ok",
                    "n_chunks": s.get("n_chunks"),
                    "n_with_refs": s.get("n_chunks_with_refs"),
                    "dur_s": s.get("total_duration_s"),
                    "precision": pick("precision_wmean", "precision_mean"),
                    "recall": pick("recall_wmean", "recall_mean"),
                    "f1": pick("f1_wmean", "f1_mean"),
                    "redun_vlm": pick("redundancy_vlm_wmean", "redundancy_vlm_mean"),
                    "redun_sim": pick("redundancy_sim_wmean", "redundancy_sim_mean"),
                    "sel_eff": pick("selection_efficiency_wmean",
                                    s.get("selection_efficiency_mean", "efficiency_mean")),
                    "budget": s.get("budget_avg_refs_per_chunk_with_refs"),
                    "retr_ms": s.get("retrieval_ms_mean"),
                    "score_ms": s.get("score_ms_mean"),
                    # keep equal-weight macro alongside for robustness reference
                    "precision_macro": s.get("precision_mean"),
                    "recall_macro": s.get("recall_mean"),
                    "f1_macro": s.get("f1_mean"),
                })

    # ---- corpus rows per (system, mode): duration-weighted across movies ----------
    corpus = []
    for sys in SYSTEMS:
        for mode, _ in MODES:
            grp = [r for r in rows if r["system"] == sys and r["mode"] == mode
                   and r.get("status") == "ok" and r.get("dur_s")]
            if not grp:
                continue
            W = sum(r["dur_s"] for r in grp)
            def wavg(k):
                vs = [(r.get(k), r["dur_s"]) for r in grp if r.get(k) is not None]
                return round(sum(v * w for v, w in vs) / sum(w for _, w in vs), 4) if vs else None
            corpus.append({
                "movie": "CORPUS(dur-w)", "system": sys, "mode": mode, "status": "ok",
                "n_chunks": sum(r.get("n_chunks") or 0 for r in grp),
                "dur_s": round(W, 1),
                "precision": wavg("precision"), "recall": wavg("recall"), "f1": wavg("f1"),
                "redun_vlm": wavg("redun_vlm"), "redun_sim": wavg("redun_sim"),
                "sel_eff": wavg("sel_eff"), "budget": wavg("budget"),
                "retr_ms": wavg("retr_ms"), "score_ms": wavg("score_ms"),
            })

    (args.out / "results.json").write_text(
        json.dumps({"per_movie": rows, "corpus_dur_weighted": corpus},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ["movie", "system", "mode", "n_chunks", "n_with_refs", "dur_s", "precision",
           "recall", "f1", "redun_vlm", "redun_sim", "sel_eff", "budget",
           "retr_ms", "score_ms", "status"]
    lines = ["# Overnight two-movie benchmark results", "",
             "systems: MemStrata + 4 external baselines + 4 retrieval baselines",
             "modes: name_anchored only | judge: qwen3-vl-32b | 480p",
             "**metrics are DURATION-WEIGHTED** (segments are 3-27s, not equal length); "
             "macro (equal-segment) kept in results.json.",
             "", "| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k)) for k in hdr) + " |")
    if corpus:
        lines += ["", "## Corpus (duration-weighted across movies)", "",
                  "| " + " | ".join(hdr) + " |",
                  "|" + "|".join(["---"] * len(hdr)) + "|"]
        for r in corpus:
            lines.append("| " + " | ".join(_fmt(r.get(k)) for k in hdr) + " |")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    lines += ["", f"completed: {ok}/{len(rows)} combos scored."]
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
