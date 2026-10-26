#!/usr/bin/env python3
"""Write tables/experiments/trackb_second_pipeline.tex from tables/experiments/h3/.

    python3 scripts/make_second_pipeline_table.py

The six Track-B conditions on 8 stories (seed 2026) with the paper's FLUX.2 -> Wan2.2 realization and two
MiniMax-H3 realizations. Inputs are copies of Montage experiments/benchmarks/vmem-bench/trackb_h3_ac/results/
(<P>/analysis.json as h3/analysis_<P>.json; near_dup_per_story.txt summed per pipeline and condition as
h3/near_dup.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
PIPES = [("reference_flux2_wan22", "FLUX.2\\,$\\rightarrow$\\,Wan2.2 (main pipeline)"),
         ("C", "FLUX.2\\,$\\rightarrow$\\,H3 FL2VA"), ("A", "H3 Ref2VA, subject crops (no keyframe)")]
CONDS = [("frame_text", "Text-frame"), ("seg_uniform", "Seg+uniform"), ("seg_dinokey", "Seg+DINO-key"),
         ("seg_framererank", "Seg+frame"), ("last_seen", "\\textit{Last-seen refs}")]
METRICS = ["recall", "precision", "avoid_ok"]


def cell(r: dict) -> str:
    return ci_cell(f"{r['diff']:+.3f}", f"{r['lo']:+.2f}", f"{r['hi']:+.2f}",
                   bold=r['lo'] > 0 or r['hi'] < 0)


def main() -> int:
    dup = json.loads((T / "h3/near_dup.json").read_text())["counts"]
    body = []
    for p, label in PIPES:
        pooled = json.loads((T / f"h3/analysis_{p}.json").read_text())["pooled"]
        n = dup[p]["none"]
        body.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\itshape {label}}}\\\\")
        body.append(" & ".join(["\\textit{No memory} (level)"] + [f"\\textit{{{pooled[m]['none']:.3f}}}" for m in METRICS]
                               + [f"{n['near_dup']}/{n['pairs']}"]) + " \\\\[2pt]")
        for key, name in CONDS:
            d = dup[p][key]
            body.append(" & ".join([name] + [cell(pooled[m][key]) for m in METRICS]
                                   + [f"{d['near_dup']}/{d['pairs']}"]) + " \\\\[2pt]")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    tex = "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{\\textbf{Track-B memory effects on two MiniMax-H3 realizations} (the same 8 stories, 767 segments per "
        "condition, seed 2026, same judge). Differences to no memory with paired story-bootstrap 95\\% CI, gray subscripts show full CI widths (upper minus lower); bold when it "
        "excludes zero. Near-dup: adjacent segments whose middle frames nearly coincide (a symptom of the video model "
        "copying its references instead of following the instruction).}",
        "\\label{tab:second-pipeline}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}l *{3}{c} c@{}}", "\\toprule",
        "Condition & Recall & Precision & Avoid-OK & Near-dup \\\\", "\\midrule", *body,
        "\\end{tabular}", "\\end{table}"])
    (T / "trackb_second_pipeline.tex").write_text(tex + "\n")
    print("wrote", T / "trackb_second_pipeline.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
