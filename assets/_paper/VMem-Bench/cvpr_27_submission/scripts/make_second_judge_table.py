#!/usr/bin/env python3
"""Write tables/experiments/trackb_second_judge.tex from tables/experiments/trackb_judge2.json.

    python3 scripts/make_second_judge_table.py

trackb_judge2.json is written by Montage experiments/benchmarks/vmem-bench/trackb_judge2/analyze.py: the
testbed rerun scored by the pinned Qwen3-VL-32B judge and, on the same views, GT and prompts, by
InternVL3.5-30B-A3B; differences to no memory use the paper's story bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
JUDGES = [("judge1", "Qwen3-VL-32B (pinned judge)"), ("judge2", "InternVL3.5-30B-A3B")]
CONDS = [("retr_frame_text", "Text-frame"), ("retr_seg_uniform", "Seg+uniform"),
         ("retr_seg_dinokey", "Seg+DINO-key"), ("retr_seg_framererank", "Seg+frame"),
         ("last_seen", "\\textit{Last-seen refs}")]
METRICS = ["recall", "precision", "avoid_ok"]


def cell(r: dict) -> str:
    return ci_cell(f"{r['diff']:+.3f}", f"{r['lo']:+.2f}", f"{r['hi']:+.2f}",
                   bold=r['lo'] > 0 or r['hi'] < 0)


def agree(g: dict, group: str) -> str:
    return f"{g[group]['agree']:.3f}/{g[group]['kappa']:.2f}"


def main() -> int:
    d = json.loads((T / "trackb_judge2.json").read_text())
    body = []
    for j, label in JUDGES:
        pt = d["point"][j]["no_memory"]
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\itshape {label}}}\\\\")
        all_agree = agree(d["agreement"]["no_memory"], "all") if j == "judge1" else ""
        present_agree = agree(d["agreement"]["no_memory"], "present") if j == "judge1" else ""
        body.append(" & ".join(["\\textit{No memory}"] + [f"\\textit{{{pt[m]:.3f}}}" for m in METRICS]
                               + [all_agree, present_agree]) + " \\\\[2pt]")
        for key, name in CONDS:
            all_agree = agree(d["agreement"][key], "all") if j == "judge1" else ""
            present_agree = agree(d["agreement"][key], "present") if j == "judge1" else ""
            body.append(" & ".join([name] + [cell(d["diff_vs_no_memory"][j][key][m]) for m in METRICS]
                                   + [all_agree, present_agree]) + " \\\\[2pt]")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    tex = "\n".join([
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{\\textbf{{Track-B memory effects under a second judge}} (the same videos, {d['n_stories']} stories, "
        "same roster, prompts, and clips). Differences to no memory with paired story-bootstrap 95\\% CI, gray subscripts show full CI widths (upper minus lower); bold when it "
        "excludes zero. Agreement / $\\kappa$ is shown for the full roster and required-present entities.}",
        "\\label{tab:second-judge}", "\\scriptsize", "\\setlength{\\tabcolsep}{1pt}",
        "\\begin{tabular}{@{}l *{3}{c} cc@{}}", "\\toprule",
        "Condition & Recall & Precision & Avoid-OK & \\shortstack{All\\\\A/$\\kappa$} & \\shortstack{Present\\\\A/$\\kappa$} \\\\", "\\midrule", *body,
        "\\end{tabular}", "\\end{table}"])
    (T / "trackb_second_judge.tex").write_text(tex + "\n")
    print("wrote", T / "trackb_second_judge.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
