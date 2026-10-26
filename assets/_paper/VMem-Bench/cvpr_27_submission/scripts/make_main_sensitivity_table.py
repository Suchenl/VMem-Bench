#!/usr/bin/env python3
"""Summarize existing judge/backend analyses for the main paper; no new scoring."""
from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

TABLES = Path(__file__).resolve().parents[1] / "tables" / "experiments"
METRICS = ("recall", "precision", "avoid_ok")


def cell(result: dict) -> str:
    return ci_cell(f"{result['diff']:+.3f}", f"{result['lo']:+.3f}", f"{result['hi']:+.3f}")


def main() -> None:
    judges = json.loads((TABLES / "trackb_judge2.json").read_text())
    rows = [r"\multicolumn{4}{@{}l}{\textit{Judge check: Wan2.2, 30 stories}}\\"]
    for key, label in (("judge1", "Qwen3-VL"),
                       ("judge2", "InternVL3.5")):
        values = judges["diff_vs_no_memory"][key]["retr_seg_dinokey"]
        rows.append(" & ".join([label] +
                               [cell(values[m]) for m in METRICS]) + r" \\")
    supported = []
    policies = ["retr_frame_text", "retr_seg_uniform", "retr_seg_dinokey", "retr_seg_framererank"]
    for metric in METRICS:
        count = 0
        for policy in policies:
            a, b = [judges["diff_vs_no_memory"][j][policy][metric] for j in ["judge1", "judge2"]]
            count += (a["lo"] > 0 and b["lo"] > 0) or (a["hi"] < 0 and b["hi"] < 0)
        arrow = r"\uparrow" if metric == "recall" else r"\downarrow"
        supported.append(f"${count}/4\\ {arrow}$")
    rows.append(r"Cross-judge support & " + " & ".join(supported) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{@{}l}{\textit{Backend check: Qwen3-VL, 8 matched stories}}\\")
    for key, label in (("reference_flux2_wan22", "Wan2.2"),
                       ("C", "MiniMax-H3")):
        analysis = json.loads((TABLES / f"h3/analysis_{key}.json").read_text())
        pooled = analysis["pooled"]
        rows.append(" & ".join([label] +
                               [cell(pooled[m]["seg_dinokey"]) for m in METRICS]) + r" \\")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Judge and backend sensitivity of Seg+DINO-key.} Deltas from no memory; gray subscripts give full paired story-bootstrap 95\% CI widths, conditional on one rollout per story. Cross-judge support counts policies with same-sign CIs excluding zero. Backend rows retain FLUX.2 keyframes. Limits: Sec.~\ref{sec:exp-sensitivity}.}",
        r"\label{tab:main-sensitivity}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Judge / backend & $\Delta R$ & $\Delta P$ & $\Delta A$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
    target = TABLES / "trackb_main_sensitivity.tex"
    target.write_text(tex + "\n")
    print("wrote", target)


if __name__ == "__main__":
    main()
