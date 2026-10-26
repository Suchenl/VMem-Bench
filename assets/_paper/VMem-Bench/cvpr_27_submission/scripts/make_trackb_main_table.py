#!/usr/bin/env python3
"""Write tables/experiments/trackb_main_results.tex from the checked result manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
COLS = [("gap_ge30", "long_gap"), ("lookalike", "lookalike"), ("false_friend", "false_fr"), ("count", "count"),
        ("temporal", "temporal"), ("state_change", "state_chg"), ("persist", "persist"),
        ("deprecation", "avoid_dep"), ("indirect", "indirect"), ("recall", "recall")]
RETR = [("retr_frame_text", "Text-frame ret."), ("retr_seg_uniform", "Seg+uniform ret."),
        ("retr_seg_dinokey", "Seg+DINO-key ret."), ("retr_seg_framererank", "Seg+frame ret.")]
NATIVE = [("memflow", "MemFlow"), ("longlive_rag", "LongLive-RAG"), ("iamflow", "IAMFlow")]

HEAD = r"""\begin{table*}[!t]
\definecolor{memup}{HTML}{E0ECF4}\definecolor{memdown}{HTML}{F6DDD5}
\centering
\caption{\textbf{Track-B: generated-video outcomes by stress} (30 stories/system; higher is better). Cells are absolute rates. In shared-generator policy rows, blue/+ and red/$-$ mark paired story-bootstrap 95\% CIs above/below the first-row no-memory baseline; unmarked cells have no supported direction. Long-gap pools characters and props; Table~\ref{tab:main-kind-gap} separates them. Overall is pooled character/prop recall. The separate released-system block uses different generators and is not a memory-only comparison. Stress-level judge sensitivity: Sec.~\ref{sec:exp-sensitivity}.}
\label{tab:trackb-results}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.06}
\begin{tabular}{@{}l cccccccccc@{}}
\toprule
& \multicolumn{5}{c}{\textbf{Presence, identity, and count}} & \multicolumn{2}{c}{\textbf{State correctness}} & \multicolumn{2}{c}{\textbf{Entity exclusion}} & \textbf{All} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-11}
System & Long-gap & Look-alike & False-fr. & Count & Temporal & State-chg & Persist & Avoid & Indirect & Overall$\uparrow$ \\
 & {rec.} & {acc.} & {acc.} & {acc.} & {rec.} & {acc.} & {acc.} & {avoid.} & {avoid.} & {rec.} \\
\midrule
\multicolumn{11}{@{}l}{\itshape Shared generator: no-memory baseline, four retrieval policies, Last-seen control}\\
"""
MID = r"""\cmidrule(lr){1-11}
\rowcolor{black!5}\multicolumn{11}{@{}l}{\itshape Separate setting: released systems with their own generators}\\
"""
TAIL = r"""\bottomrule
\end{tabular}
\end{table*}
"""


def color(ci: dict | None) -> str:
    if ci is None:
        return ""
    return "\\cellcolor{memup}" if ci["lo"] > 0 else ("\\cellcolor{memdown}" if ci["hi"] < 0 else "")


def sign(ci: dict) -> str:
    if ci["lo"] > 0:
        return r"\textsuperscript{+}"
    if ci["hi"] < 0:
        return r"\textsuperscript{$-$}"
    return ""


def row(label: str, cells: list[str]) -> str:
    return f"{label} & " + " & ".join(cells) + " \\\\\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=T, help="directory with the input JSONs")
    ap.add_argument("--out", type=Path, default=T / "trackb_main_results.tex")
    a = ap.parse_args()
    mg = json.loads((a.tables / "memory_gain_bootstrap.json").read_text())
    ls = json.loads((a.tables / "last_seen_bootstrap.json").read_text())
    nat = json.loads((a.tables / "trackb_native_pooled_20260926.json").read_text())
    pt = {c: {m: round(v, 3) for m, v in p.items()} for c, p in ls["point"].items()}
    body = row(r"No memory ($K{=}0$)", [f"{pt['no_memory'][m]:.3f}" for m, _ in COLS])
    for c, label in RETR:
        cells = []
        for m, _ in COLS:
            v = f"{pt[c][m]:.3f}"
            ci = mg["diff_vs_no_memory"][c][m]
            cells.append(color(ci) + v + sign(ci))
        body += row(label, cells)
    body += row(r"Last-seen refs", [color(ls["oracle_minus_no_memory"][m]) + f"{pt['oracle'][m]:.3f}" + sign(ls["oracle_minus_no_memory"][m]) for m, _ in COLS])
    body += MID
    for c, label in NATIVE:
        body += row(label, [f"{nat[c][k]:.3f}" for _, k in COLS])
    a.out.write_text(HEAD + body + TAIL)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
