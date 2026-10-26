#!/usr/bin/env python3
"""Write tables/experiments/trackb_kind_gap.tex from tables/experiments/trackb_kind_gap.json.

    python3 scripts/make_kind_gap_table.py

Recall by entity kind and absence gap for no memory, and each condition minus no memory with a paired story
bootstrap 95% CI (produced by experiments/benchmarks/vmem-bench/trackb_oracle/kind_gap_analysis.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
CONDS = [("retr_frame_text", "Text"), ("retr_seg_uniform", "Uniform"), ("retr_seg_dinokey", "DINO-key"),
         ("retr_seg_framererank", "Frame"), ("last_seen", "Last-seen")]
ROWS = [("character", "0", "Character", "just seen"), ("character", "1-4", "", "1--4"),
        ("character", "5-29", "", "5--29"), ("character", ">=30", "", "$\\geq$30 (returns changed)"),
        ("prop", "0", "Prop", "just seen"), ("prop", "1-4", "", "1--4"), ("prop", "5-29", "", "5--29"),
        ("prop", ">=30", "", "$\\geq$30")]


def cell(r: dict, *, width_digits: int | None = None) -> str:
    return ci_cell(f"{r['diff']:+.3f}", f"{r['lo']:+.2f}", f"{r['hi']:+.2f}",
                   bold=r['lo'] > 0 or r['hi'] < 0, width_digits=width_digits)


def write_main_table(d: dict) -> None:
    """Combine diagnostic gap slices and the existing prompt intervention."""
    from make_nolook_table import diff, rate
    prompt = json.loads((T / "trackb_nolook.json").read_text())
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Return recall and prompt information} (30 stories). No memory: absolute recall; DINO-key $\Delta R$: difference from it. Gray subscripts: full paired story-bootstrap 95\% CI width; bold: CI excludes zero. $+/-/?$: counts of four retrieval policies with CIs above/below/including zero. Gaps: segments; prompts describe changed looks in 61/62 long-gap character returns. Only (b) changes prompts; see Supplementary Table~\ref{tab:kind-gap}.}",
        r"\label{tab:main-kind-gap}", r"\label{tab:nolook}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrcc@{}}", r"\toprule",
        r"\multicolumn{6}{@{}l}{\textit{(a) Original prompts: observational groups}}\\",
        r"Entity & Gap & $n$ & \shortstack{No\\memory} & \shortstack{DINO-key\\$\Delta R$} & \shortstack{Policies\\$+ / - / ?$} \\", r"\midrule",
    ]
    for kind, gap, label, gap_label in [
        ("character", "1-4", "Character", "1--4"),
        ("character", ">=30", "", r"$\geq$30"),
        ("prop", "1-4", "Prop", "1--4"),
        ("prop", ">=30", "", r"$\geq$30"),
    ]:
        r = d[f"{kind}|{gap}"]
        gain = sum(r[c]["lo"] > 0 for c, _ in CONDS[:4])
        loss = sum(r[c]["hi"] < 0 for c, _ in CONDS[:4])
        if kind == "prop" and gap == "1-4":
            lines.append(r"\midrule")
        lines.append(f"{label} & {gap_label} & {r['n']} & {r['no_memory']:.3f} & "
                     + cell(r["retr_seg_dinokey"], width_digits=3) + f" & {gain}/{loss}/{4-gain-loss}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip",
              r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
              r"\multicolumn{5}{@{}l}{\textit{(b) Appearance intervention: same 62 character returns}}\\",
              r"Prompt & \shortstack{No\\memory} & \shortstack{DINO-key\\$R$} & \shortstack{Last-seen\\$R$} & \shortstack{DINO-key\\$\Delta R$} \\", r"\midrule"]
    for arm, label in [("look", "Described"), ("nolook", "Stripped")]:
        lines.append(" & ".join([label, rate(prompt, arm, "no_memory"),
                                  rate(prompt, arm, "retr_seg_dinokey"),
                                  rate(prompt, arm, "last_seen"),
                                  diff(prompt, arm, "retr_seg_dinokey")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "trackb_main_kind_gap.tex").write_text("\n".join(lines) + "\n")


def main() -> int:
    d = json.loads((T / "trackb_kind_gap.json").read_text())
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{\\textbf{Track-B recall by entity kind and absence gap.} No-memory recall and each condition "
         "minus no memory (paired story-bootstrap 95\\% CI; gray subscripts show full CI widths (upper minus lower); bold when it excludes zero). Just seen means the entity "
         "was on screen in the previous segment. Characters are named only by name after their first appearance; "
         "61 of the 62 character returns after 30 or more segments come back changed, with the new look in the "
         "instruction. Props often carry their look in the name. Text, Uniform, DINO-key, and Frame are the four retrieval memories.}",
         "\\label{tab:kind-gap}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular}{@{}l l r r *{5}{c}@{}}", "\\toprule",
         "Kind & Gap (segments) & $n$ & \\shortstack{No\\\\memory} & " + " & ".join(n for _, n in CONDS) + " \\\\", "\\midrule"]
    for kind, b, klab, blab in ROWS:
        r = d[f"{kind}|{b}"]
        if kind == "prop" and b == "0":
            L.append("\\midrule")
        L.append(f"{klab} & {blab} & {r['n']} & {r['no_memory']:.3f} & "
                 + " & ".join(cell(r[c]) for c, _ in CONDS) + " \\\\[2pt]")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (T / "trackb_kind_gap.tex").write_text("\n".join(L) + "\n")
    print("wrote", T / "trackb_kind_gap.tex")
    write_main_table(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
