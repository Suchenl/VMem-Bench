#!/usr/bin/env python3
"""Write tables/experiments/trackb_robustness.tex from tables/experiments/trackb_rigor_checks.json.

    python3 scripts/make_robustness_table.py

(a) condition minus no memory after dropping every segment on which either arm's judge output failed to parse;
(b) last-seen minus no memory split by the number of references the control used. Paired story bootstrap,
95% CIs, as produced by experiments/benchmarks/vmem-bench/trackb_oracle/rigor_checks.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
NAMES = {"retr_frame_text": "Text-frame", "retr_seg_uniform": "Seg+uniform", "retr_seg_dinokey": "Seg+DINO-key",
         "retr_seg_framererank": "Seg+frame", "last_seen": "Last-seen refs"}
COLS_A = [("recall", "Recall"), ("precision", "Precision"), ("avoid_ok", "Avoid-OK"), ("gap_1_4", "1--4"),
          ("gap_5_29", "5--29"), ("gap_ge30", "$\\geq$30")]
COLS_B = [("recall", "Recall"), ("precision", "Precision"), ("avoid_ok", "Avoid-OK")]


CAPTION = ("\\caption{\\textbf{Robustness of the Track-B memory effects.} Differences to no memory with paired "
          "story-bootstrap 95\\% CIs; gray subscripts show full CI widths (upper minus lower); bold when the CI excludes zero. (a) Dropping every segment on which either "
          "arm's judge output failed to parse leaves all main effects in place; the three segment memories also gain "
          "significantly at 5--29 segments. (b) Last-seen effects are stratified by reference count; "
          "groups contain different segments, so this is not a count intervention.}")


def cell(r: dict) -> str:
    return ci_cell(f"{r['diff']:+.3f}", f"{r['lo']:+.2f}", f"{r['hi']:+.2f}",
                   bold=r['lo'] > 0 or r['hi'] < 0)


def main() -> int:
    d = json.loads((T / "trackb_rigor_checks.json").read_text())
    L = ["\\begin{table*}[t]", "\\centering", CAPTION, "\\label{tab:robustness}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l r *{6}{c}@{}}", "\\toprule",
         "\\multicolumn{8}{@{}l}{\\textit{(a) Minus no memory, segments with a parsable judge output in both arms}}\\\\",
         "Condition & dropped & " + " & ".join(c for _, c in COLS_A) + " \\\\", "\\midrule"]
    for k, name in NAMES.items():
        r = d["parse_sensitivity"][k]
        L.append(f"{name} & {r['dropped_segments']} & " + " & ".join(cell(r[m]) for m, _ in COLS_A) + " \\\\[2pt]")
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{(b) Last-seen refs minus no memory, by number of references used}}\\\\",
          "References & segments & " + " & ".join(c for _, c in COLS_B) + " & & & \\\\", "\\midrule"]
    for n, r in d["last_seen_by_ref_count"].items():
        L.append(f"{n} & {r['segments']} & " + " & ".join(cell(r[m]) for m, _ in COLS_B) + " & & & \\\\[2pt]")
    L += ["\\bottomrule", "\\end{tabular}",
"\\end{table*}"]
    (T / "trackb_robustness.tex").write_text("\n".join(L) + "\n")
    print("wrote", T / "trackb_robustness.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
