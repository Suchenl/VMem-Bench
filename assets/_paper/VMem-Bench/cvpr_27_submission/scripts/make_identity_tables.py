#!/usr/bin/env python3
"""Write the Track-B identity-consistency tables from the benchmark scorers' JSON.

    python3 scripts/make_identity_tables.py

Inputs:  tables/experiments/trackb_identity_vlm.json    (trackB-identity-1.0)
         tables/experiments/trackb_identity_embed.json  (trackB-embed-1.0)
Outputs: tables/experiments/trackb_identity_calibration.tex
         tables/experiments/trackb_identity_results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
T = PAPER / "tables" / "experiments"
CONDS = [("text_frame", "Text"), ("seg_uniform", "Uniform"), ("seg_dinokey", "DINO-key"),
         ("seg_frame_clean", "Frame"), ("retr_oracle", "Last-seen")]
ROWS = [("main|all", "All returns"), ("main|character", "Characters"), ("main|prop", "Props"),
        ("main|all|1-4", "Gap 1--4"), ("main|all|5-29", "Gap 5--29"), ("main|all|>=30", "Gap $\\geq$30"),
        ("transform_ge30|all", "Returns changed, $\\geq$30")]


def diff_cell(d) -> str:
    v, lo, hi = d
    s = f"{v:+.2f}"
    if lo > 0 or hi < 0:
        s = f"\\textbf{{{s}}}"
    return s


def block(title: str, cells: dict, metric: str, rows) -> list[str]:
    out = [f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\"]
    for key, lab in rows:
        if key not in cells:
            continue
        c = cells[key]["conditions"]
        base = c["no_memory"][metric]
        out.append(f"{lab} & {cells[key]['n']} & {base:.2f} & "
                   + " & ".join(diff_cell(c[k]["diff_vs_no_memory"][metric]) for k, _ in CONDS) + " \\\\")
    return out


def main() -> int:
    v = json.loads((T / "trackb_identity_vlm.json").read_text())
    e = json.loads((T / "trackb_identity_embed.json").read_text())

    cal = e["calibration"]
    names = {"dino_crop": "DINOv3 crop", "face": "ArcFace face",
             "location_megaloc": "MegaLoc scene", "location_dino": "DINOv3 scene"}
    ctl = v["controls"]["summary"] if "summary" in v["controls"] else v["controls"]
    rr = [min(x["exact_answer_agreement"] for x in v["test_retest"].values()),
          max(x["exact_answer_agreement"] for x in v["test_retest"].values())]
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{\\textbf{Calibration of the identity-consistency metrics.} Positive controls pair two "
         "disjoint frame sets of the same clip; negative controls pair two different entities of the same kind "
         "and story. For the embeddings, the threshold is the Youden-optimal cut on these controls, fixed before "
         f"any condition was scored. The VLM judge answers the same question twice with {rr[0]:.3f}--{rr[1]:.3f} "
         "exact agreement.}", "\\label{tab:identity-calibration}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
         "Metric & pos./neg. pairs & AUC & threshold & pos. median & neg. median \\\\", "\\midrule"]
    for m, lab in names.items():
        c = cal[m]
        L.append(f"{lab} & {c['counts']['ctl_pos_both']}/{c['counts']['ctl_neg_both']} & {c['auc']:.3f} & "
                 f"{c['youden']['threshold']:.2f} & {c['pos']['p50']:.2f} & {c['neg']['p50']:.2f} \\\\")
    L += ["\\midrule",
          f"VLM pairwise judge & \\multicolumn{{5}}{{l}}{{same when both visible: {ctl['pos_same_given_both']:.2f} "
          f"(pos.), {1 - ctl['neg_different_given_both']:.2f} (neg.); neg.\\ not consistent: "
          f"{ctl['neg_not_consistent_rate']:.2f}}} \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (T / "trackb_identity_calibration.tex").write_text("\n".join(L) + "\n")

    R = ["\\begin{table*}[t]", "\\centering",
         "\\caption{\\textbf{Track-B identity consistency.} No-memory value and each condition minus no memory; $N$ counts eligible events, while policy-specific detected/visible subsets determine the analyzed means "
         "(bold: paired story-bootstrap 95\\% CI excludes zero). Embeddings: mean cosine similarity between the "
         "previous and current appearance (DINOv3 on the localized entity; MegaLoc on the whole frame for "
         "locations), over returns detected on both sides. VLM: share judged the same individual among returns "
         "visible in both clips (lenient, Table~\\ref{tab:identity-calibration}). Returns are character and prop "
         "re-appearances after at least one segment away; returns changed are characters whose look changed after "
         "30 or more segments, for which a higher similarity means the return looks more like its old look.}",
         "\\label{tab:identity-results}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l r r *{5}{r}@{}}", "\\toprule",
         r"Stratum & \shortstack{Eligible\\$N$} & \shortstack{No\\memory} & " + " & ".join(r"\shortstack{" + n + r"\\$\Delta$}" for _, n in CONDS) + " \\\\", "\\midrule"]
    R += block("DINOv3 crop similarity (trackB-embed-1.0)", e["modalities"]["dino_crop"]["cells"], "mean_similarity", ROWS)
    R += ["\\midrule"]
    R += block("VLM pairwise judge: same individual when visible in both (trackB-identity-1.0)", v["cells"],
               "identity_given_both", ROWS[:-1])
    R += ["\\midrule"]
    R += block("ArcFace face similarity, characters", e["modalities"]["face"]["cells"], "mean_similarity",
               [("main|character", "Characters"), ("transform_ge30|character", "Returns changed, $\\geq$30")])
    R += ["\\midrule"]
    R += block("MegaLoc scene similarity, locations", e["modalities"]["location_megaloc"]["cells"], "mean_similarity",
               [("main|location", "Locations")])
    R += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (T / "trackb_identity_results.tex").write_text("\n".join(R) + "\n")
    print("wrote identity tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
