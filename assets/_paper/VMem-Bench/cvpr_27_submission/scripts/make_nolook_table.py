#!/usr/bin/env python3
"""Generate the look-stripped Track-B sensitivity table from its analysis JSON."""
from __future__ import annotations

import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / "tables" / "experiments"


def rate(d: dict, arm: str, condition: str) -> str:
    return f"{d['rate'][f'{arm}|{condition}']:.3f}"


def diff(d: dict, arm: str, condition: str) -> str:
    x = d["memory_minus_none"][f"{arm}|{condition}"]
    return ci_cell(f"{x['diff']:+.3f}", f"{x['lo']:+.3f}", f"{x['hi']:+.3f}")


def main() -> None:
    d = json.loads((TABLES / "trackb_nolook.json").read_text())
    rows = []
    for arm, prompt in (("look", "Described"), ("nolook", "Stripped")):
        rows.append(" & ".join([
            prompt,
            rate(d, arm, "no_memory"),
            rate(d, arm, "retr_seg_dinokey"),
            rate(d, arm, "last_seen"),
            diff(d, arm, "retr_seg_dinokey"),
        ]) + r" \\")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{\textbf{{Prompt-description sensitivity.}} Presence on {d['n_units']['look|no_memory']} long-gap character returns ($\geq$30 segments; {d['n_stories']} stories). $\Delta R$: Seg+DINO-key minus no memory; gray subscripts give full paired story-bootstrap 95\% CI widths. Both CIs include zero; stripping appearance descriptions establishes no reliable retrieval gain.}}",
        r"\label{tab:nolook}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"Prompt & None & DINO-key & Last-seen & $\Delta R$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
    out = TABLES / "trackb_nolook.tex"
    out.write_text(tex + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
