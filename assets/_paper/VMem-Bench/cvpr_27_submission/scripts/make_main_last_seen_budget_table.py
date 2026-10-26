#!/usr/bin/env python3
"""Summarize frozen last-seen reference-count strata; no rescoring or resampling."""
import json
from pathlib import Path
from ci_format import ci_cell

PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / 'tables' / 'experiments'


def main():
    data = json.loads((TABLES / 'trackb_rigor_checks.json').read_text())['last_seen_by_ref_count']
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{\textbf{Last-seen is not budget matched.} Deltas from no memory by references used. Gray subscripts: full paired story-bootstrap 95\% CI widths; bold: CI excludes zero. Different segments per group preclude causal count comparisons.}',
             r'\label{tab:main-last-seen-budget}', r'\footnotesize',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}rrccc@{}}', r'\toprule',
             r'Refs & Segments & $\Delta R$ & $\Delta P$ & $\Delta A$ \\', r'\midrule']
    for count, row in data.items():
        cells = []
        for metric in ('recall', 'precision', 'avoid_ok'):
            r = row[metric]
            cells.append(ci_cell(f"{r['diff']:+.3f}", str(r['lo']), str(r['hi']),
                                 bold=r['lo'] > 0 or r['hi'] < 0, width_digits=3))
        lines.append(f"{count} & {row['segments']:,} & " + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (TABLES / 'trackb_main_last_seen_budget.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
