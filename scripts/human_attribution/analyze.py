#!/usr/bin/env python3
"""Analyse closed-loop failure-attribution labels deterministically.

    python3 scripts/human_attribution/analyze.py --labels <labels_*.json> [--study results/human_attribution/v1]
        [--tex assets/_paper/VMem-Bench/iclr_27_final/tables/experiments/human_attribution.tex]

Stage rule per scored unit (first occurrence; repeats are used only for consistency):
  miss / wrong   clip=yes -> evaluator; clip=no & refs=no -> history if the earlier clip
                 also lacks it (writing), else selection; clip=no & refs=yes -> synthesis
                 (keyframe_loss if keyframe=no, else video_loss); any needed 'unsure' -> unresolved
  forbidden      clip=no -> evaluator; clip=yes & refs=yes -> context (obsolete evidence
                 was passed to the generator); clip=yes & refs=no -> generator
  control        clip=yes -> confirmed; clip=no -> evaluator
Shares are inverse-inclusion-probability weighted within each system; 95% CIs come from a
story-clustered bootstrap with a fixed seed. Unresolved units are reported, not imputed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"text_frame": "Text-frame", "seg_uniform": "Seg+uniform", "seg_dinokey": "Seg+DINO-key", "seg_frame": "Seg+frame"}
FAIL_CAUSES = ["history", "selection", "synthesis", "evaluator"]
FORB_CAUSES = ["context", "generator", "evaluator"]
N_BOOT = 2000
BOOT_SEED = 20260926


def stage(kind: str, a: dict) -> tuple[str, str | None]:
    e, r, k, c = (a.get(p) for p in ("earlier", "refs", "keyframe", "clip"))
    if kind in ("miss", "wrong"):
        if c == "yes":
            return "evaluator", None
        if c != "no" or r not in ("yes", "no"):
            return "unresolved", None
        if r == "no":
            return ("history", None) if e == "no" else ("selection", None)
        return "synthesis", "keyframe_loss" if k == "no" else ("video_loss" if k == "yes" else "unresolved_stage")
    if kind == "forbidden":
        if c == "no":
            return "evaluator", None
        if c != "yes" or r not in ("yes", "no"):
            return "unresolved", None
        return ("context" if r == "yes" else "generator"), None
    if kind == "control":
        return {"yes": "confirmed", "no": "evaluator"}.get(c, "unresolved"), None
    raise ValueError(kind)


def weighted_shares(rows: list[dict], causes: list[str]) -> dict:
    tot = sum(r["w"] for r in rows if r["cause"] in causes)
    return {c: (sum(r["w"] for r in rows if r["cause"] == c) / tot if tot else None) for c in causes}


def boot_ci(rows: list[dict], causes: list[str], rng: random.Random) -> dict:
    by_story = defaultdict(list)
    for r in rows:
        by_story[r["story"]].append(r)
    stories = sorted(by_story)
    draws = {c: [] for c in causes}
    for _ in range(N_BOOT):
        sample = [r for s in (rng.choice(stories) for _ in stories) for r in by_story[s]]
        sh = weighted_shares(sample, causes)
        for c in causes:
            if sh[c] is not None:
                draws[c].append(sh[c])
    out = {}
    for c, v in draws.items():
        v.sort()
        out[c] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else None
    return out


def summarize(rows: list[dict], causes: list[str], rng: random.Random) -> dict:
    return {
        "n": len(rows),
        "n_resolved": sum(r["cause"] in causes for r in rows),
        "counts": dict(Counter(r["cause"] for r in rows)),
        "subtypes": dict(Counter(r["sub"] for r in rows if r["sub"])),
        "share": weighted_shares(rows, causes),
        "ci95": boot_ci(rows, causes, rng) if rows else None,
    }


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--study", type=Path, default=ROOT / "results" / "human_attribution" / "v2")
    ap.add_argument("--tex", type=Path, default=None)
    args = ap.parse_args()
    sealed = json.loads((args.study / "sealed_manifest.json").read_text())
    labels = json.loads(args.labels.read_text())
    answers = labels["answers"]
    rng = random.Random(BOOT_SEED)

    first, repeats, missing = {}, [], []
    for u in sealed["units"]:
        if u["practice"]:
            continue
        a = answers.get(u["display_id"])
        if not a or not all(a.get(p) for p in ("earlier", "refs", "keyframe", "clip")):
            missing.append(u["display_id"])
            continue
        if u["is_repeat"]:
            repeats.append((u, a))
        else:
            first[u["base_unit"]] = [(u, a)]

    rows = []
    for base, [(u, a)] in sorted(first.items()):
        cause, sub = stage(u["kind"], a)
        rows.append({"base": base, "system": u["system"], "kind": u["kind"], "stratum": u["stratum"],
                     "story": u["story_id"], "cause": cause, "sub": sub,
                     "w": 1.0 / u["inclusion_prob"] if u.get("inclusion_prob") else 1.0})

    rep_same_answers = rep_same_cause = 0
    for u, a in repeats:
        if u["base_unit"] not in first:
            continue
        a0 = first[u["base_unit"]][0][1]
        rep_same_answers += all(a0.get(p) == a.get(p) for p in ("earlier", "refs", "keyframe", "clip"))
        rep_same_cause += stage(u["kind"], a0)[0] == stage(u["kind"], a)[0]
    n_rep = sum(1 for u, _ in repeats if u["base_unit"] in first)

    report = {"annotator": labels.get("annotator"), "n_scored_units": len(rows), "missing_items": missing,
              "repeat": {"n": n_rep, "same_all_answers": rep_same_answers, "same_cause": rep_same_cause},
              "median_seconds_per_item": None, "failures": {}, "forbidden": {}, "controls": {}}
    ms = sorted(a.get("ms", 0) for a in answers.values() if a.get("ms"))
    report["median_seconds_per_item"] = round(ms[len(ms) // 2] / 1000, 1) if ms else None

    for sysname in list(LABELS) + ["pooled"]:
        sel = [r for r in rows if sysname == "pooled" or r["system"] == sysname]
        report["failures"][sysname] = summarize([r for r in sel if r["kind"] in ("miss", "wrong")], FAIL_CAUSES, rng)
        report["forbidden"][sysname] = summarize([r for r in sel if r["kind"] == "forbidden"], FORB_CAUSES, rng)
        ctl = [r for r in sel if r["kind"] == "control"]
        report["controls"][sysname] = {"n": len(ctl), "confirmed": sum(r["cause"] == "confirmed" for r in ctl)}
    by_stratum = defaultdict(list)
    for r in rows:
        if r["kind"] in ("miss", "wrong"):
            by_stratum[r["stratum"]].append(r)
    report["failures_by_stratum"] = {k: summarize(v, FAIL_CAUSES, rng) for k, v in sorted(by_stratum.items())}

    out = args.study / f"analysis_{labels.get('annotator', 'H')}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("annotator", "n_scored_units", "repeat", "median_seconds_per_item")}, indent=1))
    for s in list(LABELS) + ["pooled"]:
        f, g = report["failures"][s], report["forbidden"][s]
        print(f"{s:12s} fail n={f['n_resolved']}/{f['n']} " + " ".join(f"{c}={pct(f['share'][c])}%" for c in FAIL_CAUSES)
              + f" | forbidden n={g['n_resolved']}/{g['n']} " + " ".join(f"{c}={pct(g['share'][c])}%" for c in FORB_CAUSES))

    if args.tex:
        lines = [
            "% Generated by scripts/human_attribution/analyze.py; do not edit by hand.",
            "\\begin{tabular}{@{}l cccc c ccc c@{}}",
            "\\toprule",
            " & \\multicolumn{5}{c}{Required entity missing or wrong} & \\multicolumn{4}{c}{Forbidden entity rendered} \\\\",
            "\\cmidrule(lr){2-6}\\cmidrule(l){7-10}",
            "Memory & Hist. & Select. & Synth. & Judge & $n$ & Context & Gen. & Judge & $n$ \\\\",
            "\\midrule",
        ]
        for s in list(LABELS) + ["pooled"]:
            f, g = report["failures"][s], report["forbidden"][s]
            name = "Pooled" if s == "pooled" else LABELS[s]
            if s == "pooled":
                lines.append("\\midrule")
            lines.append(f"{name} & " + " & ".join(pct(f["share"][c]) for c in FAIL_CAUSES) + f" & {f['n_resolved']} & "
                         + " & ".join(pct(g["share"][c]) for c in FORB_CAUSES) + f" & {g['n_resolved']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        args.tex.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
