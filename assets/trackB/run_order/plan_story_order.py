#!/usr/bin/env python3
"""Rank Track B English stories by segment count and estimate cumulative Stage-1
runtime, so a cutoff (top-K shortest stories) can be chosen to fit a wall-clock
budget.

Writes ``outputs/evaluation/trackB/story_run_order.{md,csv,json}``.

Per-segment seconds are ESTIMATES (frames_per_segment=21). They are CLI knobs so
the table can be regenerated with measured rates once the pilot logs land.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[3]
ASSETS_EN = BENCH_ROOT / "assets" / "trackB" / "en" / "sut_prompts"
OUT_DIR = BENCH_ROOT / "outputs" / "evaluation" / "trackB"

# system -> (per-segment seconds @ fps=21, steps)
DEFAULT_RATES = {"memflow": 20.0, "longlive_rag": 55.0, "iamflow": 60.0}
LOAD_SEC = 150.0  # per-job model load / warmup


def stories() -> list[tuple[str, int]]:
    rows = []
    for p in sorted(ASSETS_EN.glob("*_name_anchored.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n = len(data.get("segments") or [])
        if n <= 0:
            continue
        rows.append((str(data.get("story_id") or p.stem), n))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slots", type=int, default=30, help="parallel DiT GPU slots")
    ap.add_argument("--realistic-factor", type=float, default=1.6,
                    help="multiplier on the ideal packed time for imbalance/VLM/retry")
    ap.add_argument("--memflow-sec", type=float, default=DEFAULT_RATES["memflow"])
    ap.add_argument("--longlive-sec", type=float, default=DEFAULT_RATES["longlive_rag"])
    ap.add_argument("--iamflow-sec", type=float, default=DEFAULT_RATES["iamflow"])
    ap.add_argument("--load-sec", type=float, default=LOAD_SEC)
    args = ap.parse_args(argv)

    rates = {"memflow": args.memflow_sec, "longlive_rag": args.longlive_sec, "iamflow": args.iamflow_sec}
    rows = stories()

    def story_gpu_sec(n: int) -> float:
        return sum(r * n + args.load_sec for r in rates.values())

    def job_sec(system: str, n: int) -> float:
        return rates[system] * n + args.load_sec

    records = []
    cum_seg = 0
    cum_gpu = 0.0
    longest_job = 0.0
    for i, (sid, n) in enumerate(rows, start=1):
        cum_seg += n
        cum_gpu += story_gpu_sec(n)
        longest_job = max(longest_job, max(job_sec(s, n) for s in rates))
        n_jobs = i * 3
        eff_slots = min(args.slots, n_jobs)
        ideal_h = max(longest_job, cum_gpu / eff_slots) / 3600.0
        real_h = ideal_h * args.realistic_factor
        records.append({
            "rank": i, "story_id": sid, "n_segments": n,
            "cum_stories": i, "cum_segments": cum_seg,
            "cum_gpu_hours": round(cum_gpu / 3600.0, 1),
            "wall_ideal_h": round(ideal_h, 1), "wall_realistic_h": round(real_h, 1),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "story_run_order.json").write_text(
        json.dumps({"assumptions": {"slots": args.slots, "realistic_factor": args.realistic_factor,
                                    "per_seg_sec": rates, "load_sec": args.load_sec,
                                    "frames_per_segment": 21, "steps": {"memflow": 5, "longlive_rag": 10, "iamflow": 5}},
                    "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_lines = ["rank,story_id,n_segments,cum_stories,cum_segments,cum_gpu_hours,wall_ideal_h,wall_realistic_h"]
    for r in records:
        csv_lines.append(f"{r['rank']},{r['story_id']},{r['n_segments']},{r['cum_stories']},"
                         f"{r['cum_segments']},{r['cum_gpu_hours']},{r['wall_ideal_h']},{r['wall_realistic_h']}")
    (OUT_DIR / "story_run_order.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = [
        "# Track B Stage-1 — English story run order (shortest first)",
        "",
        f"Assumptions: {args.slots} parallel GPU slots, frames_per_segment=21, "
        f"steps memflow=5/longlive=10/iamflow=5, per-segment sec "
        f"memflow={rates['memflow']:.0f}/longlive={rates['longlive_rag']:.0f}/iamflow={rates['iamflow']:.0f}, "
        f"load={args.load_sec:.0f}s/job, realistic factor x{args.realistic_factor}.",
        "These per-segment rates are ESTIMATES; regenerate with measured rates after the pilot.",
        "",
        "Pick a cutoff K = the number of shortest stories to run (each story runs all 3 systems).",
        "",
        "| K (stories) | story_id | n_seg | cum_seg | cum GPU·h | wall ideal (h) | wall realistic (h) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        md.append(f"| {r['rank']} | {r['story_id']} | {r['n_segments']} | {r['cum_segments']} | "
                  f"{r['cum_gpu_hours']} | {r['wall_ideal_h']} | {r['wall_realistic_h']} |")
    (OUT_DIR / "story_run_order.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Console: suggest cutoffs at a few realistic-hour budgets.
    print(f"wrote {OUT_DIR/'story_run_order.md'} (+ .csv/.json)")
    for budget in (6.0, 7.0, 8.0):
        best = None
        for r in records:
            if r["wall_realistic_h"] <= budget:
                best = r
        if best:
            print(f"  <= {budget:.0f}h realistic  -> K={best['rank']} stories "
                  f"(up to {best['n_segments']} seg/story, cum {best['cum_gpu_hours']} GPU-h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
