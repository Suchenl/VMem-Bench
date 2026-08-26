#!/usr/bin/env python3
"""Aggregate per-movie Track A results under outputs/evaluation/trackA/.

The causal runner (``baseline_adapters/causal/runner.py``) + scorer
(``vmem_bench.scoring.visual_coverage``) write each run under
``outputs/evaluation/trackA/<system>/<dataset>/<sample>/``. This tool is the
deterministic, model-free collector that normalizes those system run names into
paper baseline names and computes dataset-level + overall macro-averages per baseline.

Per movie it reads (all produced by Stage 1 + Stage 2, never by this script):

    <outputs>/<system>/<dataset>/<sample>/_visual_score/<system>/score.json
    <outputs>/<system>/<dataset>/<sample>/visual_selections/<system>.json
    <outputs>/<system>/<dataset>/<sample>/_adapter_work/<system>/finalize.json
    <outputs>/<system>/<dataset>/<sample>/_ref_frames/<system>/

``<system>`` is the runner's run name: ``<adapter.name>`` optionally suffixed with the input
mode (``__descprov`` / ``__desconly``) and budget (``__B<n>``). It is parsed back into
``(baseline, input_mode, budget)`` and the artifacts are written to:

    <outputs>/<baseline>/<dataset>/<sample>/<input_mode>[/B<budget>]/{score.json,visual_selections.json,finalize.json,meta.json}

plus per-baseline ``aggregate.json`` / ``aggregate.md`` (dataset + overall macro-averages) and a
top-level ``leaderboard.json`` / ``leaderboard.md`` ranking baselines by headline f1.

This is a pure filesystem/JSON pass: no GPU, no model calls, idempotent (re-running overwrites).
Reference frames are NOT copied by default (they can be large); pass ``--copy-frames`` to also
copy them. Run::

    PYTHONPATH=src python scripts/evaluate_baselines/trackA/aggregate_trackA_outputs.py
    # or a custom destination/source root:
    ... --outputs outputs/evaluation/trackA
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

_BENCH = Path(__file__).resolve().parents[3]

# Metric columns pulled from each score.json "summary" block (headline first); mirrors
# scripts/get_trackA_assets/compare/build_leaderboard_v2.py so the two stay consistent.
_METRIC_COLS = [
    "f1_mean",              # HEADLINE
    "recall_mean",          # continuity / memory recall
    "precision_mean",
    "redundancy_vlm_mean",
    "redundancy_sim_mean",
    "selection_efficiency_mean",
    "budget_avg_refs_per_chunk_with_refs",
    "retrieval_ms_mean",
    "write_ms_mean",
    "score_ms_mean",
    "total_duration_s",
    "n_chunks",
    "n_chunks_with_refs",
]
_RANK_KEY = "f1_mean"

# run_name (adapter.name) -> canonical baseline output-dir name. The four retrieval families use
# their paper names; retrieval diagnostics get a ``control_*`` dir; other baselines map by prefix.
_RETRIEVAL_MAP = {
    "retrieval_frame_text_ablation": "text_frame_retrieval",
    "retrieval_seg_uniform_ablation": "text_segment_retrieval_then_uniform_sampling",
    "retrieval_seg_dinokey_ablation": "text_segment_retrieval_then_dino_keyframe_sampling",
    "retrieval_seg_framererank_ablation": "text_segment_retrieval_then_frame_retrieval",
    "retrieval_recency_ctrl": "control_recency",
    "retrieval_bm25_desc_ctrl": "control_bm25_desc",
    "retrieval_random_ctrl": "control_random",
}
# Longest / most specific first (memflow_sma before memflow).
_BASELINE_PREFIXES = ["memstrata", "longlive_rag", "memflow_sma", "memflow", "iamflow", "slotmem"]

_BUDGET_RE = re.compile(r"__B(\d+)$")

# Normalized-store layout: <baseline>/<dataset>/<sample>/<input_mode>[/B<budget>]/score.json.
_INPUT_MODES = {"name_anchored", "description_provided", "description_only"}
_BUDGET_DIR_RE = re.compile(r"^B(\d+)$")


def _parse_system(system: str) -> tuple[str, str, int | None]:
    """run_name -> (baseline_dir, input_mode, budget). Order of suffixes: <base>[__descprov]__B<n>."""
    name = system
    budget: int | None = None
    m = _BUDGET_RE.search(name)
    if m:
        budget = int(m.group(1))
        name = name[: m.start()]
    if name.endswith("__descprov"):
        input_mode, base = "description_provided", name[: -len("__descprov")]
    elif name.endswith("__desconly"):
        input_mode, base = "description_only", name[: -len("__desconly")]
    else:
        input_mode, base = "name_anchored", name
    return _baseline_of(base), input_mode, budget


def _baseline_of(base: str) -> str:
    if base in _RETRIEVAL_MAP:
        return _RETRIEVAL_MAP[base]
    if base.startswith("retrieval_"):
        # Unknown retrieval variant: keep a stable, self-describing dir.
        return "retrieval_" + base[len("retrieval_"):]
    for prefix in _BASELINE_PREFIXES:
        if base == prefix or base.startswith(prefix + "_") or base.startswith(prefix):
            return prefix
    return base  # fallback: use the run name verbatim as its own dir


def _load_summary(score_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(score_path.read_text(encoding="utf-8")).get("summary", {})
    except Exception:  # noqa: BLE001 - a malformed score.json should not abort the whole pass
        return None


def _iter_run_dirs(outputs: Path, datasets: list[str] | None) -> list[tuple[str, str, Path]]:
    """Yield (dataset, sample, run_dir) for scored Stage-1/2 system runs."""
    out: list[tuple[str, str, Path]] = []
    if not outputs.is_dir():
        return out
    for system_dir in sorted(p for p in outputs.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            if datasets and ds_dir.name not in datasets:
                continue
            for sample_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                if (sample_dir / "_visual_score").is_dir():
                    out.append((ds_dir.name, sample_dir.name, sample_dir))
    return out


def _iter_normalized(
    outputs: Path, datasets: list[str] | None
) -> list[tuple[str, str, int | None, str, str, dict[str, Any] | None]]:
    """Yield (baseline, input_mode, budget, dataset, sample, summary) from the canonical
    normalized store: ``<baseline>/<dataset>/<sample>/<input_mode>[/B<budget>]/score.json``.

    This is the source of truth for ranking. It survives even when a run's *raw*
    ``_visual_score`` dir was later cleaned (e.g. to reclaim disk for heavy ``_ref_frames``),
    so a baseline that was scored once no longer silently drops off the leaderboard — the very
    failure that once hid the two strongest retrieval baselines. Raw run dirs (score.json under
    ``.../_visual_score/<system>/``) and ``*.pre_smoke_backup*`` samples are excluded.
    """
    out: list[tuple[str, str, int | None, str, str, dict[str, Any] | None]] = []
    if not outputs.is_dir():
        return out
    for score_path in outputs.rglob("score.json"):
        rel = score_path.relative_to(outputs).parts
        if "_visual_score" in rel:
            continue  # raw run dir, not the normalized store
        if any("pre_smoke_backup" in p for p in rel):
            continue
        # (baseline, dataset, sample, input_mode[, B<budget>], "score.json")
        if len(rel) not in (5, 6):
            continue
        baseline, dataset, sample, input_mode = rel[0], rel[1], rel[2], rel[3]
        if input_mode not in _INPUT_MODES:
            continue
        if datasets and dataset not in datasets:
            continue
        budget: int | None = None
        if len(rel) == 6:
            m = _BUDGET_DIR_RE.match(rel[4])
            if not m:
                continue
            budget = int(m.group(1))
        out.append((baseline, input_mode, budget, dataset, sample, _load_summary(score_path)))
    return out


def _prev_leaderboard_baselines(outputs: Path) -> set[str]:
    """Baselines listed in the existing leaderboard.json (for vanish detection)."""
    try:
        rows = json.loads((outputs / "leaderboard.json").read_text(encoding="utf-8")).get("rows", [])
        return {r.get("baseline") for r in rows if r.get("baseline")}
    except Exception:  # noqa: BLE001 - missing/malformed prior board is not fatal
        return set()


def _copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def collect(data_root: Path, outputs: Path, *, datasets: list[str] | None,
            copy_frames: bool, dry_run: bool) -> dict[str, Any]:
    n_samples = 0
    del data_root  # retained for CLI compatibility; Stage-2 now scans outputs directly.
    for dataset, sample, run in _iter_run_dirs(outputs, datasets):
        vsdir = run / "_visual_score"
        if not vsdir.is_dir():
            continue
        for sysdir in sorted(p for p in vsdir.iterdir() if p.is_dir()):
            system = sysdir.name
            score_path = sysdir / "score.json"
            baseline, input_mode, budget = _parse_system(system)
            summary = _load_summary(score_path) if score_path.is_file() else None

            leaf = outputs / baseline / dataset / sample / input_mode
            if budget is not None:
                leaf = leaf / f"B{budget}"
            sel_src = run / "visual_selections" / f"{system}.json"
            fin_src = run / "_adapter_work" / system / "finalize.json"
            frames_src = run / "_ref_frames" / system

            meta = {
                "baseline": baseline,
                "dataset": dataset,
                "sample": sample,
                "system": system,
                "input_mode": input_mode,
                "budget": budget,
                "has_score": score_path.is_file(),
                "source": {
                    "score": str(score_path),
                    "visual_selections": str(sel_src) if sel_src.is_file() else None,
                    "finalize": str(fin_src) if fin_src.is_file() else None,
                    "ref_frames_dir": str(frames_src) if frames_src.is_dir() else None,
                },
            }
            if summary:
                meta["summary"] = {c: summary.get(c) for c in _METRIC_COLS}
                meta["metric_version"] = summary.get("metric_version")

            if not dry_run:
                leaf.mkdir(parents=True, exist_ok=True)
                _copy(score_path, leaf / "score.json")
                _copy(sel_src, leaf / "visual_selections.json")
                _copy(fin_src, leaf / "finalize.json")
                (leaf / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                if copy_frames and frames_src.is_dir():
                    dst_frames = leaf / "ref_frames"
                    if dst_frames.exists():
                        shutil.rmtree(dst_frames)
                    shutil.copytree(frames_src, dst_frames)

            n_samples += 1

    # Rank from the CANONICAL normalized store (refreshed by the raw pass above), not the raw
    # runs alone: a baseline whose raw _visual_score was cleaned still has its normalized
    # score.json and must stay on the board. rows keyed by (baseline, input_mode, budget) ->
    # dataset -> metric rows.
    index: dict[tuple[str, str, int | None], dict[str, list[dict[str, Any]]]] = {}
    for baseline, input_mode, budget, dataset, sample, summary in _iter_normalized(outputs, datasets):
        row = {"sample": sample, "system": baseline, "ok": bool(summary)}
        if summary:
            for c in _METRIC_COLS:
                row[c] = summary.get(c)
        index.setdefault((baseline, input_mode, budget), {}).setdefault(dataset, []).append(row)

    now_baselines = {b for (b, _m, _bud) in index}
    vanished = sorted(_prev_leaderboard_baselines(outputs) - now_baselines)
    if vanished:
        print(f"[aggregate_trackA] WARNING: {len(vanished)} baseline(s) on the previous "
              f"leaderboard have no score.json in the normalized store now and would drop off: "
              f"{', '.join(vanished)}")

    if not dry_run:
        _write_aggregates(outputs, index)
    return {"n_systems_collected": n_samples,
            "n_baselines_ranked": len(now_baselines),
            "n_baseline_mode_budget_groups": len(index),
            "outputs": str(outputs)}


def _macro_avg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n_samples": len(rows),
                           "n_scored": sum(1 for r in rows if r.get("ok"))}
    for c in _METRIC_COLS:
        vals = [r[c] for r in rows if r.get("ok") and isinstance(r.get(c), (int, float))]
        agg[c] = round(mean(vals), 6) if vals else None
    return agg


def _write_aggregates(outputs: Path,
                      index: dict[tuple[str, str, int | None], dict[str, list[dict[str, Any]]]]) -> None:
    leaderboard: list[dict[str, Any]] = []
    per_baseline: dict[str, dict[str, Any]] = {}

    for (baseline, input_mode, budget), by_ds in sorted(index.items(), key=lambda kv: kv[0]):
        all_rows = [r for rows in by_ds.values() for r in rows]
        overall = _macro_avg(all_rows)
        entry = {
            "baseline": baseline,
            "input_mode": input_mode,
            "budget": budget,
            "overall": overall,
            "by_dataset": {ds: _macro_avg(rows) for ds, rows in sorted(by_ds.items())},
            "samples": {ds: sorted(r["sample"] for r in rows) for ds, rows in sorted(by_ds.items())},
        }
        per_baseline.setdefault(baseline, {"baseline": baseline, "runs": []})["runs"].append(entry)
        leaderboard.append({
            "baseline": baseline, "input_mode": input_mode, "budget": budget,
            **{c: overall.get(c) for c in _METRIC_COLS},
            "n_samples": overall["n_samples"], "n_scored": overall["n_scored"],
        })

    for baseline, payload in per_baseline.items():
        bdir = outputs / baseline
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "aggregate.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (bdir / "aggregate.md").write_text(_baseline_md(payload), encoding="utf-8")

    leaderboard.sort(key=lambda r: (r.get(_RANK_KEY) is None, -(r.get(_RANK_KEY) or 0.0)))
    board = {"metric": "visual-coverage-v2", "headline": _RANK_KEY, "rows": leaderboard}
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "leaderboard.json").write_text(
        json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    (outputs / "leaderboard.md").write_text(_leaderboard_md(leaderboard), encoding="utf-8")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _leaderboard_md(rows: list[dict[str, Any]]) -> str:
    hdr = "| # | baseline | mode | B | f1 | recall | prec | redun_vlm | redun_sim | sel_eff | budget | retr_ms | write_ms | score_ms | n |"
    al = "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = ["# Track A — visual-coverage-v2 leaderboard (aggregated)", "",
             "Headline = **f1** (precision × continuity-recall), macro-averaged over samples. "
             "One row per (baseline, input_mode, budget). See `docs/trackA.md`.", "", hdr, al]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['baseline']} | {r['input_mode']} | {_fmt(r.get('budget'))} | "
            f"{_fmt(r.get('f1_mean'))} | {_fmt(r.get('recall_mean'))} | {_fmt(r.get('precision_mean'))} | "
            f"{_fmt(r.get('redundancy_vlm_mean'))} | {_fmt(r.get('redundancy_sim_mean'))} | "
            f"{_fmt(r.get('selection_efficiency_mean'))} | {_fmt(r.get('budget_avg_refs_per_chunk_with_refs'))} | "
            f"{_fmt(r.get('retrieval_ms_mean'))} | {_fmt(r.get('write_ms_mean'))} | {_fmt(r.get('score_ms_mean'))} | "
            f"{_fmt(r.get('n_samples'))} |")
    lines.append("")
    return "\n".join(lines)


def _baseline_md(payload: dict[str, Any]) -> str:
    lines = [f"# {payload['baseline']} — Track A aggregate", ""]
    for run in payload["runs"]:
        b = "" if run["budget"] is None else f" · B{run['budget']}"
        lines.append(f"## input_mode={run['input_mode']}{b}")
        ov = run["overall"]
        lines.append(f"- overall (n={ov['n_samples']}, scored={ov['n_scored']}): "
                     f"f1={_fmt(ov.get('f1_mean'))} recall={_fmt(ov.get('recall_mean'))} "
                     f"prec={_fmt(ov.get('precision_mean'))} sel_eff={_fmt(ov.get('selection_efficiency_mean'))} "
                     f"retr_ms={_fmt(ov.get('retrieval_ms_mean'))} "
                     f"write_ms={_fmt(ov.get('write_ms_mean'))} "
                     f"score_ms={_fmt(ov.get('score_ms_mean'))}")
        for ds, agg in run["by_dataset"].items():
            lines.append(f"  - {ds} (n={agg['n_samples']}): f1={_fmt(agg.get('f1_mean'))} "
                         f"recall={_fmt(agg.get('recall_mean'))} prec={_fmt(agg.get('precision_mean'))} "
                         f"retr_ms={_fmt(agg.get('retrieval_ms_mean'))} "
                         f"write_ms={_fmt(agg.get('write_ms_mean'))} "
                         f"score_ms={_fmt(agg.get('score_ms_mean'))}")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=_BENCH / "assets" / "trackA",
                    help="kept for CLI compatibility; Stage-2 aggregation scans --outputs")
    ap.add_argument("--outputs", type=Path, default=_BENCH / "outputs" / "evaluation" / "trackA",
                    help="destination root (default: outputs/evaluation/trackA)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="only these dataset dirs (default: all under data-root)")
    ap.add_argument("--copy-frames", action="store_true",
                    help="also copy each system's _ref_frames/ (large; off by default)")
    ap.add_argument("--dry-run", action="store_true", help="scan + report, write nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = collect(args.data_root.resolve(), args.outputs.resolve(),
                     datasets=args.datasets, copy_frames=args.copy_frames, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["n_systems_collected"] == 0:
        print("[aggregate_trackA] no scored systems found under "
              f"{args.outputs} — run Stage 1 (runner.py) + Stage 2 (visual_coverage) first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
