#!/usr/bin/env python3
"""Build the v2 (visual-coverage) leaderboard from per-system VLM-judge scorecards.

Consumes the scores written by ``vmem_bench.scoring.visual_coverage`` at
``<movie>/benchmark_run/_visual_score/<system>/score.json`` and emits, per movie:

  benchmark_run/leaderboard_v2.json   ranked rows (one per system) + provenance
  benchmark_run/leaderboard_v2.md     human-readable table

With ``--all-blender`` it also writes a corpus-level macro-average across movies to
``data/BlenderOpenMovies/_leaderboard_v2_corpus.json`` / ``.md``.

This REPLACES the v1 ID/embedder leaderboard for headline reporting (see
``docs/benchmark/scoring_v2.md``). It does NOT run any model — scoring is a separate pass;
this tool only aggregates already-written scorecards. Systems without a score.json are listed
as ``missing`` so gaps are visible rather than silently dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BLENDER = ROOT / "data" / "BlenderOpenMovies"

# columns pulled from each score.json "summary" block (headline first)
COLS = [
    "f1_mean",              # HEADLINE
    "recall_mean",          # HEADLINE (continuity / memory recall)
    "precision_mean",
    "redundancy_vlm_mean",
    "redundancy_sim_mean",
    "efficiency_mean",
    "budget_avg_refs_per_chunk_with_refs",
    "n_chunks",
    "n_chunks_with_refs",
]
RANK_KEY = "f1_mean"


def _rows_for_movie(movie_dir: Path) -> list[dict[str, Any]]:
    vsdir = movie_dir / "benchmark_run" / "_visual_score"
    rows: list[dict[str, Any]] = []
    if not vsdir.is_dir():
        return rows
    for sysdir in sorted(p for p in vsdir.iterdir() if p.is_dir()):
        sp = sysdir / "score.json"
        if not sp.is_file():
            rows.append({"system": sysdir.name, "ok": False, "note": "no score.json"})
            continue
        try:
            summ = json.loads(sp.read_text()).get("summary", {})
        except Exception as exc:  # noqa: BLE001
            rows.append({"system": sysdir.name, "ok": False, "note": f"bad json: {exc}"})
            continue
        row = {"system": sysdir.name, "ok": True,
               "metric_version": summ.get("metric_version")}
        for c in COLS:
            row[c] = summ.get(c)
        rows.append(row)
    rows.sort(key=lambda r: (r.get(RANK_KEY) is None, -(r.get(RANK_KEY) or 0)))
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_md(title: str, rows: list[dict[str, Any]], path: Path) -> None:
    hdr = "| # | system | f1 | recall | prec | redun_vlm | redun_sim | eff | budget | chunks | note |"
    al = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [f"# {title} — visual-coverage v2 leaderboard", "",
             "Headline = **f1** (precision × continuity-recall). "
             "redun_vlm / redun_sim = two redundancy variants; eff = efficiency; "
             "budget = avg refs/chunk (descriptive). See `docs/benchmark/scoring_v2.md`.", "",
             hdr, al]
    for i, r in enumerate(rows, 1):
        if not r.get("ok"):
            lines.append(f"| | {r['system']} | | | | | | | | | {r.get('note', 'missing')} |")
            continue
        lines.append("| {i} | {s} | {f1} | {rc} | {pr} | {rv} | {rs} | {ef} | {bg} | {ck} | {mv} |".format(
            i=i, s=r["system"], f1=_fmt(r.get("f1_mean")), rc=_fmt(r.get("recall_mean")),
            pr=_fmt(r.get("precision_mean")), rv=_fmt(r.get("redundancy_vlm_mean")),
            rs=_fmt(r.get("redundancy_sim_mean")), ef=_fmt(r.get("efficiency_mean")),
            bg=_fmt(r.get("budget_avg_refs_per_chunk_with_refs")), ck=_fmt(r.get("n_chunks")),
            mv=r.get("metric_version") or ""))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_movie(movie_dir: Path) -> dict[str, Any]:
    rows = _rows_for_movie(movie_dir)
    out = movie_dir / "benchmark_run"
    board = {"movie_id": movie_dir.name, "metric": "visual-coverage-v2",
             "headline": RANK_KEY, "rows": rows}
    if out.is_dir():
        (out / "leaderboard_v2.json").write_text(
            json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_md(movie_dir.name, rows, out / "leaderboard_v2.md")
    return board


def build_corpus(movies: list[Path]) -> dict[str, Any]:
    """Macro-average each metric per system across movies where the system was scored."""
    from collections import defaultdict
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[str, int] = defaultdict(int)
    for md in movies:
        for r in _rows_for_movie(md):
            if not r.get("ok"):
                continue
            seen[r["system"]] += 1
            for c in COLS:
                v = r.get(c)
                if isinstance(v, (int, float)):
                    acc[r["system"]][c].append(float(v))
    rows = []
    for sysname, cols in acc.items():
        row = {"system": sysname, "ok": True, "n_movies": seen[sysname]}
        for c in COLS:
            row[c] = round(mean(cols[c]), 4) if cols[c] else None
        rows.append(row)
    rows.sort(key=lambda r: (r.get(RANK_KEY) is None, -(r.get(RANK_KEY) or 0)))
    board = {"corpus": "BlenderOpenMovies", "metric": "visual-coverage-v2",
             "headline": RANK_KEY, "n_movies": len(movies), "rows": rows}
    (BLENDER / "_leaderboard_v2_corpus.json").write_text(
        json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md("BlenderOpenMovies (corpus macro-avg)", rows, BLENDER / "_leaderboard_v2_corpus.md")
    return board


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--movie-dir", type=Path)
    g.add_argument("--all-blender", action="store_true")
    args = ap.parse_args(argv)
    movies = ([args.movie_dir] if args.movie_dir
              else sorted(p.parent.parent for p in BLENDER.glob("*/benchmark_run/_visual_score")))
    for md in movies:
        b = build_movie(md)
        n_ok = sum(1 for r in b["rows"] if r.get("ok"))
        print(f"[{md.name}] {n_ok}/{len(b['rows'])} systems scored "
              f"-> {md / 'benchmark_run/leaderboard_v2.json'}")
    if args.all_blender:
        c = build_corpus(movies)
        print(f"[corpus] {len(c['rows'])} systems across {c['n_movies']} movies "
              f"-> {BLENDER / '_leaderboard_v2_corpus.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
