#!/usr/bin/env python
"""Offline recompute of the NON-VLM redundancy column (``redundancy_sim``).

The VLM judge pass (``scoring.visual_coverage``) already persisted, per chunk, which
reference image was judged present and which gold entity it maps to
(``_visual_score/<system>/details.json``). ``redundancy_sim`` only needs that grouping
plus the pinned DINOv3 embedder -- it does NOT need the VLM. So we can fill the column
for runs that were scored while the DINOv3 embedder was unavailable, WITHOUT re-running
the judge. The grouping and the pair-weighted mean here are byte-for-byte the same logic
as ``score_chunk`` (visual_coverage.py), so the number is identical to a full re-score.

Usage:
  PUBLIC_MODELS_ROOT=/.../_public_models \
  python recompute_redundancy_sim.py --movie <movie_dir> [--systems a b c]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_SRC = Path(__file__).resolve().parents[3] / "src"  # trackA/../../../ -> repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vmem_bench.scoring.embedder import build_scoring_embedder  # noqa: E402
from vmem_bench.scoring.visual_coverage import _mean_pairwise_sim  # noqa: E402


def _chunk_sim(refs: list[dict], emb, movie: Path) -> float | None:
    """Replicate score_chunk's per-entity DINOv3 self-similarity aggregation."""
    present = [r for r in refs if r.get("pred_present") is True]
    groups: dict[str, list[str]] = {}
    for idx, r in enumerate(present):
        e = r.get("pred_entity")
        key = e if (e and e != "none") else f"__none_{idx}"
        p = r.get("crop", "")
        # crop paths were written absolute by the scorer; fall back to movie-relative.
        pp = Path(p)
        if not pp.is_absolute() and not pp.exists():
            pp = (movie / p)
        groups.setdefault(key, []).append(str(pp))
    sim_num, sim_pairs = 0.0, 0
    for key, imgs in groups.items():
        if len(imgs) >= 2 and not key.startswith("__none_"):
            msim = _mean_pairwise_sim(imgs, emb)
            if msim is not None:
                pairs = len(imgs) * (len(imgs) - 1) // 2
                sim_num += msim * pairs
                sim_pairs += pairs
    return round(sim_num / sim_pairs, 4) if sim_pairs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True, type=Path)
    ap.add_argument("--systems", nargs="*", default=None,
                    help="default: every dir under benchmark_run/_visual_score")
    ap.add_argument("--write", action="store_true", default=True,
                    help="write recomputed redundancy_sim back into score.json (default on)")
    args = ap.parse_args()

    movie = args.movie.resolve()
    vs_root = movie / "benchmark_run" / "_visual_score"
    systems = args.systems or sorted(p.name for p in vs_root.iterdir() if p.is_dir())

    emb = build_scoring_embedder("dinov3")
    print(f"embedder: {getattr(emb, 'name', emb)}\n")

    print(f"{'system':<16} {'chunks_w_refs':>13} {'redundancy_sim_mean':>20}")
    print("-" * 52)
    for sysname in systems:
        det_p = vs_root / sysname / "details.json"
        sc_p = vs_root / sysname / "score.json"
        if not det_p.exists():
            print(f"{sysname:<16} {'--':>13} {'(no details.json)':>20}")
            continue
        details = json.loads(det_p.read_text())
        per_chunk_sim: dict[int, float | None] = {}
        vals = []
        for det in details:
            cid = int(det["chunk_id"])
            refs = det.get("refs", [])
            sim = _chunk_sim(refs, emb, movie) if refs else None
            per_chunk_sim[cid] = sim
            if refs and sim is not None:
                vals.append(sim)
        mean_sim = round(mean(vals), 4) if vals else None
        print(f"{sysname:<16} {len(vals):>13} {str(mean_sim):>20}")

        if args.write and sc_p.exists():
            sc = json.loads(sc_p.read_text())
            sc["summary"]["redundancy_sim_mean"] = mean_sim
            for row in sc.get("per_chunk", []):
                cid = int(row["chunk_id"])
                if cid in per_chunk_sim and per_chunk_sim[cid] is not None:
                    row["redundancy_sim"] = per_chunk_sim[cid]
            sc_p.write_text(json.dumps(sc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
