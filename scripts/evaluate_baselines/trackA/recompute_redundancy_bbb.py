#!/usr/bin/env python3
"""Fill the DINOv3 ``redundancy_sim`` column for the flat TrackA output layout.

``recompute_redundancy_sim.py`` assumes the old ``movie/benchmark_run/_visual_score``
layout. The current fan-out writes the flat layout::

    outputs/evaluation/trackA/<system>/<dataset>/<movie>/_visual_score/<system>/{score,details}.json

This driver walks that layout for the given systems/dataset and, for every score.json
whose ``summary.redundancy_sim_mean`` is null, recomputes it from the persisted
``details.json`` (grouping + crops) with the pinned DINOv3 embedder. The math is byte-for-byte
the same as ``score_segment`` (visual_coverage.py), so the number equals a full re-score,
WITHOUT the VLM. Movies that are null because no chunk has a >=2 same-entity present group are
left null (nothing to measure). Idempotent: pass --force to also recompute already-filled ones.

Weights resolve via the repo-local fallback in ``resolve_pinned_weights`` (no env needed);
set CUDA_VISIBLE_DEVICES to pick the GPU. Run::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
      python scripts/evaluate_baselines/trackA/recompute_redundancy_bbb.py \
        --dataset BlenderOpenMovies
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

_BENCH = Path(__file__).resolve().parents[3]
_SRC = _BENCH / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vmem_bench.scoring.embedder import build_scoring_embedder, resolve_pinned_weights  # noqa: E402
from vmem_bench.scoring.visual_coverage import _mean_pairwise_sim  # noqa: E402

IN_SCOPE = [
    "memstrata__B16", "longlive_rag__B16", "memflow__B16",
    "retrieval_frame_text_ablation__B16", "retrieval_seg_framererank_ablation__B16",
    "retrieval_seg_uniform_ablation__B16", "retrieval_seg_dinokey_ablation__B16",
]


def _chunk_sim(refs: list[dict], emb) -> float | None:
    """Replicate score_segment's per-entity DINOv3 self-similarity aggregation."""
    present = [r for r in refs if r.get("pred_present") is True]
    groups: dict[str, list[str]] = {}
    for idx, r in enumerate(present):
        e = r.get("pred_entity")
        key = e if (e and e != "none") else f"__none_{idx}"
        groups.setdefault(key, []).append(str(r.get("crop", "")))
    sim_num, sim_pairs = 0.0, 0
    for key, imgs in groups.items():
        if len(imgs) >= 2 and not key.startswith("__none_"):
            msim = _mean_pairwise_sim(imgs, emb)
            if msim is not None:
                pairs = len(imgs) * (len(imgs) - 1) // 2
                sim_num += msim * pairs
                sim_pairs += pairs
    return round(sim_num / sim_pairs, 4) if sim_pairs else None


def process_movie(vs_sys_dir: Path, emb, *, force: bool) -> str:
    sc_p, det_p = vs_sys_dir / "score.json", vs_sys_dir / "details.json"
    if not sc_p.is_file():
        return "no_score"
    sc = json.loads(sc_p.read_text(encoding="utf-8"))
    if sc.get("summary", {}).get("redundancy_sim_mean") is not None and not force:
        return "already_filled"
    if not det_p.is_file():
        return "no_details"
    details = json.loads(det_p.read_text(encoding="utf-8"))
    per_chunk_sim: dict[int, float | None] = {}
    vals: list[float] = []
    for det in details:
        cid = int(det["chunk_id"])
        refs = det.get("refs", [])
        sim = _chunk_sim(refs, emb) if refs else None
        per_chunk_sim[cid] = sim
        if refs and sim is not None:
            vals.append(sim)
    mean_sim = round(mean(vals), 4) if vals else None
    sc["summary"]["redundancy_sim_mean"] = mean_sim
    for row in sc.get("per_chunk", []):
        cid = int(row["chunk_id"])
        if per_chunk_sim.get(cid) is not None:
            row["redundancy_sim"] = per_chunk_sim[cid]
    sc_p.write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"filled={mean_sim}" if mean_sim is not None else "no_pairs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path,
                    default=_BENCH / "outputs" / "evaluation" / "trackA")
    ap.add_argument("--dataset", default="BlenderOpenMovies")
    ap.add_argument("--systems", nargs="*", default=IN_SCOPE)
    ap.add_argument("--force", action="store_true",
                    help="recompute even already-filled score.json")
    args = ap.parse_args()

    emb = build_scoring_embedder("dinov3")
    print(f"embedder: {getattr(emb, 'name', emb)} | resolved_weights={resolve_pinned_weights()!r}",
          flush=True)

    tally: Counter[str] = Counter()
    for system in args.systems:
        base = args.outputs / system / args.dataset
        if not base.is_dir():
            print(f"[skip] {system}: no {args.dataset} dir", flush=True)
            continue
        movies = sorted(p for p in base.iterdir() if p.is_dir())
        for mv in movies:
            vs = mv / "_visual_score" / system
            status = process_movie(vs, emb, force=args.force)
            tally[status.split("=")[0]] += 1
            if status.startswith("filled"):
                print(f"[fill] {system}/{mv.name}: redundancy_sim_mean={status.split('=')[1]}",
                      flush=True)
    print(f"\n[recompute_redundancy_bbb] done: {dict(tally)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
