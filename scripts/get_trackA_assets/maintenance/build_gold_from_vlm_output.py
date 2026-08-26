#!/usr/bin/env python3
"""Build frozen MemStrata gold from a web-MLLM ``vlm_output.json`` (video-free), and write the three
gold files into ``<movie-dir>/gold/``.

This is an annotation / gold-maintenance driver (``scripts/vmem_bench/``): it imports only
``vmem_bench`` and never ``memstrata`` (benchmarks/MemStrata/AGENTS.md Rule 2).

Usage:

    cd benchmarks/MemStrata
    PYTHONPATH=src python scripts/vmem_bench/maintenance/build_gold_from_vlm_output.py \
        --vlm-output data/BlenderOpenMovies/big_buck_bunny_720p/vlm_output.json \
        --movie-dir  data/BlenderOpenMovies/big_buck_bunny_720p \
        --movie-id   big_buck_bunny [--model qwen3-vl-web] [--candidate]

``--candidate`` emits an unfrozen candidate (human_reviewed=false) for the normal review gate;
the default trusts the VLM entity table + segment presence and freezes (human_reviewed=true).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm-output", required=True, type=Path)
    ap.add_argument("--movie-dir", required=True, type=Path)
    ap.add_argument("--movie-id", required=True)
    ap.add_argument("--model", default="qwen3-vl-web")
    ap.add_argument("--candidate", action="store_true",
                    help="emit unfrozen candidate (human_reviewed=false) instead of trust-freeze")
    args = ap.parse_args(argv)

    from vmem_bench.annotation.pipeline_vlm_dominant.postprocess_segments import (
        build_gold_from_segments,
    )
    from vmem_bench.common.gold_lint import lint_movie_dir, summarize_violations
    from vmem_bench.common.paths import MovieDirs

    vlm = json.loads(Path(args.vlm_output).read_text(encoding="utf-8"))
    registry, annotations, chunk_index = build_gold_from_segments(
        vlm, movie_id=args.movie_id, model_name=args.model, trust_entities=not args.candidate)

    dirs = MovieDirs(Path(args.movie_dir), write=True)
    dirs.mkdirs()
    dirs.registry_json.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    dirs.annotations_json.write_text(
        json.dumps(annotations.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    dirs.chunk_index.write_text(
        json.dumps(chunk_index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote gold: {dirs.registry_json}", file=sys.stderr)
    print(f"  entities={len(registry.entities)} chunks={len(annotations.chunks)} "
          f"layout_hash={chunk_index['layout_hash']} human_reviewed={registry.human_reviewed}",
          file=sys.stderr)

    violations = lint_movie_dir(Path(args.movie_dir), strict_review=not args.candidate)
    summary = summarize_violations(violations)
    print("gold_lint: " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    for v in violations:
        if v.severity == "error":
            print(f"  ERROR [{v.code}] {v.path}: {v.message}", file=sys.stderr)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
