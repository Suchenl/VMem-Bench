#!/usr/bin/env python3
"""Build frozen MemStrata gold from the **S4 human-reviewed** annotation.

Why this exists (and why NOT ``build_gold_from_vlm_output.py``): the sibling script builds gold
straight from the raw S1 ``vlm_output.json`` (the web-MLLM's first draft), which bypasses the
S2 postprocess / S3 auto-review / **S4 human review** gates. The authoritative gold must be the
**S4 human-reviewed** segment annotation, not the S1 draft. This driver reads
``tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json`` and runs the SAME
deterministic segment->gold conversion (``build_gold_from_segments``), so the roster + per-chunk
presence come from the reviewed truth.

It is an annotation / gold-maintenance driver (``scripts/vmem_bench/``): imports only
``vmem_bench`` and never ``memstrata`` (benchmarks/MemStrata/AGENTS.md Rule 2).

Output is the standard 3-file gold, each single-purpose (no duplicated rows):
``gold/chunk_index.json`` = thin LAYOUT (chunk_id + shot/frame/seconds spans + layout_hash);
``gold/chunk_annotations.json`` = rich per-chunk GT (present / first_appearances / prompt / ...),
which the visual-coverage scorer reads; ``gold/entity_registry.json`` = the entity roster.

Usage::

    cd benchmarks/MemStrata
    PYTHONPATH=src python scripts/vmem_bench/maintenance/build_gold_from_s4_review.py \
        --movie-dir data/BlenderOpenMovies/big_buck_bunny

By default it writes into ``<movie>/gold/`` and, on the first run, moves any pre-existing ``gold/``
aside to ``gold.pre_s4_bak/`` (reversible; never silently clobbered). Use ``--out-dir`` to write
elsewhere, or ``--no-backup`` to overwrite the three gold JSONs in place without archiving.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# S4 human-reviewed annotation, relative to the movie dir.
S4_REL = Path("tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json")
PIPELINE_STATE_REL = Path("tmp/pipeline/state.json")


def _load_s4_annotation(movie_dir: Path, override: Path | None) -> tuple[dict[str, Any], Path]:
    path = override or (movie_dir / S4_REL)
    if not path.is_file():
        raise FileNotFoundError(
            f"S4 human-reviewed annotation not found: {path}\n"
            f"(run the S4 segment-sampling human review first, or pass --annotation)")
    return json.loads(path.read_text(encoding="utf-8")), path


def _check_s4_reviewed(movie_dir: Path) -> str:
    """Return the S4 review status from the pipeline state (advisory; warns if not reviewed)."""
    state_path = movie_dir / PIPELINE_STATE_REL
    if not state_path.is_file():
        return "unknown (no tmp/pipeline/state.json)"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    s4 = (state.get("stages") or {}).get("s4_segment_sampling_human_review") or {}
    return str(s4.get("status", "unknown"))


def _present_by_chunk(chunk_index: dict[str, Any]) -> dict[int, list[str]]:
    return {int(c["chunk_id"]): sorted(c.get("present") or []) for c in chunk_index.get("chunks", [])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movie-dir", required=True, type=Path)
    ap.add_argument("--movie-id", default=None, help="default: movie dir name")
    ap.add_argument("--annotation", type=Path, default=None,
                    help="override path to the S4 human_revised_annotation.json")
    ap.add_argument("--model", default="s4-human-review",
                    help="annotation_provenance.model tag")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="gold output dir (default: <movie>/gold)")
    ap.add_argument("--no-backup", action="store_true",
                    help="overwrite the 3 gold JSONs in place, do NOT archive the old gold/")
    args = ap.parse_args(argv)

    from vmem_bench.annotation.pipeline_vlm_dominant.postprocess_segments import (
        build_gold_from_segments,
    )
    from vmem_bench.common.schemas import SCHEMA_VERSION

    movie_dir: Path = args.movie_dir.resolve()
    movie_id = args.movie_id or movie_dir.name
    out_dir: Path = (args.out_dir or (movie_dir / "gold")).resolve()

    s4_status = _check_s4_reviewed(movie_dir)
    if s4_status != "human_reviewed":
        print(f"warning: S4 stage status is {s4_status!r} (expected 'human_reviewed'); "
              f"proceeding with the annotation on disk anyway", file=sys.stderr)

    s4, s4_path = _load_s4_annotation(movie_dir, args.annotation)
    print(f"S4 annotation: {s4_path}", file=sys.stderr)

    # S4 is human-reviewed -> trust & freeze (human_reviewed=true on the gold files).
    registry, annotations, chunk_index_min = build_gold_from_segments(
        s4, movie_id=movie_id, model_name=args.model, trust_entities=True)

    # chunk_index.json is the thin LAYOUT file (chunk_id + shot/frame/seconds spans + layout_hash).
    # The rich per-chunk GT (present/first_appearances/prompt/...) lives ONLY in chunk_annotations.json
    # to avoid duplicating the same rows in two files; the visual-coverage scorer reads present/prompt
    # from chunk_annotations.json (see visual_coverage._load_gold).
    chunk_index = {
        "schema_version": SCHEMA_VERSION,
        "movie_id": movie_id,
        "layout_hash": chunk_index_min["layout_hash"],
        "fps": chunk_index_min.get("fps"),
        "time_unit": "seconds",
        "source": "s4_human_review",
        "chunks": chunk_index_min["chunks"],
    }

    # Diff vs any pre-existing gold (present-set changes), so the review effect is visible.
    # present now lives in chunk_annotations.json (chunk_index is thin layout), so diff on that.
    old_ca_path = out_dir / "chunk_annotations.json"
    if old_ca_path.is_file():
        try:
            old_present = _present_by_chunk(json.loads(old_ca_path.read_text(encoding="utf-8")))
            new_present = _present_by_chunk(annotations.to_dict())
            changed = [cid for cid in new_present if old_present.get(cid) != new_present.get(cid)]
            print(f"diff vs existing gold: {len(changed)}/{len(new_present)} chunks change 'present'",
                  file=sys.stderr)
            for cid in changed[:8]:
                print(f"  chunk {cid}: {old_present.get(cid)} -> {new_present.get(cid)}",
                      file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - diagnostics only
            print(f"  (could not diff existing gold: {e})", file=sys.stderr)

    # Backup existing gold/ once (unless writing elsewhere or --no-backup).
    if out_dir.is_dir() and not args.no_backup and out_dir.name == "gold":
        bak = out_dir.parent / "gold.pre_s4_bak"
        if not bak.exists():
            shutil.move(str(out_dir), str(bak))
            print(f"archived existing gold -> {bak}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entity_registry.json").write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "chunk_annotations.json").write_text(
        json.dumps(annotations.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "chunk_index.json").write_text(
        json.dumps(chunk_index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote S4 gold -> {out_dir}", file=sys.stderr)
    print(f"  entities={len(registry.entities)} chunks={len(annotations.chunks)} "
          f"layout_hash={chunk_index['layout_hash']} human_reviewed={registry.human_reviewed} "
          f"source=s4_human_review", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
