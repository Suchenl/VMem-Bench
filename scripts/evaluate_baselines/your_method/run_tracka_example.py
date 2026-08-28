"""Drive the Track A memory loop for YOUR method and emit a scorer-ready selection.

Runs the strict per-segment protocol (compose from current memory, THEN observe the
real segment) over a bundled/frozen gold movie, and writes
``visual_selections/<system>.json`` — the exact file
``vmem_bench.scoring.visual_coverage`` reads.

Example (CPU, bundled gold, placeholder frames)::

    python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
        --movie-dir assets/trackA/BlenderOpenMovies/charge \
        --limit 5

Then score (needs a VLM judge endpoint + the real source video; see README)::

    PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage \
        --movie assets/trackA/BlenderOpenMovies/charge \
        --system your_method-recency \
        --video /path/to/charge.mp4 --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from example_method import build  # noqa: E402
from sut_interface import Reference, load_segments, write_tracka_selections  # noqa: E402


def run(movie_dir: Path, system: str, out_path: Path, mem_dir: Path,
        budget: int, limit: int | None, ffmpeg: str) -> Path:
    segments = load_segments(movie_dir, limit=limit)
    method = build(mem_dir=mem_dir, budget=budget, ffmpeg=ffmpeg)
    per_segment: list[tuple[int, list[Reference]]] = []
    for seg in segments:
        refs = list(method.compose(seg))   # recall from memory built by PRIOR segments
        per_segment.append((seg.chunk_id, refs))
        method.observe(seg)                # THEN write this segment into memory
        print(f"chunk {seg.chunk_id:>3}: composed {len(refs)} ref(s)")
    written = write_tracka_selections(out_path, system, per_segment)
    print(f"\nwrote {written}")
    print("score with:\n"
          f"  PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage \\\n"
          f"      --movie {movie_dir} --system {system} --video <source_video.mp4>")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track A example driver for an external method")
    ap.add_argument("--movie-dir", type=Path,
                    default=Path("assets/trackA/BlenderOpenMovies/charge"),
                    help="a gold movie dir containing gold/chunk_annotations.json")
    ap.add_argument("--system", default="your_method-recency",
                    help="visual_selections/<system>.json basename")
    ap.add_argument("--out", type=Path, default=None,
                    help="output json path (default: <movie>/benchmark_run/visual_selections/<system>.json)")
    ap.add_argument("--mem-dir", type=Path, default=None,
                    help="where the example method stores its frames (default: <out_dir>/_mem)")
    ap.add_argument("--budget", type=int, default=3, help="max refs recalled per segment")
    ap.add_argument("--limit", type=int, default=None, help="only drive the first N segments")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args(argv)

    out_path = a.out or (a.movie_dir / "benchmark_run" / "visual_selections" / f"{a.system}.json")
    mem_dir = a.mem_dir or (out_path.parent / "_mem")
    run(a.movie_dir, a.system, out_path, mem_dir, a.budget, a.limit, a.ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
