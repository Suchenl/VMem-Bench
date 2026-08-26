#!/usr/bin/env python3
"""Print which Track A source videos are present vs missing.

Stats expected paths only. Does not recurse video trees (no find/du).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CAUSAL = (
    Path(__file__).resolve().parent
    / "evaluate_baselines"
    / "trackA"
    / "baseline_adapters"
    / "causal"
)
sys.path.insert(0, str(CAUSAL))
from _local_roots import BENCH_ROOT, expand_dataset_root  # noqa: E402

EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
ASSETS = BENCH_ROOT / "assets" / "trackA"


def _movie_ids(dataset: str) -> list[str]:
    root = ASSETS / dataset
    if not root.is_dir():
        return []
    ids = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "gold" / "entity_registry.json").is_file():
            ids.append(p.name)
    return ids


def _first_video_in_dir(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    vids = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    return vids[0] if vids else None


def _flat_video(root: Path, movie_id: str) -> Path | None:
    if not root.is_dir():
        return None
    hits = sorted(
        p for p in root.glob(f"{movie_id}.*") if p.suffix.lower() in EXTS
    )
    return hits[0] if hits else None


def main() -> int:
    blender_root = expand_dataset_root("${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos")
    lsmdc_root = expand_dataset_root("${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched")
    env = os.environ.get("VMEM_DATASETS_ROOT", "")
    print("VMem-Bench source-video check")
    print(f"  VMEM_DATASETS_ROOT={env or '(unset → repo data/)'}")
    print(f"  Blender dir={blender_root}")
    print(f"  LSMDC dir  ={lsmdc_root}")
    print()

    missing = 0
    present = 0

    print("BlenderOpenMovies / CC corpus (layout: <root>/<movie_id>/<file>)")
    for mid in _movie_ids("BlenderOpenMovies"):
        hit = _first_video_in_dir(blender_root / mid)
        if hit:
            print(f"  OK  {mid}  {hit}")
            present += 1
        else:
            print(f"  MISSING  {mid}")
            print(f"           expected: {blender_root / mid}/<video>")
            missing += 1

    print()
    print("LSMDC (layout: <root>/<movie_id>.<ext>)")
    for mid in _movie_ids("LSMDC"):
        hit = _flat_video(lsmdc_root, mid)
        if hit:
            print(f"  OK  {mid}  {hit}")
            present += 1
        else:
            print(f"  MISSING  {mid}")
            print(f"           expected: {lsmdc_root / (mid + '.mp4')}")
            missing += 1

    print()
    print(f"present={present}  missing={missing}")
    print("How to obtain files: docs/DATA.md")
    if missing:
        print("Smoke only needs BBB: bash scripts/prepare_blender.sh")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
