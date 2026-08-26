#!/usr/bin/env python3
"""Migrate a per-movie annotation directory from the legacy layout to the new scheme.

Legacy:  layout/ (chunk_index.json, boundaries.csv), build/, derived/ (candidates,
         frames, clips, assets), gold/ (unchanged)
New:     gold/ (chunk_index.json, shot_boundaries.csv, entity_registry.json,
         chunk_annotations.json, embeddings.safetensors), assets/, review.html, tmp/

Moves performed (idempotent; each move is printed):
  layout/chunk_index.json   -> gold/chunk_index.json
  layout/boundaries.csv     -> gold/shot_boundaries.csv   (RENAME)
  build/                    -> tmp/                        (whole-dir rename)
  derived/candidates        -> tmp/candidates
  derived/frames            -> tmp/frames
  derived/clips             -> tmp/clips
  derived/assets            -> assets/                     (if not already present)
Emptied layout/ and derived/ directories are removed.

Refuses to overwrite: when a source and its target both exist with different sizes,
the script aborts without moving anything else.

Usage:
    python scripts/vmem_bench/maintenance/migrate_flat_layout.py --movie-dir <root> [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _same_size(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size
    except OSError:
        return False


def _move_file(src: Path, dst: Path, *, dry_run: bool) -> bool:
    """Move one file; return True if a move happened. Raises on conflicting target."""
    if not src.is_file():
        return False
    if dst.is_file():
        if _same_size(src, dst):
            print(f"skip (target exists, same size): {src} -> {dst}")
            if not dry_run:
                src.unlink()
            return False
        raise SystemExit(f"REFUSING: both exist with different sizes: {src} vs {dst}")
    print(f"move: {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True


def _move_dir(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Merge-move a directory file-by-file (safe when dst already exists)."""
    if not src.is_dir():
        return
    if not dst.exists():
        print(f"move dir: {src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return
    for f in sorted(src.rglob("*")):
        if f.is_file():
            _move_file(f, dst / f.relative_to(src), dry_run=dry_run)
    _remove_if_empty(src, dry_run=dry_run)


def _remove_if_empty(d: Path, *, dry_run: bool) -> None:
    if not d.is_dir():
        return
    if any(d.rglob("*")):
        return
    print(f"remove empty dir: {d}")
    if not dry_run:
        shutil.rmtree(d, ignore_errors=True)


def migrate(root: Path, *, dry_run: bool = False) -> None:
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    gold = root / "gold"
    if not dry_run:
        gold.mkdir(parents=True, exist_ok=True)

    # layout/ files into gold/ (boundaries.csv is renamed).
    _move_file(root / "layout" / "chunk_index.json", gold / "chunk_index.json",
               dry_run=dry_run)
    _move_file(root / "layout" / "boundaries.csv", gold / "shot_boundaries.csv",
               dry_run=dry_run)
    _remove_if_empty(root / "layout", dry_run=dry_run)

    # build/ -> tmp/ (whole dir; merge if tmp/ already exists).
    _move_dir(root / "build", root / "tmp", dry_run=dry_run)

    # derived/{candidates,frames,clips} -> tmp/; derived/assets -> assets/.
    derived = root / "derived"
    for name in ("candidates", "frames", "clips"):
        _move_dir(derived / name, root / "tmp" / name, dry_run=dry_run)
    if (derived / "assets").is_dir():
        _move_dir(derived / "assets", root / "assets", dry_run=dry_run)
    _remove_if_empty(derived, dry_run=dry_run)

    print(f"done ({'dry-run' if dry_run else 'migrated'}): {root}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate a legacy MemStrata movie dir to the gold/+tmp/ layout.")
    parser.add_argument("--movie-dir", type=Path, required=True, help="movie root dir")
    parser.add_argument("--dry-run", action="store_true", help="print moves, change nothing")
    args = parser.parse_args()
    migrate(args.movie_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
