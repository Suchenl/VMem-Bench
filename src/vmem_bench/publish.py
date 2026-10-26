"""Assemble a shippable MemStrata-Bench release from an annotated movie directory.

A release contains only the version-worthy parts — the download recipe, the chunking
layout, the frozen annotations, and the published asset bank — never video bytes.
Everything under ``tmp/`` (legacy ``derived/`` and ``build/``) is re-derivable from the
source video (see ``manifest.json``) and is left out.

Shipped (release is always written in the NEW scheme; legacy movie dirs are accepted):
    manifest.json                 # source download recipe + sha256 + fps/duration + license
    gold/chunk_index.json         # chunking logic (frame spans, closed inclusive) + layout_hash
    gold/shot_boundaries.csv
    gold/entity_registry.json     # frozen annotations (crop_path/frame_index/bbox -> re-derivable)
    gold/chunk_annotations.json
    gold/embeddings.safetensors    # not re-derivable without the exact embedder -> must ship
    assets/                       # published per-entity crop bank (when present)
    SHA256SUMS                     # checksum of every shipped file

Usage:
    python -m vmem_bench.publish --movie-dir <annotated_dir> --out <release_dir>
    python -m vmem_bench.publish --movie-dir <annotated_dir> --out <release_dir> --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from vmem_bench.common.gold_lint import lint_movie_dir
from vmem_bench.common.paths import MovieDirs

# Release-relative target path -> MovieDirs property that resolves the source (new scheme
# with legacy fallback). Directories are shipped whole; files must exist unless optional.
SHIP_FILES = {"manifest.json": None,  # always at the movie root
              "gold/chunk_index.json": "chunk_index",
              "gold/shot_boundaries.csv": "shot_boundaries",
              "gold/entity_registry.json": "registry_json",
              "gold/chunk_annotations.json": "annotations_json"}
SHIP_OPTIONAL = {"gold/embeddings.safetensors": "embeddings"}


def _ship_src(movie_dir: Path, rel: str, prop: str | None) -> Path:
    return movie_dir / rel if prop is None else getattr(MovieDirs(movie_dir), prop)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_frozen(movie_dir: Path) -> list[str]:
    """Return reasons the gold is not publishable (empty list = OK)."""
    problems: list[str] = []
    dirs = MovieDirs(movie_dir)
    for rel, prop in SHIP_FILES.items():
        if not _ship_src(movie_dir, rel, prop).is_file():
            problems.append(f"missing required file: {rel}")
    # schema_version presence + cross-file consistency (contract §0: every JSON top-level carries
    # schema_version; a frozen release must agree across layout + gold files). Catches a stale
    # layout shipped against newer gold (or vice versa) before it reaches the scoring harness
    # (Pitfall_Notes: F7).
    versions: dict[str, str] = {}
    for rel, path in (("gold/chunk_index.json", dirs.chunk_index),
                      ("gold/entity_registry.json", dirs.registry_json),
                      ("gold/chunk_annotations.json", dirs.annotations_json)):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        v = data.get("schema_version")
        if not v:
            problems.append(f"{rel} missing schema_version")
        else:
            versions[rel] = str(v)
    if len(set(versions.values())) > 1:
        problems.append(f"schema_version mismatch across files: {versions}")
    # layout_hash presence (contract §5.3: chunk layout is frozen with gold; the scoring harness
    # checks the run-side layout hash matches gold's — a release without it cannot be validated).
    ci = dirs.chunk_index
    if ci.is_file():
        ci_data = json.loads(ci.read_text(encoding="utf-8"))
        if not ci_data.get("layout_hash"):
            problems.append("gold/chunk_index.json missing layout_hash")
    for name, path in (("entity_registry.json", dirs.registry_json),
                       ("chunk_annotations.json", dirs.annotations_json)):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("human_reviewed", False):
            problems.append(f"gold/{name} is not frozen (human_reviewed=false)")
    if dirs.registry_json.is_file() and dirs.annotations_json.is_file():
        for violation in lint_movie_dir(movie_dir, strict_review=True):
            if violation.severity == "error":
                problems.append(f"gold lint {violation.code}: {violation.message}")
    return problems


def publish(movie_dir: Path, out_dir: Path, *, force: bool = False) -> Path:
    movie_dir = Path(movie_dir).resolve()
    if not movie_dir.is_dir():
        raise FileNotFoundError(f"movie dir does not exist: {movie_dir}")

    problems = _check_frozen(movie_dir)
    blocking = [p for p in problems if p.startswith("missing")]
    unfrozen = [p for p in problems if "not frozen" in p]
    quality = [p for p in problems if p not in blocking and p not in unfrozen]
    if blocking:
        raise SystemExit("cannot publish:\n  " + "\n  ".join(blocking))
    if quality:
        raise SystemExit("cannot publish (gold lint failed):\n  " + "\n  ".join(quality))
    if unfrozen and not force:
        raise SystemExit(
            "cannot publish (gold not human-reviewed):\n  " + "\n  ".join(unfrozen)
            + "\nFreeze it first (vmem_bench.annotation.pipeline_track_first.review.freeze) or pass --force.")

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"output dir already exists (refusing to overwrite): {out_dir}")
    out_dir.mkdir(parents=True)

    shipped: list[Path] = []
    for rel, prop in {**SHIP_FILES, **SHIP_OPTIONAL}.items():
        src = _ship_src(movie_dir, rel, prop)
        if not src.is_file():
            continue  # optional
        dst = out_dir / rel  # release is always the NEW scheme
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        shipped.append(dst)

    assets_src = MovieDirs(movie_dir).assets
    if assets_src.is_dir():
        for src in assets_src.rglob("*"):
            if not src.is_file():
                continue
            rel = Path("assets") / src.relative_to(assets_src)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            shipped.append(dst)

    lines = [f"{_sha256(p)}  {p.relative_to(out_dir)}" for p in sorted(shipped)]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(p.stat().st_size for p in shipped)
    print(f"published {len(shipped)} files ({total / 1e6:.2f} MB) to {out_dir}")
    for p in shipped:
        print(f"  {p.relative_to(out_dir)}")
    if unfrozen:
        print("WARNING: published UNFROZEN gold (--force); not valid for official scoring.")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a shippable MemStrata-Bench release.")
    parser.add_argument("--movie-dir", type=Path, required=True,
                        help="annotated movie dir (contains manifest.json + gold/; "
                             "legacy layout/ dirs accepted)")
    parser.add_argument("--out", type=Path, required=True, help="release output dir (must not exist)")
    parser.add_argument("--force", action="store_true",
                        help="publish even if gold is not human_reviewed (debug only)")
    args = parser.parse_args()
    publish(args.movie_dir, args.out, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
