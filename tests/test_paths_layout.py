"""MovieDirs layout resolver + migrate_flat_layout end-to-end.

Covers: new-scheme write targets, legacy fallback reads, write=True ignoring legacy,
and the migration script (old layout in -> new layout out; second run is a no-op).
Run: PYTHONPATH=benchmarks/MemStrata/src python3 benchmarks/MemStrata/tests/test_paths_layout.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from vmem_bench.common.paths import (
    MovieDirs, asset_crop_relpath, entity_asset_dir, is_entity_asset_path, movie_root_from)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "get_trackA_assets"
    / "maintenance"
    / "migrate_flat_layout.py"
)


def test_new_scheme_write_targets() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        dirs = MovieDirs(root, write=True)
        assert dirs.legacy is False
        assert dirs.gold == root / "gold"
        assert dirs.registry_json == root / "gold" / "entity_registry.json"
        assert dirs.annotations_json == root / "gold" / "chunk_annotations.json"
        assert dirs.embeddings == root / "gold" / "embeddings.safetensors"
        assert dirs.chunk_index == root / "gold" / "chunk_index.json"
        assert dirs.shot_boundaries == root / "gold" / "shot_boundaries.csv"
        assert dirs.assets == root / "assets"
        assert dirs.review_html == root / "review.html"
        assert dirs.tmp == root / "tmp"
        assert dirs.checkpoint == root / "tmp" / "checkpoint"
        assert dirs.events == root / "tmp" / "events.jsonl"
        assert dirs.candidates == root / "tmp" / "candidates"
        assert dirs.frames == root / "tmp" / "frames"
        assert dirs.clips == root / "tmp" / "clips"
        assert dirs.services_manifest == root / "tmp" / "services.json"
        assert dirs.qa_report == root / "tmp" / "annotation_qa.json"
        assert dirs.auto_review_json == root / "tmp" / "auto_review.json"
        assert dirs.review_queue == root / "tmp" / "review_queue.json"
        assert dirs.auto_review_patch == root / "tmp" / "auto_review_patch.json"
        assert dirs.merge_proposals == root / "tmp" / "merge_proposals.json"
        assert dirs.review_patch_draft == root / "tmp" / "review_patch.draft.json"
        assert dirs.review_patch_applied == root / "tmp" / "review_patch.applied.json"
        dirs.mkdirs()
        assert (root / "gold").is_dir() and (root / "tmp").is_dir() and (root / "assets").is_dir()
        assert (root / "assets" / "characters").is_dir()
        assert (root / "assets" / "props").is_dir()
        assert (root / "assets" / "locations").is_dir()


def test_kind_asset_paths_accept_canonical_and_legacy_layouts() -> None:
    root = Path("/movie/assets")
    assert entity_asset_dir(root, "char_bunny", "character") == root / "characters" / "char_bunny"
    assert asset_crop_relpath("loc_meadow", "location", "cover.jpg") == (
        "assets/locations/loc_meadow/cover.jpg")
    assert is_entity_asset_path("assets/props/prop_apple/c000.jpg", "prop_apple", "prop")
    assert is_entity_asset_path("assets/prop_apple/c000.jpg", "prop_apple", "prop")
    assert is_entity_asset_path("derived/assets/prop_apple/c000.jpg", "prop_apple", "prop")
    assert not is_entity_asset_path("assets/characters/char_bunny/c000.jpg", "prop_apple", "prop")


def _legacy_movie(root: Path) -> None:
    (root / "layout").mkdir(parents=True)
    (root / "layout" / "chunk_index.json").write_text("{}", encoding="utf-8")
    (root / "layout" / "boundaries.csv").write_text("shot_idx,start_frame,last_frame\n",
                                                    encoding="utf-8")
    (root / "gold").mkdir()
    (root / "gold" / "entity_registry.json").write_text("{}", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "events.jsonl").write_text('{"kind":"run_start"}\n', encoding="utf-8")
    (root / "build" / "services.json").write_text("{}", encoding="utf-8")
    (root / "build" / "checkpoint").mkdir()
    (root / "build" / "checkpoint" / "roster.json").write_text("{}", encoding="utf-8")
    (root / "derived" / "candidates").mkdir(parents=True)
    (root / "derived" / "candidates" / "c0.jpg").write_bytes(b"x")
    (root / "derived" / "frames").mkdir()
    (root / "derived" / "frames" / "f0.jpg").write_bytes(b"y")
    (root / "derived" / "assets" / "char_a").mkdir(parents=True)
    (root / "derived" / "assets" / "char_a" / "cover.jpg").write_bytes(b"z")
    (root / "manifest.json").write_text("{}", encoding="utf-8")


def test_legacy_fallback_read() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _legacy_movie(root)
        dirs = MovieDirs(root)  # read mode
        assert dirs.legacy is True
        assert dirs.chunk_index == root / "layout" / "chunk_index.json"
        assert dirs.shot_boundaries == root / "layout" / "boundaries.csv"
        assert dirs.events == root / "build" / "events.jsonl"
        assert dirs.services_manifest == root / "build" / "services.json"
        assert dirs.checkpoint == root / "build" / "checkpoint"
        assert dirs.candidates == root / "derived" / "candidates"
        assert dirs.frames == root / "derived" / "frames"
        assert dirs.assets == root / "derived" / "assets"
        # gold files live in gold/ in both schemes.
        assert dirs.registry_json == root / "gold" / "entity_registry.json"
        # A new-scheme file that exists wins over the legacy one.
        (root / "gold" / "chunk_index.json").write_text("{}", encoding="utf-8")
        assert MovieDirs(root).chunk_index == root / "gold" / "chunk_index.json"


def test_write_true_ignores_legacy() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _legacy_movie(root)
        dirs = MovieDirs(root, write=True)
        assert dirs.legacy is True
        assert dirs.chunk_index == root / "gold" / "chunk_index.json"
        assert dirs.shot_boundaries == root / "gold" / "shot_boundaries.csv"
        assert dirs.events == root / "tmp" / "events.jsonl"
        assert dirs.candidates == root / "tmp" / "candidates"
        assert dirs.assets == root / "assets"


def test_movie_root_from() -> None:
    assert movie_root_from(Path("/a/b/gold")) == Path("/a/b")
    assert movie_root_from(Path("/a/b")) == Path("/a/b")


def test_migration_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _legacy_movie(root)
        run = subprocess.run([sys.executable, str(SCRIPT), "--movie-dir", str(root)],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert (root / "gold" / "chunk_index.json").is_file()
        assert (root / "gold" / "shot_boundaries.csv").is_file()
        assert (root / "gold" / "entity_registry.json").is_file()
        assert (root / "tmp" / "events.jsonl").is_file()
        assert (root / "tmp" / "services.json").is_file()
        assert (root / "tmp" / "checkpoint" / "roster.json").is_file()
        assert (root / "tmp" / "candidates" / "c0.jpg").is_file()
        assert (root / "tmp" / "frames" / "f0.jpg").is_file()
        assert (root / "assets" / "char_a" / "cover.jpg").is_file()
        assert not (root / "layout").exists()
        assert not (root / "build").exists()
        assert not (root / "derived").exists()
        # Migrated dir is no longer detected as legacy.
        assert MovieDirs(root).legacy is False
        # Second run is a no-op.
        rerun = subprocess.run([sys.executable, str(SCRIPT), "--movie-dir", str(root)],
                               capture_output=True, text=True)
        assert rerun.returncode == 0, rerun.stderr
        assert "move" not in rerun.stdout, rerun.stdout
        assert (root / "gold" / "chunk_index.json").is_file()


def test_migration_dry_run_changes_nothing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _legacy_movie(root)
        run = subprocess.run([sys.executable, str(SCRIPT), "--movie-dir", str(root), "--dry-run"],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert "move" in run.stdout
        assert (root / "layout" / "boundaries.csv").is_file()
        assert (root / "build" / "events.jsonl").is_file()
        assert not (root / "tmp").exists()


def main() -> int:
    test_new_scheme_write_targets()
    test_kind_asset_paths_accept_canonical_and_legacy_layouts()
    test_legacy_fallback_read()
    test_write_true_ignores_legacy()
    test_movie_root_from()
    test_migration_end_to_end()
    test_migration_dry_run_changes_nothing()
    print("test_paths_layout: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
