"""Per-movie directory layout resolver (gold/ + assets/ + review.html + tmp/).

New layout (five top-level entries)::

    <movie>/
      manifest.json
      gold/                    # ALL frozen/publishable annotation files
        chunk_index.json           (was layout/chunk_index.json)
        shot_boundaries.csv        (was layout/boundaries.csv — renamed)
        entity_registry.json
        chunk_annotations.json
        embeddings.safetensors
      assets/                  # entity crops + covers
      review.html
      tmp/                     # ALL process files (was build/ + derived/)

Legacy layout (still readable): ``layout/``, ``build/``, ``derived/``.

Writers always use the new scheme (``MovieDirs(root, write=True)``). Readers use
``write=False``: the new path is returned unless it does not exist AND the legacy
one does.
"""

from __future__ import annotations

from pathlib import Path

_ASSET_KIND_DIRS = {
    "character": "characters",
    "prop": "props",
    "location": "locations",
}


def asset_kind_dir(kind: str) -> str:
    """Canonical published asset subdirectory for an entity kind."""
    try:
        return _ASSET_KIND_DIRS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown entity kind for assets: {kind!r}") from exc


def entity_asset_dir(assets_root: Path, entity_id: str, kind: str) -> Path:
    """Directory for one entity in the canonical kind-segregated asset bank."""
    return Path(assets_root) / asset_kind_dir(kind) / entity_id


def entity_asset_relprefix(entity_id: str, kind: str) -> str:
    """Movie-root-relative directory prefix for an entity's canonical assets."""
    return f"assets/{asset_kind_dir(kind)}/{entity_id}/"


def asset_crop_relpath(entity_id: str, kind: str, leaf: str) -> str:
    """Portable, movie-root-relative canonical asset path."""
    return entity_asset_relprefix(entity_id, kind) + Path(leaf).name


def is_entity_asset_path(crop_path: str, entity_id: str, kind: str) -> bool:
    """Accept canonical paths plus flat and derived legacy asset layouts."""
    return crop_path.startswith((
        entity_asset_relprefix(entity_id, kind),
        f"assets/{entity_id}/",
        f"derived/assets/{entity_id}/",
    ))


def movie_root_from(path: Path) -> Path:
    """Accept a movie root or its ``gold/`` subdirectory; return the movie root.

    ``apply_patch`` / ``freeze`` historically took ``<out>/gold``; callers may now
    pass the movie root instead. If ``path.name == "gold"``, use its parent.
    """
    path = Path(path)
    return path.parent if path.name == "gold" else path


class MovieDirs:
    """Resolve publish + process paths for one annotated movie directory."""

    def __init__(self, root: Path, *, write: bool = False) -> None:
        self.root = Path(root)
        self.write = bool(write)
        # Detected once at construction; write=True still forces new-scheme targets.
        self.legacy = ((self.root / "layout").is_dir()
                       or (self.root / "build").is_dir()
                       or (self.root / "derived").is_dir()
                       or (self.root / "layout" / "boundaries.csv").is_file())

    def _pick(self, new: Path, legacy: Path) -> Path:
        if self.write or not self.legacy:
            return new
        if not new.exists() and legacy.exists():
            return legacy
        return new

    # --- publishable (gold/; layout files were legacy layout/) ------------------------------

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def registry_json(self) -> Path:
        return self.root / "gold" / "entity_registry.json"

    @property
    def annotations_json(self) -> Path:
        return self.root / "gold" / "chunk_annotations.json"

    @property
    def embeddings(self) -> Path:
        return self.root / "gold" / "embeddings.safetensors"

    @property
    def chunk_index(self) -> Path:
        return self._pick(self.root / "gold" / "chunk_index.json",
                          self.root / "layout" / "chunk_index.json")

    @property
    def shot_boundaries(self) -> Path:
        return self._pick(self.root / "gold" / "shot_boundaries.csv",
                          self.root / "layout" / "boundaries.csv")

    @property
    def assets(self) -> Path:
        return self._pick(self.root / "assets", self.root / "derived" / "assets")

    @property
    def review_html(self) -> Path:
        return self.root / "review.html"

    # --- process / scratch (tmp/; legacy build/ or derived/) -------------------------------

    @property
    def tmp(self) -> Path:
        return self._pick(self.root / "tmp", self.root / "build")

    @property
    def checkpoint(self) -> Path:
        return self._pick(self.root / "tmp" / "checkpoint",
                          self.root / "build" / "checkpoint")

    @property
    def events(self) -> Path:
        return self._pick(self.root / "tmp" / "events.jsonl",
                          self.root / "build" / "events.jsonl")

    @property
    def candidates(self) -> Path:
        return self._pick(self.root / "tmp" / "candidates",
                          self.root / "derived" / "candidates")

    @property
    def frames(self) -> Path:
        return self._pick(self.root / "tmp" / "frames",
                          self.root / "derived" / "frames")

    @property
    def clips(self) -> Path:
        return self._pick(self.root / "tmp" / "clips",
                          self.root / "derived" / "clips")

    @property
    def services_manifest(self) -> Path:
        return self._pick(self.root / "tmp" / "services.json",
                          self.root / "build" / "services.json")

    @property
    def qa_report(self) -> Path:
        return self._pick(self.root / "tmp" / "annotation_qa.json",
                          self.root / "build" / "annotation_qa.json")

    @property
    def auto_review_json(self) -> Path:
        return self._pick(self.root / "tmp" / "auto_review.json",
                          self.root / "build" / "auto_review.json")

    @property
    def review_queue(self) -> Path:
        """Read-only, reproducible human-review queue derived from process artifacts."""
        return self._pick(self.root / "tmp" / "review_queue.json",
                          self.root / "build" / "review_queue.json")

    @property
    def review_dispositions(self) -> Path:
        """Non-published human decisions for auto-review's must-review queue."""
        return self._pick(self.root / "tmp" / "review_dispositions.json",
                          self.root / "build" / "review_dispositions.json")

    @property
    def state_event_dispositions(self) -> Path:
        return self._pick(self.root / "tmp" / "state_event_dispositions.json",
                          self.root / "build" / "state_event_dispositions.json")

    @property
    def state_event_pairs(self) -> Path:
        return self._pick(self.root / "tmp" / "state_event_review_pairs.jsonl",
                          self.root / "build" / "state_event_review_pairs.jsonl")

    @property
    def auto_review_patch(self) -> Path:
        return self._pick(self.root / "tmp" / "auto_review_patch.json",
                          self.root / "build" / "auto_review_patch.json")

    @property
    def merge_proposals(self) -> Path:
        return self._pick(self.root / "tmp" / "merge_proposals.json",
                          self.root / "build" / "merge_proposals.json")

    @property
    def identity_candidates(self) -> Path:
        """Non-published ranked identity evidence for human review only."""
        return self._pick(self.root / "tmp" / "identity_candidates.json",
                          self.root / "build" / "identity_candidates.json")

    @property
    def review_patch_draft(self) -> Path:
        return self._pick(self.root / "tmp" / "review_patch.draft.json",
                          self.root / "build" / "review_patch.draft.json")

    @property
    def review_patch_applied(self) -> Path:
        return self._pick(self.root / "tmp" / "review_patch.applied.json",
                          self.root / "build" / "review_patch.applied.json")

    def mkdirs(self) -> None:
        """Create gold / tmp / canonical kind-segregated asset directories for writers."""
        (self.root / "gold").mkdir(parents=True, exist_ok=True)
        (self.root / "tmp").mkdir(parents=True, exist_ok=True)
        for kind_dir in _ASSET_KIND_DIRS.values():
            (self.root / "assets" / kind_dir).mkdir(parents=True, exist_ok=True)
