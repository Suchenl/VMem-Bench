"""Build a shallow, restart-safe catalog of annotation movie inputs.

The catalog reads the authoritative BlenderOpenMovies and stitched LSMDC index
JSON files.  It never recursively walks source dataset roots or LSMDC clips.

Path policy (keep these separate):

* **BlenderOpenMovies** — ``Videos/<movie_id>/<video_file>``: match the folder
  name under ``Videos/``, then take a video file inside (no recursion).
* **LSMDC** — use the stitched ``output_file`` from the LSMDC index only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.orchestration.contracts import MovieManifest

# Default read path = uncompressed sources (same policy as LSMDC_Videos_Stitched).
# Do NOT default to Videos_Compressed_360p_CRF26.
DEFAULT_BLENDER_VIDEOS_ROOT = Path(
    "${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos"
)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _vlm_path(movie_dir: Path) -> Path:
    return movie_dir / "vlm_output.json"


def resolve_blender_source_video(
    *,
    movie_id: str,
    filename: str = "",
    videos_root: Path,
) -> Path:
    """Resolve Blender source as ``Videos/<movie_id>/<video>``.

    1. Match ``videos_root / movie_id`` (folder name = sample ``movie_id``).
    2. Prefer ``filename`` when it exists in that folder.
    3. Otherwise take the first video file directly under the folder.
    4. Legacy flat fallbacks only if the folder is missing.

    LSMDC must not use this helper — it resolves via stitched index paths.
    """
    videos_root = Path(videos_root)
    folder = videos_root / movie_id
    name = Path(str(filename or "").strip()).name

    if folder.is_dir():
        if name:
            preferred = folder / name
            if preferred.is_file():
                return preferred
        videos = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
        )
        if videos:
            return videos[0]
        return folder / (name or f"{movie_id}.mp4")

    # Legacy: flat file under Videos/, or Videos_parent/<id>/<file>.
    if name:
        flat = videos_root / name
        if flat.is_file():
            return flat
        legacy = videos_root.parent / movie_id / name
        if legacy.is_file():
            return legacy
        return videos_root / movie_id / name
    return videos_root / movie_id / f"{movie_id}.mp4"


def build_catalog(
    *,
    data_root: Path,
    blender_index: Path,
    lsmdc_index: Path,
    blender_videos_root: Path | None = None,
) -> list[MovieManifest]:
    """Return manifests for all shallow movie dirs under the two datasets."""
    blender = _read_json(blender_index)
    blender_by_id = {str(item["id"]): item for item in blender.get("items", [])}
    lsmdc = _read_json(lsmdc_index)
    lsmdc_by_id = {str(item["movie_id"]): item for item in lsmdc.get("movies", [])}
    videos_root = Path(blender_videos_root) if blender_videos_root else DEFAULT_BLENDER_VIDEOS_ROOT

    manifests: list[MovieManifest] = []
    for dataset in ("BlenderOpenMovies", "LSMDC"):
        dataset_dir = data_root / dataset
        if not dataset_dir.is_dir():
            continue
        for movie_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            movie_id = movie_dir.name
            vlm = _vlm_path(movie_dir)
            notes: list[str] = []
            if dataset == "BlenderOpenMovies":
                # Folder match under Videos/ is authoritative; index is metadata only.
                source = blender_by_id.get(movie_id)
                filename = str((source or {}).get("filename") or (source or {}).get("file") or "")
                video_path = resolve_blender_source_video(
                    movie_id=movie_id,
                    filename=filename,
                    videos_root=videos_root,
                )
                video = str(video_path) if video_path.is_file() else ""
                if source is None:
                    notes.append("missing_blender_catalog_entry")
                if not video_path.is_file():
                    notes.append("missing_blender_source_video")
                duration = source.get("duration_sec") if source else None
                fps = source.get("fps") if source else None
            else:
                # LSMDC: stitched output path from index (not Videos/<id>/).
                source = lsmdc_by_id.get(movie_id)
                stitched = dict(source.get("stitched") or {}) if source else {}
                video = str(stitched.get("output_file") or "")
                duration = stitched.get("stitched_duration_sec") if source else None
                fps = None
                if source is None:
                    notes.append("missing_lsmdc_catalog_entry")
                elif stitched.get("status") != "complete":
                    notes.append("lsmdc_stitch_incomplete")

            status = "source_ready"
            if not vlm.is_file():
                status = "s1_missing"
            elif vlm.stat().st_size == 0:
                status = "s1_incomplete"
                notes.append("empty_vlm_output")
            manifests.append(
                MovieManifest(
                    dataset=dataset,
                    movie_id=movie_id,
                    movie_dir=str(movie_dir),
                    source_video=video,
                    vlm_output=str(vlm),
                    source_duration_seconds=float(duration) if duration is not None else None,
                    source_fps=float(fps) if fps is not None else None,
                    status=status,
                    notes=notes,
                )
            )
    return manifests


def write_catalog(items: list[MovieManifest], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in items)
        + ("\n" if items else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--blender-index", type=Path, required=True)
    parser.add_argument("--lsmdc-index", type=Path, required=True)
    parser.add_argument(
        "--blender-videos-root",
        type=Path,
        default=DEFAULT_BLENDER_VIDEOS_ROOT,
        help="Uncompressed Blender videos root (default: .../BlenderOpenMovies/Videos). "
        "Do not point this at Videos_Compressed_360p_CRF26.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_catalog(
        build_catalog(
            data_root=args.data_root,
            blender_index=args.blender_index,
            lsmdc_index=args.lsmdc_index,
            blender_videos_root=args.blender_videos_root,
        ),
        args.out,
    )


if __name__ == "__main__":
    main()
