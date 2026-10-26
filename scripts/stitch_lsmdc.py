#!/usr/bin/env python3
"""Stitch chronologically ordered LSMDC clips into one movie file.

The LSMDC download is gated and must be obtained by the user. This helper only
performs the local, post-download concatenation step; it never downloads or
redistributes LSMDC media.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _resolve_ffmpeg(value: str | None) -> str:
    if value:
        return value
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return bundled
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg was not found; install ffmpeg, set FFMPEG_BIN, "
        "or pass --ffmpeg /path/to/ffmpeg"
    )


def _read_manifest(path: Path) -> list[Path]:
    manifest = path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")

    clips: list[Path] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        clip = Path(entry).expanduser()
        if not clip.is_absolute():
            clip = manifest.parent / clip
        clip = clip.resolve()
        if not clip.is_file():
            raise FileNotFoundError(
                f"manifest line {line_number} points to a missing file: {clip}"
            )
        clips.append(clip)

    if not clips:
        raise ValueError(f"manifest contains no clip paths: {manifest}")
    return clips


def _concat_entry(path: Path) -> str:
    # ffmpeg's concat demuxer accepts shell-style single-quoted paths.
    escaped = str(path).replace("\\", "\\\\").replace("'", "'\\''")
    return f"file '{escaped}'\n"


def stitch(
    manifest: Path,
    output: Path,
    *,
    ffmpeg: str | None = None,
    overwrite: bool = False,
    reencode: bool = False,
    dry_run: bool = False,
) -> Path:
    clips = _read_manifest(manifest)
    destination = output.expanduser().resolve()
    if destination in clips:
        raise ValueError("output must not be one of the input clips")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination} (pass --overwrite to replace it)"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vmem_lsmdc_", dir=str(destination.parent)
    ) as temp_dir:
        concat_file = Path(temp_dir) / "concat.txt"
        concat_file.write_text(
            "".join(_concat_entry(clip) for clip in clips), encoding="utf-8"
        )
        command = [
            _resolve_ffmpeg(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-map",
            "0",
        ]
        if reencode:
            command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
            command.extend(["-c:a", "aac"])
        else:
            command.extend(["-c", "copy"])
        command.append(str(destination))

        print(f"[stitch_lsmdc] clips={len(clips)}")
        print(f"[stitch_lsmdc] output={destination}")
        print(f"[stitch_lsmdc] command={shlex.join(command)}")
        if not dry_run:
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"cannot execute ffmpeg: {command[0]}"
                ) from exc
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate an explicitly ordered LSMDC clip manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="text file with one clip path per line, in chronological order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="stitched output movie, normally .../LSMDC_Videos_Stitched/<movie_id>.mp4",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="ffmpeg executable (default: FFMPEG_BIN, PATH, or imageio-ffmpeg)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="re-encode to H.264/AAC instead of stream-copying compatible clips",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the manifest and print the command without running ffmpeg",
    )
    args = parser.parse_args(argv)
    try:
        stitch(
            args.manifest,
            args.output,
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
            reencode=args.reencode,
            dry_run=args.dry_run,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[stitch_lsmdc] error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
