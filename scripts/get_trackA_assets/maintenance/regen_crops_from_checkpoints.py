"""Regenerate pruned candidate crops from tracklet checkpoints + the source video.

After a run, ``tmp/candidates`` is pruned; the tracklet checkpoints keep every
detection's ``frame_index`` + normalized bbox + recorded ``crop_path``. To resume a run
in a NEW output dir (e.g. re-running post-tracking stages with new logic), this script:

  1. copies ``tmp/checkpoint/roster.json`` + ``tracklets/`` from --src to --dst,
     rewriting absolute crop_path prefixes from the old out dir to the new one;
  2. re-extracts each needed frame from --video (ffmpeg, same ``extract_frame`` used by
     the pipeline) and re-crops each detection with the same ``_crop`` helper, writing
     to the rewritten crop_path.

Deterministic; crops are near-identical re-encodes of the originals (same decode+crop
path). Names/chunk-draft caches are intentionally NOT copied: entity ids and present
sets change under new re-ID logic, so stale caches would corrupt gold.

Usage:
    python scripts/vmem_bench/maintenance/regen_crops_from_checkpoints.py \
        --src <old_out_dir> --dst <new_out_dir> --video <source.mp4> [--fps 24.0]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vmem_bench.common.media import extract_frame
from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import _crop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="old annotated out dir")
    ap.add_argument("--dst", type=Path, required=True, help="new out dir to prepare")
    ap.add_argument("--video", type=Path, required=True, help="source video")
    ap.add_argument("--fps", type=float, default=None,
                    help="video fps (default: read src gold/chunk_index.json)")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="optional inclusive frame ceiling for short-clip/pilot reuse")
    args = ap.parse_args()

    fps = args.fps
    if fps is None:
        from vmem_bench.common.paths import MovieDirs
        fps = float(json.loads(MovieDirs(args.src).chunk_index.read_text(encoding="utf-8"))["fps"])

    from vmem_bench.common.paths import MovieDirs
    src_ckpt = MovieDirs(args.src).checkpoint
    dst_ckpt = MovieDirs(args.dst, write=True).checkpoint
    (dst_ckpt / "tracklets").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_ckpt / "roster.json", dst_ckpt / "roster.json")

    old_prefix = str(args.src.resolve())
    new_prefix = str(args.dst.resolve())
    frames_dir = MovieDirs(args.dst, write=True).frames / "regen"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    frame_cache: dict[int, Path] = {}

    def frame_path(idx: int) -> Path:
        p = frame_cache.get(idx)
        if p is None:
            p = frames_dir / f"f{idx:07d}.jpg"
            if not p.is_file():
                extract_frame(args.video, p, frame_index=idx, fps=fps)
            frame_cache[idx] = p
        return p

    n_shots = n_crops = n_skipped = 0
    for ck in sorted((src_ckpt / "tracklets").glob("*.json")):
        payload = json.loads(ck.read_text(encoding="utf-8"))
        if args.max_frame is not None and int(payload.get("first", 0)) > args.max_frame:
            continue
        kept_tracklets = []
        for tk in payload.get("tracklets", []):
            detections = [
                det for det in tk.get("detections", [])
                if args.max_frame is None or int(det["frame_index"]) <= args.max_frame
            ]
            if not detections:
                continue
            tk["detections"] = detections
            kept_tracklets.append(tk)
            for det in detections:
                cp = det.get("crop_path")
                if not cp:
                    continue
                if cp.startswith(old_prefix):
                    cp = new_prefix + cp[len(old_prefix):]
                    det["crop_path"] = cp
                out_path = Path(cp)
                if out_path.is_file():
                    n_skipped += 1
                    continue
                pil = Image.open(frame_path(int(det["frame_index"]))).convert("RGB")
                _crop(pil, det["bbox"], out_path)
                n_crops += 1
        payload["tracklets"] = kept_tracklets
        if args.max_frame is not None:
            payload["last"] = min(int(payload.get("last", args.max_frame)), args.max_frame)
        (dst_ckpt / "tracklets" / ck.name).write_text(
            json.dumps(payload), encoding="utf-8")
        n_shots += 1
    print(f"shots={n_shots} crops_written={n_crops} crops_existing={n_skipped} "
          f"frames_extracted={len(frame_cache)}")


if __name__ == "__main__":
    main()
