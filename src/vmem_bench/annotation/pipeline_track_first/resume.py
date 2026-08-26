"""Result-level resume for the track-first pipeline (fine-grained, atomic checkpointing).

The user asked to "resume 标注的结果" -- resume the produced annotation RESULTS, not just the
frame cache. So the four EXPENSIVE result artifacts are checkpointed as they are produced, under
``<out>/tmp/checkpoint/`` (legacy: ``build/checkpoint/``); a killed run reloads finished work
and only fills the gap:

  roster.json                -- merged cast roster           (VLM; once per movie)
  tracklets/shot_NNNNN.json  -- one shot's tracklets + scene vector + per-tracklet face signature
                                (GPU: detect+embed+face -- the dominant cost; also the unit the
                                 multi-GPU pool parallelizes, so parallelism and resume share it)
  names/<entity_id>.json     -- one entity's VLM name/description  (VLM; once per entity)
  chunks/chunk_NNNN.json     -- one chunk's drafted ChunkAnnotation (VLM; once per chunk)

Cheap deterministic steps (re-ID, presence, time metadata) are NOT checkpointed: they are recomputed
from the tracklets on every resume (CPU cosine, milliseconds) and entity_ids come out identical, so
the ``names/*`` cache still applies. Writes are atomic (tmp + os.replace) so a crash mid-write can
never leave a half-written file that a later resume would trust. Self-contained: stdlib only.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet


def ckpt_dir(out: Path) -> Path:
    from vmem_bench.common.paths import MovieDirs
    return MovieDirs(Path(out), write=False).checkpoint


def _write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp in the same dir + os.replace) so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # a truncated/partial file is treated as "not done" -> recomputed


# --- roster ---------------------------------------------------------------------------------

def save_roster(out: Path, roster: list[dict]) -> None:
    _write_json(ckpt_dir(out) / "roster.json", {"roster": roster})


def load_roster(out: Path) -> list[dict] | None:
    d = _read_json(ckpt_dir(out) / "roster.json")
    return list(d["roster"]) if d and "roster" in d else None


# --- tracklets (per shot) -------------------------------------------------------------------

def _det_to_dict(d: Detection) -> dict:
    return {"frame_index": d.frame_index, "bbox": list(d.bbox), "score": d.score,
            "phrase": d.phrase, "embedding": d.embedding, "crop_path": d.crop_path}


def _det_from_dict(d: dict) -> Detection:
    return Detection(frame_index=int(d["frame_index"]), bbox=[int(x) for x in d["bbox"]],
                     score=float(d["score"]), phrase=str(d["phrase"]),
                     embedding=d.get("embedding"), crop_path=d.get("crop_path"))


def shot_ckpt_path(out: Path, shot_idx: int) -> Path:
    return ckpt_dir(out) / "tracklets" / f"shot_{shot_idx:05d}.json"


def save_shot(out: Path, payload: dict) -> None:
    """Persist one shot checkpoint. ``payload`` = shot_payload(...) output (JSON-able)."""
    _write_json(shot_ckpt_path(out, int(payload["shot_idx"])), payload)


def load_shot(out: Path, shot_idx: int) -> dict | None:
    return _read_json(shot_ckpt_path(out, shot_idx))


def shot_done(out: Path, shot_idx: int) -> bool:
    return shot_ckpt_path(out, shot_idx).is_file()


def shot_payload(shot_idx: int, first: int, last: int, tracklets: list[Tracklet],
                 face_sigs: list[list[float] | None], scene_frame: int,
                 scene_vec: list[float]) -> dict:
    """Build the JSON-able per-shot checkpoint (tracklets aligned with ``face_sigs``)."""
    return {"shot_idx": shot_idx, "first": first, "last": last,
            "scene_frame": scene_frame, "scene_vec": scene_vec,
            "tracklets": [{"track_id": tk.track_id, "phrase": tk.phrase,
                           "detections": [_det_to_dict(d) for d in tk.detections],
                           "face_sig": fs}
                          for tk, fs in zip(tracklets, face_sigs)]}


def shot_tracklets(payload: dict) -> tuple[list[Tracklet], list[list[float] | None]]:
    """Rebuild (tracklets, face_sigs) from a loaded shot checkpoint (aligned lists)."""
    tks, sigs = [], []
    for t in payload.get("tracklets", []):
        tks.append(Tracklet(track_id=int(t["track_id"]), phrase=str(t["phrase"]),
                            detections=[_det_from_dict(d) for d in t["detections"]]))
        sigs.append(t.get("face_sig"))
    return tks, sigs


# --- naming (per entity) --------------------------------------------------------------------

def save_name(out: Path, entity_id: str, name: str, description: str) -> None:
    _write_json(ckpt_dir(out) / "names" / f"{entity_id}.json",
                {"name": name, "description": description})


def load_name(out: Path, entity_id: str) -> dict | None:
    cached = _read_json(ckpt_dir(out) / "names" / f"{entity_id}.json")
    # A failed naming attempt may have checkpointed empty fields (BBB v13: resumed entities
    # kept the raw roster phrase and an empty description). Empty cache = no cache.
    if cached is not None and not str(cached.get("name") or "").strip():
        return None
    return cached


# --- chunk drafts (per chunk) ---------------------------------------------------------------

def save_chunk(out: Path, chunk_id: int, ann_dict: dict) -> None:
    _write_json(ckpt_dir(out) / "chunks" / f"chunk_{chunk_id:04d}.json", ann_dict)


def load_chunk(out: Path, chunk_id: int) -> dict | None:
    return _read_json(ckpt_dir(out) / "chunks" / f"chunk_{chunk_id:04d}.json")
