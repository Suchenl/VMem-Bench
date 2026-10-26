"""Minimal SUT (system-under-test) interface for evaluating YOUR method on VMem-Bench.

This module is intentionally tiny and dependency-light (stdlib only). It gives an
external method two things:

1. A typed contract (:class:`Segment`, :class:`Reference`, :class:`Method`) that
   mirrors the fairness rules in ``docs/benchmark/running_eval.md`` §1: the bench
   hands your method **only** a prompt and (optionally) the real segment video; it
   never hands you ``present`` / ``first_appearances`` / roster ids.
2. Writers that emit the exact artifacts the release scorers ingest:
   - Track A: ``visual_selections/<system>.json`` read by
     ``vmem_bench.scoring.visual_coverage`` (per-segment reference images).
   - Track B: a run dir (``progress.json`` + ``review/segments/<segment_id>.mp4``)
     read by ``vmem_bench.scoring.end2end_coverage``.

Nothing here imports ``vmem_bench``; the packages stay decoupled. Copy this folder,
implement :class:`Method`, and score with the two documented commands.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Segment:
    """One chunk the bench drives your method with, in chronological order.

    ``video_path`` is the real segment clip; it *stands in for* your generator's
    pixels so the benchmark isolates the memory mechanism (running_eval §1). It is
    ``None`` when you have not obtained source videos yet (see ``docs/DATA.md``) —
    your method must still emit a (possibly empty) selection for every segment.
    """

    chunk_id: int
    prompt: str
    seconds_span: tuple[float, float] | None = None
    video_path: Path | None = None


@dataclass
class Reference:
    """One reference image your method selected as context for a segment.

    Provide EITHER ``crop_abspath`` (an image you already materialized) OR
    ``source_seconds`` (a timestamp into the source video; the bench's Track A frame
    materializer cuts the frame for you). The visual scorer is name-blind: it reads
    ``crop_abspath`` and judges it against the video, so ``entity_id`` is optional and
    only kept for your own bookkeeping.
    """

    crop_abspath: str | None = None
    source_seconds: float | None = None
    entity_id: str | None = None

    def to_representation(self) -> dict:
        rep: dict = {}
        if self.crop_abspath is not None:
            rep["crop_abspath"] = str(self.crop_abspath)
        if self.source_seconds is not None:
            rep["source_seconds"] = float(self.source_seconds)
        if self.entity_id is not None:
            rep["entity_id"] = str(self.entity_id)
        if not rep:
            raise ValueError("Reference needs crop_abspath or source_seconds")
        return rep


@runtime_checkable
class Method(Protocol):
    """The two hooks every SUT implements. Drive order per segment is strict:
    ``compose`` (recall from CURRENT memory) BEFORE ``observe`` (write the real
    segment into memory). Never peek at segment t's video while composing t."""

    def compose(self, seg: Segment) -> Sequence[Reference]:
        """Return the reference images recalled from memory for this prompt (may be empty)."""

    def observe(self, seg: Segment) -> None:
        """Update your memory from the real segment video (detect/crop/embed/store)."""


# --------------------------------------------------------------------------- gold
def load_segments(movie_dir: Path, limit: int | None = None) -> list[Segment]:
    """Read a bundled/frozen gold movie into the SUT-visible fields ONLY.

    Deliberately exposes just ``chunk_id`` / ``prompt`` / ``seconds_span``. It does
    NOT read ``present`` / ``first_appearances`` — feeding those to a SUT is gold
    leakage (running_eval fairness rule 3) and invalidates the numbers.
    """
    movie_dir = Path(movie_dir)
    ca = json.loads((movie_dir / "gold" / "chunk_annotations.json").read_text(encoding="utf-8"))
    videos_root = _maybe_video_root(movie_dir)
    segs: list[Segment] = []
    for c in ca["chunks"]:
        span = c.get("seconds_span")
        segs.append(Segment(
            chunk_id=int(c["chunk_id"]),
            prompt=str(c.get("prompt", "")),
            seconds_span=(float(span[0]), float(span[1])) if span else None,
            video_path=None if videos_root is None else _segment_video(videos_root, int(c["chunk_id"])),
        ))
    segs.sort(key=lambda s: s.chunk_id)
    return segs[:limit] if limit else segs


def _maybe_video_root(movie_dir: Path) -> Path | None:
    """Optional: a sibling ``segments/`` dir with ``chunk_<id>.mp4`` if you pre-cut clips."""
    root = movie_dir / "segments"
    return root if root.is_dir() else None


def _segment_video(videos_root: Path, chunk_id: int) -> Path | None:
    cand = videos_root / f"chunk_{chunk_id:05d}.mp4"
    return cand if cand.is_file() else None


# ----------------------------------------------------------------- Track A writer
def write_tracka_selections(
    out_path: Path,
    system: str,
    per_segment: Sequence[tuple[int, Sequence[Reference]]],
) -> Path:
    """Emit ``visual_selections/<system>.json`` exactly as the Track A scorer reads it.

    Schema (see ``vmem_bench.scoring.visual_coverage._load_selection``)::

        {"system": "<system>",
         "chunks": [{"chunk_id": 0,
                     "selected": [{"representations": [{"crop_abspath": "/abs.png"}]}]}]}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk_id, refs in per_segment:
        chunks.append({
            "chunk_id": int(chunk_id),
            "selected": [{"representations": [r.to_representation() for r in refs]}] if refs else [],
        })
    out_path.write_text(
        json.dumps({"system": system, "chunks": chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ----------------------------------------------------------------- Track B writer
def write_trackb_run(run_dir: Path, segment_videos: Sequence[tuple[str, Path]]) -> Path:
    """Lay out a Track B SUT run dir as ``end2end_coverage`` expects.

    Writes ``review/segments/<segment_id>.mp4`` (copies your generated clips) and a
    ``progress.json`` index. ``segment_id`` must match the GT segment ids in
    ``assets/trackB/en/gt/<story>.json``. The scorer judges the RENDERED pixels, so
    these must be your generator's real outputs, not placeholders.
    """
    run_dir = Path(run_dir)
    seg_dir = run_dir / "review" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for idx, (seg_id, src) in enumerate(segment_videos):
        dst = seg_dir / f"{seg_id}.mp4"
        dst.write_bytes(Path(src).read_bytes())
        chunks.append({"chunk_id": idx, "segment_id": seg_id, "video": str(dst.resolve())})
    (run_dir / "progress.json").write_text(
        json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
