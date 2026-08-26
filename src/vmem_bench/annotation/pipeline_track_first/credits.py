"""Opening/ending credits exclusion (cheap deterministic prefilter + one VLM confirmation).

Credits and title/logo cards are non-diegetic: annotating them wastes budget and, worse, the
scrolling-text mega-shot dominates tracking time. Detection is intentionally conservative:

1. Only shots overlapping the head/tail window (``head_tail_ratio`` of the film) are candidates —
   overlap, not containment, because a credits scroll can be one mega-shot far longer than the
   window itself (Big Buck Bunny's end credits start at 82% of the film).
2. Deterministic prefilter: a candidate shot whose sampled frames are dark on average (mean gray
   level <= ``max_luminance``) looks like a text-on-black card.
3. Optional single VLM batch confirms the prefiltered shots' middle frames (guards against night
   scenes); without a confirm function the prefilter result stands.
4. Kept shots are merged into contiguous segments and a segment must be anchored at the very
   first or very last shot — credits never float in the middle of the film.

Every excluded segment is recorded (frames + seconds + reason) in ``chunk_index.json`` /
``manifest.json`` so evaluation can prove the SUT never saw those spans.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path


def frame_mean_luminance(path: Path) -> float:
    """Mean gray level (0-255) of one frame; cheap and deterministic."""
    from PIL import Image, ImageStat
    with Image.open(path) as img:
        return float(ImageStat.Stat(img.convert("L")).mean[0])


def detect_credit_segments(
    shots: Sequence[tuple[int, int]], *, total_frames: int, fps: float,
    frame_path: Callable[[int], Path],
    confirm_fn: Callable[[list[Path]], list[bool]] | None = None,
    head_tail_ratio: float = 0.08, max_luminance: float = 45.0,
    luminance_fn: Callable[[Path], float] | None = None,
) -> list[dict]:
    """Return excluded segments ``[{shot_span, frame_span, seconds_span, reason}]``.

    ``shots`` are closed-inclusive ``(first, last)`` spans covering the video (the pipeline's
    convention). ``confirm_fn`` receives the prefiltered candidate shots' middle frames (one
    batch) and returns one bool per frame.
    """
    if not shots or total_frames <= 0 or head_tail_ratio <= 0:
        return []
    lum = luminance_fn or frame_mean_luminance
    window = int(total_frames * head_tail_ratio)
    head_end, tail_start = window, total_frames - window

    candidates: list[int] = []
    for i, (start, last) in enumerate(shots):
        if last < start:
            continue
        overlaps_head = start < head_end
        overlaps_tail = last >= tail_start
        if not (overlaps_head or overlaps_tail):
            continue
        sample = sorted({start, (start + last) // 2, last})
        try:
            mean = sum(lum(frame_path(f)) for f in sample) / len(sample)
        except Exception:  # noqa: BLE001 — an undecodable frame must not kill the run
            continue
        if mean <= max_luminance:
            candidates.append(i)

    if candidates and confirm_fn is not None:
        mids = [frame_path((shots[i][0] + shots[i][1]) // 2) for i in candidates]
        try:
            verdicts = list(confirm_fn(mids))
        except Exception:  # noqa: BLE001 — a VLM hiccup falls back to the deterministic prefilter
            verdicts = [True] * len(candidates)
        candidates = [i for i, ok in zip(candidates, verdicts) if ok]

    kept = set(candidates)
    segments: list[dict] = []
    i, n = 0, len(shots)
    while i < n:
        if i not in kept:
            i += 1
            continue
        j = i
        while j + 1 < n and (j + 1) in kept:
            j += 1
        # Anchor rule: a credits segment must include the film's first or last shot.
        if i == 0 or j == n - 1:
            first, last = shots[i][0], shots[j][1]
            segments.append({
                "shot_span": [i, j],
                "frame_span": [int(first), int(last)],
                "seconds_span": [round(first / fps, 3), round((last + 1) / fps, 3)],
                "reason": "opening_credits" if i == 0 else "end_credits",
            })
        i = j + 1
    return segments


def filter_shots(shots: Sequence[tuple[int, int]],
                 segments: Sequence[dict]) -> list[tuple[int, int]]:
    """Drop closed-inclusive shots fully covered by an excluded segment (index-agnostic)."""
    spans = [tuple(seg["frame_span"]) for seg in segments]
    return [(s, e) for (s, e) in shots
            if not any(lo <= s and e <= hi for lo, hi in spans)]
