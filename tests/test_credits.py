"""CPU-only checks for opening/ending credits exclusion (annotation/credits.py)."""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.credits import detect_credit_segments, filter_shots

# 1000-frame film, closed-inclusive shots (sorted): 0:(0,29) 1:(30,59) dark head cards,
# 3:(450,549) a dark NIGHT scene mid-film (must never be excluded), 5:(900,999) dark end scroll.
SHOTS = [(0, 29), (30, 59), (60, 449), (450, 549), (550, 899), (900, 999)]
DARK = {0, 1, 3, 5}  # shot indices whose frames are dark (incl. the mid-film night scene)


def _lum_for(shots):
    def lum(path: Path) -> float:
        frame = int(path.stem)
        for i, (s, e) in enumerate(shots):
            if s <= frame <= e:
                return 10.0 if i in DARK else 120.0
        return 120.0
    return lum


def _fp(i: int) -> Path:
    return Path(f"/nonexistent/{i:07d}.jpg")


def test_detect_anchored_head_and_tail_only() -> None:
    segs = detect_credit_segments(SHOTS, total_frames=1000, fps=24.0, frame_path=_fp,
                                  head_tail_ratio=0.08, luminance_fn=_lum_for(SHOTS))
    # head window = 80 frames -> dark shots 0-1 overlap it and merge; tail -> shot 5 (900-999).
    assert [s["reason"] for s in segs] == ["opening_credits", "end_credits"]
    assert segs[0]["frame_span"] == [0, 59]
    assert segs[1]["frame_span"] == [900, 999]
    assert segs[1]["seconds_span"][0] == round(900 / 24.0, 3)


def test_confirm_fn_vetoes_prefilter() -> None:
    # VLM says the head card is a real (night) scene -> only the tail is excluded.
    segs = detect_credit_segments(SHOTS, total_frames=1000, fps=24.0, frame_path=_fp,
                                  head_tail_ratio=0.08, luminance_fn=_lum_for(SHOTS),
                                  confirm_fn=lambda frames: [False] + [True] * (len(frames) - 1))
    assert [s["reason"] for s in segs] == ["end_credits"]


def test_mid_film_dark_scene_never_excluded_and_filter_shots() -> None:
    segs = detect_credit_segments(SHOTS, total_frames=1000, fps=24.0, frame_path=_fp,
                                  head_tail_ratio=0.08, luminance_fn=_lum_for(SHOTS))
    excluded_spans = {tuple(s["frame_span"]) for s in segs}
    assert (450, 549) not in excluded_spans  # dark night scene mid-film stays
    kept = filter_shots(SHOTS, segs)
    assert (0, 29) not in kept and (30, 59) not in kept and (900, 999) not in kept
    assert (450, 549) in kept and len(kept) == len(SHOTS) - 3


def _run_all() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_credits: OK")


if __name__ == "__main__":
    _run_all()
