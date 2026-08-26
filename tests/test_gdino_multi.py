"""Self-check for cross-entity GroundingDINO batching (the per-shot speedup).

The old gdino_track called GDINO once per (frame x phrase); now it calls detect_all_multi once per
frame (all phrases in one forward, boxes mapped back to phrases by text label). This checks the
deterministic glue that makes that safe:
  - _match_phrase: partial-label attribution, tie -> first, zero-overlap -> None.
  - _phrase_groups: caption char-budget splitting (huge rosters -> few forwards, not one/phrase).
  - GdinoTrackBackend.track_shot: ONE detect_all_multi call per frame, boxes land on the right
    phrase, locations skipped, a frame is decoded once (crop from an open PIL).

Run: PYTHONPATH=benchmarks/MemStrata/src python3 benchmarks/MemStrata/tests/test_gdino_multi.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.grounding import _match_phrase, _phrase_groups
from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import _keep_box


def test_match_phrase() -> None:
    phrases = ["white rabbit", "red fox", "flying squirrel"]
    assert _match_phrase("white rabbit", phrases) == "white rabbit"
    assert _match_phrase("rabbit", phrases) == "white rabbit"      # partial label
    assert _match_phrase("fox", phrases) == "red fox"
    assert _match_phrase("", phrases) is None                       # empty
    assert _match_phrase("dragon", phrases) is None                 # zero overlap -> dropped
    # tie on shared head noun -> first phrase (deterministic).
    assert _match_phrase("squirrel", ["red squirrel", "flying squirrel"]) == "red squirrel"


def test_phrase_groups() -> None:
    assert _phrase_groups([]) == []
    assert _phrase_groups(["a", "b", "c"]) == [["a", "b", "c"]]     # small roster -> one forward
    long = [f"phrase_number_{i}" for i in range(40)]
    groups = _phrase_groups(long, max_chars=60)
    assert len(groups) > 1                                          # huge roster splits
    assert [p for g in groups for p in g] == long                  # lossless, order preserved
    assert all(sum(len(p) + 2 for p in g) <= 60 or len(g) == 1 for g in groups)


def test_keep_box() -> None:
    W, H = 1280, 720             # 720p frame
    big = [0, 0, 500, 500]       # 0.5*720=360 px tall, 0.5*1280=640 px wide
    tiny = [0, 0, 20, 20]        # 0.02*720=14.4 px tall/wide -> junk
    # min_box_px=24 (absolute px): big kept, tiny (<24 px per side) dropped.
    assert _keep_box(big, 0.9, W, H, 0.0, 24) is True
    assert _keep_box(tiny, 0.9, W, H, 0.0, 24) is False
    # same normalized box, but on a huge frame the "tiny" box is now large in px -> kept
    # (this is exactly why absolute px beats a frame-fraction).
    assert _keep_box(tiny, 0.9, 8000, 8000, 0.0, 24) is True  # 0.02*8000=160 px
    # score floor drops low-confidence even if large.
    assert _keep_box(big, 0.2, W, H, 0.3, 0) is False
    assert _keep_box(big, 0.4, W, H, 0.3, 0) is True
    # all-off keeps everything.
    assert _keep_box(tiny, 0.01, W, H, 0.0, 0) is True


def test_prune_scratch() -> None:
    from vmem_bench.annotation.pipeline_track_first import prune_scratch
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for sub in ("derived/frames", "derived/candidates", "assets", "gold"):
            (root / sub).mkdir(parents=True)
        (root / "derived/frames/f1.jpg").write_bytes(b"x" * 100)
        (root / "derived/candidates/c1.jpg").write_bytes(b"y" * 50)
        (root / "assets/keep.jpg").write_bytes(b"z" * 10)
        (root / "gold/entity_registry.json").write_text("{}")
        freed = prune_scratch(root)
        assert freed == 150, freed
        assert not (root / "derived").exists()                 # entire derived/ tree deleted
        assert (root / "assets/keep.jpg").is_file()            # top-level assets untouched
        assert (root / "gold/entity_registry.json").is_file()
        assert prune_scratch(root) == 0                        # idempotent


class _FakeDetector:
    """Records the phrases seen per frame; returns fixed boxes keyed by the input phrase."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def detect_all_multi(self, image, phrases):
        self.calls.append(list(phrases))
        out = {p: [] for p in phrases}
        if "white rabbit" in out:
            out["white rabbit"] = [([0, 0, 500, 500], 0.9)]
        if "red fox" in out:
            out["red fox"] = [([100, 100, 300, 300], 0.8), ([200, 200, 400, 400], 0.6)]
        return out


class _FakeEmbedder:
    def __init__(self) -> None:
        self.n = 0

    def embed_image(self, path):
        self.n += 1
        assert Path(path).is_file()  # crop was actually written
        return [1.0, 2.0, 3.0]


def test_track_shot_one_call_per_frame() -> None:
    try:
        from PIL import Image
    except ImportError:
        print("test_gdino_multi: PIL missing, skipping track_shot integration part")
        return
    from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
    from vmem_bench.annotation.pipeline_track_first.perception.gdino_track import GdinoTrackBackend

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        frame_paths = []
        for fi in (10, 20):
            p = root / f"frame_{fi}.jpg"
            Image.new("RGB", (640, 480), (fi, fi, fi)).save(p)
            frame_paths.append(Frame(frame_index=fi, path=p))
        roster = [
            RosterEntry(name="Bunny", kind="character", grounding_phrase="white rabbit"),
            RosterEntry(name="Fox", kind="character", grounding_phrase="red fox"),
            RosterEntry(name="Meadow", kind="location", grounding_phrase="green meadow"),
        ]
        det = _FakeDetector()
        emb = _FakeEmbedder()
        backend = GdinoTrackBackend(det, emb, crop_dir=root / "crops",
                                    track_min_len=1, use_motion=True)
        backend.track_shot(frame_paths, roster, next_track_id=0)

        # ONE detect call per frame (2 frames), never per-phrase.
        assert len(det.calls) == 2, det.calls
        # Location phrase is excluded from what the detector is asked for.
        assert det.calls[0] == ["white rabbit", "red fox"], det.calls[0]
        # rabbit(1) + fox(2) = 3 detections per frame -> 6 crops/embeds total.
        assert emb.n == 6, emb.n
        crops = list((root / "crops").glob("*.jpg"))
        assert len(crops) == 6, crops


def main() -> int:
    test_match_phrase()
    test_phrase_groups()
    test_keep_box()
    test_prune_scratch()
    test_track_shot_one_call_per_frame()
    print("test_gdino_multi: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
