"""Route B (exemplar-grounded) unit tests — fakes only, no SAM3/GPU."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.annotation.pipeline_track_first.perception.sam3_track import (
    Sam3ExemplarTrackBackend, assign_by_exemplar, head_noun, mask_to_bbox, px_to_norm_bbox,
)


def test_geometry_helpers() -> None:
    assert head_noun("red apple") == "apple"
    assert head_noun("gliding_squirrel") == "squirrel"
    assert px_to_norm_bbox([64.0, 36.0, 128.0, 72.0], 640, 360) == [100, 100, 200, 200]
    import numpy as np
    m = np.zeros((10, 20), dtype=bool)
    m[2:5, 4:8] = True
    assert mask_to_bbox(m) == [200, 200, 500, 400]
    assert mask_to_bbox(np.zeros((4, 4), dtype=bool)) is None


def test_assign_by_exemplar_floor_and_argmax() -> None:
    exemplars = {"white rabbit": [1.0, 0.0], "red creature": [0.0, 1.0]}
    assert assign_by_exemplar([0.9, 0.1], exemplars, sim_floor=0.28)[0] == "white rabbit"
    assert assign_by_exemplar([0.1, 0.9], exemplars, sim_floor=0.28)[0] == "red creature"
    # Orthogonal-ish candidate below floor -> dropped, not force-assigned.
    assert assign_by_exemplar([0.2, -0.9], exemplars, sim_floor=0.28) is None


class _FakeSegmenter:
    """Deterministic candidates: one 'animal' box left, one 'apple' box right."""

    def __init__(self):
        self.calls = []

    def segment(self, image, concept):
        self.calls.append(concept)
        if concept == "animal":
            return [([10.0, 10.0, 60.0, 90.0], 0.95, None)]
        if concept == "apple":
            return [([70.0, 70.0, 95.0, 95.0], 0.9, None)]
        return []


class _FakeEmbedder:
    """Left half of the image embeds like the rabbit exemplar, right half like the apple."""

    def embed_image(self, path):
        name = Path(path).name
        if "exemplar_rabbit" in name:
            return [1.0, 0.0]
        if "c0_" in name:  # character candidate crop (left box)
            return [0.95, 0.05]
        return [0.0, 1.0]


def test_backend_assigns_characters_by_exemplar_and_props_by_class() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sam3trk_"))
    from PIL import Image
    frame_paths = []
    for i in range(2):
        p = tmp / f"fr{i}.png"
        Image.new("RGB", (100, 100), "gray").save(p)
        frame_paths.append(p)
    exemplar = tmp / "exemplar_rabbit.jpg"
    Image.new("RGB", (32, 32), "white").save(exemplar)

    backend = Sam3ExemplarTrackBackend(
        _FakeEmbedder(), crop_dir=tmp / "crops", segmenter=_FakeSegmenter(),
        character_concepts=("animal",), exemplar_sim_floor=0.28,
        track_min_len=2, min_box_px=4)
    roster = [
        RosterEntry(name="White Rabbit", kind="character", grounding_phrase="white rabbit",
                    exemplar_crop=str(exemplar)),
        RosterEntry(name="Apple", kind="prop", grounding_phrase="red apple"),
    ]
    frames = [Frame(frame_index=i * 8, path=p) for i, p in enumerate(frame_paths)]
    tracklets = backend.track_shot(frames, roster, next_track_id=0)
    phrases = sorted({t.phrase for t in tracklets})
    assert phrases == ["red apple", "white rabbit"]  # exemplar match + class identity
    assert all(len(t.detections) == 2 for t in tracklets)  # tracked across both frames


def test_backend_without_exemplars_skips_characters() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sam3trk_noex_"))
    from PIL import Image
    p = tmp / "fr0.png"
    Image.new("RGB", (100, 100), "gray").save(p)
    seg = _FakeSegmenter()
    backend = Sam3ExemplarTrackBackend(
        _FakeEmbedder(), crop_dir=tmp / "crops", segmenter=seg,
        character_concepts=("animal",), track_min_len=1, min_box_px=4)
    roster = [RosterEntry(name="R", kind="character", grounding_phrase="white rabbit")]
    tracklets = backend.track_shot([Frame(frame_index=0, path=p)], roster)
    assert tracklets == []          # no exemplar -> no character identity source
    assert "animal" not in seg.calls  # and no wasted SAM3 forwards


def test_multi_view_exemplar_max_similarity() -> None:
    # Entity with two anchors (front/back view): a back-view candidate must still match.
    exemplars = {"white rabbit": [[1.0, 0.0], [0.0, 1.0]], "apple": [[-1.0, 0.0]]}
    got = assign_by_exemplar([0.05, 0.95], exemplars, sim_floor=0.28)
    assert got is not None and got[0] == "white rabbit"


class _FakeDetector:
    """Route A proposer: claims a 'white rabbit' box on the RIGHT side of the frame —
    but the right side embeds like the APPLE, so fusion identity must NOT trust the phrase."""

    def detect_all_multi(self, image, phrases):
        return {p: [([700, 700, 950, 950], 0.8)] for p in phrases}  # [ymin,xmin,ymax,xmax] 0-1000


def test_fusion_gdino_proposes_but_never_names() -> None:
    from vmem_bench.annotation.pipeline_track_first.perception.fusion_track import FusionTrackBackend
    tmp = Path(tempfile.mkdtemp(prefix="fusion_"))
    from PIL import Image
    p = tmp / "fr0.png"
    Image.new("RGB", (100, 100), "gray").save(p)
    exemplar = tmp / "exemplar_rabbit.jpg"
    Image.new("RGB", (32, 32), "white").save(exemplar)

    backend = FusionTrackBackend(
        _FakeEmbedder(), crop_dir=tmp / "crops", detector=_FakeDetector(),
        segmenter=_FakeSegmenter(), character_concepts=("animal",),
        exemplar_sim_floor=0.28, track_min_len=1, min_box_px=4)
    roster = [RosterEntry(name="White Rabbit", kind="character",
                          grounding_phrase="white rabbit", exemplar_crop=str(exemplar))]
    tracklets = backend.track_shot([Frame(frame_index=0, path=p)], roster)
    # SAM3's left-side candidate matches the rabbit exemplar and survives; GDINO's right-side
    # "white rabbit" proposal embeds like the apple -> fails the exemplar gate -> dropped.
    assert [t.phrase for t in tracklets] == ["white rabbit"]
    assert all(d.bbox[1] < 700 for t in tracklets for d in t.detections)


def _run_all() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_sam3_track: OK")


if __name__ == "__main__":
    _run_all()
