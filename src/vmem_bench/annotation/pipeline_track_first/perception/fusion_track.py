"""v12 fusion perception: union of proposers, ONE discriminative identity judge.

R-CNN philosophy applied to the two routes' complementary failure modes:

- Route A (GroundingDINO phrases) is viewpoint-stable and zero-shot but dies on vocabulary
  mismatch ("red fox" -> 0 tracklets) and mislabels across phrases (chinchilla -> "white
  rabbit").
- Route B (SAM3 generic concepts + exemplars) is language-free but weakens under extreme
  viewpoint change and vanishes entirely when an exemplar failed to anchor.

Fusion keeps proposals maximally broad (SAM3 concepts ∪ GDINO phrases, both routes' native
thresholds) while identity stays exactly route B's discriminative assignment: every candidate
box — regardless of which model proposed it — is cropped, DINOv3-embedded and matched against
exemplar anchors. A GDINO detection may PROPOSE a region; its phrase never names a character.
Props keep class identity (head noun / prop phrase), where language is safe by construction.
"""

from __future__ import annotations

from pathlib import Path

from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.annotation.pipeline_track_first.perception.sam3_track import Sam3ExemplarTrackBackend


class FusionTrackBackend(Sam3ExemplarTrackBackend):
    """Sam3ExemplarTrackBackend + GroundingDINO as an additional character proposer."""

    name = "fusion_track"

    def __init__(self, embedder, *, crop_dir: Path, detector, **kwargs) -> None:
        super().__init__(embedder, crop_dir=crop_dir, **kwargs)
        self.detector = detector  # GroundingDino (route A component, proposals only)

    def _extra_proposals(self, fr: Frame, characters: list[RosterEntry],
                         props: list[RosterEntry],
                         ) -> list[tuple[list[float], float, bool, str | None]]:
        """GDINO proposals for character phrases; identity is still exemplar similarity.

        Character boxes are marked is_character=True so the caller routes them through the
        discriminative gate (unmatched -> dropped). Prop phrases are NOT re-proposed here:
        SAM3 head-noun concepts already cover them and class identity has no mislabel risk."""
        if not characters:
            return []
        from PIL import Image
        phrases = [e.grounding_phrase for e in characters]
        try:
            boxes_by_phrase = self.detector.detect_all_multi(fr.path, phrases)
        except Exception:  # noqa: BLE001 — a proposer failure degrades to single-proposer mode
            return []
        with Image.open(fr.path) as pil:
            w, h = pil.size
        out: list[tuple[list[float], float, bool, str | None]] = []
        for phrase in phrases:
            for bbox_1000, score in boxes_by_phrase.get(phrase, []):
                y0, x0, y1, x1 = bbox_1000  # route A convention: [ymin,xmin,ymax,xmax] 0-1000
                bbox_px = [x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h]
                out.append((bbox_px, float(score), True, None))
        return out
