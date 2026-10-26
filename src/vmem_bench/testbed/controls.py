"""Track-B controls on the shared generator: no memory (budget 0) and last-seen references.

``none`` hands the backend zero references, so every keyframe is text-to-image.

``last_seen`` is a gold-selected control, not a system under test. Gold picks WHICH of this rollout's
own earlier shots supply reference stills; the selector sees only an id-only plan
(``{"rule": ..., "plan": {"<shot>": [<earlier shot>, ...]}}``) and takes the middle frame of each
chosen shot's own generated clip. ``oracle`` is the deprecated name of the same control.

Plan rule (fixed): at GT segment n, take the required character/prop entities of n that were seen
before (``op != introduce``) and carry a ``last_seen`` segment; order them by ``gap`` descending (ties
by ``eid``); each contributes the production shot of its ``last_seen`` segment. GT ``seg_k`` is
production shot ``k-1``. The selector keeps the top K of that list (deduped), fills remaining slots
with the immediately preceding own shot, and takes the middle frame of each shot's own clip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vmem_bench.testbed.retrieval.retrievers import RetrievedRef
from vmem_bench.testbed.store import extract_still, probe_duration

LAST_SEEN_VARIANT = "last_seen"
DEPRECATED_ALIASES = {"oracle": LAST_SEEN_VARIANT}

RULE = (
    "GT seg n -> production shot n-1; entities = cast[] with kind in {character, prop}, "
    "op != introduce, last_seen set; order gap desc, eid asc; source = production shot of last_seen; "
    "method keeps top-K deduped, fills with preceding own shot, middle frame of own clip"
)
ENTITY_KINDS = ("character", "prop")


def _shot(gt_segment_id: str) -> int:
    return int(str(gt_segment_id).rsplit("_", 1)[-1]) - 1


def gold_path(bench_root: Path, story_id: str) -> Path:
    return bench_root / "assets" / "trackB" / "en" / "gt" / f"{story_id}.json"


def build_last_seen_plan(gold: dict[str, Any], prompt_segment_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(plan, audit)``: the id-only plan for the selector and a bench-side audit record."""
    kinds = {eid: str((ent or {}).get("kind") or "") for eid, ent in (gold.get("entities") or {}).items()}
    gold_ids = [str(seg["segment_id"]) for seg in gold["segments"]]
    if gold_ids[: len(prompt_segment_ids)] != list(prompt_segment_ids):
        raise ValueError("gold segments do not align with the prompt stream")
    plan: dict[str, list[int]] = {}
    audit: dict[str, list[dict[str, Any]]] = {}
    for seg in gold["segments"][: len(prompt_segment_ids)]:
        shot = _shot(seg["segment_id"])
        rows = []
        for item in seg.get("cast") or []:
            if item.get("op") == "introduce" or not item.get("last_seen"):
                continue
            if kinds.get(item["eid"]) not in ENTITY_KINDS:
                continue
            source = _shot(item["last_seen"])
            if not 0 <= source < shot:
                raise ValueError(f"non-causal last_seen {item['last_seen']} at {seg['segment_id']}")
            rows.append({"eid": item["eid"], "gap": int(item.get("gap") or 0), "source_shot": source})
        rows.sort(key=lambda r: (-r["gap"], r["eid"]))
        plan[str(shot)] = [r["source_shot"] for r in rows]
        audit[str(shot)] = rows
    return (
        {"rule": RULE, "plan": plan},
        {"rule": RULE, "story_id": gold.get("story_id"), "gt_version": gold.get("gt_version"), "entities": audit},
    )


def load_plan(path: str | Path) -> dict[int, list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): [int(s) for s in v] for k, v in data["plan"].items()}


class LastSeenReferenceSelector:
    """Middle frames of this run's own earlier shots, chosen by an id-only plan."""

    name = "last_seen_reference"

    def __init__(self, plan: dict[int, list[int]], *, store: Any, run_dir: Path | None, budget: int) -> None:
        self.plan = {int(k): [int(s) for s in v] for k, v in plan.items()}
        self.store = store
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        self.budget = max(0, int(budget))
        self._recorded: dict[int, str] | None = None

    def _recorded_videos(self) -> dict[int, str]:
        """Clips this run closed before a resume, as recorded in its own progress.json."""
        if self._recorded is None:
            self._recorded = {}
            progress = self.run_dir / "progress.json" if self.run_dir is not None else None
            if progress is not None and progress.is_file():
                try:
                    records = json.loads(progress.read_text(encoding="utf-8")).get("segments") or []
                except (OSError, ValueError):
                    records = []
                for rec in records:
                    if isinstance(rec, dict) and rec.get("video"):
                        try:
                            self._recorded[int(rec["segment_id"])] = str(rec["video"])
                        except (KeyError, TypeError, ValueError):
                            continue
        return self._recorded

    def own_video(self, source: int, *, current: int) -> str | None:
        """The run's own clip for ``source``, or None if it is not strictly earlier / not own."""
        if source < 0 or source >= current:
            return None
        video = self.store.video_for(source) or self._recorded_videos().get(source)
        if not video:
            return None
        path = Path(video).resolve()
        if not path.is_file():
            return None
        if self.run_dir is not None and self.run_dir not in path.parents:
            return None
        return str(path)

    def _middle_frame(self, source: int, video: str) -> str | None:
        out = Path(self.store.work_dir) / "last_seen_frames" / f"c{source:05d}_mid.png"
        if out.is_file() and out.stat().st_size > 0:
            return str(out)
        ffmpeg = self.store.config.ffmpeg
        return str(out) if extract_still(ffmpeg, video, probe_duration(video, ffmpeg) / 2.0, out) else None

    def sources(self, *, segment_id: int) -> list[int]:
        """Top-``budget`` planned entities' shots (deduped), then the preceding shot as filler."""
        chosen: list[int] = []
        for source in self.plan.get(int(segment_id), [])[: self.budget]:
            if source not in chosen and self.own_video(source, current=segment_id):
                chosen.append(source)
        prev = int(segment_id) - 1
        if len(chosen) < self.budget and prev not in chosen and self.own_video(prev, current=segment_id):
            chosen.append(prev)
        return chosen

    def select(self, *, segment_id: int) -> list[RetrievedRef]:
        refs: list[RetrievedRef] = []
        for source in self.sources(segment_id=segment_id):
            video = self.own_video(source, current=segment_id)
            image = self._middle_frame(source, video) if video else None
            if image:
                refs.append(RetrievedRef(score=1.0, arm="last_seen", representation_id=f"c{source:05d}@mid",
                                         image_ref=image))
        return refs
