"""Exemplar collection for route B: anchor each roster character to a visual crop.

Design (probe-backed, see experiments/results/probes/sam3_exemplar_bbb): SAM3 enumerates
class-level candidates ("animal") on the roster keyframes — a task it is reliable at — and the
judge VLM only answers a MULTIPLE-CHOICE question ("which numbered crop is the white rabbit?").
The VLM never regresses box coordinates (unreliable on 8B) and never invents grounding language
for the detector (the failure mode that killed route A on BBB v10: "red fox" -> 0 tracklets).

Output: ``tmp/exemplars/<slug>.jpg`` per anchored entity + a manifest JSON; RosterEntry.exemplar_crop
carries the path into the perception backend.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from vmem_bench.common.vecmath import cosine_similarity

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 24  # vLLM serves --limit-mm-per-prompt image=24; one request max
_DEDUP_SIM = 0.90     # near-duplicate candidate crops collapse to the best-scored one
# Multi-view anchor augmentation band: candidates this similar to a primary exemplar are the
# same entity seen differently; above the band they add no information, below it they are
# probably someone else. Guarded further by the judge having left them unassigned.
_AUG_SIM_LOW, _AUG_SIM_HIGH, _AUG_MAX_EXTRA = 0.55, 0.92, 2


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "entity"


def collect_exemplars(
    out: Path,
    key_frames: Sequence[Path],
    roster: Sequence[dict],
    *,
    segmenter,
    embed_image: Callable[[Path], list[float]],
    judge_role,
    concepts: tuple[str, ...] = ("animal", "person"),
    max_frames: int = 40,
) -> dict[str, str]:
    """Return {character name -> exemplar crop path}; unanchored characters are simply absent.

    ``judge_role`` needs ``pick_exemplars(labeled_names, candidate_crops)`` ->
    {name: candidate index or -1}. Deterministic given the same frames/weights.
    """
    characters = [e for e in roster if e.get("kind") == "character"]
    if not characters:
        return {}
    ex_dir = Path(out) / "tmp" / "exemplars"
    ex_dir.mkdir(parents=True, exist_ok=True)

    # 1) Class-level candidates across (subsampled) keyframes, deduped by appearance.
    frames = list(key_frames)
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    candidates: list[dict] = []  # {crop, vec, score}
    for frame in frames:
        for concept in concepts:
            try:
                instances = segmenter.segment(frame, concept)
            except Exception:  # noqa: BLE001 — a bad frame must not kill roster
                logger.exception("SAM3 segment failed on %s", frame)
                continue
            for bi, (bbox_px, score, _mask) in enumerate(instances):
                x0, y0, x1, y1 = (int(v) for v in bbox_px)
                if (x1 - x0) < 16 or (y1 - y0) < 16:
                    continue
                from PIL import Image
                crop_path = ex_dir / f"_cand_{frame.stem}_{concept}_{bi}.jpg"
                Image.open(frame).convert("RGB").crop((x0, y0, x1, y1)).save(crop_path)
                vec = list(embed_image(crop_path))
                dup = next((c for c in candidates
                            if cosine_similarity(vec, c["vec"]) >= _DEDUP_SIM), None)
                if dup is not None:
                    if score > dup["score"]:
                        dup.update(crop=crop_path, vec=vec, score=score)
                    continue
                candidates.append({"crop": crop_path, "vec": vec, "score": float(score)})
    # Keep the candidate pool DIVERSE, not merely high-scored: top-N by score drowns rare
    # characters under dozens of protagonist crops (BBB: chipmunk/flying-squirrel lost to 24
    # rabbit boxes). Greedy farthest-point in DINOv3 space keeps one seat per look.
    if len(candidates) > _MAX_CANDIDATES:
        from vmem_bench.annotation.pipeline_track_first.roster import farthest_point_sample
        keep = farthest_point_sample([c["vec"] for c in candidates], _MAX_CANDIDATES)
        candidates = [candidates[i] for i in keep]
    candidates.sort(key=lambda c: -c["score"])
    if not candidates:
        logger.warning("exemplar collection: SAM3 produced no candidates on %d keyframes",
                       len(frames))
        return {}

    # 2) Judge assigns candidates to roster names (multiple choice; -1 = not present).
    names = [str(e.get("name") or "") for e in characters]
    try:
        assignment = judge_role.pick_exemplars(names, [c["crop"] for c in candidates])
    except Exception:  # noqa: BLE001 — no exemplars is a soft degrade (backend falls back)
        logger.exception("exemplar assignment failed")
        return {}

    result: dict[str, str] = {}
    manifest: dict[str, dict] = {}
    used_candidates: set[int] = set()
    for name in names:
        idx = assignment.get(name, -1)
        # A candidate may anchor only one entity; first name wins on conflicts.
        if not isinstance(idx, int) or not (0 <= idx < len(candidates)) or idx in used_candidates:
            continue
        crop_src = Path(candidates[idx]["crop"])
        if not crop_src.exists():
            continue
        used_candidates.add(idx)
        final = ex_dir / f"{_slug(name)}.jpg"
        crop_src.replace(final)
        candidates[idx]["crop"] = final
        anchors = [str(final)]
        # Multi-view augmentation: unclaimed candidates that look like the SAME entity from a
        # DIFFERENT view (mid similarity band: high enough to be it, low enough to add info)
        # become extra anchors — a back-view anchor rescues what a front-view anchor misses.
        primary_vec = candidates[idx]["vec"]
        extras = sorted(
            (c for ci, c in enumerate(candidates)
             if ci not in used_candidates and Path(c["crop"]).exists()
             and _AUG_SIM_LOW <= cosine_similarity(primary_vec, c["vec"]) <= _AUG_SIM_HIGH),
            key=lambda c: -c["score"])[:_AUG_MAX_EXTRA]
        for vi, extra in enumerate(extras):
            used_candidates.add(candidates.index(extra))
            aug = ex_dir / f"{_slug(name)}_v{vi + 2}.jpg"
            Path(extra["crop"]).replace(aug)
            extra["crop"] = aug
            anchors.append(str(aug))
        result[name] = ";".join(anchors)
        manifest[name] = {"crop": str(final), "anchors": anchors,
                          "sam3_score": candidates[idx]["score"]}
    (ex_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for c in candidates:  # clean unassigned scratch crops
        p = Path(c["crop"])
        if p.exists() and p.name.startswith("_cand_"):
            p.unlink()
    logger.info("exemplars anchored for %d/%d characters", len(result), len(names))
    return result
