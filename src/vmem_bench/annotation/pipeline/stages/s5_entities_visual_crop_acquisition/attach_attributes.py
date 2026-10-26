"""Attach crop attribute packs to accepted S5 proposals."""

from __future__ import annotations

from typing import Any

from vmem_bench.common.crop_attributes import (
    CropAttributeClassifier,
    build_crop_attribute_classifier,
)
from vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition.task_planner import (
    CropTask,
)


def attach_crop_attributes(
    proposal: dict[str, Any],
    *,
    task: CropTask,
    classifier: CropAttributeClassifier | None = None,
    seconds: float | None = None,
) -> dict[str, Any]:
    """Mutate ``proposal`` with ``crop_attributes`` when a crop was accepted.

    Default classifier is null (schema present, all labels ``unknown``) so S5
    runs do not require an extra VLM call. Set ``MEMSTRATA_BENCH_CROP_ATTR_CLASSIFIER=vlm``
    or pass a classifier to populate via closed-enum MCQ.
    """

    if not proposal.get("accepted"):
        return proposal
    crop_path = proposal.get("crop_path")
    if not crop_path:
        return proposal

    clf = classifier or build_crop_attribute_classifier()
    frame_index = proposal.get("frame_index")
    pack = clf.classify(
        str(crop_path),
        kind=task.kind,
        name=task.name,
        chunk_id=task.chunk_id,
        frame_index=int(frame_index) if frame_index is not None else None,
        seconds=(
            float(seconds)
            if seconds is not None
            else float(getattr(task, "start_seconds", 0.0) or 0.0)
        ),
    )
    proposal.update(pack.to_annotations())
    return proposal


__all__ = ["attach_crop_attributes"]
