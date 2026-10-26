"""Deterministic subject crops for Ref2VA references (client side, optional ``subject_crops``).

Ref2VA is trained on subject references (a person, an object); a whole earlier still carries its setting,
framing and poses, and the model tends to continue that shot instead of the instruction. This module turns
each still into crops of what the *current instruction* names:

1. :func:`phrases` extracts query phrases from the instruction text alone (the backend never sees gold):
   ``person`` (characters are often named only by name) + the <=3-word head before each parenthetical
   description + ``the X [Y [Z]]`` noun phrases, cut at stopwords.
2. LocateAnything-3B grounds all phrases in one call (``scripts/testbed/locateanything/serve.py``, a
   separate process: its remote code needs transformers 4.57.x and its weights are NVIDIA non-commercial;
   greedy decoding, deterministic within one stack). It answers ``None`` for phrases absent from the still.
   It is deliberately not GroundingDINO, which the frozen ``trackB-embed-1.0`` scorer uses for its crops.
3. :func:`select_boxes` keeps person boxes up to ``MAX_PERSON_AREA_FRAC`` of the frame (close-ups are the
   point) and other boxes up to ``MAX_AREA_FRAC`` (larger ones are the setting), drops tiny boxes, applies
   greedy IoU suppression in answer order, and keeps ``PER_IMAGE`` boxes per still.

Crops are PNGs labelled with their query phrase. Same inputs and service stack, same crops.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DETECTOR = "locateanything-3b"
ENDPOINT_ENV = "VMEM_LOCATE_URL"
MAX_PHRASES = 8
MAX_AREA_FRAC = 0.35
MAX_PERSON_AREA_FRAC = 0.9
MIN_SIDE_PX = 48
NMS_IOU = 0.5
PAD_FRAC = 0.08
PER_IMAGE = 3
PERSON = "person"
STOP = {"and", "with", "into", "onto", "the", "a", "an", "of", "to", "in", "on", "at", "by", "from", "his",
        "her", "their", "its", "for", "as", "while", "then", "beside", "under", "over", "toward", "towards",
        "around", "through", "behind", "near", "up", "down", "out", "off", "back", "is", "are", "has"}
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class SubjectCrop:
    path: Path
    label: str
    source: Path
    box: tuple[int, int, int, int]


def phrases(instruction: str) -> list[str]:
    """Query phrases from a normalized instruction (see module doc); ``person`` first, at most MAX_PHRASES."""
    out = [PERSON]

    def add(words: list[str]) -> None:
        p = " ".join(words).strip(" ,'.;:").lower()
        if p and p not in out and p not in STOP:
            out.append(p)

    for m in re.finditer(r"([A-Za-z][A-Za-z' -]{0,60}?)\s*\(", instruction):
        tail: list[str] = []
        for w in reversed(m.group(1).split()):
            if w.lower() in STOP or len(tail) == 3:
                break
            tail.insert(0, w)
        add(tail)
    body = re.sub(r"\([^)]*\)", " ", instruction)
    for m in re.finditer(r"\bthe ((?:[A-Za-z][A-Za-z'-]*[ ,.;:]?){1,3})", body):
        words: list[str] = []
        for w in m.group(1).split():
            if w.lower() in STOP or len(words) == 3:
                break
            words.append(w)
        add(words)
    return out[:MAX_PHRASES]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_boxes(dets: list[tuple[str, tuple[float, float, float, float]]], width: int, height: int,
                 per_image: int = PER_IMAGE) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Answer order (the phrase order the detector followed), size rules by label, greedy NMS."""
    keep: list[tuple[str, tuple[float, float, float, float]]] = []
    for label, box in dets:
        w, h = box[2] - box[0], box[3] - box[1]
        cap = MAX_PERSON_AREA_FRAC if label == PERSON else MAX_AREA_FRAC
        if not label or w * h > cap * width * height or min(w, h) < MIN_SIDE_PX:
            continue
        if any(_iou(box, k[1]) > NMS_IOU for k in keep):
            continue
        keep.append((label, box))
        if len(keep) == per_image:
            break
    return keep


class SubjectCropper:
    def __init__(self, endpoint: str | None = None, timeout: float = 300.0) -> None:
        self.endpoint = (endpoint or os.environ.get(ENDPOINT_ENV, "")).rstrip("/")
        if not self.endpoint:
            raise RuntimeError(f"subject_crops needs a LocateAnything endpoint (subject_detector= or ${ENDPOINT_ENV})")
        self.timeout = timeout

    def detect(self, image_path: str | Path, queries: list[str]) -> list[tuple[str, tuple[float, float, float, float]]]:
        body = json.dumps({"image_path": str(Path(image_path).resolve()), "phrases": queries}).encode()
        req = urllib.request.Request(self.endpoint + "/detect", data=body, headers={"Content-Type": "application/json"})
        with _OPENER.open(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        allowed = set(queries)
        return [(b["label"], tuple(float(v) for v in b["box"])) for b in res.get("boxes", []) if b["label"] in allowed]

    def crops(self, references: list[str | Path], instruction: str, out_dir: Path, prefix: str) -> list[SubjectCrop]:
        from PIL import Image

        queries = phrases(instruction)
        out: list[SubjectCrop] = []
        for i, ref in enumerate(references):
            im = Image.open(ref).convert("RGB")
            for j, (label, box) in enumerate(select_boxes(self.detect(ref, queries), im.width, im.height)):
                px, py = PAD_FRAC * (box[2] - box[0]), PAD_FRAC * (box[3] - box[1])
                b = (max(0, int(box[0] - px)), max(0, int(box[1] - py)),
                     min(im.width, int(round(box[2] + px))), min(im.height, int(round(box[3] + py))))
                path = Path(out_dir) / f"{prefix}_ref{i}_crop{j}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                im.crop(b).save(path, format="PNG")
                out.append(SubjectCrop(path=path, label=label, source=Path(ref), box=b))
        return out
