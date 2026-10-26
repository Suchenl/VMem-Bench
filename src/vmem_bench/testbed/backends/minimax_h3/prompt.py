"""Deterministic Track-B instruction -> MiniMax-H3 prompt conversion.

H3 is trained on rewritten prompts, not raw instructions. With reference images the Ref2VA
full-reference layout applies (``subject_definitions`` / ``summary`` / ``retention_analysis`` /
``detailed_description`` / audio sections, docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md of the H3
release); without references the T2VA layout applies (``integrated_multimodal_description`` /
``overall_soundscape`` / ``non_diegetic_music``, VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

The backend never learns which entity a reference shows (that would need gold or perception the
backend must not have), so each ``<Picture k>`` is bound to a generic ``<Subject k>``: "the
recurring characters, objects and setting shown in <Picture k>". The instruction text itself is
passed through verbatim apart from whitespace and full-width punctuation normalisation, so every
condition sees the same wording and only the reference count changes the frame around it.
"""

from __future__ import annotations

import re

PROMPT_FORMAT_VERSION = "h3-trackb-prompt/1"
MAX_REFERENCES = 9

_STYLE = "The target video is in a realistic, cinematic live-action style."
_SOUNDSCAPE = "Natural ambient sound of the scene, with the physical sounds of the described actions."
_MUSIC = "N/A"

_PUNCT = str.maketrans({
    "\uff08": " (",
    "\uff09": ") ",
    "\uff0c": ", ",
    "\uff1b": "; ",
    "\uff1a": ": ",
    "\u3002": ". ",
    "\uff01": "! ",
    "\uff1f": "? ",
    "\u3001": ", ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
})


def normalize_instruction(text: str) -> str:
    """Full-width punctuation to ASCII, whitespace collapsed, terminal punctuation ensured."""
    s = str(text).translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)
    s = re.sub(r"\)(?=[A-Za-z0-9])", ") ", s)
    if not s:
        raise ValueError("empty instruction")
    if s[-1] not in ".!?\"'":
        s += "."
    return s


def _labels(prefix: str, n: int) -> list[str]:
    return [f"<{prefix} {k}>" for k in range(1, n + 1)]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


PROMPT_FORMATS = {1: PROMPT_FORMAT_VERSION, 2: "h3-trackb-prompt/2"}


def _t2va(body: str, dur: str) -> str:
    return (
        f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, a single "
        f"continuous {dur}-second shot. {body}\n\n"
        f"overall_soundscape: {_SOUNDSCAPE}\n\n"
        f"non_diegetic_music: {_MUSIC}"
    )


def build_h3_i2v_prompt(instruction: str, *, duration_sec: float = 5.0) -> str:
    """FL2VA first-frame prompt (the release's i2va example layout): frame 0 is <Picture 1>, then T2VA."""
    body = normalize_instruction(instruction)
    return ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is "
            "fully referenced.\n\n" + _t2va(body, f"{float(duration_sec):g}"))


def build_h3_prompt_v2(instruction: str, n_references: int, *, duration_sec: float = 5.0,
                       board_stills: int = 0) -> str:
    """Guide-conformant Ref2VA layout (``h3-trackb-prompt/2``).

    ``<Picture k>`` appears only inside its ``<Subject k>`` definition, because a standalone picture label
    marks a concrete target frame (keyframe completion). Each subject is the look of the people, objects,
    and places in its picture; composition, camera, poses, and action always come from the instruction.
    ``board_stills`` > 0 means the single picture is a reference board tiling that many earlier stills.
    """
    n = int(n_references)
    if n < 0 or n > MAX_REFERENCES:
        raise ValueError(f"n_references must be in [0, {MAX_REFERENCES}], got {n_references}")
    body = normalize_instruction(instruction)
    dur = f"{float(duration_sec):g}"
    if n == 0:
        return _t2va(body, dur)
    subjects = _labels("Subject", n)
    source = (f"a reference board of {int(board_stills)} stills from earlier shots of the same story"
              if board_stills else "a still from an earlier shot of the same story")
    definitions = "\n".join(
        f"{s} is the people, objects, and places shown in <Picture {k}>, {source}; only their faces, hair, "
        f"clothing, colors, shapes, materials, and architecture are referenced."
        for k, s in enumerate(subjects, 1)
    )
    new = "composition, camera angle, framing, poses, and action"
    retention = "\n".join(
        f"{s} (appears in [Shot 1]): partially_preserved - only the look of the people, objects, and places "
        f"this shot describes is kept; the {new} are new."
        for s in subjects
    )
    return (
        "subject_definitions:\n"
        f"{definitions}\n\n"
        "summary:\n"
        f"[reference generation] The target video is a new single continuous {dur}-second shot. {body} "
        f"Whatever this shot describes that also appears in {_join(subjects)} keeps that look; the {new} "
        f"follow this description.\n\n"
        "retention_analysis:\n"
        f"{retention}\n\n"
        "detailed_description:\n"
        f"{_STYLE}\n"
        f"[Shot 1] {body}\n\n"
        "overall_soundscape:\n"
        f"{_SOUNDSCAPE}\n\n"
        "non_diegetic_music:\n"
        f"{_MUSIC}"
    )


def build_h3_prompt_subjects(instruction: str, labels: list[str], *, duration_sec: float = 5.0) -> str:
    """``h3-trackb-prompt/2`` for labelled subject crops: image ``k`` is the crop of ``labels[k-1]``."""
    n = len(labels)
    if n > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} subject crops, got {n}")
    body = normalize_instruction(instruction)
    dur = f"{float(duration_sec):g}"
    if n == 0:
        return _t2va(body, dur)
    subjects = _labels("Subject", n)
    names = [re.sub(r"\s+", " ", str(lab)).strip() for lab in labels]
    definitions = "\n".join(
        f"{s} is the {name} in <Picture {k}>, cropped from a still of an earlier shot of the same story; its "
        f"face, hair, clothing, colors, shapes, and materials are referenced."
        for k, (s, name) in enumerate(zip(subjects, names), 1)
    )
    retention = "\n".join(
        f"{s} (appears in [Shot 1]): partially_preserved - the look of the {name} is kept; its pose, position, "
        f"action, and the camera framing follow this shot."
        for s, name in zip(subjects, names)
    )
    return (
        "subject_definitions:\n"
        f"{definitions}\n\n"
        "summary:\n"
        f"[reference generation] The target video is a new single continuous {dur}-second shot. {body} "
        f"Whatever this shot describes that matches {_join(subjects)} keeps that look; the setting, "
        f"composition, camera, poses, and action follow this description.\n\n"
        "retention_analysis:\n"
        f"{retention}\n\n"
        "detailed_description:\n"
        f"{_STYLE}\n"
        f"[Shot 1] {body}\n\n"
        "overall_soundscape:\n"
        f"{_SOUNDSCAPE}\n\n"
        "non_diegetic_music:\n"
        f"{_MUSIC}"
    )


def build_h3_prompt(instruction: str, n_references: int, *, duration_sec: float = 5.0) -> str:
    """Render one Track-B segment instruction as an H3 prompt for ``n_references`` images.

    ``n_references`` must equal the number of image references sent with the request, in order:
    image ``k`` (1-based) is ``<Picture k>``.
    """
    n = int(n_references)
    if n < 0 or n > MAX_REFERENCES:
        raise ValueError(f"n_references must be in [0, {MAX_REFERENCES}], got {n_references}")
    body = normalize_instruction(instruction)
    dur = f"{float(duration_sec):g}"
    if n == 0:
        return _t2va(body, dur)

    pictures = _labels("Picture", n)
    subjects = _labels("Subject", n)
    definitions = "\n".join(
        f"{s} is the recurring characters, objects, and setting shown in {p}, a still from an "
        f"earlier shot of the same story; their faces, clothing, colors, shapes, and materials "
        f"follow {p}."
        for s, p in zip(subjects, pictures)
    )
    retention = "\n".join(
        f"{s} (appears in [Shot 1]): partially_preserved - whatever of {s} the shot describes keeps "
        f"its appearance; composition, pose, and camera follow the description, not {p}."
        for s, p in zip(subjects, pictures)
    )
    return (
        "subject_definitions:\n"
        f"{definitions}\n\n"
        "summary:\n"
        f"[reference generation] The target video is a single continuous {dur}-second shot. {body} "
        f"{_join(subjects)} guide the appearance of the recurring characters, objects, and setting; "
        f"{_join(pictures)} are appearance references, not frames of the target video.\n\n"
        "retention_analysis:\n"
        f"{retention}\n\n"
        "detailed_description:\n"
        f"{_STYLE}\n"
        f"[Shot 1] {body} Wherever they appear, the characters, objects, and setting keep the "
        f"appearance of {_join(subjects)}.\n\n"
        "overall_soundscape:\n"
        f"{_SOUNDSCAPE}\n\n"
        "non_diegetic_music:\n"
        f"{_MUSIC}"
    )
