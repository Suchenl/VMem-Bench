"""Model-specific prompt formatting of the FLUX.2 Klein -> Wan2.2-TI2V-5B-Turbo pipeline.

Byte-for-byte the formatting the Track-B paper runs applied before each model call (FLUX receives a
JSON scene prompt with the de-AI buzzword filter; Wan receives the prose prompt, prefixed with a
"cinematic video" lead when it names neither video nor motion). Only the ``flux`` and ``wan`` targets
of the original standardizer are kept, since this pipeline calls nothing else.
"""

from __future__ import annotations

import json
from typing import Any

_SCREENPLAY_KEYS = ("Narrative goal", "Visible actions", "Scene state")

AI_BUZZWORDS = [
    "photorealistic",
    "hyperrealistic",
    "highly detailed",
    "8k",
    "flawless skin",
    "studio lighting",
]

FILM_ANCHORS = (
    "raw photo, un-retouched, shot on 35mm film, Fujifilm Superia, natural skin texture, "
    "visible pores, slight film grain"
)


def strip_ai_buzzwords(text: str) -> str:
    for word in AI_BUZZWORDS:
        text = text.replace(f", {word}", "").replace(word, "")
    return text


def preprocess_de_ai_prompt(prompt: str, quality_preset: str | None) -> str:
    try:
        data = json.loads(prompt)

        def process_node(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: process_node(v) for k, v in node.items()}
            if isinstance(node, list):
                return [process_node(item) for item in node]
            if isinstance(node, str):
                return strip_ai_buzzwords(node)
            return node

        processed = process_node(data)
        if quality_preset == "raw_film" and isinstance(processed, dict):
            if "style" in processed and isinstance(processed["style"], str):
                processed["style"] = f"{processed['style']}, {FILM_ANCHORS}" if processed["style"].strip() else FILM_ANCHORS
            elif "scene" in processed and isinstance(processed["scene"], str):
                processed["scene"] += f", {FILM_ANCHORS}"
            else:
                processed["style"] = FILM_ANCHORS
        return json.dumps(processed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        if quality_preset != "raw_film":
            return prompt
        return strip_ai_buzzwords(prompt) + f", {FILM_ANCHORS}"


def _parse_screenplay(prompt: str) -> dict[str, str]:
    data = {}
    for line in prompt.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()
    return data


def _screenplay_fields(prompt: str) -> dict[str, str]:
    data = _parse_screenplay(prompt)

    def clean(s: str) -> str:
        return s.strip().rstrip(".").rstrip(",")

    goal = clean(data.get("Narrative goal", ""))
    for prefix in ["A video of ", "A video showing ", "A shot of ", "Video of ", "A photo of ", "Photo of "]:
        if goal.lower().startswith(prefix.lower()):
            goal = goal[len(prefix):]
    loc = lighting = style = ""
    scene_state = data.get("Scene state", "")
    if scene_state:
        kv = {k.strip(): v.strip() for k, v in (p.split("=") for p in scene_state.split(";") if "=" in p)}
        loc = clean(kv.get("location") or kv.get("place", ""))
        lighting = clean(kv.get("lighting") or kv.get("light", ""))
        style = clean(kv.get("style") or kv.get("genre", ""))
    return {"goal": goal, "actions": clean(data.get("Visible actions", "")),
            "characters": clean(data.get("Active characters or subjects", "")),
            "loc": loc, "lighting": lighting, "style": style}


def flux_prompt(prompt: str, quality_preset: str | None = None) -> str:
    if any(key in prompt for key in _SCREENPLAY_KEYS):
        f = _screenplay_fields(prompt)
        structured: dict[str, Any] = {"scene": f["goal"], "subjects": [], "action": f["actions"], "context": {},
                                      "style": f["style"] or "cinematic film still"}
        if f["characters"]:
            structured["subjects"].append({"description": f["characters"]})
        if f["loc"]:
            structured["context"]["location"] = f["loc"]
        if f["lighting"]:
            structured["context"]["lighting"] = f["lighting"]
        return preprocess_de_ai_prompt(json.dumps(structured, ensure_ascii=False), quality_preset)
    try:
        json.loads(prompt)
        return preprocess_de_ai_prompt(prompt, quality_preset)
    except (json.JSONDecodeError, TypeError):
        structured = {"scene": prompt, "subjects": [], "style": "cinematic film still"}
        return preprocess_de_ai_prompt(json.dumps(structured, ensure_ascii=False), quality_preset)


def wan_prompt(prompt: str) -> str:
    if any(key in prompt for key in _SCREENPLAY_KEYS):
        f = _screenplay_fields(prompt)
        setting = []
        if f["loc"]:
            setting.append(f"in a {f['loc']}")
        if f["lighting"]:
            setting.append(f"with {f['lighting']} lighting")
        if f["style"]:
            setting.append(f"styled as {f['style']}")
        setting_str = " " + ", ".join(setting) if setting else ""
        char_str = f" {f['characters']}" if f["characters"] else " a subject"
        action_str = f" performing: {f['actions']}" if f["actions"] else ""
        if f["goal"]:
            return f"A cohesive cinematic video showing {f['goal']} as{char_str}{action_str}{setting_str}.".strip()
        return f"A beautiful cinematic video showing{char_str}{action_str}{setting_str}.".strip()
    if not any(p in prompt.lower() for p in ["video", "motion"]):
        return f"A beautiful cinematic video of {prompt}"
    return prompt
