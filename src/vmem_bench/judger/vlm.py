"""VLM Judger Client using standard library urllib.request for OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Default configuration
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3-vl-32b"


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to a base64 data URL."""
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class VlmJudger:
    """OpenAI-compatible VLM client for performing structured visual and video evaluations."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        fps: float = 4.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("MONTAGE_CONTEXT_JUDGER_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.fps = fps
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        # Per-judger token accounting (Pitfall_Notes: cost optimization needs a feedback signal).
        # Accumulated across all _call_api successes; the pipeline sums these across annotator +
        # verifier judgers into run_done / summary so each run reports what it spent.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_api(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None = None,
                  fps: float | None = None, *, temperature: float = 0.0) -> dict[str, Any]:
        """POST to the OpenAI-compatible API.

        HTTP 4xx (e.g. 404 model-not-found) is non-retryable: fail fast with a message pointing
        at a base_url / served-model-name mismatch (Pitfall_Notes root cause #6). 5xx and
        transport errors retry with exponential backoff. A malformed JSON body (rare under
        json_schema mode) gets cheap no-backoff retries before falling through.

        ``temperature`` defaults to 0.0 (deterministic). The pipeline raises it on QA retry
        (attempt >= 2) and on redundancy branches (branch >= 1) so parallel candidates do not
        return near-identical outputs — temperature=0 across branches would defeat the point of
        branches_per_chunk>1 (correlated errors, principle #11). Scoring is unaffected: the
        bench metrics are deterministic set operations over gold, never over VLM output."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        initial_max_tokens = int(os.environ.get("MEMSTRATA_VLM_MAX_TOKENS", "4096"))
        max_retry_tokens = int(os.environ.get("MEMSTRATA_VLM_MAX_RETRY_TOKENS", "8192"))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            # Structured annotation responses are compact. Without an explicit cap, a backend
            # that misses EOS can generate up to its full context window and hold the pipeline
            # in one roster/draft request for tens of minutes.
            "max_tokens": initial_max_tokens,
            "mm_processor_kwargs": {
                "fps": fps if fps is not None else self.fps,
                "do_sample_frames": True,
            },
        }

        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_response",
                    "schema": schema,
                    "strict": True,
                },
            }

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    choice = res_data["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    usage = res_data.get("usage") or {}
                    self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
                    self.total_completion_tokens += int(usage.get("completion_tokens", 0))
                try:
                    return json.loads(content)
                except json.JSONDecodeError as je:
                    last_err = je
                    # A schema-constrained response can still be cut mid-object when the output
                    # reaches max_tokens. Retrying with the identical cap deterministically
                    # reproduces the same invalid JSON, so grow the budget for the next attempt.
                    if finish_reason == "length":
                        if payload["max_tokens"] < max_retry_tokens:
                            payload["max_tokens"] = min(max_retry_tokens, payload["max_tokens"] * 2)
                        else:
                            # Budget is maxed out yet output still hits the cap: the model is in a
                            # degenerate repetition loop, and at temperature=0 every retry replays
                            # it verbatim. Nudge sampling just enough to break the loop.
                            payload["temperature"] = max(float(payload["temperature"]), 0.3)
                            payload["frequency_penalty"] = 0.3
                    print(
                        f"  [VLM API] non-JSON response (attempt {attempt + 1}/{self.max_retries}, "
                        f"finish_reason={finish_reason!r}); next max_tokens={payload['max_tokens']}"
                    )
                    if attempt < self.max_retries - 1:
                        continue
                    raise RuntimeError(
                        f"VLM returned non-JSON after {self.max_retries} attempts: {je}\n"
                        f"--content--\n{content[:500]}")
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise RuntimeError(
                        f"non-retryable HTTP {exc.code} from {url}: check base_url and that "
                        f"--vlm-model matches the vLLM --served-model-name "
                        f"(model={self.model!r}).") from exc
                last_err = exc
                print(f"  [VLM API] HTTP {exc.code} (attempt {attempt + 1}/{self.max_retries}); retrying")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                break
            except urllib.error.URLError as exc:
                last_err = exc
                print(f"  [VLM API] transport error (attempt {attempt + 1}/{self.max_retries}): {exc}; retrying")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                break
        raise RuntimeError(f"VLM API call failed after {self.max_retries} attempts: {last_err}")

    def judge_same_entity(self, img1: str, img2: str, kind: str) -> bool:
        """Ask VLM if two images, or an image and a text description, represent the exact same entity."""
        try:
            img1_url = encode_image(img1)
        except Exception:
            return False

        # Check if img2 is a file path or a text description
        is_img2_file = False
        try:
            if Path(img2).is_file():
                is_img2_file = True
        except Exception:
            pass

        if is_img2_file:
            try:
                img2_url = encode_image(img2)
            except Exception:
                return False

            prompt = (
                f"You are an expert visual judge. Compare the two provided images of a {kind}. "
                f"Decide if they represent the exact same entity (e.g., the same character, same location, "
                f"or same prop). Return a JSON object with a single boolean field 'same_entity'."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img1_url}},
                        {"type": "image_url", "image_url": {"url": img2_url}},
                    ],
                }
            ]
        else:
            # img2 is a text description. Strict: default to False (do NOT merge) unless the image
            # clearly shows the described entity. Consolidation is biased toward "split rather than
            # merge": a bad merge pollutes the asset bank and is hard to catch in human review,
            # while a bad split is repairable by a merge patch (Pitfall_Notes root cause #2/#3).
            prompt = (
                f"You are a strict visual identity judge. Decide if the provided image of a {kind} "
                f"plausibly shows the entity described in the text.\n"
                f"Text Description: {img2}\n\n"
                f"Answer true ONLY if the visible evidence (colors, shape, species, object type) "
                f"is consistent with the description. The entity may be partially visible, "
                f"occluded, cropped, or from an unusual angle - do not fail on framing alone. "
                f"But if any stated attribute (color, species, object type, size class) clearly "
                f"contradicts what is visible, or the image shows a plainly different entity, "
                f"answer false. Cite the specific visual evidence behind your decision. "
                f"Return a JSON object with a single boolean field 'same_entity'."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img1_url}},
                    ],
                }
            ]

        schema = {
            "type": "object",
            "properties": {
                "same_entity": {"type": "boolean"},
            },
            "required": ["same_entity"],
            "additionalProperties": False,
        }

        try:
            result = self._call_api(messages, schema)
            return bool(result.get("same_entity", False))
        except Exception as e:
            print(f"  [VLM API] judge_same_entity failed: {e}")
            return False

    def extract_video_entities(self, video_path: str, candidate_entities: list[dict[str, Any]]) -> list[str]:
        """Ask VLM to identify which candidate entities are present in the video."""
        # Format the candidate list for the prompt
        candidates_str = ""
        for entity in candidate_entities:
            candidates_str += f"- ID: {entity['id']}, Name: {entity['name']}, Kind: {entity['kind']}, Description: {entity.get('description', '')}\n"

        prompt = (
            f"You are an expert video analyzer. You are given a video and a list of candidate entities. "
            f"Analyze the video and identify which of these candidate entities are present in the video.\n\n"
            f"Candidate Entities:\n{candidates_str}\n"
            f"Return a JSON object with a list of matched entity IDs under the field 'matched_entity_ids'."
        )

        video_url = str(Path(video_path).resolve())
        if not (video_url.startswith("http://") or video_url.startswith("https://") or video_url.startswith("file://")):
            video_url = f"file://{video_url}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]

        schema = {
            "type": "object",
            "properties": {
                "matched_entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["matched_entity_ids"],
            "additionalProperties": False,
        }

        try:
            # Use 1.0 FPS for entity extraction to make it fast
            result = self._call_api(messages, schema, fps=1.0)
            return list(result.get("matched_entity_ids", []))
        except Exception:
            return []

    def verify_instructions(self, video_path: str, instructions: list[str]) -> list[bool]:
        """Ask VLM to verify if each instruction clause is successfully executed in the video."""
        if not instructions:
            return []

        instructions_str = ""
        for i, inst in enumerate(instructions):
            instructions_str += f"{i}. {inst}\n"

        prompt = (
            f"You are an expert video verifier. You are given a video and a list of instruction clauses "
            f"describing actions, camera movements, or continuity requirements. For each instruction clause, "
            f"verify if it is successfully executed or present in the video.\n\n"
            f"Instruction Clauses:\n{instructions_str}\n"
            f"Return a JSON object with a list of boolean values under the field 'results', where each boolean "
            f"corresponds to the verification result (true for success, false for failure) of the instruction clause "
            f"at the same index."
        )

        video_url = str(Path(video_path).resolve())
        if not (video_url.startswith("http://") or video_url.startswith("https://") or video_url.startswith("file://")):
            video_url = f"file://{video_url}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]

        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "boolean"},
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        }

        try:
            result = self._call_api(messages, schema)
            results = result.get("results", [])
            # Ensure results length matches instructions length
            if len(results) < len(instructions):
                results.extend([False] * (len(instructions) - len(results)))
            return results[:len(instructions)]
        except Exception:
            return [False] * len(instructions)

    def caption_image(self, image_path: str | Path, kind: str) -> str:
        """Ask VLM to act as a captioner and generate a concise description of the entity in the image."""
        try:
            img_url = encode_image(image_path)
        except Exception:
            return f"An asset of kind {kind}"

        prompt = (
            f"You are an expert image captioner. Describe the {kind} shown in the provided image. "
            f"Provide a concise, detailed, and highly descriptive caption focusing on key visual features (e.g., appearance, colors, style, textures). "
            f"Return a JSON object with a single string field 'caption'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        schema = {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
            },
            "required": ["caption"],
            "additionalProperties": False,
        }

        try:
            result = self._call_api(messages, schema)
            return str(result.get("caption", f"An asset of kind {kind}"))
        except Exception:
            return f"An asset of kind {kind}"

    def discover_entities(self, video_path: str | Path) -> list[dict[str, Any]]:
        """Ask VLM to analyze the video chunk and discover all key entities (characters, locations, props)."""
        prompt = (
            "You are an expert video analyzer. Analyze the provided video chunk and identify all key entities.\n"
            "For each entity, determine its kind (character, location, or prop), assign it a name, and generate a concise, "
            "highly descriptive caption focusing on its key visual features (appearance, colors, style, textures).\n"
            "For ALL entities (including 'character', 'prop', and 'location'), you MUST provide a bounding box in the format [ymin, xmin, ymax, xmax] "
            "normalized to 1000 (0-1000 range) representing where the entity (or the most defining visual region/landmark of that location) is located in the first frame of the video. "
            "Avoid using [0, 0, 1000, 1000] for locations unless the entire frame is absolutely uniform and has no specific defining landmark.\n\n"
            "Return a JSON object with a list of discovered entities under the field 'entities', where each entity has "
            "the fields 'name', 'kind', 'description', and 'box' (an array of 4 integers, e.g., [ymin, xmin, ymax, xmax])."
        )

        video_url = str(Path(video_path).resolve())
        if not (video_url.startswith("http://") or video_url.startswith("https://") or video_url.startswith("file://")):
            video_url = f"file://{video_url}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "enum": ["character", "location", "prop"]},
                            "description": {"type": "string"},
                            "box": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["name", "kind", "description", "box"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities"],
            "additionalProperties": False,
        }

        try:
            # Use 1.0 FPS for entity discovery to make it extremely fast and efficient
            result = self._call_api(messages, schema, fps=1.0)
            return list(result.get("entities", []))
        except Exception:
            return []
