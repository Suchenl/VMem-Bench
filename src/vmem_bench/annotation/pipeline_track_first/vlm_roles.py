"""VLM annotator/verifier roles (workflow steps 3, 6-8).

Both roles ride on the OpenAI-compatible ``VlmJudger`` client but use different prompts
(and the pipeline feeds them different frame samples) to reduce correlated errors
(principle #11). All outputs are discrete/structured (principle #6).
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path
from typing import Any

from vmem_bench.judger.vlm import VlmJudger, encode_image

KINDS = ["character", "location", "prop"]


class RoundRobinRole:
    """Generic endpoint-pool proxy: round-robins EVERY method call across N underlying role
    objects (e.g. several ``AnnotatorRole(VlmJudger(base_url=...))`` instances on different vLLM
    replicas/GPUs).

    Motivation (principle #8, extreme parallelization): a single vLLM server already batches
    concurrent requests internally, but that is still bounded by ONE GPU's compute. When multiple
    physical replicas are available (e.g. the identity-resolution-v2 cluster/merge calls firing
    dozens of concurrent ``verify_cluster``/``group_same_individuals`` requests via a thread pool,
    see identity_resolution.py), spreading them across real GPUs via this proxy gets genuine
    multi-GPU parallelism instead of queueing behind one server's batch scheduler.

    Deliberately does NOT enumerate method names: it forwards ANY attribute access (verify_cluster,
    group_same_individuals, name_entity, discover_roster, judge_same_entity, ...) to whichever role
    the round-robin counter currently points at, so it stays correct if new role methods are added.
    Thread-safe (a lock guards the counter, matching how ``identity_resolution.py``'s
    ``ThreadPoolExecutor`` calls fire genuinely concurrent requests)."""

    def __init__(self, roles: list[Any]) -> None:
        if not roles:
            raise ValueError("RoundRobinRole requires at least one underlying role")
        self._roles = list(roles)
        self._cycle = itertools.cycle(range(len(self._roles)))
        self._lock = threading.Lock()

    def _next(self) -> Any:
        with self._lock:
            idx = next(self._cycle)
        return self._roles[idx]

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for attributes NOT found normally (so self._roles etc. above
        # resolve directly without recursing here).
        def _dispatch(*args: Any, **kwargs: Any) -> Any:
            return getattr(self._next(), name)(*args, **kwargs)
        return _dispatch


def _image_messages(prompt: str, images: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": encode_image(p)}} for p in images]
    return [{"role": "user", "content": content}]


class AnnotatorRole:
    """Drafting role: entity discovery, same-entity adjudication, prompt/state-event drafting."""

    def __init__(self, judger: VlmJudger) -> None:
        self.judger = judger

    def discover_entities(self, frames: list[Path], known_entities: list[dict[str, str]],
                          feedback: list[str], *, temperature: float = 0.0) -> list[dict[str, str]]:
        # Show static_attributes so the VLM can reuse identities by stable keys (species/color/
        # size), not just by name — a same-named entity with conflicting attrs must NOT be reused
        # (the consolidation gate enforces it, but telling the VLM up-front avoids spurious reuse).
        known = "\n".join(
            f"- {e['name']} ({e['kind']}): {e['description']}"
            + (f" [attrs: {', '.join(f'{k}={v}' for k, v in e.get('static_attributes', {}).items())}]"
               if e.get('static_attributes') else "")
            for e in known_entities)
        fb = ("\nPrevious attempts had these problems; fix them:\n" +
              "\n".join(f"- {f}" for f in feedback)) if feedback else ""
        prompt = (
            "You are annotating one shot-chunk of a film using the sampled frames provided.\n"
            "List every salient entity visible in these frames: main characters, the location, "
            "and story-relevant props. Ignore fleeting background clutter.\n"
            f"Known entities from earlier chunks (REUSE their exact names if the same entity "
            f"reappears; do not invent a new name for a known entity):\n{known or '(none yet)'}\n"
            "For each entity give:\n"
            "- name (short, unique, natural, e.g. 'Big Buck Bunny');\n"
            "- kind (character|location|prop);\n"
            "- description (concise appearance-focused, usable as a text reference for image "
            "generation);\n"
            "- grounding_phrase (a SHORT noun phrase naming the entity for open-vocabulary "
            "detection, e.g. 'grey rabbit', 'red apple', 'wooden tree trunk'. MUST be nouns only: "
            "NO actions, NO moods, NO time/order words, NO neighboring objects - those bias the "
            "detector onto the wrong subject in multi-entity frames);\n"
            "- static_attributes (object of STABLE identity keys only: species/subcategory, "
            "primary_color, size_class, object_type, location_type, ...). These are identity "
            "attributes that do not change across chunks; never put actions, pose, or mood here.\n"
            "CRITICAL: the description must state ONLY what is visible in THESE frames. "
            "Do NOT use prior knowledge of the film, franchise, or species (fur/eye colors, "
            "sizes, moods must come from the pixels, not from what you know about the story)."
            f"{fb}\n"
            "Return JSON: {\"entities\": [{\"name\", \"kind\", \"description\", "
            "\"grounding_phrase\", \"static_attributes\"}]}"
        )
        schema = {"type": "object", "properties": {"entities": {"type": "array", "items": {
            "type": "object", "properties": {
                "name": {"type": "string"}, "kind": {"type": "string", "enum": KINDS},
                "description": {"type": "string"},
                "grounding_phrase": {"type": "string"},
                "static_attributes": {"type": "object",
                    "additionalProperties": {"type": "string"}}},
            "required": ["name", "kind", "description", "grounding_phrase", "static_attributes"],
            "additionalProperties": False}}},
            "required": ["entities"], "additionalProperties": False}
        result = self.judger._call_api(_image_messages(prompt, frames), schema, temperature=temperature)
        return list(result.get("entities", []))

    def judge_same_entity(self, crop: Path, description: str, kind: str) -> bool:
        return self.judger.judge_same_entity(str(crop), description, kind)

    def group_same_individuals(self, labeled: list[tuple[str, str]], crops: list[Path], *,
                               temperature: float = 0.0) -> list[list[int]]:
        """Global identity adjudication: which numbered crops show the SAME individual?

        ``labeled`` pairs (entity_id, display_name) align 1:1 with ``crops``. Returns groups of
        crop indices; the caller treats them as review recommendations, never auto-merges."""
        roster = "\n".join(f"[{i}] {name or eid}" for i, (eid, name) in enumerate(labeled))
        prompt = (
            "Each numbered image is ONE candidate entity's best crop from the SAME film. "
            "Your job is to find which crops show the SAME individual.\n"
            f"Candidates (image order matches this list):\n{roster}\n"
            "Work in two steps INSIDE the JSON:\n"
            "1. identities: for EACH image index, state species/object type and 2-3 distinctive "
            "visual features (fur color, body size, markings) visible in that crop.\n"
            "2. groups: partition the indices. Two indices belong to one group ONLY IF the same "
            "individual is shown (same species AND consistent distinctive features; lighting, "
            "pose, scale and color grading may differ). NEVER group different species or clearly "
            "different individuals; when unsure, keep them apart. Singletons are expected.\n"
            "Return JSON {\"identities\": [{\"index\": int, \"who\": str}], "
            "\"groups\": [{\"indices\": [int, ...], \"who\": str}]}"
        )
        schema = {"type": "object", "properties": {
            "identities": {"type": "array", "items": {"type": "object", "properties": {
                "index": {"type": "integer"}, "who": {"type": "string"}},
                "required": ["index", "who"], "additionalProperties": False}},
            "groups": {"type": "array", "items": {"type": "object", "properties": {
                "indices": {"type": "array", "items": {"type": "integer"}},
                "who": {"type": "string"}},
                "required": ["indices", "who"], "additionalProperties": False}}},
            "required": ["identities", "groups"], "additionalProperties": False}
        result = self.judger._call_api(_image_messages(prompt, crops), schema,
                                       temperature=temperature)
        return [list(g.get("indices", [])) for g in result.get("groups", [])]

    def verify_cluster(self, crops: list[Path], *, temperature: float = 0.0) -> dict[str, Any]:
        """Authoritative identity check for ONE deterministically pre-clustered candidate group.

        Identity resolution redesign (pitfalls / annotation_tracking_internals §identity): a candidate
        cluster comes from ``identity_clustering.py``'s complete/average-link grouping over noisy
        appearance embeddings -- treat it as a HYPOTHESIS, not ground truth. This call is the
        AUTHORITATIVE decision (not a gray-zone fallback): does every crop show the same individual?
        If not, partition into subgroups by 0-based crop index. Bias mirrors the rest of re-ID: when
        unsure, split rather than merge (an under-split is cheaply repaired by the cross-cluster
        merge pass; a silent over-merge corrupts gold and is invisible to reviewers)."""
        prompt = (
            f"These {len(crops)} numbered images (0-indexed, in order) are candidate crops that a "
            "deterministic tracker + appearance-matching pipeline BELIEVES are all the SAME "
            "individual/instance from one film. Verify this claim; do not trust it blindly.\n"
            "Step 1 (features): for EACH image, note species/object type and 2-3 distinctive "
            "visible features (fur/color pattern, size, markings, damage/state, pose is NOT "
            "distinctive).\n"
            "Step 2 (verdict): if ALL images show the same individual, set coherent=true and leave "
            "subgroups empty. If they mix DIFFERENT individuals, set coherent=false and partition "
            "ALL indices into subgroups (each subgroup = one individual; a singleton subgroup is "
            "fine, e.g. one odd crop apart from the rest). When unsure whether two images show the "
            "same or a different individual, treat them as different (bias toward splitting).\n"
            "Return JSON {\"features\": [{\"index\": int, \"note\": str}], \"coherent\": bool, "
            "\"subgroups\": [[int, ...]]}"
        )
        schema = {"type": "object", "properties": {
            "features": {"type": "array", "items": {"type": "object", "properties": {
                "index": {"type": "integer"}, "note": {"type": "string"}},
                "required": ["index", "note"], "additionalProperties": False}},
            "coherent": {"type": "boolean"},
            "subgroups": {"type": "array", "items": {
                "type": "array", "items": {"type": "integer"}}}},
            "required": ["features", "coherent", "subgroups"], "additionalProperties": False}
        return self.judger._call_api(_image_messages(prompt, crops), schema, temperature=temperature)

    def pick_exemplars(self, names: list[str], candidate_crops: list[Path], *,
                       temperature: float = 0.0) -> dict[str, int]:
        """Multiple choice: for each roster character name, which numbered crop shows it?

        Returns {name: 0-based crop index or -1 when absent}. The judge only CHOOSES among
        SAM3-proposed crops — it never regresses coordinates nor invents detector language."""
        roster = "\n".join(f"- {n}" for n in names)
        prompt = (
            f"The {len(candidate_crops)} numbered images are candidate creature/character crops "
            "detected across one film's keyframes.\n"
            f"Film characters to anchor:\n{roster}\n"
            "For EACH character name, pick the SINGLE crop index (0-based, matching image order) "
            "that best shows that character clearly; use -1 if no crop shows it. Different "
            "characters must get different indices.\n"
            "Return JSON {\"assignments\": [{\"name\": str, \"index\": int}]}"
        )
        schema = {"type": "object", "properties": {"assignments": {
            "type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "index": {"type": "integer"}},
                "required": ["name", "index"], "additionalProperties": False}}},
            "required": ["assignments"], "additionalProperties": False}
        result = self.judger._call_api(_image_messages(prompt, candidate_crops), schema,
                                       temperature=temperature)
        return {str(a.get("name")): int(a.get("index", -1))
                for a in result.get("assignments", []) if isinstance(a, dict)}

    def classify_prop_relevance(self, props: list[dict], *,
                                temperature: float = 0.0) -> dict[str, str]:
        """Text-only judgment: which roster props are STORY props vs background dressing?

        BBB v11 evidence: tracking background classes ("rock") entity-izes every boulder in the
        film (122 props). A prop is worth individual tracking only if characters interact with
        it or it can undergo a state event; scenery-filler goes to location context instead.
        Returns {grounding_phrase: "story" | "background"}."""
        listing = "\n".join(
            f"- {p.get('grounding_phrase')}: {p.get('description') or p.get('name') or ''}"
            for p in props)
        prompt = (
            "Classify each film prop as 'story' or 'background'.\n"
            "story: characters pick it up / use it / eat it / destroy it, or the plot depends "
            "on it (a held apple, a crafted weapon, a chased butterfly).\n"
            "background: static scenery dressing that exists in many copies or is never "
            "interacted with (rocks, grass tufts, generic leaves, clouds).\n"
            f"Props:\n{listing}\n"
            "Return JSON {\"props\": [{\"grounding_phrase\": str, \"relevance\": "
            "\"story\"|\"background\"}]}"
        )
        schema = {"type": "object", "properties": {"props": {
            "type": "array", "items": {"type": "object", "properties": {
                "grounding_phrase": {"type": "string"},
                "relevance": {"type": "string", "enum": ["story", "background"]}},
                "required": ["grounding_phrase", "relevance"], "additionalProperties": False}}},
            "required": ["props"], "additionalProperties": False}
        result = self.judger._call_api([{"role": "user", "content": prompt}], schema,
                                       temperature=temperature)
        return {str(p.get("grounding_phrase")): str(p.get("relevance"))
                for p in result.get("props", []) if isinstance(p, dict)}

    def judge_same_individual_pair(self, a_crops: list[Path], b_crops: list[Path],
                                   a_label: str, b_label: str, *,
                                   temperature: float = 0.0) -> dict[str, Any]:
        """One focused call: do groups A and B show the SAME individual?

        Used as the VLM vote of the three-vote auto-merge (embedding support + this + the
        deterministic species/kind guard). Two-step reasoning inside the JSON keeps the
        judgment grounded on visible features rather than the labels."""
        prompt = (
            f"Images 1-{len(a_crops)} are crops of candidate entity A ({a_label!r}); the remaining "
            f"{len(b_crops)} images are crops of candidate entity B ({b_label!r}), all from the "
            "same film.\n"
            "Step 1 (features): describe A's and B's species/object type and 2-3 distinctive "
            "visible features each.\n"
            "Step 2 (verdict): same=true ONLY IF A and B are the same individual (same species "
            "AND consistent distinctive features; lighting/pose/scale may differ). Different "
            "individuals of the same species -> same=false. When unsure -> same=false.\n"
            "Return JSON {\"a_features\": str, \"b_features\": str, \"same\": bool, "
            "\"reason\": str}"
        )
        schema = {"type": "object", "properties": {
            "a_features": {"type": "string"}, "b_features": {"type": "string"},
            "same": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["a_features", "b_features", "same", "reason"],
            "additionalProperties": False}
        return self.judger._call_api(_image_messages(prompt, a_crops + b_crops), schema,
                                     temperature=temperature)

    def classify_credit_frames(self, frames: list[Path], *,
                               temperature: float = 0.0) -> list[bool]:
        """One batch: is each frame a non-diegetic title/credits/logo card (not a story scene)?

        Used to confirm the deterministic dark-card prefilter so night scenes are not excluded."""
        prompt = (
            "For each provided frame IN ORDER, answer whether it is a NON-DIEGETIC card: opening "
            "titles, end credits, studio logo, or a black/plain screen with overlaid text. A dark "
            "or dim STORY scene (night, cave, silhouette) is NOT a card.\n"
            f"There are exactly {len(frames)} frames; return exactly one boolean per frame.\n"
            "Return JSON: {\"credits\": [bool, ...]}"
        )
        schema = {"type": "object", "properties": {"credits": {
            "type": "array", "items": {"type": "boolean"}}},
            "required": ["credits"], "additionalProperties": False}
        result = self.judger._call_api(_image_messages(prompt, frames), schema,
                                       temperature=temperature)
        verdicts = [bool(v) for v in list(result.get("credits", []))[:len(frames)]]
        return verdicts + [False] * (len(frames) - len(verdicts))

    def discover_roster(self, frames: list[Path], known: list[dict[str, Any]], *,
                        temperature: float = 0.0) -> list[dict[str, Any]]:
        """Track-first §3.2: discover the global cast roster from movie-wide keyframes (ONE pass,
        batched by the pipeline). Unlike per-chunk discovery this is allowed to recognize a
        recurring character across the whole film (identity is later fixed by re-ID, not by name),
        so it does NOT forbid prior-knowledge identity — but descriptions stay pixel-grounded (D)."""
        known_txt = "\n".join(
            f"- {e['name']} ({e['kind']})"
            + (f" [attrs: {', '.join(f'{k}={v}' for k, v in e.get('static_attributes', {}).items())}]"
               if e.get('static_attributes') else "")
            for e in known)
        prompt = (
            "You are building the CAST ROSTER of an entire film from the representative keyframes "
            "provided (sampled across the whole movie). List every recurring or salient entity: "
            "main characters, distinct locations, and story-relevant props. Merge the same entity "
            "seen in different frames into ONE row (a character shown in several shots is one "
            "entry).\n"
            "A LOCATION is a narrative STAGE where action takes place — a place a character could "
            "walk around in (a meadow, a cave mouth, a riverbank, a forest clearing). A film has "
            "FEW locations (typically 3-6). Large objects VISIBLE in frame are NOT locations: a "
            "tree trunk, a branch, the sky, a canopy, a boulder are props or scenery, never "
            "locations.\n"
            f"Already-known roster entries:\n{known_txt or '(none yet)'}\n"
            "Return ONLY entities that are NOT in the already-known list. Do NOT repeat, rewrite "
            "or re-describe known entries — output them zero times. If these frames show nothing "
            "new, return an empty entities array.\n"
            "For each entity give:\n"
            "- name (short, unique, natural);\n"
            "- kind (character|location|prop);\n"
            "- grounding_phrase (a SHORT noun phrase for open-vocabulary detection, e.g. 'grey "
            "rabbit', 'red apple'. Nouns only: NO actions/moods/neighboring objects);\n"
            "- description (concise, appearance-only, usable as an image-generation reference);\n"
            "- static_attributes (STABLE identity keys only: species/subcategory, primary_color, "
            "size_class, object_type, location_type). Never actions/pose/mood.\n"
            "Descriptions and attributes must state ONLY what is visible in the frames (no "
            "franchise prior knowledge for colors/sizes).\n"
            "Return JSON: {\"entities\": [{\"name\", \"kind\", \"grounding_phrase\", "
            "\"description\", \"static_attributes\"}]}"
        )
        schema = {"type": "object", "properties": {"entities": {"type": "array", "items": {
            "type": "object", "properties": {
                "name": {"type": "string"}, "kind": {"type": "string", "enum": KINDS},
                "grounding_phrase": {"type": "string"}, "description": {"type": "string"},
                "static_attributes": {"type": "object",
                    "additionalProperties": {"type": "string"}}},
            "required": ["name", "kind", "grounding_phrase", "description", "static_attributes"],
            "additionalProperties": False}}},
            "required": ["entities"], "additionalProperties": False}
        result = self.judger._call_api(_image_messages(prompt, frames), schema, temperature=temperature)
        return list(result.get("entities", []))

    def name_entity(self, crops: list[Path], kind: str, static_attributes: dict[str, str], *,
                    known_names: list[str] | None = None,
                    temperature: float = 0.0) -> dict[str, str]:
        """Track-first §3.2: name+describe ONE already-identified entity from its best crop(s).

        Identity is already fixed by tracking+re-ID; the VLM only produces the authoritative
        display name and a pixel-grounded appearance description (used inline on first appearance).
        ``known_names`` is the film's already-assigned name vocabulary: if this entity visibly IS
        one of those individuals (a re-ID split), reusing the exact name makes the split trivially
        detectable downstream instead of hiding it behind a fresh synonym."""
        attrs = ", ".join(f"{k}={v}" for k, v in (static_attributes or {}).items())
        loc_note = (
            " Describe the environment/scenery only and ignore any characters or creatures in frame."
            if kind == "location" else "")
        location_options = (static_attributes or {}).get("roster_location_options", "")
        loc_taxonomy_note = (
            f" Choose the closest canonical setting name from this roster taxonomy: {location_options}."
            if kind == "location" and location_options else "")
        # NOTE: never instruct "reuse the same name for the same individual" — an 8B model
        # latches onto the list and echoes one name for everything (observed: an entire film
        # named "Grumpy Rabbit"). Same-individual fragments are handled by the visual identity
        # adjudication pass instead; names only need to be unique and descriptive.
        known_note = (
            f"\nNames already TAKEN by OTHER entities in this film: {', '.join(known_names)}.\n"
            "Your name MUST be different from every taken name and must describe THIS entity's "
            "own appearance."
            if known_names else "")
        prompt = (
            f"These crop(s) are all the SAME {kind} (identity already established by tracking). "
            f"Stable attributes: {attrs or '(none)'}.\n"
            "Give it ONE authoritative name and a concise appearance-only description usable as an "
            "image-generation reference. The name must be PLAIN and DESCRIPTIVE — category word "
            "plus visible attributes (\"White Rabbit\", \"Red Squirrel\", \"Grassy Meadow\"). "
            "NEVER poetic or whimsical nicknames (\"Snowball Goliath\", \"Lavender Perch\" are "
            "wrong: a reader must infer what the entity IS from its name alone). Describe ONLY "
            f"what is visible (no franchise prior knowledge).{loc_note}{loc_taxonomy_note}"
            f"{known_note} Return JSON: {{\"name\": str, \"description\": str}}"
        )
        schema = {"type": "object", "properties": {
            "name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "description"], "additionalProperties": False}
        return self.judger._call_api(_image_messages(prompt, crops), schema, temperature=temperature)

    def audit_entity(self, crops: list[Path], name: str, description: str, *,
                     temperature: float = 0.0) -> dict:
        """Machine review: check whether ALL crops show one entity consistent with name/description."""
        prompt = (
            f"These crops are filed as ONE entity named {name!r} with description: {description}\n"
            "Answer strictly whether ALL crops show the same single entity consistent with that "
            "name and description. If any crop shows a different entity or is inconsistent, set "
            "coherent=false and list 0-based wrong_crop_indices.\n"
            'Return JSON: {"coherent": bool, "wrong_crop_indices": [int], "note": str}'
        )
        schema = {"type": "object", "properties": {
            "coherent": {"type": "boolean"},
            "wrong_crop_indices": {"type": "array", "items": {"type": "integer"}},
            "note": {"type": "string"}},
            "required": ["coherent", "wrong_crop_indices"], "additionalProperties": False}
        return self.judger._call_api(_image_messages(prompt, crops), schema, temperature=temperature)

    def draft_chunk(self, frames: list[Path], present: list[dict[str, Any]],
                    prev_prompt: str, feedback: list[str], *, temperature: float = 0.0,
                    frame_indices: list[int] | None = None) -> dict[str, Any]:
        roster = "\n".join(
            f"- {p['name']} (id={p['entity_id']}, {p['kind']}, this_chunk_rep={p['representation_id']}"
            f", prior_reps={p.get('prior_representations', [])}"
            f", identity_scope={p.get('identity_scope') or 'unspecified'}"
            f", allowed_state_events={p.get('allowed_state_events', [])}"
            f", evidence_crops={len(p.get('crops') or [p])}"
            f"{', FIRST APPEARANCE — inline its appearance description: ' + p['description'] if p['first_appearance'] else ''})"
            for p in present)
        evidence_order = "; ".join(
            f"{p['name']}: " + ", ".join(
                c.get("representation_id", "crop") +
                (" (historical continuity reference)" if c.get("continuity_reference") else "")
                for c in p.get("crops", []))
            for p in present if p.get("crops")) or "(no entity crop evidence available)"
        fb = ("\nPrevious attempts had these problems; fix them:\n" +
              "\n".join(f"- {f}" for f in feedback)) if feedback else ""
        # Advisory timing (Q3): expose the sampled frames' absolute indices so the model can point a
        # state event at the nearest one. Best-effort only; chunk_id remains authoritative + scored.
        idx_line = (f"The sampled frames correspond, in order, to these absolute frame indices: "
                    f"{list(frame_indices)}.\n" if frame_indices else "")
        idx_rule = (
            "7. For EACH state event, also give event_frame = the absolute frame index, chosen from "
            "the sampled-index list above, of the sampled frame where the change is most clearly "
            "enacted (pick the closest sampled frame; this is a best-effort timestamp, not exact). "
            "If truly unsure, use the first index in the list.\n" if frame_indices else "")
        event_ret = (", \"event_frame\": int" if frame_indices else "")
        prompt = (
            "You are writing the generation prompt for one shot-chunk of a film, plus its "
            "state-change events. Sampled frames are followed by labeled entity evidence crops. "
            "The entity list is authoritative: do not discover, add, remove, or rename entities.\n"
            f"{idx_line}"
            f"Previous chunk prompt (context only): {prev_prompt or '(none)'}\n"
            f"Entities present in this chunk:\n{roster}\n\n"
            f"Entity evidence-crop order after the sampled frames: {evidence_order}.\n\n"
            "Rules for the prompt (STRICT):\n"
            "1. Screenplay style, present tense, one paragraph describing exactly what happens "
            "in this chunk (action, setting, camera if notable).\n"
            "2. Refer to every present entity BY ITS EXACT NAME above. For entities marked "
            "FIRST APPEARANCE, weave their appearance description into the sentence.\n"
            "3. If any entity is irreversibly changed or destroyed in this chunk (eaten, broken, "
            "killed...), the prompt MUST narrate that event, and you must also report it as a "
            "state event. Only report IRREVERSIBLE appearance/existence changes - not position "
            "moves or reversible actions. Judge a state event only from that entity's provided "
            "current evidence crops plus its listed prior_reps; never infer it from free-form "
            "whole-chunk narration, camera changes, pose, or lighting.\n"
            "   The roster's allowed_state_events list is a hard policy: use only those event "
            "types for that entity. An empty list means report NO state event for it.\n"
            "4. For each state event, list in deprecates_representations the specific prior_reps "
            "of that entity whose APPEARANCE is superseded by this event (use the rep ids from "
            "the roster). Leave it empty to deprecate ALL prior reps of that entity (the default "
            "when the whole appearance is gone, e.g. the entity is destroyed).\n"
            "5. NEVER mention chunk numbers, asset ids, memory, or where an entity appeared "
            "before. The prompt only describes what the viewer sees now.\n"
            "6. Describe ONLY what is visible in these frames - no film/franchise prior "
            "knowledge, no invented lighting, foliage, or actions that the frames do not show.\n"
            f"{idx_rule}"
            f"{fb}\n"
            "Return JSON: {\"prompt\": str, \"state_events\": [{\"entity_id\": str (from the "
            f"roster ids), \"event_type\": one of destroyed|consumed|broken|acquired|attached|"
            f"detached|appearance_changed, \"description\": str, "
            f"\"deprecates_representations\": [str]{event_ret}}}]}}. "
            "Use an empty list when nothing irreversible happens."
        )
        ev_props = {"entity_id": {"type": "string"},
                    "event_type": {"type": "string", "enum": [
                        "destroyed", "consumed", "broken", "acquired", "attached", "detached",
                        "appearance_changed"]},
                    "description": {"type": "string"},
                    "deprecates_representations": {"type": "array", "items": {"type": "string"}}}
        ev_required = ["entity_id", "event_type", "description", "deprecates_representations"]
        if frame_indices:
            ev_props["event_frame"] = {"type": "integer"}
            ev_required.append("event_frame")
        schema = {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "state_events": {"type": "array", "items": {"type": "object", "properties": ev_props,
                "required": ev_required, "additionalProperties": False}}},
            "required": ["prompt", "state_events"], "additionalProperties": False}
        evidence: list[Path] = []
        # First-appearance entities first: their crops are the ones the prompt MUST inline.
        for p in sorted(present, key=lambda p: not p.get("first_appearance")):
            for crop in p.get("crops") or [{"crop_path": p.get("crop_path")}]:
                path = Path(crop.get("crop_path") or "")
                if path.is_file():
                    evidence.append(path)
        # vLLM serves --limit-mm-per-prompt image=24; one image over returns a non-retryable
        # HTTP 400 (BBB v13: 7 chunks died exactly at the boundary). Sampled frames win;
        # evidence crops truncate by the priority order above.
        budget = 24 - len(frames)
        return self.judger._call_api(_image_messages(prompt, frames + evidence[:budget]),
                                     schema, temperature=temperature)


class VerifierRole:
    """Independent checklist verification (workflow step 8). Binary checks only."""

    def __init__(self, judger: VlmJudger, *, crop_audit_score_threshold: float = 0.60) -> None:
        self.judger = judger
        self.crop_audit_score_threshold = crop_audit_score_threshold

    def _should_audit_crop(self, p: dict[str, Any]) -> bool:
        """Skip per-crop VLM audit for location-kind, full-frame, and high-grounding-score crops
        (pitfalls: low value, high cost). presence_recall/precision already cover whether
        the scene/entity is there; re-judging a full-frame location crop or a 0.9-score grounded
        crop against its description adds cost without catching real asset-pollution bugs."""
        if not p.get("crop_path"):
            return False
        if p.get("kind") == "location":
            return False
        # full_frame (location) AND vlm_fallback (character/prop grounding miss -> whole frame)
        # are both whole-frame crops: judge_same_entity(whole_frame, description) almost always
        # returns true because the frame contains the entity, so the audit is pure VLM spend with
        # no signal. Skip both; presence_precision + human review catch the real grounding-miss
        # cases (pitfalls: per-crop audit cost).
        if p.get("bbox_source") in ("full_frame", "vlm_fallback"):
            return False
        if p.get("first_appearance") and p.get("kind") in ("character", "prop"):
            return True
        if p.get("grounding_score", 0.0) >= self.crop_audit_score_threshold:
            return False
        return True

    def _checklist(self, annotation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        present = annotation["present"]
        roster = "\n".join(f"- {p['name']} ({p['kind']}): {p['description']}" for p in present)
        events = "; ".join(e["description"] for e in annotation.get("state_events", [])) or "(none)"
        prompt = (
            "You are auditing a film-chunk annotation against the provided frames/video. "
            "Answer every question strictly from what is visible.\n"
            "IMPORTANT: when given a SPARSE frame sample (a handful of frames out of hundreds), "
            "brief actions and in-between motion may fall between samples. Fail a check ONLY on a "
            "clear contradiction with the frames (wrong color, wrong object, entity plainly "
            "absent) - NOT because a transient action or a described detail is merely not "
            "captured in those particular samples. When given the full chunk VIDEO, audit against "
            "everything visible across the clip.\n"
            f"Claimed entity roster:\n{roster or '(empty)'}\n"
            f"Claimed generation prompt:\n{annotation['prompt']}\n"
            f"Claimed irreversible state events: {events}\n\n"
            "Checklist (answer all):\n"
            "1. presence_recall: Is any salient visible entity (main character / location / "
            "story-relevant prop) MISSING from the roster? List missing ones.\n"
            "2. presence_precision: Is any roster entity NOT actually visible? List spurious ones.\n"
            "3. prompt_completeness: Does the prompt mention every roster entity by name and "
            "narrate every claimed state event? List omissions.\n"
            "4. prompt_faithful: Does the prompt describe only what is visible (no invented "
            "actions/objects, no references to other chunks or memory)? List violations.\n"
            "Return JSON: {\"checks\": [{\"check\": str, \"passed\": bool, \"detail\": str}]} "
            "with exactly the four checks above; detail lists the offending items or is empty."
        )
        schema = {"type": "object", "properties": {"checks": {"type": "array", "items": {
            "type": "object", "properties": {
                "check": {"type": "string"}, "passed": {"type": "boolean"}, "detail": {"type": "string"}},
            "required": ["check", "passed", "detail"], "additionalProperties": False}}},
            "required": ["checks"], "additionalProperties": False}
        return prompt, schema

    def _crop_checks(self, present: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(p: dict[str, Any]) -> dict[str, Any]:
            crop = Path(p["crop_path"])
            rid = p.get("representation_id", "")
            name = p.get("name", "")
            if not crop.is_file():
                return {"check": "crop_match", "passed": False,
                        "representation_id": rid, "name": name,
                        "detail": f"{name}: crop file missing"}
            ok = self.judger.judge_same_entity(str(crop), p["description"], p["kind"])
            return {"check": "crop_match", "passed": bool(ok),
                    "representation_id": rid, "name": name,
                    "detail": "" if ok else f"{name}: crop does not match description"}

        crops = [p for p in present if self._should_audit_crop(p)]
        if not crops:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(crops))) as pool:
            return list(pool.map(_one, crops))

    def verify_chunk(self, frames: list[Path], annotation: dict[str, Any],
                     *, temperature: float = 0.0) -> list[dict[str, Any]]:
        prompt, schema = self._checklist(annotation)
        result = self.judger._call_api(_image_messages(prompt, frames), schema, temperature=temperature)
        checks = list(result.get("checks", []))
        checks += self._crop_checks(annotation["present"])
        return checks

    def verify_chunk_video(self, video_path: Path, annotation: dict[str, Any],
                           *, temperature: float = 0.0) -> list[dict[str, Any]]:
        """Audit against the full chunk video clip (used on QA retry for flagged / multi-entity
        chunks where sparse frames structurally miss in-between actions). Same checklist; the
        prompt tells the model it has the full clip."""
        prompt, schema = self._checklist(annotation)
        video_url = str(Path(video_path).resolve())
        if not (video_url.startswith("http://") or video_url.startswith("https://")
                or video_url.startswith("file://")):
            video_url = f"file://{video_url}"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]}]
        result = self.judger._call_api(messages, schema, temperature=temperature)
        checks = list(result.get("checks", []))
        checks += self._crop_checks(annotation["present"])
        return checks


def to_feedback(checks: list[dict[str, Any]]) -> list[str]:
    """Failed checks -> counter-example strings for the prompt-optimizer retry."""
    return [f"{c['check']}: {c['detail'] or 'failed'}" for c in checks if not c["passed"]]
