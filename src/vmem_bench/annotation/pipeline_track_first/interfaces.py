"""Model-backend protocols for the annotation pipeline.

The pipeline is written against these interfaces so the whole orchestration can be
self-checked end-to-end with deterministic stubs (no GPU / no VLM service).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AnnotatorVlm(Protocol):
    """Drafting role: proposes entities, prompts and state events (discrete outputs only)."""

    def discover_entities(self, frames: list[Path], known_entities: list[dict[str, str]],
                          feedback: list[str], *, temperature: float = 0.0) -> list[dict[str, str]]:
        """Return [{"name", "kind", "description"}]. ``known_entities`` carries the
        authoritative names discovered so far (naming consistency); ``feedback`` carries
        verifier counter-examples for the prompt-optimizer retry loop. ``temperature`` is raised
        by the pipeline on retry / redundancy branches to decorrelate candidates."""
        ...

    def judge_same_entity(self, crop: Path, description: str, kind: str) -> bool:
        ...

    def draft_chunk(self, frames: list[Path], present: list[dict[str, Any]],
                    prev_prompt: str, feedback: list[str], *,
                    temperature: float = 0.0) -> dict[str, Any]:
        """Return {"prompt": str, "state_events": [{"entity_id", "description", "deprecates_states": [str]}]}.
        The prompt must narrate every present entity (inline description on first
        appearance) and every state event, with no provenance hints (D3/D3b)."""
        ...


@runtime_checkable
class VerifierVlm(Protocol):
    """Independent verification role (different prompts / frame sampling than the annotator)."""

    def verify_chunk(self, frames: list[Path], annotation: dict[str, Any],
                     *, temperature: float = 0.0) -> list[dict[str, Any]]:
        """Return checklist results: [{"check", "passed", "detail"}]. Checks:
        presence_recall, presence_precision, prompt_completeness, crop_match."""
        ...


@runtime_checkable
class RosterVlm(Protocol):
    """Track-first §3.2: discover the global cast roster ONCE from movie-wide keyframes.

    Returns [{"name", "kind", "grounding_phrase", "static_attributes", "description"}]. Called on
    small batches of keyframes; the pipeline merges batches by (name, kind)."""

    def discover_roster(self, frames: list[Path], known: list[dict[str, Any]], *,
                        temperature: float = 0.0) -> list[dict[str, Any]]:
        ...


@runtime_checkable
class NamerVlm(Protocol):
    """Track-first §3.2: give ONE authoritative name+description to an already-identified entity
    (identity is fixed by tracking+re-ID; the VLM only names its best crop(s))."""

    def name_entity(self, crops: list[Path], kind: str, static_attributes: dict[str, str], *,
                    temperature: float = 0.0) -> dict[str, str]:
        """Return {"name": str, "description": str} from the entity's representative crop(s)."""
        ...


@runtime_checkable
class Grounder(Protocol):
    def ground(self, image: Path, phrase: str) -> tuple[list[int], float] | None:
        """Locate ``phrase`` in ``image``; return ([ymin, xmin, ymax, xmax] in 0-1000, score)."""
        ...


@runtime_checkable
class ImageEmbedder(Protocol):
    def embed_image(self, image: Path) -> list[float]:
        ...
