"""Human-confirmed canonical roster contract for production gold annotation.

Automatic roster discovery remains useful for proposing candidates, but it is not allowed to
define stable benchmark identities.  A production run consumes this small JSON contract instead:
humans decide *which* benchmark-relevant entities exist and provide a few representative crops;
the pipeline then does the expensive localization/tracking work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vmem_bench.common.vecmath import cosine_similarity


SEED_VERSION = 1
IDENTITY_SCOPES = frozenset({"individual", "category", "scene"})
STATE_EVENT_TYPES = frozenset({
    "destroyed",
    "consumed",
    "broken",
    "acquired",
    "attached",
    "detached",
    "appearance_changed",
})
_ID_RE = re.compile(r"^(char|prop|loc)_[a-z0-9]+(?:_[a-z0-9]+)*$")
_PREFIX = {"character": "char_", "prop": "prop_", "location": "loc_"}


@dataclass(frozen=True, slots=True)
class CanonicalEntitySeed:
    entity_id: str
    name: str
    kind: str
    identity_scope: str
    description: str
    grounding_phrases: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    exemplar_crops: tuple[str, ...] = ()
    static_attributes: dict[str, str] = field(default_factory=dict)
    allowed_state_events: tuple[str, ...] = ()

    @property
    def primary_grounding_phrase(self) -> str:
        return self.grounding_phrases[0]

    def to_roster_record(self) -> dict[str, Any]:
        """Return the plain-data shape consumed by the existing track-first pipeline."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "kind": self.kind,
            "identity_scope": self.identity_scope,
            "grounding_phrase": self.primary_grounding_phrase,
            "grounding_phrases": list(self.grounding_phrases),
            "aliases": list(self.aliases),
            "description": self.description,
            "static_attributes": {
                **self.static_attributes,
                "description": self.description,
            },
            "exemplar_crops": list(self.exemplar_crops),
            "allowed_state_events": list(self.allowed_state_events),
        }


@dataclass(frozen=True, slots=True)
class CanonicalRosterSeed:
    movie_id: str
    human_confirmed: bool
    entities: tuple[CanonicalEntitySeed, ...]
    source_path: Path
    version: int = SEED_VERSION
    ignored_tracks: tuple[str, ...] = ()

    def to_roster(self) -> list[dict[str, Any]]:
        return [entity.to_roster_record() for entity in self.entities]

    @property
    def by_id(self) -> dict[str, CanonicalEntitySeed]:
        return {entity.entity_id: entity for entity in self.entities}


@dataclass(frozen=True, slots=True)
class SeedAssignment:
    entity_id: str | None
    reason: str
    best_score: float | None
    margin: float | None
    scores: dict[str, float] = field(default_factory=dict)


def assign_closed_set(
    signature: list[float] | None,
    *,
    kind: str,
    phrase_owner_id: str,
    seed: CanonicalRosterSeed,
    exemplar_embeddings: dict[str, list[list[float]]],
    min_similarity: float,
    min_margin: float,
) -> SeedAssignment:
    """Assign one tracklet to a canonical seed entity, or explicitly reject it.

    Category props are phrase-owned (their ontology intentionally collapses instances). Individual
    identities use multi-view exemplar similarity across all same-kind candidates.  Low absolute
    score and ambiguous top-two margins produce ``unknown/reject`` rather than a forced gold label.
    """
    owner = seed.by_id.get(phrase_owner_id)
    if owner is None or owner.kind != kind:
        return SeedAssignment(None, "unknown_phrase_owner", None, None)
    if owner.identity_scope == "category":
        return SeedAssignment(owner.entity_id, "category_phrase", None, None)
    if signature is None:
        return SeedAssignment(None, "missing_track_signature", None, None)
    scores: dict[str, float] = {}
    for candidate in seed.entities:
        if candidate.kind != kind or candidate.identity_scope != "individual":
            continue
        vectors = exemplar_embeddings.get(candidate.entity_id, [])
        if vectors:
            scores[candidate.entity_id] = max(cosine_similarity(signature, vec) for vec in vectors)
    if not scores:
        return SeedAssignment(None, "no_candidate_exemplars", None, None)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score
    if best_score < min_similarity:
        return SeedAssignment(None, "below_similarity_floor", best_score, margin, scores)
    if len(ranked) > 1 and margin < min_margin:
        return SeedAssignment(None, "ambiguous_margin", best_score, margin, scores)
    return SeedAssignment(best_id, "exemplar_match", best_score, margin, scores)


def _strings(value: object, *, field_name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if value is None and allow_empty:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    cleaned = tuple(item.strip() for item in value if item.strip())
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one non-empty string")
    return cleaned


def _resolve_exemplars(raw: object, *, base_dir: Path, field_name: str) -> tuple[str, ...]:
    paths = _strings(raw, field_name=field_name)
    resolved: list[str] = []
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"{field_name} exemplar does not exist: {path}")
        resolved.append(str(path))
    return tuple(resolved)


def _parse_entity(raw: object, *, index: int, base_dir: Path) -> CanonicalEntitySeed:
    if not isinstance(raw, dict):
        raise ValueError(f"entities[{index}] must be an object")
    entity_id = str(raw.get("entity_id") or "").strip()
    name = str(raw.get("name") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    scope = str(raw.get("identity_scope") or "").strip()
    description = str(raw.get("description") or "").strip()
    if not _ID_RE.fullmatch(entity_id):
        raise ValueError(f"entities[{index}].entity_id is not canonical snake_case: {entity_id!r}")
    if kind not in _PREFIX or not entity_id.startswith(_PREFIX[kind]):
        raise ValueError(f"entities[{index}] kind/id prefix mismatch: {kind!r}, {entity_id!r}")
    if not name or not description:
        raise ValueError(f"entities[{index}] requires non-empty name and description")
    if scope not in IDENTITY_SCOPES:
        raise ValueError(f"entities[{index}].identity_scope must be one of {sorted(IDENTITY_SCOPES)}")
    if (kind == "location") != (scope == "scene"):
        raise ValueError("location seeds must use identity_scope='scene'; scene scope is location-only")
    grounding = _strings(
        raw.get("grounding_phrases"),
        field_name=f"entities[{index}].grounding_phrases",
        allow_empty=False,
    )
    aliases = _strings(raw.get("aliases"), field_name=f"entities[{index}].aliases")
    exemplars = _resolve_exemplars(
        raw.get("exemplar_crops"),
        base_dir=base_dir,
        field_name=f"entities[{index}].exemplar_crops",
    )
    attrs_raw = raw.get("static_attributes") or {}
    if not isinstance(attrs_raw, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in attrs_raw.items()):
        raise ValueError(f"entities[{index}].static_attributes must be string-to-string")
    events = _strings(
        raw.get("allowed_state_events"),
        field_name=f"entities[{index}].allowed_state_events",
    )
    unknown_events = sorted(set(events) - STATE_EVENT_TYPES)
    if unknown_events:
        raise ValueError(f"entities[{index}] has unsupported state event types: {unknown_events}")
    if scope != "individual" and events:
        raise ValueError(
            f"entities[{index}] only identity_scope='individual' may allow lifecycle events")
    if scope == "individual" and kind in ("character", "prop") and not exemplars:
        raise ValueError(f"entities[{index}] individual seed requires at least one exemplar crop")
    return CanonicalEntitySeed(
        entity_id=entity_id,
        name=name,
        kind=kind,
        identity_scope=scope,
        description=description,
        grounding_phrases=grounding,
        aliases=aliases,
        exemplar_crops=exemplars,
        static_attributes=dict(attrs_raw),
        allowed_state_events=events,
    )


def load_roster_seed(
    path: Path,
    *,
    expected_movie_id: str | None = None,
    require_confirmed: bool = True,
) -> CanonicalRosterSeed:
    """Load and validate one human-curated roster seed.

    Relative exemplar paths are resolved beside the seed JSON so the seed directory is portable.
    No model output is trusted here: malformed IDs, duplicate aliases/phrases, missing exemplars,
    and non-confirmed production seeds fail before any GPU work starts.
    """
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load roster seed {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("roster seed root must be an object")
    version = int(payload.get("version", 0))
    if version != SEED_VERSION:
        raise ValueError(f"unsupported roster seed version {version}; expected {SEED_VERSION}")
    movie_id = str(payload.get("movie_id") or "").strip()
    if not movie_id:
        raise ValueError("roster seed requires movie_id")
    if expected_movie_id and movie_id != expected_movie_id:
        raise ValueError(
            f"roster seed movie_id={movie_id!r} does not match run movie_id={expected_movie_id!r}")
    confirmed = bool(payload.get("human_confirmed", False))
    if require_confirmed and not confirmed:
        raise ValueError("production roster seed must set human_confirmed=true")
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ValueError("roster seed requires a non-empty entities list")
    entities = tuple(
        _parse_entity(raw, index=index, base_dir=path.parent)
        for index, raw in enumerate(raw_entities)
    )
    ids = [entity.entity_id for entity in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("roster seed entity_id values must be unique")
    names = [(entity.kind, entity.name.casefold()) for entity in entities]
    if len(names) != len(set(names)):
        raise ValueError("roster seed canonical names must be unique within each kind")
    phrase_owner: dict[tuple[str, str], str] = {}
    for entity in entities:
        for phrase in entity.grounding_phrases:
            key = (entity.kind, phrase.casefold())
            owner = phrase_owner.setdefault(key, entity.entity_id)
            if owner != entity.entity_id:
                raise ValueError(
                    f"grounding phrase {phrase!r} is shared by {owner} and {entity.entity_id}")
    return CanonicalRosterSeed(
        movie_id=movie_id,
        human_confirmed=confirmed,
        entities=entities,
        source_path=path,
        version=version,
        ignored_tracks=_strings(payload.get("ignored_tracks"), field_name="ignored_tracks"),
    )
