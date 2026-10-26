"""v2 data contracts for MemStrata-Bench.

Authoritative field definitions live in ``docs/benchmark/schemas_and_contracts.md`` (schema-first);
this module must mirror that document field-by-field, no additions.

Self-containment (design principle #7/III): zero imports from `memstrata` or any SUT code.
Embeddings are never inlined here (principle #10): JSON carries only `embedding_key`
references into `gold/embeddings.safetensors`. `crop_path` is relative to the movie dir
(`derived/assets/<entity_id>/…`), so gold stays portable and video-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "2.0.0"

REQUIRED = "required"
OPTIONAL = "optional"

CONTINUITY = "continuity"
INTRODUCE = "introduce"


# ---------------------------------------------------------------------------
# SUT-facing contracts (bench <-> SUT, pure JSON)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromptPacket:
    """bench -> SUT, before scoring. The prompt is the complete generation source
    (principle #9) and never contains provenance/retrieval hints (principle #3)."""

    chunk_id: int
    prompt: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromptPacket:
        return cls(chunk_id=int(d["chunk_id"]), prompt=str(d["prompt"]),
                   schema_version=str(d.get("schema_version", SCHEMA_VERSION)))


@dataclass(slots=True)
class SelectedAsset:
    """One asset slot in the SUT's composed context."""

    asset_id: str
    representation_ids: list[str] = field(default_factory=list)
    function: str = ""  # subject | background | style-anchor | ...
    strength: str = OPTIONAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelectedAsset:
        return cls(asset_id=str(d["asset_id"]),
                   representation_ids=[str(x) for x in d.get("representation_ids", [])],
                   function=str(d.get("function", "")),
                   strength=str(d.get("strength", OPTIONAL)))


@dataclass(slots=True)
class InstructionItem:
    asset_ref: str
    requirement: str  # continuity | introduce

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstructionItem:
        return cls(asset_ref=str(d["asset_ref"]), requirement=str(d["requirement"]))


@dataclass(slots=True)
class Instruction:
    per_asset: list[InstructionItem] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)  # representation ids the SUT rules out

    def to_dict(self) -> dict[str, Any]:
        return {"per_asset": [i.to_dict() for i in self.per_asset],
                "exclusions": list(self.exclusions)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Instruction:
        return cls(per_asset=[InstructionItem.from_dict(i) for i in d.get("per_asset", [])],
                   exclusions=[str(x) for x in d.get("exclusions", [])])


@dataclass(slots=True)
class ComposedContextRecord:
    """SUT -> bench, before scoring. IDs must be the authoritative IDs obtained
    through ingestion feedback (naming-authority principle #4)."""

    chunk_id: int
    selected: list[SelectedAsset] = field(default_factory=list)
    instruction: Instruction = field(default_factory=Instruction)
    memory_keys: list[str] = field(default_factory=list)
    timing_ms: float = 0.0
    model_calls: int = 0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "chunk_id": self.chunk_id,
                "selected": [s.to_dict() for s in self.selected],
                "instruction": self.instruction.to_dict(),
                "memory_keys": list(self.memory_keys),
                "timing_ms": self.timing_ms, "model_calls": self.model_calls}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ComposedContextRecord:
        return cls(chunk_id=int(d["chunk_id"]),
                   selected=[SelectedAsset.from_dict(s) for s in d.get("selected", [])],
                   instruction=Instruction.from_dict(d.get("instruction", {})),
                   memory_keys=[str(x) for x in d.get("memory_keys", [])],
                   timing_ms=float(d.get("timing_ms", 0.0)),
                   model_calls=int(d.get("model_calls", 0)),
                   schema_version=str(d.get("schema_version", SCHEMA_VERSION)))


@dataclass(slots=True)
class Observation:
    """One authoritative entity observation inside the post-chunk reference feedback."""

    entity_id: str
    kind: str  # character | location | prop
    name: str
    representation_id: str
    crop_path: str
    description: str = ""  # only provided on first appearance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        return cls(entity_id=str(d["entity_id"]), kind=str(d["kind"]), name=str(d["name"]),
                   representation_id=str(d["representation_id"]),
                   crop_path=str(d["crop_path"]), description=str(d.get("description", "")))


@dataclass(slots=True)
class StateEvent:
    """Irreversible appearance/existence change; ground truth for Avoidance (D4).
    Visible to the SUT (narrated in the prompt per principle #9 and echoed here);
    the materialized per-chunk forbidden table is scoring-only and never shared."""

    event_id: str
    chunk_id: int
    description: str
    deprecates: list[str] = field(default_factory=list)  # representation ids
    # Optional precise timing (Q3): the frame/second the change is enacted. chunk_id is authoritative
    # and always present; frame_index/seconds are best-effort (null when not localizable).
    frame_index: int | None = None
    seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StateEvent:
        fi = d.get("frame_index")
        sec = d.get("seconds")
        return cls(event_id=str(d["event_id"]), chunk_id=int(d["chunk_id"]),
                   description=str(d.get("description", "")),
                   deprecates=[str(x) for x in d.get("deprecates", [])],
                   frame_index=(int(fi) if fi is not None else None),
                   seconds=(float(sec) if sec is not None else None))


@dataclass(slots=True)
class ObservationPacket:
    """bench -> SUT ingester, after scoring (generator-reference feedback)."""

    chunk_id: int
    chunk_video: str
    observations: list[Observation] = field(default_factory=list)
    state_events: list[StateEvent] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "chunk_id": self.chunk_id,
                "chunk_video": self.chunk_video,
                "observations": [o.to_dict() for o in self.observations],
                "state_events": [e.to_dict() for e in self.state_events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationPacket:
        return cls(chunk_id=int(d["chunk_id"]), chunk_video=str(d["chunk_video"]),
                   observations=[Observation.from_dict(o) for o in d.get("observations", [])],
                   state_events=[StateEvent.from_dict(e) for e in d.get("state_events", [])],
                   schema_version=str(d.get("schema_version", SCHEMA_VERSION)))


# ---------------------------------------------------------------------------
# Gold contracts (frozen annotation, scoring-only)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Representation:
    representation_id: str
    chunk_id: int
    crop_path: str
    bbox: list[int] = field(default_factory=list)  # [ymin, xmin, ymax, xmax], 0-1000
    bbox_source: str = ""  # grounding_dino | vlm_fallback | full_frame
    frame_index: int = -1
    embedding_key: str = ""
    state: str = "default"
    qa: dict[str, Any] = field(default_factory=dict)  # {"verified": bool, "rounds": int, "flagged": bool}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Representation:
        return cls(representation_id=str(d["representation_id"]), chunk_id=int(d["chunk_id"]),
                   crop_path=str(d["crop_path"]), bbox=[int(x) for x in d.get("bbox", [])],
                   bbox_source=str(d.get("bbox_source", "")), frame_index=int(d.get("frame_index", -1)),
                   embedding_key=str(d.get("embedding_key", "")), state=str(d.get("state", "default")),
                   qa=dict(d.get("qa", {})))


@dataclass(slots=True)
class Entity:
    entity_id: str
    kind: str
    name: str
    description: str
    first_chunk: int
    representations: list[Representation] = field(default_factory=list)
    state_events: list[StateEvent] = field(default_factory=list)
    # Stable identity attributes (species/subcategory, primary_color, size_class, ...) used by
    # the consolidation identity funnel (principle: static-identity vs dynamic-state decoupling).
    # Free-form string dict; never carries appearance descriptions. Backward compatible: old
    # gold without this field deserializes to {}.
    static_attributes: dict[str, str] = field(default_factory=dict)
    # Time metadata (Q3): computed deterministically from tracklet/​shot presence spans (§4.1). All
    # optional/backward-compatible (old gold -> empty/None). Not consumed by SUT; drives the temporal
    # MemRecall metric + human review. presence_spans are closed-inclusive absolute frame ranges.
    presence_spans: list[list[int]] = field(default_factory=list)
    first_frame: int | None = None
    first_seconds: float | None = None
    last_frame: int | None = None
    last_seconds: float | None = None
    screen_time_seconds: float | None = None
    max_absence_frames: int | None = None
    max_absence_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "kind": self.kind, "name": self.name,
                "description": self.description, "first_chunk": self.first_chunk,
                "presence_spans": [list(s) for s in self.presence_spans],
                "first_frame": self.first_frame, "first_seconds": self.first_seconds,
                "last_frame": self.last_frame, "last_seconds": self.last_seconds,
                "screen_time_seconds": self.screen_time_seconds,
                "max_absence_frames": self.max_absence_frames,
                "max_absence_seconds": self.max_absence_seconds,
                "representations": [r.to_dict() for r in self.representations],
                "state_events": [e.to_dict() for e in self.state_events],
                "static_attributes": dict(self.static_attributes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        def _opt_i(v):
            return int(v) if v is not None else None

        def _opt_f(v):
            return float(v) if v is not None else None

        # Canonical field is ``representations``. S7 ``build_gold`` historically also
        # wrote a parallel ``reps`` list and left ``representations`` empty; accept
        # either so frozen gold remains scorible without a rewrite.
        raw_reps = d.get("representations") or d.get("reps") or []
        return cls(entity_id=str(d["entity_id"]), kind=str(d["kind"]), name=str(d["name"]),
                   description=str(d.get("description", "")), first_chunk=int(d["first_chunk"]),
                   representations=[Representation.from_dict(r) for r in raw_reps],
                   state_events=[StateEvent.from_dict(e) for e in d.get("state_events", [])],
                   static_attributes={str(k): str(v) for k, v in d.get("static_attributes", {}).items()},
                   presence_spans=[[int(x) for x in s] for s in d.get("presence_spans", [])],
                   first_frame=_opt_i(d.get("first_frame")), first_seconds=_opt_f(d.get("first_seconds")),
                   last_frame=_opt_i(d.get("last_frame")), last_seconds=_opt_f(d.get("last_seconds")),
                   screen_time_seconds=_opt_f(d.get("screen_time_seconds")),
                   max_absence_frames=_opt_i(d.get("max_absence_frames")),
                   max_absence_seconds=_opt_f(d.get("max_absence_seconds")))


@dataclass(slots=True)
class EntityRegistry:
    movie_id: str
    entities: list[Entity] = field(default_factory=list)
    human_reviewed: bool = False
    annotation_provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "movie_id": self.movie_id,
                "human_reviewed": self.human_reviewed,
                "annotation_provenance": dict(self.annotation_provenance),
                "entities": [e.to_dict() for e in self.entities]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntityRegistry:
        return cls(movie_id=str(d["movie_id"]),
                   entities=[Entity.from_dict(e) for e in d.get("entities", [])],
                   human_reviewed=bool(d.get("human_reviewed", False)),
                   annotation_provenance=dict(d.get("annotation_provenance", {})),
                   schema_version=str(d.get("schema_version", SCHEMA_VERSION)))

    def all_state_events(self) -> list[StateEvent]:
        return [e for entity in self.entities for e in entity.state_events]


@dataclass(slots=True)
class GoldInstruction:
    entity_id: str
    requirement: str  # continuity | introduce
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldInstruction:
        return cls(entity_id=str(d["entity_id"]), requirement=str(d["requirement"]),
                   note=str(d.get("note", "")))


@dataclass(slots=True)
class ForbiddenRep:
    representation_id: str
    reason: str = ""  # event_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForbiddenRep:
        return cls(representation_id=str(d["representation_id"]), reason=str(d.get("reason", "")))


@dataclass(slots=True)
class ChunkAnnotation:
    chunk_id: int
    shot_span: list[int] = field(default_factory=list)
    frame_span: list[int] = field(default_factory=list)
    prompt: str = ""
    present: list[str] = field(default_factory=list)
    first_appearances: list[str] = field(default_factory=list)
    gold_instructions: list[GoldInstruction] = field(default_factory=list)
    forbidden: list[ForbiddenRep] = field(default_factory=list)
    scenario_tags: list[str] = field(default_factory=list)
    # Q3 metadata (not scored): chunk time window in seconds, and Q2 prompt-completeness report.
    seconds_span: list[float] = field(default_factory=list)
    prompt_completeness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "shot_span": list(self.shot_span),
                "frame_span": list(self.frame_span), "seconds_span": list(self.seconds_span),
                "prompt": self.prompt,
                "present": list(self.present), "first_appearances": list(self.first_appearances),
                "gold_instructions": [g.to_dict() for g in self.gold_instructions],
                "forbidden": [f.to_dict() for f in self.forbidden],
                "prompt_completeness": dict(self.prompt_completeness),
                "scenario_tags": list(self.scenario_tags)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkAnnotation:
        return cls(chunk_id=int(d["chunk_id"]), shot_span=[int(x) for x in d.get("shot_span", [])],
                   frame_span=[int(x) for x in d.get("frame_span", [])], prompt=str(d.get("prompt", "")),
                   present=[str(x) for x in d.get("present", [])],
                   first_appearances=[str(x) for x in d.get("first_appearances", [])],
                   gold_instructions=[GoldInstruction.from_dict(g) for g in d.get("gold_instructions", [])],
                   forbidden=[ForbiddenRep.from_dict(f) for f in d.get("forbidden", [])],
                   scenario_tags=[str(x) for x in d.get("scenario_tags", [])],
                   seconds_span=[float(x) for x in d.get("seconds_span", [])],
                   prompt_completeness=dict(d.get("prompt_completeness", {})))


@dataclass(slots=True)
class ChunkAnnotations:
    movie_id: str
    chunks: list[ChunkAnnotation] = field(default_factory=list)
    human_reviewed: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "movie_id": self.movie_id,
                "human_reviewed": self.human_reviewed,
                "chunks": [c.to_dict() for c in self.chunks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkAnnotations:
        return cls(movie_id=str(d["movie_id"]),
                   chunks=[ChunkAnnotation.from_dict(c) for c in d.get("chunks", [])],
                   human_reviewed=bool(d.get("human_reviewed", False)),
                   schema_version=str(d.get("schema_version", SCHEMA_VERSION)))
