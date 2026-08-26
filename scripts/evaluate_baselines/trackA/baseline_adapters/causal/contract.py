"""New-protocol causal SUT contract for MemStrata-Bench baselines.

This is the **only** contract for causal baselines. The retired gold-replay /
online-gold / gold-id-mapping path (which fed gold pixels or gold entity ids to
the SUT) has been removed; see ``README.md``.

Protocol (three iron rules, mirror of ``docs/benchmark/running_eval.md`` §0/§1):

1. Gold is S4-human-reviewed **text only**. The bench never hands the SUT any
   pixels it produced, nor any ``present`` / ``first_appearances`` / roster ids.
2. Every reference image scored is produced by the SUT itself. Here the SUT
   observes the **real segment** (which replaces its generator output, to
   remove generation noise), runs its **own** perception / memory write, then at
   the next prompt runs its **own** retrieval.
3. Per segment, bench gives the SUT exactly two things: the segment ``prompt`` text
   and the raw ``segment`` video. Nothing else.

Each baseline adapter implements :class:`CausalMemoryAdapter`. The adapter owns
its native memory space (latents / KV / role-wise slots / frames). It never sees
gold. When it retrieves, it returns items carrying a **temporal identity**
(absolute source-video seconds and/or legacy source chunk id); latent / KV systems stay
on that timestamp path, while image-native systems may additionally provide the
absolute path to their own composed reference image. The bench-side
:mod:`frame_materializer` either uses that guarded SUT image directly or renders
a fallback frame from the source video -- this is the "map retrieved memory to
pixels via temporal consistency" step. The adapter itself does not read or write
gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class MovieContext:
    """Immutable per-movie context handed to the adapter at ``reset``.

    ``seconds_span_by_chunk`` and ``fps`` come from the gold *layout*
    (``chunk_index.json`` / ``chunk_annotations.json`` seconds spans) -- layout
    is bench-side metadata, not an answer, so exposing it is allowed. No
    ``present`` / roster / crop is ever included here.
    """

    movie_id: str
    source_video: str  # absolute path to the real source video
    fps: float
    seconds_span_by_chunk: dict[int, tuple[float, float]]
    work_dir: str  # scratch dir the adapter may use for its own memory artifacts
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SegmentObservation:
    """The raw chunk segment handed to the SUT to build memory from.

    ``segment_video`` is a clip cut from the real source video for this chunk;
    it stands in for the SUT generator's output so evaluation isolates the memory
    mechanism. ``seconds_span`` locates the clip inside the source video, so the
    adapter can tag anything it stores with absolute source seconds.

    ``prompt_text`` is this chunk's prompt. Text/entity-conditioned memory writers
    (MemFlow's text-saliency bank, IAMFlow's entity attention) must condition the
    write on it -- the same prompt the real generator would have used for this
    chunk. Purely visual writers (LongLive-RAG's AE descriptors) ignore it.
    """

    chunk_id: int
    segment_video: str
    seconds_span: tuple[float, float]
    fps: float
    prompt_text: str = ""


@dataclass(slots=True, frozen=True)
class ComposeRequest:
    """A prompt handed to the SUT to compose context from current memory."""

    chunk_id: int
    prompt_text: str
    seconds_span: tuple[float, float]


@dataclass(slots=True)
class RetrievedItem:
    """One historical memory item the SUT retrieved into the current context.

    The adapter must supply a temporal identity. Provide ``source_seconds``
    (absolute in the source video) when known; otherwise provide
    ``source_chunk_id`` and the materializer will fall back to that chunk's span.
    Image-native systems may also provide ``image_path``: an absolute path to the
    SUT's own composed reference image. Latent / KV / timestamp-only systems
    leave it ``None`` and keep the source-frame materialization path.
    ``latent_index`` / ``kv_slot`` / ``raw_ref`` are native debug handles and
    never used for scoring.
    """

    evidence_kind: str  # frame | latent | kv | slot | reference_image
    source_seconds: float | None = None
    source_chunk_id: int | None = None
    latent_index: int | None = None
    score: float | None = None  # native retrieval score (debug/analysis only)
    raw_ref: str = ""
    image_path: str | None = None  # SUT-composed reference image, absolute path.

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_kind": self.evidence_kind,
            "source_seconds": self.source_seconds,
            "source_chunk_id": self.source_chunk_id,
            "latent_index": self.latent_index,
            "score": self.score,
            "raw_ref": self.raw_ref,
            "image_path": self.image_path,
        }


@dataclass(slots=True)
class RetrievedMemory:
    """The SUT's per-segment retrieval decision, before frame materialization."""

    chunk_id: int
    items: list[RetrievedItem] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "items": [it.to_dict() for it in self.items],
            "extras": dict(self.extras),
        }


@runtime_checkable
class CausalMemoryAdapter(Protocol):
    """Per-baseline SUT. Owns a native memory space; never sees gold."""

    name: str

    def reset(self, movie: MovieContext) -> None:
        """Start a fresh rollout for one movie."""
        ...

    def observe_segment(self, obs: SegmentObservation) -> None:
        """Ingest the real chunk segment into the native memory (memory WRITE).

        Must be called for chunk t **after** :meth:`compose` for chunk t, so the
        SUT cannot peek at chunk t's video while composing chunk t's context.
        """
        ...

    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        """Run native retrieval over current memory and return temporal items."""
        ...

    def finalize(self) -> dict[str, Any] | None:
        """Optional run-level metadata (config, memory sizes, retrieval mode)."""
        ...
