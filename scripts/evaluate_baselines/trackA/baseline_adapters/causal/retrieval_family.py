"""Causal thin adapter for the self-built retrieval control/ablation family."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation

# Self-contained: these four retrieval baselines use the BENCH-owned reimplementation under
# vmem_bench (they never import the SUT package `memstrata`). baseline_adapters live at
# scripts/evaluate_baselines/trackA/baseline_adapters/causal/; the benchmark src is parents[5]/src.
_BENCH_SRC = Path(__file__).resolve().parents[5] / "src"
if str(_BENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_BENCH_SRC))

from vmem_bench.baseline_adapters.external.retrieval.retrieval_baselines import (  # noqa: E402
    MemoryRetrievalStore,
    RetrievedRef,
    Retriever,
    RetrieverConfig,
    build_retriever,
)

_DEFAULT_BUDGET = 16


def _canonical_variant(name: str) -> str:
    key = (name or "seg_framererank").strip().lower()
    if key in {"seg_uniform", "seg_dinokey", "seg_framererank", "frame_text"}:
        return f"{key}_ablation"
    if key in {"recency", "bm25_desc", "random"}:
        return f"{key}_ctrl"
    return key


def _item_from_ref(ref: RetrievedRef) -> RetrievedItem:
    return RetrievedItem(
        evidence_kind="frame",
        source_seconds=ref.source_seconds,
        score=float(ref.score),
        raw_ref=f"{ref.arm}:{ref.representation_id or ref.asset_id or ref.source_seconds}",
    )


class RetrievalFamilyAdapter:
    """Adapter boundary: MemStrata skill refs -> causal bench temporal items."""

    def __init__(
        self,
        *,
        variant: str = "seg_framererank",
        config: RetrieverConfig | None = None,
        budget: int = _DEFAULT_BUDGET,
    ) -> None:
        self.variant = _canonical_variant(variant)
        self.config = config or RetrieverConfig.from_env()
        self.budget = int(budget)
        self.store = MemoryRetrievalStore(self.config)
        self.retriever: Retriever = build_retriever(self.variant, store=self.store)
        self.name = f"retrieval_{getattr(self.retriever, 'name', self.variant)}"
        self._movie: MovieContext | None = None

    def set_budget(self, budget: int) -> None:
        self.budget = int(budget)

    def reset(self, movie: MovieContext) -> None:
        self._movie = movie
        self.store.reset(
            movie_id=movie.movie_id,
            source_video=movie.source_video,
            work_dir=movie.work_dir,
        )

    def observe_segment(self, obs: SegmentObservation) -> None:
        self.store.observe_segment(
            chunk_id=obs.chunk_id,
            prompt_text=obs.prompt_text,
            seconds_span=obs.seconds_span,
            segment_video=obs.segment_video,
        )

    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        rec = RetrievedMemory(chunk_id=req.chunk_id)
        rankings_fn = getattr(self.retriever, "rankings", None)
        if callable(rankings_fn):
            rankings = rankings_fn(
                req.prompt_text,
                as_of_seconds=float(req.seconds_span[0]),
                budget=self.budget,
            )
            rec.extras["rrf_rankings"] = [
                [_item_from_ref(ref) for ref in ranking]
                for ranking in rankings
            ]
            rec.extras["fusion"] = "rrf"
            rec.extras["rrf_k"] = 60
            return rec
        refs = self.retriever.retrieve(
            req.prompt_text,
            as_of_seconds=float(req.seconds_span[0]),
            budget=self.budget,
        )
        rec.items = [_item_from_ref(ref) for ref in refs]
        return rec

    def finalize(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "status": "ok",
            "positioning": "self-built control/ablation only; not a headline baseline",
            "variant": self.variant,
            "budget": self.budget,
            "uniform_fps": self.config.uniform_fps,
            "dense_fps": self.config.dense_fps,
            "key_per_segment": self.config.key_per_segment,
            "topk_segment": self.config.topk_segment,
            "text_provider": self.config.text_provider,
            "frame_provider": self.config.frame_provider,
            "keyframe_provider": self.config.keyframe_provider,
            "observed_segments": len(self.store.segments),
            "observed_frames": len(self.store.frames),
        }


def build_adapter() -> RetrievalFamilyAdapter:
    variant = os.environ.get("MEMSTRATA_RETRIEVAL_VARIANT") or os.environ.get("RETR_VARIANT")
    budget = int(os.environ.get("RETR_BUDGET", str(_DEFAULT_BUDGET)))
    return RetrievalFamilyAdapter(variant=variant or "seg_framererank", budget=budget)
