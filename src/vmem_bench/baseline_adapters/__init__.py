"""Bench-owned baseline code for Track A (new causal protocol).

The former **gold-replay** protocol (``run_gold_replay.py`` + ``registry`` + ``convert`` +
``common/`` + ``diagnostics/`` + ``external/causal/`` + ``external/scripted/``) has been
removed. Track A now runs exclusively under the causal protocol: the bench hands the SUT a
prompt, the SUT composes+returns context (persisted for scoring and paper figures), the bench
hands back the *real* historical segment for the SUT to memorise, and retrieved refs are
materialised into real frames scored by ``vmem_bench.scoring.visual_coverage``. The causal
runner and thin adapters live under
``scripts/evaluate_baselines/trackA/baseline_adapters/`` (outside the ``vmem_bench`` package).

The only baseline code that stays inside ``vmem_bench`` is the self-contained retrieval family
(:mod:`vmem_bench.baseline_adapters.external.retrieval`), a bench-owned reimplementation that
never imports the SUT package (``memstrata``).
"""

from vmem_bench.baseline_adapters.external import retrieval as retrieval
from vmem_bench.baseline_adapters.external.retrieval import (
    MemoryRetrievalStore,
    RetrievedRef,
    Retriever,
    RetrieverConfig,
    build_retriever,
)

__all__ = [
    "retrieval",
    "MemoryRetrievalStore",
    "RetrievedRef",
    "Retriever",
    "RetrieverConfig",
    "build_retriever",
]
