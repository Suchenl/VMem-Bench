"""Self-contained causal retrieval-family baselines.

Four retrieval baselines (``frame_text`` and three ``seg_*`` text→segment variants) plus
controls, driven under the causal bench protocol by
``scripts/evaluate_baselines/trackA/baseline_adapters/causal/retrieval_family.py``. These are a
bench-owned reimplementation that never imports ``memstrata`` (their encoder substrate lives
in ``._retrieval_encoders``); see ``retrieval_baselines.py``.

The former gold-replay retrieval adapters (``text_retrieval`` / ``frame_retrieval`` /
``textframe_fusion``) were removed with the old gold-replay protocol; they are no longer
registered as online adapters.
"""

from vmem_bench.baseline_adapters.external.retrieval import retrieval_baselines
from vmem_bench.baseline_adapters.external.retrieval.retrieval_baselines import (
    MemoryRetrievalStore,
    RetrievedRef,
    Retriever,
    RetrieverConfig,
    build_retriever,
)

__all__ = [
    "retrieval_baselines",
    "MemoryRetrievalStore",
    "RetrievedRef",
    "Retriever",
    "RetrieverConfig",
    "build_retriever",
]
