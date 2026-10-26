"""Self-contained causal retrieval-family baselines (Track A entry point).

Four retrieval baselines (``frame_text`` and three ``seg_*`` text→segment variants) plus
controls, driven under the causal bench protocol by
``scripts/evaluate_baselines/trackA/baseline_adapters/causal/retrieval_family.py``. The implementation
now lives in :mod:`vmem_bench.testbed.retrieval` and is shared with the Track-B testbed; this package
and ``retrieval_baselines`` only re-export it for existing imports.

The former gold-replay retrieval adapters (``text_retrieval`` / ``frame_retrieval`` /
``textframe_fusion``) were removed with the old gold-replay protocol; they are no longer
registered as online adapters.
"""

from vmem_bench.baseline_adapters.external.retrieval import retrieval_baselines
from vmem_bench.testbed.retrieval.retrievers import (
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
