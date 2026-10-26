"""Similarity-retrieval memories: four paper variants plus controls, and their encoders.

One implementation serves Track A (``scripts/evaluate_baselines/trackA/.../retrieval_family.py``) and
the Track-B testbed (``vmem_bench.testbed.runner``); see ``retrievers.py``.
"""

from vmem_bench.testbed.retrieval.retrievers import (
    FrameRecord,
    MemoryRetrievalStore,
    RetrievedRef,
    Retriever,
    RetrieverConfig,
    SegmentRecord,
    build_retriever,
)

# Paper Track-B row -> retrieval variant.
PAPER_VARIANTS = ("frame_text", "seg_uniform", "seg_dinokey", "seg_framererank")

__all__ = [
    "FrameRecord",
    "MemoryRetrievalStore",
    "PAPER_VARIANTS",
    "RetrievedRef",
    "Retriever",
    "RetrieverConfig",
    "SegmentRecord",
    "build_retriever",
]
