"""External baseline code.

Only the self-contained ``retrieval`` family remains here. The old gold-replay ``causal/``
(MemFlow, IAMFlow, LongLive-RAG, DecMem, Helios adapters) and ``scripted/`` (ViMax,
VideoMemory, StoryMem, …) regimes were removed with the gold-replay protocol; the live causal
baselines now live under ``scripts/evaluate_baselines/trackA/baseline_adapters/causal/``.
"""

from vmem_bench.baseline_adapters.external import retrieval as retrieval

__all__ = ["retrieval"]
