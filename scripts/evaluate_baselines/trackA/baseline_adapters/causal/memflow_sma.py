"""MemFlow (with SMA) adapter — the φ compact-vector routing variant.

Thin wrapper over :class:`memflow.MemFlowAdapter` with ``sma=True`` so the baseline
table gets a second row, MemFlow (with SMA), alongside the shipped-default MemFlow
(w/o SMA). All logic lives in ``memflow.py``; see its module docstring for how the
``dynamic_topk_routing_attention`` (φ_q·φ_k top-k chunk) read path is enabled and
captured. Run via ``runner.py --adapter memflow_sma``.
"""
from __future__ import annotations

from memflow import MemFlowAdapter


def build_adapter() -> MemFlowAdapter:
    return MemFlowAdapter(sma=True)
