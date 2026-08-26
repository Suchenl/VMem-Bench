"""Path bootstrap for VMem-Bench cross-package (baseline / SUT) tests.

These are the ONLY tests allowed to cross the split boundary: they drive the
MemStrata method (as a system-under-test baseline) through the VMem-Bench
scorer, so they import both ``vmem_bench`` (bench) and ``memstrata`` (method).
They live under ``scripts/evaluate_baselines/`` per the repo boundary rule and
are deliberately excluded from the bench's own ``pytest.ini`` test collection.

This conftest puts both ``src`` roots on ``sys.path`` so the tests run via
``python -m pytest`` from anywhere without extra PYTHONPATH wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parents[3]   # benchmarks/VMem-Bench
_REPO = _BENCH_ROOT.parents[1]                       # repo root
for _src in (_BENCH_ROOT / "src", _REPO / "methods" / "MemStrata" / "src"):
    _s = str(_src)
    if _src.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
