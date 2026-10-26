"""Realization backends: turn one segment's instruction and reference images into a clip.

Backends are loaded by name (``load_backend``); heavy modules are imported only when selected.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from vmem_bench.testbed.backends.base import (
    RealizationBackend,
    RealizationRequest,
    RealizationResult,
)

# name -> (module, class). Constructors take keyword options only.
BACKENDS: dict[str, tuple[str, str]] = {
    "flux2_wan22": ("vmem_bench.testbed.backends.flux2_wan22", "Flux2Wan22Backend"),
    "flux2_h3": ("vmem_bench.testbed.backends.flux2_h3", "Flux2H3Backend"),
    "minimax_h3_turbo": ("vmem_bench.testbed.backends.minimax_h3", "MiniMaxH3TurboBackend"),
    "synthetic": ("vmem_bench.testbed.backends.synthetic", "SyntheticBackend"),
}


def load_backend(
    name: str, options: dict[str, Any] | None = None, *, defaults: dict[str, Any] | None = None
) -> RealizationBackend:
    """Instantiate a registered backend with user ``options`` (all must be accepted) plus runner
    ``defaults`` (passed only where the constructor accepts them)."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choices: {sorted(BACKENDS)}")
    module, attr = BACKENDS[name]
    cls = getattr(importlib.import_module(module), attr)
    params = inspect.signature(cls).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    opts = {k: v for k, v in (defaults or {}).items() if takes_any or k in params}
    opts.update(options or {})
    backend = cls(**opts)
    if not isinstance(backend, RealizationBackend):
        raise TypeError(f"{module}.{attr} does not implement RealizationBackend")
    return backend


__all__ = ["BACKENDS", "RealizationBackend", "RealizationRequest", "RealizationResult", "load_backend"]
