"""Shared SAM3 transformers loader for refine + concept segmentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def vendored_deps_dir() -> str:
    """Vendored SAM3-capable transformers dir (env override, else repo-local vendor)."""
    deps = os.environ.get("MEMSTRATA_SAM3_DEPS")
    if deps:
        return deps
    # stages/s5/.../sam3_common.py -> Montage repo root is parents[8].
    repo = Path(__file__).resolve().parents[8]
    candidate = repo / "models" / "vendor" / "sam3_transformers59"
    return str(candidate) if candidate.is_dir() else ""


def import_sam3_classes():
    """Import (Sam3Model, Sam3Processor) without hot-swapping transformers versions."""
    try:
        from transformers import Sam3Model, Sam3Processor

        return Sam3Model, Sam3Processor
    except ImportError:
        pass
    deps = vendored_deps_dir()
    if not deps:
        raise RuntimeError(
            "SAM3 requires transformers>=5.9 or MEMSTRATA_SAM3_DEPS "
            "(or models/vendor/sam3_transformers59)"
        )
    if "transformers" in sys.modules:
        raise RuntimeError(
            "SAM3 cannot load after an incompatible transformers import; "
            f"launch with PYTHONPATH={deps} prepended"
        )
    sys.path.insert(0, deps)
    from transformers import Sam3Model, Sam3Processor

    return Sam3Model, Sam3Processor
