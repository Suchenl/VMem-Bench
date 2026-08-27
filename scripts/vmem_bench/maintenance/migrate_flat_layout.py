#!/usr/bin/env python3
"""Compatibility entry point for the public flat-layout migration utility."""

from __future__ import annotations

import runpy
from pathlib import Path


_IMPLEMENTATION = (
    Path(__file__).resolve().parents[2]
    / "get_trackA_assets"
    / "maintenance"
    / "migrate_flat_layout.py"
)


if __name__ == "__main__":
    runpy.run_path(str(_IMPLEMENTATION), run_name="__main__")
