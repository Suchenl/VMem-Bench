#!/usr/bin/env python3
"""Catalog blueprints for Track B stories 0004-0050.

Loads compact blueprints from trackb_catalog_blueprints.json and expands them
via build_from_blueprint into full gt_source stories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent  # scripts/ (sibling import of generate_gt_source_batch)
# Blueprint data stays at the trackB root, one level up from scripts/.
_DATA = json.loads((_HERE.parent / "trackb_catalog_blueprints.json").read_text(encoding="utf-8"))
CATALOG_IDS = list(_DATA.keys())


def build_catalog_story(story_id: str) -> dict[str, Any]:
    from generate_gt_source_batch import build_from_blueprint
    if story_id not in _DATA:
        raise KeyError(story_id)
    return build_from_blueprint(_DATA[story_id])
