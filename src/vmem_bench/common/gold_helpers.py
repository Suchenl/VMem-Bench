"""Stable deterministic helpers shared by gold materialization and linting.

This module intentionally contains no imports from annotation pipelines.  It is
the public home for small helpers that must remain usable while annotation
implementations are migrated or replaced.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from vmem_bench.common.schemas import ForbiddenRep


def normalize_entity_name(name: str) -> str:
    """Normalize display names without changing non-ASCII identity."""
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = re.sub(
        r"\s*\(\s*(character|location|prop|char|loc)\s*\)?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+-(character|location|prop|char|loc)\s*$", "", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(character|location|prop|char|loc)\s*$", "", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"[_-](character|location|prop|char|loc)$", "", cleaned,
                     flags=re.IGNORECASE)
    return cleaned.strip('"\'').strip()


def slugify(name: str) -> str:
    """Return a stable ASCII-safe slug, including collision-safe CJK fallback."""
    norm = normalize_entity_name(name)
    slug = re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_")
    if slug:
        return slug
    if re.search(r"\w", norm, re.UNICODE):
        return "u_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    return "unnamed"


def materialize_forbidden(registry: Any, chunk_id: int) -> list[ForbiddenRep]:
    """Return representations deprecated strictly before ``chunk_id``.

    ``registry`` only needs an ``entities`` mapping.  This intentionally accepts
    the light-weight shim used by gold linting as well as build registries.
    """
    forbidden: list[ForbiddenRep] = []
    for entity in registry.entities.values():
        for event in entity.state_events:
            if event.chunk_id < chunk_id:
                forbidden.extend(
                    ForbiddenRep(representation_id=rep_id, reason=event.event_id)
                    for rep_id in event.deprecates
                )
    return forbidden


def resolve_current_appearance_rep(
    reps: list[dict[str, Any]],
    *,
    chunk_id: int,
    state: str = "default",
    allow_any_state_fallback: bool = True,
) -> dict[str, Any] | None:
    """Nearest gold rep with ``chunk_id ≤ t`` and matching ``state`` (v3 rule)."""
    same = [
        rep
        for rep in reps
        if int(rep.get("chunk_id", -1)) <= chunk_id and str(rep.get("state") or "default") == state
    ]
    if same:
        return max(same, key=lambda rep: int(rep["chunk_id"]))
    if not allow_any_state_fallback:
        return None
    any_state = [rep for rep in reps if int(rep.get("chunk_id", -1)) <= chunk_id]
    if not any_state:
        return None
    return max(any_state, key=lambda rep: int(rep["chunk_id"]))
