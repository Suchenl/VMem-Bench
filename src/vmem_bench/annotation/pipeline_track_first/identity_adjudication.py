"""Global visual identity adjudication (machine recommendation for the review queue).

Fragmentation happens because re-ID merges only on embedding cosine and auto-review merges only
on text+body double agreement — but the very reason one character splits into several entities
(lighting/pose shifts + independently invented names) also kills both signals. A VLM looking at
one labeled crop per entity CAN say "these are the same individual", which is exactly the
evidence a human reviewer reconstructs by eye today.

This pass is review-only: it writes ``tmp/identity_adjudication.json`` and the review queue
attaches it as ``machine_suggestion`` evidence. It NEVER mutates gold; humans still decide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vmem_bench.common.paths import MovieDirs


def _normalize_groups(groups: Any, n: int) -> list[list[int]]:
    """Coerce VLM output into a partition of ``range(n)``: clamp to valid unseen indices, then
    add missing indices as singletons. Deterministic and total."""
    seen: set[int] = set()
    out: list[list[int]] = []
    for group in (groups if isinstance(groups, list) else []):
        clean = []
        for idx in (group if isinstance(group, list) else []):
            if isinstance(idx, (int, float)) and 0 <= int(idx) < n and int(idx) not in seen:
                clean.append(int(idx))
                seen.add(int(idx))
        if clean:
            out.append(sorted(clean))
    for idx in range(n):
        if idx not in seen:
            out.append([idx])
    return sorted(out)


def _best_crop(entity: dict, out: Path) -> Path | None:
    reps = sorted((r for r in entity.get("representations", []) if r.get("crop_path")),
                  key=lambda r: -float((r.get("qa") or {}).get("grounding_score", 0.0)))
    for rep in reps:
        path = Path(rep["crop_path"])
        path = path if path.is_absolute() else out / path
        if path.is_file():
            return path
    return None


def adjudicate_identities(out: Path, role: Any, *, kinds: tuple[str, ...] = ("character",),
                          max_images: int = 24) -> dict[str, Any]:
    """One VLM call per kind: group entity crops that show the same individual/instance."""
    out = Path(out)
    registry = json.loads(MovieDirs(out).registry_json.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"version": 1, "groups": []}
    for kind in kinds:
        entries = []
        for entity in registry.get("entities", []):
            if entity.get("kind") != kind:
                continue
            crop = _best_crop(entity, out)
            if crop is not None:
                species = str((entity.get("static_attributes") or {}).get("species") or "")
                entries.append((entity["entity_id"], entity.get("name") or "", crop, species))
        # Budget guard: adjudicate the most-evidenced entities first if over the image limit.
        entries = entries[:max_images]
        if len(entries) < 2:
            continue
        groups = role.group_same_individuals(
            [(eid, name) for eid, name, _crop, _sp in entries],
            [crop for _eid, _name, crop, _sp in entries])
        for idx_group in _normalize_groups(groups, len(entries)):
            # Deterministic species guard: a VLM grouping mistake must never suggest merging
            # across species. Missing species is its own bucket (never merged by suggestion).
            by_species: dict[str, list[int]] = {}
            for i in idx_group:
                by_species.setdefault(entries[i][3] or f"__unknown_{i}", []).append(i)
            for sub in by_species.values():
                if len(sub) > 1:
                    result["groups"].append({
                        "kind": kind,
                        "species": entries[sub[0]][3],
                        "entity_ids": [entries[i][0] for i in sub],
                        "names": [entries[i][1] for i in sub],
                    })
    dirs = MovieDirs(out, write=True)
    dirs.tmp.mkdir(parents=True, exist_ok=True)
    path = dirs.tmp / "identity_adjudication.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VLM global identity adjudication (review-only)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vlm-base-url", required=True)
    parser.add_argument("--vlm-model", default="qwen3-vl-8b")
    parser.add_argument("--kinds", default="character",
                        help="comma-separated kinds to adjudicate (default: character)")
    args = parser.parse_args(argv)
    from vmem_bench.judger.vlm import VlmJudger
    from vmem_bench.annotation.pipeline_track_first.vlm_roles import AnnotatorRole
    role = AnnotatorRole(VlmJudger(base_url=args.vlm_base_url, model=args.vlm_model))
    report = adjudicate_identities(args.out, role,
                                   kinds=tuple(k for k in args.kinds.split(",") if k))
    merged = [g for g in report["groups"] if len(g["entity_ids"]) > 1]
    print(json.dumps({"n_groups": len(merged), "groups": merged}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
