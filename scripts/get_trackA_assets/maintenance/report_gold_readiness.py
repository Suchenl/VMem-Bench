#!/usr/bin/env python3
"""Emit a deterministic per-catalog S1--S7 readiness report for gold freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _catalog(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("catalog rows must be JSON objects")
        items.append(payload)
    return items


def _sample_status(item: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(item.get("root") or ""))
    pipeline = root / "tmp" / "pipeline"
    state = _read_json(pipeline / "state.json").get("stages") or {}
    s4 = _read_json(pipeline / "s4_segment_sampling_human_review" / "review_audit.json")
    s6 = _read_json(pipeline / "s6_entities_visual_crop_human_review" / "review_audit.json")
    s7 = _read_json(pipeline / "s7_freeze_publish" / "strict_lint.json")
    manifest = _read_json(root / "gold" / "manifest.json")
    return {
        "dataset": item.get("dataset") or "",
        "movie_id": item.get("movie_id") or root.name,
        "root": str(root),
        "catalog_status": item.get("status") or "",
        "source_video": item.get("source_video") or "",
        "source_exists": Path(str(item.get("source_video") or "")).is_file(),
        "stages": state,
        "s4_human_reviewed": bool(s4.get("human_reviewed")),
        "s6_human_reviewed": bool(s6.get("human_reviewed")),
        "s7_strict_lint": s7.get("status") or "",
        "gold_manifest_exists": bool(manifest),
        "gold_human_reviewed": bool(manifest.get("human_reviewed")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="Explicit JSONL catalog to inspect")
    parser.add_argument("--out", type=Path, required=True, help="Readiness JSON output path")
    args = parser.parse_args()

    rows = [_sample_status(item) for item in _catalog(args.catalog)]
    summary = {
        "catalog": str(args.catalog),
        "n_samples": len(rows),
        "n_gold_human_reviewed": sum(1 for row in rows if row["gold_human_reviewed"]),
        "n_s7_strict_ready": sum(
            1 for row in rows if row["s7_strict_lint"] == "human_reviewed_ready"
        ),
        "samples": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
