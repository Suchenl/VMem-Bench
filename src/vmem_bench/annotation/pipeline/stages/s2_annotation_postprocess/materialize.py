"""Reusable API and CLI implementation for S2 annotation post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .entity_checks import check_entities
from .normalize import normalize_annotation
from .segment_checks import check_segments
from .structural_lint import lint_structure


_STAGE = "s2_annotation_postprocess"
_REPORT_NAMES = {
    "structural_lint": "structural_lint.json",
    "entity_checks": "entity_checks.json",
    "segment_checks": "segment_checks.json",
}


def postprocess_annotation(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Materialize S2 artifacts for one v5 JSON input.

    A blank source is explicitly skipped.  Invalid JSON and hard structural
    failures still receive all three check reports, without rewriting the source
    annotation.  Declared ``counts`` mismatches are warnings only: normalize
    rewrites them from actual collection lengths in ``normalized_annotation.json``.
    Consumers must require the returned ``status == "ok"``.
    """
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = {"stage": _STAGE, "input_path": str(source)}

    try:
        payload = source.read_bytes()
    except OSError as error:
        return _write_failure_reports(destination, base, "input_unreadable", "INPUT_UNREADABLE", str(error))
    if not payload.strip():
        return _write_failure_reports(destination, base, "skipped_empty_input", "EMPTY_INPUT", "input is empty")

    try:
        annotation = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return _write_failure_reports(destination, base, "invalid_json", "INVALID_JSON", str(error))

    structural = lint_structure(annotation)
    if not isinstance(annotation, dict):
        return _write_reports(
            destination,
            base,
            "invalid_structure",
            structural,
            {"errors": [], "warnings": [], "entity_ids": []},
            {"errors": [], "warnings": [], "segment_index": []},
        )

    entities = check_entities(annotation)
    segments = check_segments(annotation, set(entities["entity_ids"]))
    status = "ok" if not (structural["errors"] or entities["errors"] or segments["errors"]) else "invalid_structure"
    result = _write_reports(destination, base, status, structural, entities, segments)
    normalized_path = destination / "normalized_annotation.json"
    _write_json(normalized_path, normalize_annotation(annotation))
    result["normalized_annotation"] = str(normalized_path)
    return result


def _write_failure_reports(
    destination: Path, base: dict[str, str], status: str, code: str, message: str
) -> dict[str, Any]:
    failure = {"errors": [{"code": code, "path": "$", "message": message}], "warnings": []}
    return _write_reports(
        destination,
        base,
        status,
        failure,
        {"errors": [], "warnings": [], "entity_ids": []},
        {"errors": [], "warnings": [], "segment_index": []},
    )


def _write_reports(
    destination: Path,
    base: dict[str, str],
    status: str,
    structural: dict[str, Any],
    entities: dict[str, Any],
    segments: dict[str, Any],
) -> dict[str, Any]:
    reports = {
        "structural_lint": structural,
        "entity_checks": entities,
        "segment_checks": segments,
    }
    paths: dict[str, str] = {}
    for name, report in reports.items():
        content = {
            **base,
            "status": status,
            "errors": report["errors"],
            "warnings": report["warnings"],
        }
        if name == "structural_lint":
            content["actual_counts"] = report.get("actual_counts", {})
        elif name == "entity_checks":
            content["entity_ids"] = report.get("entity_ids", [])
        else:
            content["segment_index"] = report.get("segment_index", [])
        report_path = destination / _REPORT_NAMES[name]
        _write_json(report_path, content)
        paths[name] = str(report_path)
    return {"status": status, "artifacts": paths}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = postprocess_annotation(args.input_path, args.output_dir)
    print(json.dumps({"status": result["status"], "artifacts": result.get("artifacts", {})}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
