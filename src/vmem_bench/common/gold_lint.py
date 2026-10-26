"""Deterministic quality gates for MemStrata gold/checkpoint annotations.

The annotator is allowed to produce noisy build artifacts, but frozen gold and
asset-bank checkpoints need structural invariants that do not depend on VLMs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import normalize_entity_name, slugify
from vmem_bench.annotation.pipeline_track_first.drafting import materialize_forbidden
from vmem_bench.common.paths import is_entity_asset_path
from vmem_bench.common.schemas import ChunkAnnotations, EntityRegistry


PLACEHOLDER_PROMPT_RE = re.compile(r"continues in this location\s*\(chunk\s+\d+\)", re.I)
_SUFFIX_RE = re.compile(r"_(character|prop|location|char|loc)$", re.I)


@dataclass(slots=True)
class LintViolation:
    code: str
    message: str
    severity: str = "error"
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity,
                "path": self.path, "message": self.message}


def _v(code: str, message: str, *, path: str = "", severity: str = "error") -> LintViolation:
    return LintViolation(code=code, message=message, path=path, severity=severity)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = ih * iw
    if inter <= 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _bbox_area(bbox: list[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    y0, x0, y1, x1 = bbox
    return max(0, y1 - y0) * max(0, x1 - x0) / 1_000_000


def _canonical_alias_key(kind: str, name: str) -> str:
    return f"{kind}:{slugify(normalize_entity_name(name))}"


def lint_annotations(
    registry: EntityRegistry,
    chunks: ChunkAnnotations,
    *,
    layout: dict[str, Any] | None = None,
    qa_report: list[dict[str, Any]] | None = None,
    strict_review: bool = False,
    max_non_location_area: float = 0.95,  # near-full-frame only; close-ups OK
    max_same_chunk_iou: float = 0.95,
) -> list[LintViolation]:
    """Return deterministic annotation violations.

    ``strict_review`` turns build-time warnings (flagged chunks, placeholder prompts) into errors
    for freeze/publish. Checkpoint probes can keep it false to quantify unfinished runs.
    """
    violations: list[LintViolation] = []
    entities = {e.entity_id: e for e in registry.entities}
    provenance = dict(registry.annotation_provenance or {})
    roster_mode = str(provenance.get("roster_mode") or "")
    production_mode = bool(provenance.get("production_mode", False))
    if strict_review and roster_mode and roster_mode != "seeded":
        violations.append(_v(
            "unconfirmed_roster",
            "production freeze requires a human-confirmed canonical roster seed",
            severity="error"))
    if production_mode and roster_mode != "seeded":
        violations.append(_v(
            "production_roster_mode_mismatch",
            f"production_mode=true but roster_mode={roster_mode!r}",
            severity="error"))

    if len(entities) != len(registry.entities):
        violations.append(_v("duplicate_entity_id", "entity_id values must be unique"))

    alias_to_ids: dict[str, list[str]] = {}
    for entity in registry.entities:
        alias_to_ids.setdefault(_canonical_alias_key(entity.kind, entity.name), []).append(entity.entity_id)
        expected_prefix = f"{entity.kind[:4]}_"
        if entity.entity_id.endswith(("_character", "_prop", "_location")):
            base = _SUFFIX_RE.sub("", entity.entity_id)
            if base in entities:
                violations.append(_v(
                    "suffix_alias_entity",
                    f"{entity.entity_id} duplicates canonical-looking {base}",
                    path=f"entities[{entity.entity_id}]"))
        for rep in entity.representations:
            if rep.chunk_id < 0:
                violations.append(_v("bad_rep_chunk", f"{rep.representation_id} has invalid chunk_id",
                                     path=f"entities[{entity.entity_id}].representations"))
            if rep.embedding_key and rep.embedding_key != rep.representation_id:
                violations.append(_v("embedding_key_mismatch",
                                     f"{rep.representation_id} embedding_key={rep.embedding_key}",
                                     path=f"entities[{entity.entity_id}].representations"))
            if rep.bbox_source == "grounding_dino" and entity.kind != "location":
                area = _bbox_area(rep.bbox)
                if area > max_non_location_area:
                    violations.append(_v("oversized_bbox",
                                         f"{rep.representation_id} covers {area:.1%} of frame",
                                         path=f"entities[{entity.entity_id}].representations"))
            if rep.crop_path:
                if not is_entity_asset_path(rep.crop_path, entity.entity_id, entity.kind):
                    # Candidate paths are valid in checkpoints before commit; keep as warning.
                    violations.append(_v("non_asset_crop_path",
                                         f"{rep.representation_id} crop_path={rep.crop_path}",
                                         path=f"entities[{entity.entity_id}].representations",
                                         severity="warning"))
        if expected_prefix == "char_" and not entity.entity_id.startswith("char_"):
            violations.append(_v("entity_id_prefix", f"character id should start char_: {entity.entity_id}",
                                 path=f"entities[{entity.entity_id}]"))

    for key, ids in alias_to_ids.items():
        if len(ids) > 1:
            violations.append(_v("canonical_alias_split",
                                 f"{key} appears as multiple ids: {', '.join(sorted(ids))}"))

    chunk_ids = [c.chunk_id for c in chunks.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        violations.append(_v("duplicate_chunk_id", "chunk_id values must be unique"))
    if chunk_ids and sorted(chunk_ids) != list(range(min(chunk_ids), max(chunk_ids) + 1)):
        violations.append(_v("non_contiguous_chunks", f"chunk ids are not contiguous: {chunk_ids[:8]}..."))
    if layout and "chunks" in layout and len(chunks.chunks) != len(layout.get("chunks", [])):
        violations.append(_v("layout_chunk_count_mismatch",
                             f"annotations={len(chunks.chunks)} layout={len(layout.get('chunks', []))}"))

    # Registry-style shim for materialize_forbidden.
    class _RegistryShim:
        pass
    shim = _RegistryShim()
    shim.entities = entities

    reps_by_chunk: dict[int, list[tuple[str, Any]]] = {}
    for entity in registry.entities:
        for rep in entity.representations:
            reps_by_chunk.setdefault(rep.chunk_id, []).append((entity.entity_id, rep))

    for chunk in chunks.chunks:
        path = f"chunks[{chunk.chunk_id}]"
        if len(chunk.present) != len(set(chunk.present)):
            violations.append(_v("duplicate_present", "present contains duplicate entity ids", path=path))
        unknown = [eid for eid in chunk.present if eid not in entities]
        if unknown:
            violations.append(_v("unknown_present_entity", f"unknown entity ids: {unknown}", path=path))
        first_extra = [eid for eid in chunk.first_appearances if eid not in set(chunk.present)]
        if first_extra:
            violations.append(_v("first_not_present", f"first_appearances outside present: {first_extra}",
                                 path=path))
        if PLACEHOLDER_PROMPT_RE.search(chunk.prompt):
            violations.append(_v("placeholder_prompt", "fallback placeholder prompt is not publishable",
                                 path=path, severity="error" if strict_review else "warning"))
        instr = {g.entity_id: g.requirement for g in chunk.gold_instructions}
        for eid in chunk.present:
            expected = "introduce" if eid in set(chunk.first_appearances) else "continuity"
            if instr.get(eid) != expected:
                violations.append(_v("instruction_mismatch",
                                     f"{eid} instruction={instr.get(eid)!r}, expected={expected!r}",
                                     path=path))
        expected_forbidden = {(f.representation_id, f.reason) for f in materialize_forbidden(shim, chunk.chunk_id)}
        actual_forbidden = {(f.representation_id, f.reason) for f in chunk.forbidden}
        if expected_forbidden != actual_forbidden:
            violations.append(_v("forbidden_mismatch", "forbidden table differs from state_events",
                                 path=path))

        grounded = [(eid, rep) for eid, rep in reps_by_chunk.get(chunk.chunk_id, [])
                    if entities.get(eid) and entities[eid].kind != "location"
                    and rep.bbox_source == "grounding_dino"]
        for i, (eid_a, rep_a) in enumerate(grounded):
            for eid_b, rep_b in grounded[i + 1:]:
                if eid_a == eid_b:
                    continue
                iou = _bbox_iou(rep_a.bbox, rep_b.bbox)
                if iou >= max_same_chunk_iou:
                    violations.append(_v(
                        "same_chunk_bbox_conflict",
                        f"{rep_a.representation_id} and {rep_b.representation_id} IoU={iou:.3f}",
                        path=path))

    if qa_report:
        flagged = [int(q["chunk_id"]) for q in qa_report if q.get("flagged")]
        if flagged:
            violations.append(_v("flagged_chunks",
                                 f"{len(flagged)} flagged chunks require review: {flagged[:20]}",
                                 severity="error" if strict_review else "warning"))

    return violations


def load_gold_annotations(movie_dir: Path) -> tuple[EntityRegistry, ChunkAnnotations, dict[str, Any] | None]:
    from vmem_bench.common.paths import MovieDirs
    dirs = MovieDirs(Path(movie_dir))
    registry = EntityRegistry.from_dict(
        json.loads(dirs.registry_json.read_text(encoding="utf-8")))
    chunks = ChunkAnnotations.from_dict(
        json.loads(dirs.annotations_json.read_text(encoding="utf-8")))
    layout_path = dirs.chunk_index
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else None
    return registry, chunks, layout


def load_checkpoint_annotations(movie_dir: Path) -> tuple[EntityRegistry, ChunkAnnotations, dict[str, Any] | None]:
    from vmem_bench.common.paths import MovieDirs
    dirs = MovieDirs(Path(movie_dir))
    registry = EntityRegistry.from_dict(
        json.loads((dirs.tmp / "checkpoint_registry.json").read_text(encoding="utf-8")))
    checkpoint = json.loads((dirs.tmp / "checkpoint.json").read_text(encoding="utf-8"))
    chunks = ChunkAnnotations.from_dict({
        "movie_id": checkpoint.get("movie_id", registry.movie_id),
        "chunks": checkpoint.get("chunks", []),
        "human_reviewed": False,
        "schema_version": registry.schema_version,
    })
    layout_path = dirs.chunk_index
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else None
    return registry, chunks, layout


def lint_movie_dir(movie_dir: Path, *, checkpoint: bool = False,
                   strict_review: bool = False) -> list[LintViolation]:
    movie_dir = Path(movie_dir)
    if checkpoint:
        registry, chunks, layout = load_checkpoint_annotations(movie_dir)
    else:
        registry, chunks, layout = load_gold_annotations(movie_dir)
    from vmem_bench.common.paths import MovieDirs
    qa_path = MovieDirs(movie_dir).qa_report
    qa_report = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else []
    return lint_annotations(registry, chunks, layout=layout, qa_report=qa_report,
                            strict_review=strict_review)


def summarize_violations(violations: list[LintViolation]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    severity: dict[str, int] = {}
    for v in violations:
        counts[v.code] = counts.get(v.code, 0) + 1
        severity[v.severity] = severity.get(v.severity, 0) + 1
    return {"ok": not any(v.severity == "error" for v in violations),
            "n_violations": len(violations),
            "by_code": dict(sorted(counts.items())),
            "by_severity": dict(sorted(severity.items()))}


def write_lint_report(violations: list[LintViolation], out: Path, *, title: str) -> None:
    summary = summarize_violations(violations)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps({
            "summary": summary,
            "violations": [v.to_dict() for v in violations],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in summary["by_code"].items():
        lines.append(f"- `{key}`: {value}")
    if not summary["by_code"]:
        lines.append("- No violations.")
    lines.extend(["", "## Violations", ""])
    for v in violations:
        loc = f" `{v.path}`" if v.path else ""
        lines.append(f"- **{v.severity}** `{v.code}`{loc}: {v.message}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Lint VMem-Bench gold or build checkpoint annotations.")
    parser.add_argument("--movie-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="store_true",
                        help="lint tmp/checkpoint*.json instead of gold/")
    parser.add_argument("--strict-review", action="store_true",
                        help="treat review-required warnings as blocking errors")
    parser.add_argument("--out", type=Path, help="optional .json or .md report path")
    args = parser.parse_args()

    violations = lint_movie_dir(args.movie_dir, checkpoint=args.checkpoint,
                                strict_review=args.strict_review)
    summary = summarize_violations(violations)
    payload = {"summary": summary, "violations": [v.to_dict() for v in violations]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        title = f"VMem-Bench {'Checkpoint' if args.checkpoint else 'Gold'} Lint"
        write_lint_report(violations, args.out, title=title)
    return 1 if not summary["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
