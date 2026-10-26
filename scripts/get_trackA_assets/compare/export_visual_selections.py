#!/usr/bin/env python3
"""Resolve every system's per-chunk selections to concrete visual evidence.

The per-chunk records under ``benchmark_run/`` store the *selection logic* (asset_id +
representation_id, e.g. ``char_002@c00002``) but not a direct handle to the crop image or
source frame, so building a qualitative comparison figure -- or re-checking a metric by eye --
means re-joining against the gold every time. This tool does that join once and writes a
compact ``visual_selections/`` manifest:

  visual_selections/<system>.json   per-system, per-chunk resolved picks (crop path, entity
                                    name, kind, source chunk) + the chunk prompt
  visual_selections/by_chunk.json   cross-system view: for each chunk, what every system
                                    selected, side by side (for qualitative comparison figures)

Everything is resolved from the frozen gold (observations.jsonl + prompts.jsonl), so it is
deterministic and adds no model calls. Crop paths are given both relative to the movie's
``gold/`` dir and as absolute paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REP_CHUNK_RE = re.compile(r"@c0*(\d+)$")
BLENDER = Path(__file__).resolve().parents[3] / "data" / "BlenderOpenMovies"


def _load_repmap(gold: Path) -> dict[str, dict[str, Any]]:
    """representation_id -> {crop_path, name, kind, entity_id, source_chunk}."""
    repmap: dict[str, dict[str, Any]] = {}
    obs_file = gold / "observations.jsonl"
    if not obs_file.is_file():
        return repmap
    for line in obs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for o in row.get("observations", []):
            rid = o.get("representation_id")
            if not rid:
                continue
            m = REP_CHUNK_RE.search(str(rid))
            repmap[str(rid)] = {
                "crop_path": o.get("crop_path"),
                "name": o.get("name"),
                "kind": o.get("kind"),
                "entity_id": o.get("entity_id"),
                "source_chunk": int(m.group(1)) if m else None,
            }
    return repmap


def _load_prompts(gold: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    pf = gold / "prompts.jsonl"
    if not pf.is_file():
        return prompts
    for line in pf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompts[int(row["chunk_id"])] = row.get("prompt") or row.get("text") or ""
    return prompts


def _resolve_item(item: dict[str, Any], repmap: dict[str, dict[str, Any]],
                  gold: Path) -> dict[str, Any]:
    rids = item.get("representation_ids") or []
    reps = []
    for rid in rids:
        info = dict(repmap.get(str(rid), {}))
        cp = info.get("crop_path")
        info["representation_id"] = rid
        info["crop_abspath"] = str((gold / cp).resolve()) if cp else None
        reps.append(info)
    return {
        "asset_id": item.get("asset_id"),
        "function": item.get("function"),
        "requirement": (item.get("requirement")
                        or item.get("strength")),
        "representations": reps,
    }


def _iter_system_record_dirs(run: Path) -> list[tuple[str, Path]]:
    """(system_name, records_dir) for baselines and the memstrata SUT + ablations."""
    out: list[tuple[str, Path]] = []
    bl = run / "baselines"
    if bl.is_dir():
        for d in sorted(bl.iterdir()):
            rec = d / "records"
            if rec.is_dir():
                out.append((d.name, rec))
    ms = run / "memstrata"
    if ms.is_dir():
        for d in sorted(ms.iterdir()):
            if d.is_dir() and d.name.endswith("_records"):
                sysname = "memstrata:" + d.name[: -len("_records")].replace("_hash", "")
                out.append((sysname, d))
    return out


def export_movie(movie_dir: Path) -> dict[str, Any]:
    gold = movie_dir / "gold"
    run = movie_dir / "benchmark_run"
    if not run.is_dir():
        return {"movie": movie_dir.name, "ok": False, "reason": "no benchmark_run"}
    repmap = _load_repmap(gold)
    prompts = _load_prompts(gold)
    out_dir = run / "visual_selections"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_chunk: dict[int, dict[str, Any]] = {}
    systems: list[str] = []
    for sysname, rec in _iter_system_record_dirs(run):
        systems.append(sysname)
        chunks = []
        for cf in sorted(rec.glob("chunk_*.json")):
            record = json.loads(cf.read_text(encoding="utf-8"))
            cid = int(record.get("chunk_id", int(cf.stem.split("_")[1])))
            selected = [_resolve_item(it, repmap, gold) for it in record.get("selected", [])]
            chunks.append({"chunk_id": cid, "prompt": prompts.get(cid, ""),
                           "selected": selected})
            slot = by_chunk.setdefault(cid, {"chunk_id": cid, "prompt": prompts.get(cid, ""),
                                             "systems": {}})
            slot["systems"][sysname] = [
                {"name": r["representations"][0].get("name") if r["representations"] else None,
                 "kind": r["representations"][0].get("kind") if r["representations"] else None,
                 "asset_id": r["asset_id"],
                 "crop_path": r["representations"][0].get("crop_path") if r["representations"] else None}
                for r in selected
            ]
        (out_dir / f"{sysname.replace(':', '_')}.json").write_text(
            json.dumps({"movie": movie_dir.name, "system": sysname, "chunks": chunks},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "by_chunk.json").write_text(
        json.dumps({"movie": movie_dir.name, "systems": systems,
                    "chunks": [by_chunk[c] for c in sorted(by_chunk)]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"movie": movie_dir.name, "ok": True, "systems": len(systems),
            "chunks": len(by_chunk), "out": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--movie-dir", type=Path, help="one movie dir")
    g.add_argument("--all-blender", action="store_true", help="all BlenderOpenMovies with a benchmark_run")
    args = ap.parse_args(argv)

    movies = ([args.movie_dir] if args.movie_dir
              else [d for d in sorted(BLENDER.iterdir())
                    if (d / "benchmark_run").is_dir()])
    for md in movies:
        print(export_movie(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
