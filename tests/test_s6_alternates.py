"""S6 alternate listing must stay read-only and path-robust."""

from __future__ import annotations

import json
from pathlib import Path

from vmem_bench.annotation.pipeline.servers.backend import review_service as rs


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid 1x1 PNG.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_s6_alternates_does_not_materialize_and_maps_existing_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    movie = tmp_path / "movie"
    s5 = movie / "tmp" / "pipeline" / "s5_entities_visual_crop_acquisition"
    cand = s5 / "candidates" / "character" / "char_001"
    current = cand / "c00019_current.png"
    other = cand / "c00024_other.png"
    _write_png(current)
    _write_png(other)

    proposals = [
        {
            "representation_id": "char_001@c00019",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 19,
            "crop_path": str(current),
            "bbox_source": "sam3_concept",
            "sam3": {"score": 0.9},
            "frame_index": 1,
            "bbox_norm": [0, 0, 100, 100],
        },
        {
            "representation_id": "char_001@c00024",
            "entity_id": "char_001",
            "kind": "character",
            "chunk_id": 24,
            "crop_path": str(other),
        },
    ]
    (s5 / "crop_proposals.json").parent.mkdir(parents=True, exist_ok=True)
    (s5 / "crop_proposals.json").write_text(json.dumps(proposals), encoding="utf-8")

    monkeypatch.setattr(rs, "memstrata_root_from_here", lambda: tmp_path)

    sample = {
        "dataset": "BlenderOpenMovies",
        "movie_id": "movie",
        "movie_dir": str(movie),
    }
    payload = rs.s6_alternates(sample, "char_001@c00019")
    alts = payload["alternates"]
    assert alts, "expected other candidate crops"
    paths = {row["crop_path"] for row in alts}
    assert not any(Path(p).name == current.name for p in paths)
    mapped = next(row for row in alts if Path(row["crop_path"]).name == other.name)
    assert mapped["existing_representation_id"] == "char_001@c00024"
