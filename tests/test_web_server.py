"""Self-check for the review API helpers in vmem_bench.web.server.

Covers the deterministic, GPU-free glue: run_done transition, last_stage tail parsing, and
gold_payload assembling registry+chunks+sidecars. The patch/freeze endpoints just delegate to
annotation/review.py (tested there), so we only exercise the server's own read/assemble logic.
Run: PYTHONPATH=benchmarks/MemStrata/src python3 benchmarks/MemStrata/tests/test_web_server.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from vmem_bench.web import server


def _fake_run(root: Path, *, with_gold: bool) -> None:
    (root / "build").mkdir(parents=True, exist_ok=True)
    (root / "build" / "events.jsonl").write_text(
        '{"ts":1,"kind":"run_start","movie_id":"m"}\n'
        '{"ts":2,"kind":"track_progress","shot":5,"n_shots":10,"stage":"tracking"}\n',
        encoding="utf-8")
    (root / "layout").mkdir(parents=True, exist_ok=True)
    (root / "layout" / "chunk_index.json").write_text(json.dumps({"fps": 24.0}), encoding="utf-8")
    if with_gold:
        gold = root / "gold"
        gold.mkdir(parents=True, exist_ok=True)
        registry = {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": False,
                    "annotation_provenance": {}, "entities": [
                        {"entity_id": "e1", "kind": "character", "name": "Rabbit",
                         "description": "white", "first_chunk": 0}]}
        chunks = {"schema_version": "2.0.0", "movie_id": "m", "human_reviewed": False,
                  "chunks": [{"chunk_id": 0, "prompt": "a rabbit", "present": ["e1"]}]}
        (gold / "entity_registry.json").write_text(json.dumps(registry), encoding="utf-8")
        (gold / "chunk_annotations.json").write_text(json.dumps(chunks), encoding="utf-8")
        (root / "build" / "annotation_qa.json").write_text(
            json.dumps([{"chunk_id": 0, "flagged": False}]), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Before gold exists: not reviewable; last_stage reads the tail event.
        _fake_run(root, with_gold=False)
        assert server.run_done(root) is False
        st = server.last_stage(root)
        assert st["kind"] == "track_progress" and st["stage"] == "tracking", st

        # After gold exists: reviewable; gold_payload assembles everything.
        _fake_run(root, with_gold=True)
        assert server.run_done(root) is True
        payload = server.gold_payload(root)
        assert payload["done"] is True
        assert payload["human_reviewed"] is False
        assert payload["registry"]["entities"][0]["name"] == "Rabbit"
        assert payload["chunks"]["chunks"][0]["prompt"] == "a rabbit"
        assert payload["qa"][0]["chunk_id"] == 0
        assert payload["layout"]["fps"] == 24.0

        # The queue endpoint has a safe empty payload and reads only the process artifact.
        assert server.MovieDirs(root).review_queue == root / "tmp" / "review_queue.json"

        # atomic json write round-trips.
        draft = root / "build" / "review_patch.draft.json"
        server._atomic_write_json(draft, {"renames": {"e1": "Bunny"}})
        assert json.loads(draft.read_text())["renames"]["e1"] == "Bunny"

        # static path resolver: allow listed assets, reject traversal / missing.
        css = server.resolve_static_path("/static/css/app.css")
        assert css is not None and css.name == "app.css" and css.is_file(), css
        js = server.resolve_static_path("/static/js/api.js")
        assert js is not None and js.name == "api.js" and js.is_file(), js
        roster_html = server.resolve_static_path("/static/roster.html")
        assert roster_html is not None and roster_html.name == "roster.html"
        assert server.resolve_static_path("/static/../server.py") is None
        assert server.resolve_static_path("/static/js/nope.exe") is None
        assert server.resolve_static_path("/static/missing.js") is None

        # /img serves both top-level assets/ and legacy derived/ paths under OUT_DIR.
        server.OUT_DIR = root
        (root / "assets" / "char_x").mkdir(parents=True)
        (root / "assets" / "char_x" / "cover.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "derived" / "assets" / "char_y").mkdir(parents=True)
        (root / "derived" / "assets" / "char_y" / "c000.jpg").write_bytes(b"\xff\xd8\xff")

        class _FakeHandler(server.Handler):
            def __init__(self) -> None:  # noqa: D107 — bypass BaseHTTPRequestHandler.__init__
                pass

        h = _FakeHandler()
        sent: list[tuple] = []

        def _send(code, body, ctype="text/plain"):  # noqa: ANN001
            sent.append((code, body, ctype))

        h._send = _send  # type: ignore[method-assign]
        h._serve_img("assets/char_x/cover.jpg")
        assert sent[-1][0] == 200 and sent[-1][2] == "image/jpeg"
        h._serve_img("derived/assets/char_y/c000.jpg")
        assert sent[-1][0] == 200
        h._serve_img("../etc/passwd")
        assert sent[-1][0] == 404

        # Roster curation: bootstrap from gold, save a draft, then validate/promote a confirmed seed.
        proposal = server.roster_seed_editor_payload(root)
        assert proposal["source"] == "gold_proposal"
        assert proposal["seed"]["human_confirmed"] is False
        valid_seed = {
            "version": 1, "movie_id": "m", "human_confirmed": False,
            "entities": [{
                "selected": True, "entity_id": "char_rabbit", "name": "Rabbit",
                "kind": "character", "identity_scope": "individual",
                "description": "Large white rabbit.", "grounding_phrases": ["white rabbit"],
                "aliases": [], "exemplar_crops": ["assets/char_x/cover.jpg"],
                "static_attributes": {"species": "rabbit"}, "allowed_state_events": [],
            }],
        }
        saved = server.save_roster_seed(root, valid_seed, confirm=False)
        assert saved["confirmed"] is False and (root / "roster_seed.draft.json").is_file()
        promoted = server.save_roster_seed(root, valid_seed, confirm=True)
        assert promoted["confirmed"] is True and promoted["n_entities"] == 1
        assert json.loads((root / "roster_seed.json").read_text())["human_confirmed"] is True

    print("test_web_server: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
