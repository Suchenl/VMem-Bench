"""Live annotation-progress dashboard + human-review API (stdlib-only, no new dependencies).

Two roles over one stdlib ``http.server`` (dashboard_and_review_spec.md):

* **Live monitor** — tails ``<out>/tmp/events.jsonl`` (written by the pipeline, possibly on a
  GPU node, read here over the shared filesystem) and streams every event via Server-Sent Events.
* **Review & edit** — once the run finishes (gold exists), serves the gold over ``GET /gold`` and
  accepts edits as review patches (``POST /review/patch|apply|freeze``), reusing the exact patch
  semantics of ``annotation/review.py`` so gold-mutation logic lives in one place.

Self-containment: imports only ``vmem_bench`` (never ``memstrata``/SUT) and stdlib. The heavy
review imports are done lazily inside handlers so the dashboard stays light and starts even if an
optional dep is missing. The SPA shell is ``static/index.html``; CSS/JS modules are served under ``/static/``
(no build step; CDN Vue + Tailwind).

Usage:
    PYTHONPATH=src python3 -m vmem_bench.web.server \
        --out data/blender_open_movies/big_buck_bunny --port 7863
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from vmem_bench.common.paths import MovieDirs

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUT_DIR: Path = Path(".")  # set in main()
WORKSPACE_DIR: Path | None = None
_WRITE_LOCK = threading.Lock()  # serialize gold-mutating review writes (single reviewer assumed)
_STATIC_SUFFIXES = {".js", ".css", ".html", ".map", ".woff", ".woff2", ".ttf", ".svg", ".ico"}


def resolve_static_path(url_path: str, static_dir: Path = STATIC_DIR) -> Path | None:
    """Map ``/static/<rel>`` to a file under ``static_dir``; reject traversal / bad suffixes."""
    if not url_path.startswith("/static/"):
        return None
    rel = unquote(url_path[len("/static/"):]).lstrip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    static_root = Path(static_dir).resolve()
    target = (static_root / rel).resolve()
    try:
        target.relative_to(static_root)
    except ValueError:
        return None
    if not target.is_file() or target.suffix.lower() not in _STATIC_SUFFIXES:
        return None
    return target


# --- review/status helpers (module-level -> unit-testable without a live server) ------------

def _looks_like_movie_root(p: Path) -> bool:
    """True if ``p`` is (or is becoming) one movie's own output dir, not a workspace of movies.

    A movie root always ends up with ``tmp/`` (live events + scratch) and/or ``gold/`` (final
    output); legacy runs use ``build/``/``layout/``/``derived/`` instead. A workspace directory
    (multiple movies as siblings) has none of these directly under it -- only movie subdirs do.
    """
    return any((p / marker).is_dir() or (p / marker).is_file()
               for marker in ("tmp", "gold", "build", "layout", "derived"))


def run_done(out_dir: Path) -> bool:
    """A run is reviewable once gold exists (the pipeline writes gold only at the very end)."""
    return MovieDirs(Path(out_dir)).registry_json.is_file()


def last_stage(out_dir: Path) -> dict:
    """Best-effort current stage from the tail of events.jsonl (drives the Live/Review switch)."""
    ev = MovieDirs(Path(out_dir)).events
    empty = {"kind": None, "stage": None}
    if not ev.is_file():
        return empty
    last = ""
    for line in ev.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    if not last:
        return empty
    try:
        d = json.loads(last)
    except json.JSONDecodeError:
        return empty
    return {"kind": d.get("kind"), "stage": d.get("stage"), "movie_id": d.get("movie_id"),
            "n_chunks": d.get("n_chunks"), "n_entities": d.get("n_entities")}


def gold_payload(out_dir: Path) -> dict:
    """Assemble the review payload from on-disk gold + sidecars (spec §5 GET /gold)."""
    from vmem_bench.common.schemas import ChunkAnnotations, EntityRegistry
    out_dir = Path(out_dir)
    dirs = MovieDirs(out_dir)
    registry = EntityRegistry.from_dict(
        json.loads(dirs.registry_json.read_text(encoding="utf-8")))
    chunks = ChunkAnnotations.from_dict(
        json.loads(dirs.annotations_json.read_text(encoding="utf-8")))
    qa_path = dirs.qa_report
    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else []
    layout_path = dirs.chunk_index
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else {}
    return {"registry": registry.to_dict(), "chunks": chunks.to_dict(), "qa": qa,
            "layout": layout, "human_reviewed": registry.human_reviewed, "done": True}


def _atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def roster_seed_editor_payload(out_dir: Path) -> dict[str, Any]:
    """Return a seed draft/confirmed seed, or bootstrap candidates from current gold."""
    out_dir = Path(out_dir)
    confirmed = out_dir / "roster_seed.json"
    draft = out_dir / "roster_seed.draft.json"
    source = confirmed if confirmed.is_file() else draft
    candidate_crops: dict[str, list[str]] = {}
    registry_payload: dict[str, Any] = {}
    dirs = MovieDirs(out_dir)
    if dirs.registry_json.is_file():
        registry_payload = json.loads(dirs.registry_json.read_text(encoding="utf-8"))
        for entity in registry_payload.get("entities", []):
            candidate_crops[str(entity.get("entity_id") or "")] = [
                str(rep.get("crop_path") or "")
                for rep in entity.get("representations", [])
                if rep.get("crop_path")
            ][:12]
    if source.is_file():
        seed = json.loads(source.read_text(encoding="utf-8"))
    else:
        entities = []
        for entity in registry_payload.get("entities", []):
            kind = str(entity.get("kind") or "")
            if kind not in ("character", "prop"):
                continue
            entity_id = str(entity.get("entity_id") or "")
            crops = candidate_crops.get(entity_id, [])
            name = str(entity.get("name") or entity_id)
            entities.append({
                "selected": True,
                "entity_id": entity_id,
                "name": name,
                "kind": kind,
                "identity_scope": "individual" if kind == "character" else "category",
                "description": str(entity.get("description") or ""),
                "grounding_phrases": [name.replace("_", " ").lower()],
                "aliases": [],
                "exemplar_crops": crops[:3] if kind == "character" else [],
                "static_attributes": dict(entity.get("static_attributes") or {}),
                "allowed_state_events": [],
            })
        seed = {
            "version": 1,
            "movie_id": str(registry_payload.get("movie_id") or out_dir.name),
            "human_confirmed": False,
            "entities": entities,
        }
    for entity in seed.get("entities", []):
        entity.setdefault("selected", True)
        entity_id = str(entity.get("entity_id") or "")
        candidate_crops.setdefault(entity_id, [])
        for crop in entity.get("exemplar_crops") or []:
            if crop not in candidate_crops[entity_id]:
                candidate_crops[entity_id].append(crop)
    return {
        "seed": seed,
        "candidate_crops": candidate_crops,
        "source": ("confirmed" if confirmed.is_file() else "draft" if draft.is_file()
                   else "gold_proposal"),
        "confirmed": bool(seed.get("human_confirmed", False)),
    }


def save_roster_seed(out_dir: Path, payload: dict[str, Any], *, confirm: bool) -> dict[str, Any]:
    """Atomically save a roster draft or validate+promote it to the production seed."""
    out_dir = Path(out_dir)
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        raise ValueError("roster seed body requires an entities list")
    clean = dict(payload)
    clean["version"] = 1
    clean["human_confirmed"] = bool(confirm)
    clean["entities"] = [
        {k: v for k, v in entity.items() if k not in {"selected", "candidate_crops"}}
        for entity in clean["entities"]
        if isinstance(entity, dict) and entity.get("selected", True)
    ]
    if not clean["entities"]:
        raise ValueError("at least one roster entity must remain selected")
    draft_path = out_dir / "roster_seed.draft.json"
    _atomic_write_json(draft_path, clean)
    if not confirm:
        return {"ok": True, "confirmed": False, "path": str(draft_path)}
    candidate = out_dir / "roster_seed.json.tmp"
    _atomic_write_json(candidate, clean)
    try:
        from vmem_bench.annotation.pipeline_track_first.roster_seed import load_roster_seed
        seed = load_roster_seed(candidate, require_confirmed=True)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    final = out_dir / "roster_seed.json"
    candidate.replace(final)
    return {"ok": True, "confirmed": True, "path": str(final),
            "n_entities": len(seed.entities)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quiet
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _movies(self) -> list[dict]:
        if WORKSPACE_DIR is None:
            return [{"movie_id": OUT_DIR.name, "done": run_done(OUT_DIR)}]
        return [{"movie_id": p.name, "done": run_done(p)} for p in sorted(WORKSPACE_DIR.iterdir())
                if p.is_dir() and (MovieDirs(p).events.is_file() or run_done(p))]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path in ("/roster", "/roster.html"):
            self._send(200, (STATIC_DIR / "roster.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            self._serve_static(path)
        elif path == "/events":
            self._stream_events()
        elif path == "/img":
            self._serve_img(parse_qs(urlparse(self.path).query).get("p", [""])[0])
        elif path == "/status":
            st = last_stage(OUT_DIR)
            st["done"] = run_done(OUT_DIR)
            self._send_json(200, st)
        elif path == "/movies":
            self._send_json(200, {"movies": self._movies(), "selected": OUT_DIR.name})
        elif path == "/gold":
            if not run_done(OUT_DIR):
                self._send_json(409, {"ok": False, "done": False,
                                      "error": "gold not ready (run in progress)"})
                return
            self._send_json(200, gold_payload(OUT_DIR))
        elif path == "/auto_review":
            p = MovieDirs(OUT_DIR).auto_review_json
            if not p.is_file():
                self._send_json(200, {"gray_merges": [], "queue": [], "stats": {}})
                return
            self._send_json(200, json.loads(p.read_text(encoding="utf-8")))
        elif path == "/review_queue":
            p = MovieDirs(OUT_DIR).review_queue
            if not p.is_file():
                self._send_json(200, {"version": 1, "items": [], "summary": {}})
                return
            self._send_json(200, json.loads(p.read_text(encoding="utf-8")))
        elif path == "/review/patch":  # fetch the saved draft
            p = MovieDirs(OUT_DIR).review_patch_draft
            self._send_json(200, json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {})
        elif path == "/roster-seed":
            self._send_json(200, roster_seed_editor_payload(OUT_DIR))
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_static(self, url_path: str) -> None:
        target = resolve_static_path(url_path, STATIC_DIR)
        if target is None:
            self._send(404, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix.lower() == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif target.suffix.lower() == ".css":
            ctype = "text/css; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        global OUT_DIR
        path = urlparse(self.path).path
        if path == "/roster-seed":
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
                confirm = bool(body.pop("_confirm", False))
                with _WRITE_LOCK:
                    self._send_json(200, save_roster_seed(OUT_DIR, body, confirm=confirm))
            except Exception as exc:  # noqa: BLE001 - validation errors are user-facing
                self._send_json(400, {"ok": False, "error": str(exc)})
            return
        if not path.startswith("/review/"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        # Never mutate gold while the pipeline is still writing it (spec §5: 409 until done).
        if not run_done(OUT_DIR):
            self._send_json(409, {"ok": False, "error": "run not finished; gold not available yet"})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid JSON body"})
            return
        with _WRITE_LOCK:
            try:
                if path == "/review/select":
                    if WORKSPACE_DIR is None: raise ValueError("server is not in workspace mode")
                    candidate = (WORKSPACE_DIR / str(body.get("movie_id") or "")).resolve()
                    if candidate.parent != WORKSPACE_DIR.resolve() or not candidate.is_dir(): raise ValueError("unknown movie_id")
                    OUT_DIR = candidate
                    self._send_json(200, {"ok": True, "selected": candidate.name})
                elif path == "/review/patch":
                    _atomic_write_json(MovieDirs(OUT_DIR, write=True).review_patch_draft, body)
                    self._send_json(200, {"ok": True})
                elif path == "/review/apply":
                    self._apply(body)
                elif path == "/review/preview":
                    from vmem_bench.annotation.pipeline_track_first.review import preview_patch
                    self._send_json(200, preview_patch(OUT_DIR, body))
                elif path == "/review/freeze":
                    self._freeze()
                else:
                    self._send_json(404, {"ok": False, "error": "unknown review endpoint"})
            except Exception as exc:  # noqa: BLE001 - report, never crash the server thread
                self._send_json(400, {"ok": False, "error": str(exc)})

    def _apply(self, body: dict) -> None:
        from vmem_bench.annotation.pipeline_track_first.review import apply_patch
        dirs = MovieDirs(OUT_DIR)
        draft = dirs.review_patch_draft
        patch = body or (json.loads(draft.read_text(encoding="utf-8")) if draft.is_file() else {})
        patch_path = MovieDirs(OUT_DIR, write=True).review_patch_applied
        _atomic_write_json(patch_path, patch)
        apply_patch(dirs.gold, patch_path)
        self._send_json(200, gold_payload(OUT_DIR))

    def _freeze(self) -> None:
        from vmem_bench.annotation.pipeline_track_first.review import freeze
        try:
            freeze(MovieDirs(OUT_DIR).gold)
        except ValueError as exc:  # gold_lint failure -> structured-ish report
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, {"ok": True, "human_reviewed": True})

    def _serve_img(self, rel: str) -> None:
        # Images live under <out>/assets/ (published bank), <out>/tmp/candidates/ (scratch)
        # and legacy <out>/derived/. Only those subtrees are exposed; path traversal outside
        # them is rejected.
        dirs = MovieDirs(OUT_DIR)
        roots = [dirs.assets.resolve(), dirs.candidates.resolve(),
                 (OUT_DIR / "tmp" / "exemplars").resolve(),
                 (OUT_DIR / "derived").resolve()]
        rel_clean = rel
        # derived/ = legacy scratch; exemplars are route B's visual anchors.
        for marker in ("derived/", "tmp/candidates/", "tmp/exemplars/", "assets/"):
            if marker in rel:
                rel_clean = rel[rel.find(marker):]
                break
        target = (OUT_DIR / rel_clean).resolve()
        allowed = any(r == target or r in target.parents for r in roots)
        if (allowed and target.is_file()
                and target.suffix.lower() in (".jpg", ".jpeg", ".png")):
            ext = target.suffix.lower()
            ctype = "image/png" if ext == ".png" else "image/jpeg"
            self._send(200, target.read_bytes(), ctype)
        else:
            self._send(404, b"no image", "text/plain")

    def _stream_events(self) -> None:
        events_path = MovieDirs(OUT_DIR).events
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"event: reset\ndata: {}\n\n")
        self.wfile.flush()
        offset = 0
        buffer = b""
        try:
            while True:
                if events_path.is_file():
                    size = events_path.stat().st_size
                    if size < offset:  # log truncated / new run: resend from scratch
                        offset = 0
                        buffer = b""
                        self.wfile.write(b"event: reset\ndata: {}\n\n")
                    if size > offset:
                        with events_path.open("rb") as f:
                            f.seek(offset)
                            buffer += f.read(size - offset)
                            offset = size
                        *lines, buffer = buffer.split(b"\n")
                        for line in lines:
                            line = line.strip()
                            if line:
                                self.wfile.write(b"data: " + line + b"\n\n")
                        self.wfile.flush()
                    else:
                        self.wfile.write(b": ping\n\n")  # SSE comment heartbeat
                        self.wfile.flush()
                        time.sleep(5.0)
                else:
                    self.wfile.write(b": waiting for events.jsonl\n\n")
                    self.wfile.flush()
                    time.sleep(2.0)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    global OUT_DIR, WORKSPACE_DIR
    parser = argparse.ArgumentParser(description="MemStrata annotation live dashboard + review API")
    parser.add_argument("--out", type=Path, required=True,
                        help="annotation output dir (contains tmp/events.jsonl and gold/)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    requested = args.out.resolve()
    if _looks_like_movie_root(requested):
        # requested IS one movie's own output dir (has tmp/ or gold/ already) -- never treat it
        # as a workspace just because it also has sibling data dirs like assets/ (bug: a movie
        # dir with assets/ sorting before gold/tmp/ was mistaken for a multi-movie workspace and
        # OUT_DIR silently pointed at .../assets, which has no events.jsonl or gold/).
        OUT_DIR = requested
    else:
        movies = ([p for p in sorted(requested.iterdir()) if p.is_dir() and _looks_like_movie_root(p)]
                  if requested.is_dir() else [])
        if movies:
            WORKSPACE_DIR, OUT_DIR = requested, movies[0]
        else:
            OUT_DIR = requested
    print(f"serving {OUT_DIR} on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
