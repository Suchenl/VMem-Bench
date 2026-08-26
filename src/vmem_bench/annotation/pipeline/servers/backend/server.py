"""HTTP API for the MemStrata annotation pipeline console."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog_state import (
    default_data_root, get_sample, list_samples, memstrata_root_from_here,
    resolve_crop_path, sample_detail, stage_inspect,
)
from .jobs import JobStore
from .review_service import (
    build_segment_clip, build_segment_poster, get_sample_or_raise,
    resolve_review_media, s3_live_payload, s4_payload, s6_alternates,
    s6_payload, save_s4, save_s4_partial, save_s6,
)
from vmem_bench.annotation.pipeline.orchestration.catalog import DEFAULT_BLENDER_VIDEOS_ROOT
from vmem_bench.annotation.pipeline.servers.direct_http import ensure_no_proxy_env
from vmem_bench.annotation.pipeline.servers.fleet.registry import (
    default_fleet_root,
    list_fleet,
)

ensure_no_proxy_env(extra="11.0.0.0/8")

DATA_ROOT = default_data_root()
STORE = JobStore(Path(DATA_ROOT / "_services" / "annotation_jobs"), DATA_ROOT)


def parse_byte_range(value: str, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, returning inclusive ``(start, end)``."""
    text = str(value or "").strip()
    if not text:
        return None
    if not text.startswith("bytes=") or "," in text:
        raise ValueError("unsupported byte range")
    spec = text[6:].strip()
    if "-" not in spec or size <= 0:
        raise ValueError("invalid byte range")
    start_text, end_text = spec.split("-", 1)
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("invalid byte suffix")
        start = max(0, size - suffix)
        return start, size - 1
    start = int(start_text)
    if start < 0 or start >= size:
        raise ValueError("range start outside file")
    end = size - 1 if not end_text else min(int(end_text), size - 1)
    if end < start:
        raise ValueError("range end before start")
    return start, end


def _env_path(value):
    return Path(value).expanduser() if value else None

def _default_blender_index(root: Path) -> Path | None:
    """Prefer local download catalogs with ``filename`` + Videos layout."""
    candidates = [
        root / "data" / "_services" / "blender_index.json",
        Path("${VMEM_DATASETS_ROOT}/BlenderOpenMovies/download_status.json"),
        Path("${UNSET_INTERNAL_PATH}"),
        root / "data" / "BlenderOpenMovies" / "download_status.json",
        root / "data" / "BlenderOpenMovies" / "manifest.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            continue
        # Skip remote-only download manifests that only list official .mov URLs.
        sample = items[0] if isinstance(items[0], dict) else {}
        if sample.get("filename") or Path(str(sample.get("file") or "")).suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return candidate.resolve()
        videos = DEFAULT_BLENDER_VIDEOS_ROOT
        if videos.is_dir() and any((videos / str(it.get("id") or "")).is_dir() for it in items[:5] if isinstance(it, dict)):
            return candidate.resolve()
    return None


def _default_lsmdc_index(root: Path, blender_index: Path | None = None) -> Path | None:
    """Discover LSMDC stitch index without hard-coding personal dataset roots."""
    candidates = [
        root / "data" / "_services" / "lsmdc_index.json",
        root / "data" / "LSMDC" / "complete_movies.json",
        Path("${VMEM_DATASETS_ROOT}/LSMDC/complete_movies.json"),
        Path("${UNSET_INTERNAL_PATH}"),
    ]
    if blender_index is not None and blender_index.is_file():
        try:
            payload = json.loads(blender_index.read_text(encoding="utf-8"))
            storage = Path(str(payload.get("storage_root") or ""))
            if storage.name:
                candidates.append(storage.parent / "LSMDC" / "complete_movies.json")
        except Exception:  # noqa: BLE001
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, data, content_type="application/json", extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range, If-Range")
        self.send_header("Access-Control-Expose-Headers", "Accept-Ranges, Content-Range, Content-Length")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for key, value in (extra_headers or {}).items():
            self.send_header(str(key), str(value))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def _serve_file(self, path, content_type, *, head_only=False):
        target = Path(path)
        size = target.stat().st_size
        try:
            byte_range = parse_byte_range(self.headers.get("Range", ""), size)
        except (TypeError, ValueError):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, max(0, size - 1))
        length = 0 if size == 0 else end - start + 1
        self.send_response(206 if byte_range is not None else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Accept-Ranges, Content-Range, Content-Length")
        self.end_headers()
        if head_only or length == 0:
            return
        remaining = length
        with target.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                block = handle.read(min(1 << 20, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def _selected_sample(self, query):
        dataset = query.get("dataset", [""])[0]
        movie_id = query.get("movie_id", [""])[0]
        if not dataset or not movie_id:
            return None
        # Avoid full-catalog scans on every review/media hit (was crowding API).
        return get_sample(
            data_root=DATA_ROOT,
            dataset=dataset,
            movie_id=movie_id,
            blender_index=STORE.blender_index,
            lsmdc_index=STORE.lsmdc_index,
        )

    def _serve_review_get(self, stage):
        query = parse_qs(urlparse(self.path).query)
        sample = self._selected_sample(query)
        if sample is None:
            return self._json(404, {"ok": False, "error": "sample not found"})
        if stage == "s3":
            limit = int(query.get("limit", ["200"])[0] or 200)
            return self._json(200, s3_live_payload(sample, limit=limit))
        payload = s4_payload(sample) if stage == "s4" else s6_payload(sample)
        return self._json(200, payload)

    def _serve_job_log(self, job_id):
        job = STORE.refresh_job(job_id)
        log_path = Path(str(job.get("log_path") or ""))
        if not log_path.is_file():
            return self._json(404, {"ok": False, "error": "log not found"})
        size = log_path.stat().st_size
        query = parse_qs(urlparse(self.path).query)
        offset = int(query.get("offset", [max(0, size - 65536)])[0])
        offset = max(0, min(offset, size))
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read(262144)
        ctype = mimetypes.guess_type(str(log_path))[0] or "text/plain; charset=utf-8"
        return self._json(200, {"job_id": job_id, "offset": offset, "next_offset": offset + len(data), "size": size, "text": data.decode("utf-8", errors="replace"), "content_type": ctype})

    def _serve_segment_media(self, *, head_only=False):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        sample = self._selected_sample(q)
        sid = q.get("segment_id", [""])[0]
        if sample is None or not sid:
            return self._json(404, {"ok": False, "error": "segment media not found"})
        try:
            start = float(q["start_seconds"][0]) if q.get("start_seconds", [""])[0] else None
            end = float(q["end_seconds"][0]) if q.get("end_seconds", [""])[0] else None
            is_clip = "clip" in path
            media = (
                build_segment_clip(sample, sid, start_seconds=start, end_seconds=end)
                if is_clip
                else build_segment_poster(sample, sid, start_seconds=start)
            )
            return self._serve_file(
                media,
                "video/mp4" if is_clip else "image/jpeg",
                head_only=head_only,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            return self._json(400, {"ok": False, "error": str(exc)})

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path in (
            "/api/review/segment-clip",
            "/review/segment-clip",
            "/api/review/segment-poster",
            "/review/segment-poster",
        ):
            return self._serve_segment_media(head_only=True)
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path); path = parsed.path
        if path in ("/api/health", "/health"):
            fleet = list_fleet(fleet_root=STORE.fleet_root, probe=False)
            return self._json(200, {
                "ok": True,
                "data_root": str(DATA_ROOT),
                "jobs_root": str(STORE.jobs_root),
                "fleet_root": str(STORE.fleet_root),
                "fleet_online": fleet.get("online_count", 0),
                "fleet_busy": fleet.get("busy_count", 0),
                "fleet_idle": fleet.get("idle_count", 0),
                "fleet_total": fleet.get("total_count", 0),
                "timezone": "Asia/Shanghai",
                "default_vlm_base_url": fleet.get("dispatch_url") or os.environ.get("VLM_BASE_URL", ""),
                "default_reviewer_model": fleet.get("default_model") or "",
            })
        if path in ("/api/fleet", "/fleet"):
            q = parse_qs(parsed.query)
            probe = str(q.get("probe", ["0"])[0]).lower() in {"1", "true", "yes"}
            return self._json(200, list_fleet(fleet_root=STORE.fleet_root, probe=probe))
        if path in ("/api/samples", "/samples"):
            q = parse_qs(parsed.query); samples = list_samples(data_root=DATA_ROOT, blender_index=STORE.blender_index, lsmdc_index=STORE.lsmdc_index)
            dataset, status = q.get("dataset", [""])[0], q.get("status", [""])[0]; text = q.get("q", [""])[0].lower().strip()
            if dataset: samples = [s for s in samples if s["dataset"] == dataset]
            if status: samples = [s for s in samples if s["status"] == status]
            if text: samples = [s for s in samples if text in s["movie_id"].lower() or text in s["dataset"].lower()]
            return self._json(200, {"samples": samples, "count": len(samples)})
        if path in ("/api/sample-detail", "/sample-detail"):
            sample = self._selected_sample(parse_qs(parsed.query))
            return self._json(404, {"ok": False, "error": "sample not found"}) if sample is None else self._json(200, sample_detail(sample))
        if path in ("/api/review/stage", "/review/stage"):
            q = parse_qs(parsed.query)
            sample = self._selected_sample(q)
            if sample is None:
                return self._json(404, {"ok": False, "error": "sample not found"})
            return self._json(200, stage_inspect(sample, q.get("stage", [""])[0]))
        if path in ("/api/crop", "/crop"):
            q=parse_qs(parsed.query); sample=self._selected_sample(q); crop=resolve_crop_path(sample,q.get("path",[""])[0]) if sample else None
            return self._json(404, {"ok":False,"error":"crop not found"}) if crop is None else self._send(200,crop.read_bytes(),mimetypes.guess_type(str(crop))[0] or "image/jpeg")
        if path in ("/api/review/s4","/review/s4"): return self._serve_review_get("s4")
        if path in ("/api/review/s3","/review/s3"): return self._serve_review_get("s3")
        if path in ("/api/review/s6","/review/s6"): return self._serve_review_get("s6")
        if path in ("/api/review/media","/review/media"):
            q=parse_qs(parsed.query); sample=self._selected_sample(q); media=resolve_review_media(sample,q.get("path",[""])[0]) if sample else None
            return self._json(404,{"ok":False,"error":"media not found"}) if media is None else self._send(200,media.read_bytes(),mimetypes.guess_type(str(media))[0] or "image/jpeg")
        if path in ("/api/review/segment-clip","/review/segment-clip","/api/review/segment-poster","/review/segment-poster"):
            return self._serve_segment_media()
        if path in ("/api/review/s6/alts","/review/s6/alts"):
            q=parse_qs(parsed.query); sample=self._selected_sample(q); rid=q.get("representation_id",[""])[0]
            return self._json(404,{"ok":False,"error":"sample/representation not found"}) if sample is None or not rid else self._json(200,s6_alternates(sample,rid))
        if path in ("/api/jobs","/jobs"): return self._json(200,STORE.jobs_snapshot())
        if path in ("/api/jobs/active","/jobs/active"):
            snap = STORE.jobs_snapshot()
            return self._json(200,{"jobs":snap["active"],"sample_activity":snap["sample_activity"]})
        if path.startswith(("/api/jobs/","/jobs/")): return self._json(200,STORE.refresh_job(path.rstrip("/").split("/")[-1]))
        if path.startswith(("/api/job-log/","/job-log/")): return self._serve_job_log(path.rstrip("/").split("/")[-1])
        return self._json(404,{"ok":False,"error":"not found"})

    def do_POST(self):
        path=urlparse(self.path).path
        try:
            body=self._read_json_body()
            if path in ("/api/jobs","/jobs"): return self._json(200,STORE.create_job(body))
            if path in ("/api/jobs/stop-all","/jobs/stop-all"): return self._json(200,STORE.stop_all_jobs())
            if path in ("/api/jobs/stop-sample","/jobs/stop-sample"):
                dataset = str(body.get("dataset") or "").strip()
                movie_id = str(body.get("movie_id") or "").strip()
                if not dataset or not movie_id:
                    raise ValueError("dataset and movie_id are required")
                return self._json(200, STORE.stop_sample_job(dataset=dataset, movie_id=movie_id))
            if path.startswith(("/api/jobs/","/jobs/")) and path.endswith("/stop"): return self._json(200,STORE.stop_job(path.rstrip("/").split("/")[-2]))
            if path in ("/api/review/continue","/review/continue"): return self._json(200,STORE.create_continue_job(body))
            if path in ("/api/review/s4/accept-all", "/review/s4/accept-all"): return self._json(200, STORE.accept_all_s4(body))
            if path in ("/api/review/s4/apply","/review/s4/apply","/api/review/s4/draft","/review/s4/draft","/api/review/s6/apply","/review/s6/apply"):
                sample=get_sample_or_raise(data_root=DATA_ROOT,blender_index=STORE.blender_index,lsmdc_index=STORE.lsmdc_index,dataset=str(body.get("dataset") or ""),movie_id=str(body.get("movie_id") or ""))
                decisions=dict(body.get("decisions") or {})
                if "s6" in path: result=save_s6(sample=sample,decisions=decisions)
                elif "draft" in path: result=save_s4_partial(sample=sample,decisions=decisions,film_verdict=str(body.get("film_verdict") or "accept"),reason=str(body.get("reason") or ""))
                else: result=save_s4(sample=sample,decisions=decisions,film_verdict=str(body.get("film_verdict") or "accept"),reason=str(body.get("reason") or ""))
                return self._json(200,result)
            return self._json(404,{"ok":False,"error":"not found"})
        except FileNotFoundError as exc: return self._json(404,{"ok":False,"error":str(exc)})
        except ValueError as exc: return self._json(400,{"ok":False,"error":str(exc)})
        except Exception as exc: return self._json(500,{"ok":False,"error":f"{type(exc).__name__}: {exc}"})

def main():
    global DATA_ROOT, STORE
    root=memstrata_root_from_here(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=7864)
    parser.add_argument("--data-root",type=Path,default=default_data_root()); parser.add_argument("--jobs-root",type=Path,default=root/"data"/"_services"/"annotation_jobs")
    parser.add_argument("--python",default=""); parser.add_argument("--blender-index",type=Path,default=_env_path(os.environ.get("BLENDER_INDEX"))); parser.add_argument("--lsmdc-index",type=Path,default=_env_path(os.environ.get("LSMDC_INDEX")))
    args=parser.parse_args(); DATA_ROOT=args.data_root.resolve()
    bi=(args.blender_index.resolve() if args.blender_index else _env_path(os.environ.get("BLENDER_INDEX")) or _default_blender_index(root))
    li=(args.lsmdc_index.resolve() if args.lsmdc_index else _env_path(os.environ.get("LSMDC_INDEX")) or _default_lsmdc_index(root,bi))
    fleet_root=_env_path(os.environ.get("MEMSTRATA_FLEET_ROOT")) or default_fleet_root()
    STORE=JobStore(args.jobs_root.resolve(),DATA_ROOT,args.python or sys.executable,bi,li,fleet_root)
    print(f"MemStrata annotation backend: http://{args.host}:{args.port}",flush=True)
    print(f"Fleet root: {fleet_root}",flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()

if __name__ == "__main__": main()
