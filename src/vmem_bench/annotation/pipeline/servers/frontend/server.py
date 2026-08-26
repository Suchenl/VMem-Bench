"""Static frontend and API proxy for the annotation console."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request

from vmem_bench.annotation.pipeline.servers.direct_http import (
    ensure_no_proxy_env,
    ensure_no_proxy_host,
    urlopen_direct,
)

# Dev-machine http_proxy must never wrap local backend or fleet calls.
ensure_no_proxy_env(extra="11.0.0.0/8")

STATIC_DIR = Path(__file__).parent / "static"
BACKEND_URL = "http://127.0.0.1:7864"
VLM_BASE_URL = ""
_STATIC_SUFFIXES = {".html", ".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".map"}


def resolve_static_path(url_path, static_dir):
    if not url_path.startswith("/static/"):
        return None
    rel = unquote(url_path[len("/static/") :]).lstrip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    root = Path(static_dir).resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_file() or target.suffix.lower() not in _STATIC_SUFFIXES:
        return None
    return target


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, data, ctype="application/octet-stream", extra_headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(str(key), str(value))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _proxy(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        target = BACKEND_URL.rstrip("/") + self.path
        ensure_no_proxy_host(BACKEND_URL)
        headers = {"Accept": self.headers.get("Accept", "application/json")}
        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]
        for header in ("Range", "If-Range"):
            if self.headers.get(header):
                headers[header] = self.headers[header]
        req = Request(target, data=body, headers=headers, method=self.command)
        try:
            with urlopen_direct(req, timeout=120) as resp:
                ctype = (
                    resp.headers.get("Content-Type")
                    or resp.headers.get_content_type()
                    or "application/octet-stream"
                )
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                for header in ("Content-Length", "Content-Range", "Accept-Ranges"):
                    if resp.headers.get(header):
                        self.send_header(header, resp.headers[header])
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Expose-Headers", "Accept-Ranges, Content-Range, Content-Length")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command == "HEAD":
                    return
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    self.wfile.write(block)
        except HTTPError as exc:
            try:
                self._send(
                    exc.code,
                    exc.read(),
                    exc.headers.get("Content-Type") if exc.headers else "application/json",
                    {
                        header: exc.headers[header]
                        for header in ("Content-Range", "Accept-Ranges")
                        if exc.headers and exc.headers.get(header)
                    },
                )
            except (BrokenPipeError, ConnectionResetError):
                return
        except URLError as exc:
            self._send_json(502, {"ok": False, "error": f"backend unavailable: {exc.reason}"})
        except TimeoutError:
            # Always return JSON — silent return made the browser show "Failed to fetch".
            self._send_json(
                504,
                {"ok": False, "error": "backend timeout (>120s); retry or check backend load"},
            )
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_static(self, url_path):
        target = resolve_static_path(url_path, STATIC_DIR)
        if target is None:
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix.lower() == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif target.suffix.lower() == ".css":
            ctype = "text/css; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def do_HEAD(self):
        # Some previews / proxies probe with HEAD; previously returned 501.
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            return self._proxy()
        if path in {"/", "/index.html"} or path.startswith("/static/") or path.startswith("/review/"):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
                if path in {"/", "/index.html"} or path.endswith(".html")
                else "application/octet-stream",
            )
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        pages = {
            "/": "index.html",
            "/index.html": "index.html",
            "/review/s4": "review_s4.html",
            "/review/s4.html": "review_s4.html",
            "/review/s3": "review_s3.html",
            "/review/s3.html": "review_s3.html",
            "/review/s6": "review_s6.html",
            "/review/s6.html": "review_s6.html",
            "/review/stage": "review_stage.html",
            "/review/stage.html": "review_stage.html",
        }
        if path in pages:
            return self._send(200, (STATIC_DIR / pages[path]).read_bytes(), "text/html; charset=utf-8")
        if path == "/static/js/config.js":
            return self._send(
                200,
                ("window.MemStrataConfig = " + json.dumps({"defaultVlmBaseUrl": VLM_BASE_URL}) + ";\n").encode(),
                "application/javascript; charset=utf-8",
            )
        if path.startswith("/static/"):
            return self._serve_static(path)
        if path.startswith("/api/"):
            return self._proxy()
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path.startswith("/api/"):
            return self._proxy()
        self._send_json(404, {"ok": False, "error": "not found"})


def main():
    global BACKEND_URL, VLM_BASE_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--backend-url", default=BACKEND_URL)
    parser.add_argument("--vlm-base-url", default=os.environ.get("VLM_BASE_URL", ""))
    args = parser.parse_args()
    BACKEND_URL = args.backend_url
    VLM_BASE_URL = args.vlm_base_url
    ensure_no_proxy_host(BACKEND_URL)
    print(f"MemStrata annotation frontend: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
