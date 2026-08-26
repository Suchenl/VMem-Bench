"""HTTP Range support for annotation-console video proxying."""

from __future__ import annotations

import threading
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

from vmem_bench.annotation.pipeline.servers.backend.server import (
    Handler as BackendHandler,
    parse_byte_range,
)
from vmem_bench.annotation.pipeline.servers.frontend import server as frontend_server


def test_parse_byte_range_variants() -> None:
    assert parse_byte_range("", 100) is None
    assert parse_byte_range("bytes=0-9", 100) == (0, 9)
    assert parse_byte_range("bytes=90-", 100) == (90, 99)
    assert parse_byte_range("bytes=-10", 100) == (90, 99)


@contextmanager
def _server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _range_request(url: str):
    req = urllib.request.Request(url, headers={"Range": "bytes=2-5"})
    return urllib.request.urlopen(req, timeout=5)  # noqa: S310 - local test server


def test_backend_and_frontend_proxy_preserve_range(tmp_path: Path) -> None:
    payload = b"0123456789"
    media = tmp_path / "clip.mp4"
    media.write_bytes(payload)

    class FileHandler(BackendHandler):
        def do_GET(self):
            self._serve_file(media, "video/mp4")

        def log_message(self, *_args):
            return

    class QuietFrontend(frontend_server.Handler):
        def log_message(self, *_args):
            return

    with _server(FileHandler) as backend_url:
        with _range_request(backend_url + "/clip") as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.read() == b"2345"

        old_backend = frontend_server.BACKEND_URL
        frontend_server.BACKEND_URL = backend_url
        try:
            with _server(QuietFrontend) as frontend_url:
                with _range_request(frontend_url + "/api/clip") as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"] == "bytes 2-5/10"
                    assert response.headers["Accept-Ranges"] == "bytes"
                    assert response.read() == b"2345"
        finally:
            frontend_server.BACKEND_URL = old_backend

