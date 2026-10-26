"""MiniMax-H3 server app with a fake renderer: validation, output cache keyed on the request fingerprint."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from vmem_bench.testbed.backends.minimax_h3.server import MODEL_ID, build_app  # noqa: E402

MODEL = {"h3_root": "/weights/MiniMax-H3", "lora": "/weights/turbo.safetensors", "lora_alpha": 8.0}


@pytest.fixture
def app_and_calls():
    calls: list[dict] = []

    def render(body):
        calls.append(body)
        Path(body["out"]).write_bytes(b"mp4-%d" % len(calls))
        return {"generate_sec": 0.1, "delivered_frames": body["num_frames"]}

    return TestClient(build_app(render, model=MODEL)), calls


def _body(tmp_path: Path, **kw):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref")
    body = {"prompt": "p", "references": [str(ref)], "seed": 1, "out": str(tmp_path / "seg_000.mp4")}
    body.update(kw)
    return body


def test_health(app_and_calls):
    client, _ = app_and_calls
    h = client.get("/health").json()
    assert h["ok"] is True and h["model"] == MODEL_ID and h["busy"] is False


def test_realize_writes_sidecar_and_caches_identical_request(tmp_path, app_and_calls):
    client, calls = app_and_calls
    r1 = client.post("/v1/realize", json=_body(tmp_path)).json()
    assert r1["cached"] is False and r1["clip"] == str(tmp_path / "seg_000.mp4")
    assert r1["request"]["num_frames"] == 121 and r1["request"]["height"] == 480
    assert r1["geometry"] == {"name": "480x832", "render": [480, 832], "delivered": [480, 832]}
    assert (tmp_path / "seg_000.json").is_file()
    r2 = client.post("/v1/realize", json=_body(tmp_path)).json()
    assert r2["cached"] is True and r2["fingerprint"] == r1["fingerprint"] and len(calls) == 1


def test_changed_seed_or_reference_bytes_rerenders(tmp_path, app_and_calls):
    client, calls = app_and_calls
    client.post("/v1/realize", json=_body(tmp_path))
    client.post("/v1/realize", json=_body(tmp_path, seed=2))
    assert len(calls) == 2
    (tmp_path / "ref.png").write_bytes(b"other")
    ref = tmp_path / "ref.png"
    r = client.post("/v1/realize", json={"prompt": "p", "references": [str(ref)], "seed": 2,
                                         "out": str(tmp_path / "seg_000.mp4")}).json()
    assert r["cached"] is False and len(calls) == 3


def test_validation_errors(tmp_path, app_and_calls):
    client, calls = app_and_calls
    assert client.post("/v1/realize", json=_body(tmp_path, prompt="  ")).status_code == 400
    assert client.post("/v1/realize", json=_body(tmp_path, references=[str(tmp_path / "nope.png")])).status_code == 400
    assert client.post("/v1/realize", json=_body(tmp_path, height=700)).status_code == 400
    assert client.post("/v1/realize", json=_body(tmp_path, references=[str(tmp_path / "ref.png")] * 10)).status_code == 400
    assert calls == []


def test_render_failure_is_http_500(tmp_path):
    def render(body):
        raise RuntimeError("CUDA out of memory")

    client = TestClient(build_app(render, model=MODEL))
    r = client.post("/v1/realize", json=_body(tmp_path))
    assert r.status_code == 500 and "CUDA out of memory" in r.json()["detail"]
    assert not (tmp_path / "seg_000.json").exists()


def test_geometry_preset_fills_sizes_and_keys_the_cache(tmp_path, app_and_calls):
    client, calls = app_and_calls
    r480 = client.post("/v1/realize", json=_body(tmp_path)).json()
    r704 = client.post("/v1/realize", json=_body(tmp_path, geometry="704x1280")).json()
    assert len(calls) == 2 and r704["cached"] is False and r704["fingerprint"] != r480["fingerprint"]
    assert (calls[1]["height"], calls[1]["width"], calls[1]["out_height"], calls[1]["out_width"]) == (704, 1280, 480, 832)
    assert r704["geometry"] == {"name": "704x1280", "render": [704, 1280], "delivered": [480, 832]}
    again = client.post("/v1/realize", json=_body(tmp_path, geometry="704x1280")).json()
    assert again["cached"] is True and len(calls) == 2
    assert client.post("/v1/realize", json=_body(tmp_path, geometry="720p")).status_code == 400


def test_server_default_geometry_applies_when_request_omits_it(tmp_path):
    calls = []

    def render(body):
        calls.append(body)
        Path(body["out"]).write_bytes(b"mp4")
        return {}

    client = TestClient(build_app(render, model=MODEL, default_geometry="704x1280"))
    assert client.get("/health").json()["default_geometry"] == "704x1280"
    r = client.post("/v1/realize", json=_body(tmp_path)).json()
    assert calls[0]["geometry"] == "704x1280" and (calls[0]["height"], calls[0]["width"]) == (704, 1280)
    assert r["request"]["geometry"] == "704x1280"
    with pytest.raises(ValueError):
        build_app(render, model=MODEL, default_geometry="1080p")
