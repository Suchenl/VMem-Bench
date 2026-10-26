"""MiniMax-H3 Turbo backend: prompt conversion, request body, endpoint discovery, realize() over fake HTTP."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vmem_bench.testbed.backends.base import RealizationBackend, RealizationRequest
from vmem_bench.testbed.backends.minimax_h3 import (
    MiniMaxH3TurboBackend,
    build_h3_prompt,
    discover_endpoints,
    normalize_instruction,
)
from vmem_bench.testbed.backends.minimax_h3.client import candidate_urls
from vmem_bench.testbed.backends.minimax_h3.geometry import (
    DEFAULT_GEOMETRY,
    GEOMETRIES,
    MODEL_ID,
    aligned_frame_count,
)
from vmem_bench.testbed.backends.minimax_h3.pipeline import OFFLOAD_MODES, vram_placement

SEG = (
    "At dusk, lighthouse keeper Elias\uff08A lighthouse keeper in his fifties, gray-white beard\uff09 "
    "climbs the spiral iron stair into the lantern room\uff08The circular glass lantern room\uff09 and "
    "lights the central Fresnel lens\uff08A man-tall rotating Fresnel lens\uff09."
)


# ------------------------------------------------------------------ prompt conversion

def test_normalize_fullwidth_parentheses_and_spacing():
    s = normalize_instruction(SEG)
    assert "\uff08" not in s and "\uff09" not in s
    assert "Elias (A lighthouse keeper in his fifties, gray-white beard) climbs" in s
    assert "lantern room (The circular glass lantern room) and lights" in s
    assert s.endswith("Fresnel lens).")
    assert "  " not in s


def test_normalize_adds_terminal_period_and_rejects_empty():
    assert normalize_instruction("  Elias waves ") == "Elias waves."
    assert normalize_instruction("Elias asks, \u201cready?\u201d") == 'Elias asks, "ready?"'
    with pytest.raises(ValueError):
        normalize_instruction("  \n ")


def test_zero_reference_prompt_is_t2va_layout():
    p = build_h3_prompt(SEG, 0)
    assert p.startswith("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a single continuous 5-second shot. ")
    assert normalize_instruction(SEG) in p
    assert "\n\noverall_soundscape: " in p and p.endswith("non_diegetic_music: N/A")
    assert "<Picture" not in p and "<Subject" not in p and "subject_definitions" not in p


@pytest.mark.parametrize("n", [1, 2, 3])
def test_reference_prompt_is_full_reference_layout(n):
    p = build_h3_prompt(SEG, n)
    sections = ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                "overall_soundscape:", "non_diegetic_music:"]
    idx = [p.index(f"{s}\n") for s in sections]
    assert idx == sorted(idx) and p.startswith("subject_definitions:\n")
    for k in range(1, n + 1):
        assert f"<Subject {k}> is the recurring characters, objects, and setting shown in <Picture {k}>" in p
        assert f"<Subject {k}> (appears in [Shot 1]): partially_preserved" in p
    assert f"<Picture {n + 1}>" not in p and f"<Subject {n + 1}>" not in p
    assert "[reference generation] The target video is a single continuous 5-second shot." in p
    assert p.count(normalize_instruction(SEG)) == 2
    assert "[Shot 1] " + normalize_instruction(SEG) in p


def test_reference_lists_are_joined_in_english():
    assert "guide the appearance" in build_h3_prompt("x", 1)
    assert "<Subject 1> and <Subject 2> guide" in build_h3_prompt("x", 2)
    assert "<Picture 1>, <Picture 2>, and <Picture 3> are appearance references" in build_h3_prompt("x", 3)


def test_prompt_is_deterministic_and_bounded():
    assert build_h3_prompt(SEG, 2) == build_h3_prompt(SEG, 2)
    assert build_h3_prompt(SEG, 0, duration_sec=4.5).count("4.5-second") == 1
    with pytest.raises(ValueError):
        build_h3_prompt(SEG, -1)
    with pytest.raises(ValueError):
        build_h3_prompt(SEG, 10)


def test_aligned_frame_count():
    assert aligned_frame_count(121) == 124
    assert aligned_frame_count(124) == 124
    assert aligned_frame_count(1) == 5


# ------------------------------------------------------------------ request body

def _request(tmp_path: Path, n_refs: int, seed: int = 7) -> RealizationRequest:
    refs = []
    for k in range(n_refs):
        f = tmp_path / f"ref{k}.png"
        f.write_bytes(b"png")
        refs.append(f)
    return RealizationRequest(segment_id=12, prompt=SEG, references=tuple(refs), seed=seed, out_dir=tmp_path / "out")


def test_request_body_defaults_render_at_delivery_size(tmp_path):
    be = MiniMaxH3TurboBackend(["http://unused"])
    body = be.body_for(_request(tmp_path, 2))
    assert body["out"] == str((tmp_path / "out" / "seg_012.mp4").resolve())
    assert body["references"] == [str((tmp_path / "ref0.png").resolve()), str((tmp_path / "ref1.png").resolve())]
    assert body["seed"] == 7
    assert (body["num_frames"], body["fps"]) == (121, 24)
    assert body["geometry"] == "480x832" == DEFAULT_GEOMETRY
    assert (body["height"], body["width"], body["out_height"], body["out_width"]) == (480, 832, 480, 832)
    assert body["steps"] == 4 and body["ref_image_short_edge"] == 768
    assert body["prompt"] == build_h3_prompt(SEG, 2)
    assert set(body) == {"prompt", "references", "seed", "out", "num_frames", "fps", "geometry", "height",
                         "width", "out_height", "out_width", "steps", "ref_image_short_edge"}


def test_request_body_704_geometry_downscales_to_delivery(tmp_path):
    body = MiniMaxH3TurboBackend(["http://unused"], geometry="704x1280").body_for(_request(tmp_path, 1))
    assert body["geometry"] == "704x1280"
    assert (body["height"], body["width"], body["out_height"], body["out_width"]) == (704, 1280, 480, 832)


def test_unknown_geometry_rejected():
    with pytest.raises(ValueError, match="unknown H3 geometry"):
        MiniMaxH3TurboBackend(["http://unused"], geometry="720x1280")
    assert set(GEOMETRIES) == {"480x832", "544x960", "704x1280"}
    assert all(g.height % 32 == 0 and g.width % 32 == 0 for g in GEOMETRIES.values())


def test_prompt_v2_cites_pictures_only_in_subject_definitions(tmp_path):
    be = MiniMaxH3TurboBackend(["http://unused"], prompt_format=2)
    body = be.body_for(_request(tmp_path, 2))
    defs, rest = body["prompt"].split("\n\nsummary:", 1)
    assert "<Picture 1>" in defs and "<Picture 2>" in defs
    assert "<Picture" not in rest


def test_subject_phrases_are_deterministic_instruction_heads():
    from vmem_bench.testbed.backends.minimax_h3.prompt import normalize_instruction
    from vmem_bench.testbed.backends.minimax_h3.subjects import phrases

    ins = normalize_instruction("Elias closes the ship's log and returns it to the wooden box by the brass railing.")
    assert phrases(ins) == ["person", "ship's log", "wooden box", "brass railing"]
    ins = normalize_instruction("Two brass fog bells（Two brass fog bells hung under the eaves） hang there.")
    assert phrases(ins)[:2] == ["person", "brass fog bells"]


def test_subject_boxes_keep_person_closeups_but_drop_setting_boxes():
    from vmem_bench.testbed.backends.minimax_h3.subjects import select_boxes

    dets = [("person", (0, 0, 700, 440)), ("wall", (0, 0, 700, 400)), ("lamp", (10, 10, 100, 120)),
            ("lamp", (12, 12, 100, 120)), ("cup", (5, 5, 20, 20)), ("person", (0, 0, 832, 480))]
    assert select_boxes(dets, 832, 480) == [("person", (0, 0, 700, 440)), ("lamp", (10, 10, 100, 120))]


def test_reference_board_sends_one_deterministic_image(tmp_path):
    from PIL import Image

    be = MiniMaxH3TurboBackend(["http://unused"], prompt_format=2, reference_board=True)
    req = _request(tmp_path, 3)
    for k, p in enumerate(req.references):
        Image.new("RGB", (832, 480), (40 * k, 90, 200 - 40 * k)).save(p, format="PNG")
    body = be.body_for(req)
    assert len(body["references"]) == 1
    assert "a reference board of 3 stills" in body["prompt"] and "<Picture 2>" not in body["prompt"]
    first = Path(body["references"][0]).read_bytes()
    assert Path(be.body_for(req)["references"][0]).read_bytes() == first
    with pytest.raises(ValueError, match="reference_board needs prompt_format=2"):
        MiniMaxH3TurboBackend(["http://unused"], reference_board=True)


def test_request_body_zero_refs_and_limits(tmp_path):
    be = MiniMaxH3TurboBackend(["http://unused"])
    body = be.body_for(_request(tmp_path, 0))
    assert body["references"] == [] and body["prompt"] == build_h3_prompt(SEG, 0)
    with pytest.raises(ValueError):
        be.body_for(_request(tmp_path, 4))


def test_backend_satisfies_protocol():
    be = MiniMaxH3TurboBackend(["http://unused"])
    assert isinstance(be, RealizationBackend) and be.name == "minimax_h3_turbo"


# ------------------------------------------------------------------ fake server

class _Fake:
    def __init__(self, model: str = MODEL_ID, fail: bool = False, write_clip: bool = True):
        self.model, self.fail, self.write_clip = model, fail, write_clip
        self.bodies: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"ok": True, "model": fake.model, "busy": False})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                if fake.fail:
                    self._send(500, {"detail": "CUDA OOM"})
                    return
                if fake.write_clip:
                    Path(body["out"]).write_bytes(b"mp4")
                self._send(200, {"clip": body["out"], "seed": body["seed"], "cached": False, "prompt": body["prompt"]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fakes():
    made: list[_Fake] = []

    def make(**kw):
        f = _Fake(**kw)
        made.append(f)
        return f

    yield make
    for f in made:
        f.close()


@pytest.fixture(autouse=True)
def _proxy_env(monkeypatch):
    # The client must ignore proxies; a dead proxy here would break any request routed through it.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)


def test_discovery_from_endpoint_dir_skips_dead_foreign_and_broken(tmp_path, fakes):
    good, other = fakes(), fakes(model="some_other_model")
    d = tmp_path / "endpoints"
    d.mkdir()
    (d / f"{MODEL_ID}_a_1.json").write_text(json.dumps({"url": good.url + "/"}))
    (d / f"{MODEL_ID}_b_2.json").write_text(json.dumps({"url": "http://127.0.0.1:1"}))
    (d / f"{MODEL_ID}_c_3.json").write_text(json.dumps({"url": other.url}))
    (d / f"{MODEL_ID}_d_4.json").write_text("{not json")
    (d / "unrelated_e_5.json").write_text(json.dumps({"url": good.url}))
    assert candidate_urls(d) == sorted([good.url, "http://127.0.0.1:1", other.url])
    assert discover_endpoints(d, timeout=2) == [good.url]


def test_discovery_from_url_file_and_list(tmp_path, fakes):
    a, b = fakes(), fakes()
    f = tmp_path / "urls.txt"
    f.write_text(f"# pool\n{a.url}\n\n{b.url}  # second\nhttp://127.0.0.1:1\n")
    assert discover_endpoints(f, timeout=2) == sorted([a.url, b.url])
    assert discover_endpoints(f"{a.url}, http://127.0.0.1:1", timeout=2) == [a.url]
    assert discover_endpoints([b.url], timeout=2) == [b.url]


def test_realize_posts_body_and_returns_clip(tmp_path, fakes):
    srv = fakes()
    be = MiniMaxH3TurboBackend(srv.url)
    be.start()
    req = _request(tmp_path, 1, seed=2026)
    res = be.realize(req)
    assert res.clip == (tmp_path / "out" / "seg_012.mp4").resolve() and res.keyframe is None
    assert srv.bodies == [be.body_for(req)]
    assert res.meta["endpoint"] == srv.url and res.meta["n_references"] == 1
    assert res.meta["backend"] == "minimax_h3_turbo"
    assert res.meta["geometry"] == {"name": "480x832", "render": [480, 832], "delivered": [480, 832]}
    be.stop()


def test_realize_raises_on_server_error_and_missing_clip(tmp_path, fakes):
    with pytest.raises(RuntimeError, match="HTTP 500"):
        MiniMaxH3TurboBackend(fakes(fail=True).url).realize(_request(tmp_path, 0))
    with pytest.raises(RuntimeError, match="missing or empty"):
        MiniMaxH3TurboBackend(fakes(write_clip=False).url).realize(_request(tmp_path, 0))


def test_start_raises_without_healthy_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("VMEM_H3_ENDPOINTS", raising=False)
    with pytest.raises(RuntimeError, match="no MiniMax-H3 endpoints"):
        MiniMaxH3TurboBackend().start()
    with pytest.raises(RuntimeError, match="no healthy"):
        MiniMaxH3TurboBackend("http://127.0.0.1:1").start()


# ------------------------------------------------------------------ VRAM placement (no GPU needed)

def test_vram_placement_modes():
    assert OFFLOAD_MODES == ("none", "text_encoder", "all")
    assert set(vram_placement("none").values()) == {"cuda"}
    te = vram_placement("text_encoder")
    assert te["text_encoder"] == "cpu" and te["transformer"] == te["video_vae"] == te["audio_vae"] == "cuda"
    assert set(vram_placement("all").values()) == {"cpu"}
    with pytest.raises(ValueError):
        vram_placement("disk")


def test_generate_clears_cached_rope_deltas_before_each_request(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from PIL import Image

    from vmem_bench.testbed.backends.minimax_h3.pipeline import generate

    seen = []

    class FakePipe:
        text_encoder = SimpleNamespace(model=SimpleNamespace(rope_deltas="stale-from-last-image-prompt"))

        def __call__(self, **kw):
            seen.append(self.text_encoder.model.rope_deltas)
            return [Image.new("RGB", (8, 8))] * kw["num_frames"], None

    frames = generate(FakePipe(), prompt="p", references=[], seed=1, num_frames=5, height=480, width=832,
                      steps=4, ref_image_short_edge=768)
    assert len(frames) == 5 and seen == [None]
