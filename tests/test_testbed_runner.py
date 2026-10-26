"""Testbed runner: causal closed loop, artifact layout, seed pass-through, backend loading."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from vmem_bench.testbed import runner as tb_runner
from vmem_bench.testbed.backends import BACKENDS, RealizationBackend, RealizationResult, load_backend
from vmem_bench.testbed.backends import flux2_wan22
from vmem_bench.testbed.retrieval.retrievers import RetrievedRef
from vmem_bench.testbed.runner import CausalityError, RunConfig, run
from vmem_bench.testbed.store import resolve_ffmpeg

SRC = Path(__file__).resolve().parents[1] / "src" / "vmem_bench"
STORY = "0001_lighthouse_keeper"


class RecordingBackend:
    """Fake backend: logs each request and the store state visible at that moment."""

    name = "recording"

    def __init__(self) -> None:
        self.requests = []
        self.started = self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def realize(self, request):
        self.requests.append(request)
        clip = Path(request.out_dir) / "clip.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([resolve_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc2=size=64x48:rate=8:duration={2 + request.segment_id}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True, capture_output=True)
        kf = Path(request.out_dir) / "keyframe.png"
        kf.write_bytes(b"png")
        return RealizationResult(clip=clip, keyframe=kf, meta={"keyframe_mode": "fake", "seed": request.seed})


def _cfg(tmp_path: Path, variant: str, **kw) -> RunConfig:
    return RunConfig(variant=variant, out_root=tmp_path / "out", run_tag="t", segments=3, text_provider="hash",
                     frame_provider="hash", keyframe_provider="hash", **kw)


@pytest.mark.parametrize("variant", ["frame_text", "seg_uniform", "seg_dinokey", "seg_framererank", "last_seen"])
def test_three_segment_loop_is_causal_and_scorer_compatible(variant: str, tmp_path: Path) -> None:
    backend = RecordingBackend()
    (manifest,) = run(_cfg(tmp_path, variant, seed=7), [STORY], backend)
    run_dir = Path(manifest["run_dir"])
    assert run_dir == tmp_path / "out" / f"{variant}_b3" / STORY / "name_anchored" / "t" / "seed_7"
    assert backend.started == backend.stopped == 1
    assert [r.segment_id for r in backend.requests] == [0, 1, 2]
    assert [r.seed for r in backend.requests] == [7, 7, 7]
    assert backend.requests[0].references == ()
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["done"] == progress["total"] == 3
    for i, (req, rec) in enumerate(zip(backend.requests, progress["segments"])):
        assert rec["segment_id"] == i and rec["gt_segment_id"] == f"seg_{i + 1:03d}" and rec["status"] == "ok"
        assert rec["seed"] == 7 and rec["backend_meta"]["seed"] == 7
        assert set(rec["timing_s"]) == {"select", "realize", "observe"}
        assert [Path(r["image"]) for r in rec["references"]] == list(req.references)
        for ref in rec["references"]:
            assert 0 <= ref["source_segment"] < i
            assert run_dir in Path(ref["image"]).parents
        assert (run_dir / "review" / "segments" / f"seg_{i:03d}.mp4").is_file()
        assert rec["video"] == str(run_dir / "review" / "segments" / f"seg_{i:03d}.mp4")
        assert (run_dir / "review" / "keyframes" / f"seg_{i:03d}.png").is_file()
    assert progress["segments"][2]["references"], "segment 2 should read the history"
    assert manifest["gold_read"] is (variant == "last_seen")
    if variant == "last_seen":
        assert (run_dir / "input" / "last_seen_plan.json").is_file()


def test_none_passes_no_references(tmp_path: Path) -> None:
    backend = RecordingBackend()
    (manifest,) = run(_cfg(tmp_path, "none"), [STORY], backend)
    assert manifest["budget"] == 0 and not manifest["gold_read"]
    assert all(r.references == () for r in backend.requests)


def test_policy_sees_only_earlier_segments(tmp_path: Path, monkeypatch) -> None:
    seen = []
    orig = tb_runner.ReferencePolicy.select

    def spy(self, segment):
        seen.append((segment.index, [s.segment_id for s in self.store.segments]))
        return orig(self, segment)

    monkeypatch.setattr(tb_runner.ReferencePolicy, "select", spy)
    run(_cfg(tmp_path, "frame_text"), [STORY], RecordingBackend())
    assert seen == [(0, []), (1, [0]), (2, [0, 1])]


def test_guard_rejects_non_causal_references(tmp_path: Path, monkeypatch) -> None:
    def leak(self, segment):
        img = tmp_path / "out" / "leak.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
        return [RetrievedRef(score=1.0, arm="leak", representation_id=f"c{segment.index:05d}@0.500",
                             image_ref=str(img))]

    monkeypatch.setattr(tb_runner.ReferencePolicy, "select", leak)
    with pytest.raises(CausalityError):
        run(_cfg(tmp_path, "frame_text"), [STORY], RecordingBackend())


@pytest.mark.parametrize("variant", ["frame_text", "seg_uniform", "seg_dinokey", "seg_framererank"])
def test_resume_rebuilds_the_store_and_selects_as_an_uninterrupted_run(variant: str, tmp_path: Path,
                                                                        monkeypatch) -> None:
    # The frozen paper runs reopened a resumed story with an empty store; a resume must see the whole history.
    seen = []
    orig = tb_runner.ReferencePolicy.select

    def spy(self, segment):
        seen.append((segment.index, [s.segment_id for s in self.store.segments]))
        return orig(self, segment)

    monkeypatch.setattr(tb_runner.ReferencePolicy, "select", spy)
    (full,) = run(_cfg(tmp_path / "full", variant), [STORY], RecordingBackend())

    cfg = _cfg(tmp_path / "resumed", variant)
    cfg.segments = 2
    run(cfg, [STORY], RecordingBackend())
    seen.clear()
    cfg.segments, cfg.resume = 3, True
    backend = RecordingBackend()
    (resumed,) = run(cfg, [STORY], backend)

    assert [r.segment_id for r in backend.requests] == [2]
    assert seen == [(2, [0, 1])], "the reopened store must hold segments 0 and 1"
    assert resumed["segments_ok"] == 3

    def refs(manifest):
        rec = json.loads((Path(manifest["run_dir"]) / "progress.json").read_text())["segments"][2]
        return [(r["id"], r["source_segment"]) for r in rec["references"]]

    assert refs(resumed) and refs(resumed) == refs(full)


def test_flux2_wan22_seed_derivation(tmp_path: Path, monkeypatch) -> None:
    jobs = []

    def fake_submit(server_dir, request, *, timeout, poll=1.0):
        jobs.append((Path(server_dir).name, request))
        if "steps" in request:
            Path(request["save_file"]).write_bytes(b"kf")
            return {"status": "ok", "out_image": request["save_file"],
                    "mode": "flux_multi_i2i" if request["references"] else "flux_t2i"}
        return {"status": "ok", "out_video": request["save_file"], "mode": "i2v"}

    monkeypatch.setattr(flux2_wan22, "submit_job", fake_submit)
    backend = load_backend("flux2_wan22", defaults={"service_dir": tmp_path / "svc"})
    ref = tmp_path / "r.png"
    ref.write_bytes(b"r")
    for seg, refs in ((0, ()), (4, (ref,))):
        res = backend.realize(flux2_wan22.RealizationRequest(segment_id=seg, prompt="Elias climbs.",
                                                             references=refs, seed=2026, out_dir=tmp_path / str(seg)))
        assert res.meta["flux_seed"] == 2026 + seg and res.meta["wan_seed"] == 2026
    (f0, flux0), (w0, wan0), (f4, flux4), (_, wan4) = jobs
    assert (f0, w0) == ("flux", "wan22_turbo")
    assert flux0["seed"] == 2026 and flux4["seed"] == 2030 and wan0["seed"] == wan4["seed"] == 2026
    assert flux0["references"] == [] and flux4["references"] == [str(ref.resolve())]
    assert (flux0["width"], flux0["height"], flux0["steps"]) == (832, 480, 4)
    assert json.loads(flux0["prompt"]) == {"scene": "Elias climbs.", "subjects": [], "style": "cinematic film still"}
    assert wan0["prompt"] == "A beautiful cinematic video of Elias climbs."
    assert wan0["first_frame_path"] == flux0["save_file"]
    assert (wan0["height"], wan0["width"], wan0["out_height"], wan0["out_width"], wan0["num_frames"]) == \
        (704, 1280, 480, 832, 121)


@pytest.mark.parametrize("gpu", [0, "0"])
def test_flux2_wan22_gpu_index_zero_is_honoured(gpu, tmp_path: Path, monkeypatch) -> None:
    # --backend-opt flux_gpu=0 parses to the int 0; it must pin GPU 0, not fall back to the
    # runner's CUDA_VISIBLE_DEVICES (the encoder card).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(flux2_wan22, "gpu_free_mib", lambda index: None)
    backend = load_backend("flux2_wan22", {"flux_gpu": gpu}, defaults={"service_dir": tmp_path / "svc"})
    assert backend._gpu_env(backend.flux_gpu, 0, "flux")["CUDA_VISIBLE_DEVICES"] == "0"


def test_flux2_wan22_failed_start_stops_launched_servers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_MODELS_ROOT", str(tmp_path))
    monkeypatch.delenv("VMEM_WAN22_TURBO_REPO", raising=False)
    monkeypatch.setattr(flux2_wan22, "gpu_free_mib", lambda index: None)
    backend = load_backend("flux2_wan22", {"flux_gpu": "0", "wan_gpu": "0"},
                           defaults={"service_dir": tmp_path / "svc"})
    launched = []
    monkeypatch.setattr(backend, "_launch", lambda server_dir, command, env: launched.append(server_dir))
    stopped = []
    monkeypatch.setattr(backend, "stop", lambda: stopped.append(list(launched)))
    with pytest.raises(RuntimeError, match="wan_repo"):
        backend.start()
    assert launched == [backend.flux_dir] and stopped == [[backend.flux_dir]]


def test_backend_registry() -> None:
    assert {"flux2_wan22", "minimax_h3_turbo", "synthetic"} <= set(BACKENDS)
    assert isinstance(load_backend("synthetic"), RealizationBackend)
    with pytest.raises(TypeError):
        load_backend("synthetic", {"not_an_option": 1})
    with pytest.raises(ValueError):
        load_backend("nope")


def test_vmem_bench_never_imports_memstrata() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n == "memstrata" or n.startswith("memstrata.") for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == []
