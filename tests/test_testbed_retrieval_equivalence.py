"""The testbed's retrieval/store/control/prompt code selects exactly what the frozen MemStrata code did.

The Track-B paper conditions ran on MemStrata's ``memory_retrieval`` skill; the testbed now owns the one
implementation. These tests load the MemStrata source *by path, in the test only* (``src/vmem_bench``
never imports ``memstrata``) and replay a synthetic causal rollout through both, comparing every
selection field by field.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEMSTRATA_SRC = Path(os.environ.get("MEMSTRATA_SRC") or ROOT.parents[1] / "methods" / "MemStrata" / "src")
if not (MEMSTRATA_SRC / "memstrata" / "skills" / "memory_retrieval" / "retrievers.py").is_file():
    pytest.skip(f"no MemStrata retrieval source at {MEMSTRATA_SRC}; MemStrata main dropped it, so set "
                "MEMSTRATA_SRC to the src/ of a paper/iclr27-trackb-* tag checkout", allow_module_level=True)
sys.path.insert(0, str(MEMSTRATA_SRC))

from memstrata.lib.prompt_standardizer import standardize_prompt  # noqa: E402
from memstrata.skills.memory_retrieval import last_seen_ref as ms_last_seen  # noqa: E402
from memstrata.skills.memory_retrieval import retrievers as ms_retrievers  # noqa: E402
from memstrata.skills.memory_retrieval import trackb_store as ms_store  # noqa: E402

from vmem_bench.testbed.backends.prompt_format import flux_prompt, wan_prompt  # noqa: E402
from vmem_bench.testbed.controls import LastSeenReferenceSelector  # noqa: E402
from vmem_bench.testbed.retrieval import encoders as tb_encoders  # noqa: E402
from vmem_bench.testbed.retrieval import retrievers as tb_retrievers  # noqa: E402
from vmem_bench.testbed.runner import load_prompt_stream, story_prompt_path, usable_references  # noqa: E402
from vmem_bench.testbed.store import TrackBGeneratedStore, resolve_ffmpeg  # noqa: E402

VARIANTS = ["frame_text", "seg_uniform", "seg_dinokey", "seg_framererank", "recency", "bm25_desc", "random"]
DURATIONS = [3.0, 2.5, 4.0, 1.5, 3.5]


def _clips(tmp: Path) -> list[Path]:
    ffmpeg = resolve_ffmpeg()
    out = []
    for i, dur in enumerate(DURATIONS):
        clip = tmp / "clips" / f"clip_{i}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc2=size=160x96:rate=8:duration={dur}", "-vf", f"hue=h={i * 67}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bitexact", str(clip)],
                       check=True, capture_output=True, timeout=60)
        out.append(clip)
    return out


def _prompts(n: int) -> list[str]:
    stream = load_prompt_stream(story_prompt_path("0001_lighthouse_keeper"))
    return [s.prompt for s in stream.segments[:n]]


def _view(refs) -> list[tuple]:
    return [(r.representation_id, round(float(r.score), 12), r.arm, r.source_seconds) for r in refs]


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> list[Path]:
    return _clips(tmp_path_factory.mktemp("equiv"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_trackb_selection_matches_memstrata(variant: str, clips: list[Path], tmp_path: Path) -> None:
    prompts = _prompts(len(clips) + 1)
    ms = ms_store.TrackBGeneratedStore(ms_retrievers.RetrieverConfig())
    ms.work_dir = tmp_path / "ms"
    tb = TrackBGeneratedStore(tb_retrievers.RetrieverConfig(), work_dir=tmp_path / "tb")
    ms_r = ms_retrievers.build_retriever(variant, store=ms)
    tb_r = tb_retrievers.build_retriever(variant, store=tb)
    compared = 0
    for k, prompt in enumerate(prompts):
        if k:
            a = ms_r.retrieve(prompt, as_of_seconds=ms.clock, budget=3)
            b = tb_r.retrieve(prompt, as_of_seconds=tb.clock, budget=3)
            assert _view(b) == _view(a), f"{variant} segment {k}"
            assert [Path(r.image_ref).read_bytes() for r in b] == [Path(r.image_ref).read_bytes() for r in a]
            assert len(usable_references(b)) == len(b) > 0
            compared += 1
        if k < len(clips):
            ms.observe_generated_segment(segment_id=k, prompt_text=prompt, segment_video=str(clips[k]),
                                         description=prompt)
            tb.observe_generated_segment(segment_id=k, prompt_text=prompt, segment_video=str(clips[k]),
                                         description=prompt)
            assert tb.clock == ms.clock
    assert compared == len(clips)


@pytest.mark.parametrize("variant", VARIANTS)
def test_tracka_store_matches_memstrata_without_materialization(variant: str, tmp_path: Path) -> None:
    """Track-A store (source-video seconds): same ranking when no frames are materialized."""
    prompts = _prompts(6)
    ms = ms_retrievers.MemoryRetrievalStore(ms_retrievers.RetrieverConfig(extract_frames=False))
    tb = tb_retrievers.MemoryRetrievalStore(tb_retrievers.RetrieverConfig(extract_frames=False))
    ms.reset(movie_id="m", source_video="", work_dir=tmp_path / "ms")
    tb.reset(movie_id="m", source_video="", work_dir=tmp_path / "tb")
    ms_r, tb_r = ms_retrievers.build_retriever(variant, store=ms), tb_retrievers.build_retriever(variant, store=tb)
    for k, prompt in enumerate(prompts):
        span = (5.0 * k, 5.0 * k + 5.0)
        if k:
            assert _view(tb_r.retrieve(prompt, as_of_seconds=span[0], budget=4)) == \
                _view(ms_r.retrieve(prompt, as_of_seconds=span[0], budget=4))
        ms.observe_segment(segment_id=k, prompt_text=prompt, seconds_span=span)
        tb.observe_segment(segment_id=k, prompt_text=prompt, seconds_span=span)


def test_last_seen_selection_matches_memstrata(clips: list[Path], tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    own = []
    for i, clip in enumerate(clips):
        dst = run_dir / "review" / "segments" / f"seg_{i:03d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(clip.read_bytes())
        own.append(dst)
    plan = {1: [0], 2: [0, 0, 1], 3: [2, 0, 1, 1], 4: [3, 3, 0, 2], 5: [9]}
    ms = ms_store.TrackBGeneratedStore(ms_retrievers.RetrieverConfig())
    ms.work_dir = run_dir / "ms_read"
    tb = TrackBGeneratedStore(tb_retrievers.RetrieverConfig(), work_dir=run_dir / "tb_read")
    ms_sel = ms_last_seen.LastSeenReferenceSelector(plan, store=ms, run_dir=run_dir, budget=3)
    tb_sel = LastSeenReferenceSelector(plan, store=tb, run_dir=run_dir, budget=3)
    for k in range(len(own) + 1):
        assert tb_sel.sources(segment_id=k) == ms_sel.sources(segment_id=k)
        a, b = ms_sel.select(segment_id=k), tb_sel.select(segment_id=k)
        assert [r.representation_id for r in b] == [r.representation_id for r in a]
        assert [Path(r.image_ref).read_bytes() for r in b] == [Path(r.image_ref).read_bytes() for r in a]
        if k < len(own):
            for store in (ms, tb):
                store.observe_generated_segment(segment_id=k, prompt_text="p", segment_video=str(own[k]))


def test_hash_encoder_matches_memstrata(tmp_path: Path) -> None:
    from memstrata.encoders import HashEmbedding

    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG not really")
    long_text = "lighthouse " * 60  # a path component longer than NAME_MAX
    for probe in (str(img), "gen:c0:1.500", ""):
        assert tb_encoders.HashEmbedding().embed_image(probe) == HashEmbedding().embed_image(probe)
    assert tb_encoders.HashEmbedding().embed_text(long_text) == HashEmbedding().embed_text(long_text)


def test_prompt_formatting_matches_memstrata() -> None:
    prompts = []
    for path in sorted((ROOT / "assets/trackB/en/sut_prompts").glob("*_name_anchored.json")):
        prompts += [s["prompt"] for s in json.loads(path.read_text(encoding="utf-8"))["segments"]]
    prompts += [
        "A photorealistic, highly detailed 8k keeper in studio lighting",
        '{"scene": "a hyperrealistic lens", "style": "8k"}',
        "Narrative goal: A video of a keeper lighting a lamp.\nVisible actions: he climbs.\n"
        "Active characters or subjects: Elias\nScene state: location=lantern room; lighting=dusk; style=noir",
        "Slow motion of waves",
    ]
    assert len(prompts) > 1000
    for p in prompts:
        assert flux_prompt(p) == standardize_prompt(p, "flux_klein", None)
        assert flux_prompt(p, "raw_film") == standardize_prompt(p, "flux_klein", "raw_film")
        assert wan_prompt(p) == standardize_prompt(p, "wan")
