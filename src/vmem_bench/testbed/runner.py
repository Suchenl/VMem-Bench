"""Causal closed-loop runner of the Track-B shared-generator testbed.

For each story the runner walks the SUT prompt stream segment by segment:

1. **select** references with the condition under test — a similarity-retrieval memory over this
   run's own generated history (``frame_text``/``seg_uniform``/``seg_dinokey``/``seg_framererank``, plus
   the ``recency``/``bm25_desc``/``random`` diagnostics), no memory (``none``), or the gold-selected
   ``last_seen`` control;
2. **realize** the segment with the shared backend (it only renders: prompt + references + seed);
3. **observe** the new clip into the generated-history store, which is the only memory later segments
   can read.

Causal guard: every reference handed to the backend must be a still materialized under this run's
directory from a segment strictly before the current one; anything else raises ``CausalityError``.

Layout (``<out_root>/<variant>_b<budget>/<story>/<register>/<run_tag>/seed_<seed>/``) is what the Track-B
scorers read directly: ``review/segments/seg_###.mp4`` (0-indexed production shots; score through the
+1 view / ``file_index_offset: "auto"``), ``progress.json`` (per-segment records), plus
``review/keyframes/``, ``input/`` (prompt stream, last-seen plan), ``retrieval_read/`` (stills) and
``run_manifest.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vmem_bench.testbed.backends import RealizationBackend, RealizationRequest
from vmem_bench.testbed.controls import (
    DEPRECATED_ALIASES,
    LAST_SEEN_VARIANT,
    LastSeenReferenceSelector,
    build_last_seen_plan,
    gold_path,
    load_plan,
)
from vmem_bench.testbed.retrieval.retrievers import RetrievedRef, RetrieverConfig, build_retriever
from vmem_bench.testbed.store import TrackBGeneratedStore, resolve_ffmpeg

BENCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = BENCH_ROOT / "assets" / "trackB" / "en" / "sut_prompts"
RETRIEVAL_VARIANTS = ("frame_text", "seg_uniform", "seg_dinokey", "seg_framererank", "recency", "bm25_desc", "random")
NO_MEMORY_VARIANT = "none"
VARIANTS = (*RETRIEVAL_VARIANTS, NO_MEMORY_VARIANT, LAST_SEEN_VARIANT)
PAPER_SEED = 2026
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_REP_RE = re.compile(r"^c(\d{5})@")
MAX_CONSECUTIVE_FAILURES = 3


class CausalityError(RuntimeError):
    """A reference did not come from this run's own strictly earlier segments."""


@dataclass(frozen=True)
class Segment:
    index: int
    segment_id: str
    prompt: str
    transition: str
    duration_sec: float


@dataclass(frozen=True)
class PromptStream:
    path: Path
    story_id: str
    register: str
    segments: tuple[Segment, ...]


@dataclass
class RunConfig:
    variant: str
    budget: int = 3
    backend: str = "flux2_wan22"
    seed: int = PAPER_SEED
    run_tag: str = "testbed"
    out_root: Path = BENCH_ROOT / "outputs" / "evaluation" / "trackB" / "testbed"
    segments: int = 0
    resume: bool = False
    overwrite: bool = False
    text_provider: str = "qwen3_embedding"
    frame_provider: str = "siglip2"
    keyframe_provider: str = "dinov3"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.variant = DEPRECATED_ALIASES.get(self.variant, self.variant)
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; choices: {VARIANTS} (+ deprecated 'oracle')")
        self.budget = 0 if self.variant == NO_MEMORY_VARIANT else max(1, int(self.budget))
        self.out_root = Path(self.out_root)


def load_prompt_stream(path: str | Path) -> PromptStream:
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    segs = []
    for idx, row in enumerate(raw.get("segments") or []):
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{p}: empty prompt at segment {idx}")
        segs.append(Segment(index=idx, segment_id=str(row.get("segment_id") or f"seg_{idx + 1:03d}"),
                            prompt=prompt, transition=str(row.get("transition") or "cut").lower(),
                            duration_sec=float(row.get("duration_sec", 5.0))))
    story = str(raw.get("story_id") or p.stem.replace("_name_anchored", ""))
    return PromptStream(path=p, story_id=story, register=str(raw.get("register") or "name_anchored"),
                        segments=tuple(segs))


def story_prompt_path(story: str, prompts_dir: Path = PROMPTS_DIR) -> Path:
    path = Path(prompts_dir) / f"{story}_name_anchored.json"
    if not path.is_file():
        raise FileNotFoundError(f"no SUT prompt stream for story {story!r}: {path}")
    return path


def all_stories(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(p.name[: -len("_name_anchored.json")] for p in Path(prompts_dir).glob("*_name_anchored.json"))


def run_dir_for(cfg: RunConfig, stream: PromptStream) -> Path:
    return (cfg.out_root / f"{cfg.variant}_b{cfg.budget}" / stream.story_id / stream.register / cfg.run_tag
            / f"seed_{cfg.seed}")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _git_head(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": head or None, "src_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "src_dirty": None}


class ReferencePolicy:
    """Chooses the references of one segment from the run's own generated history."""

    def __init__(self, cfg: RunConfig, store: TrackBGeneratedStore, *, plan: dict[int, list[int]] | None,
                 run_dir: Path) -> None:
        self.variant, self.budget, self.store = cfg.variant, cfg.budget, store
        self.retriever = None
        self.selector = None
        if cfg.variant == LAST_SEEN_VARIANT:
            self.selector = LastSeenReferenceSelector(plan or {}, store=store, run_dir=run_dir, budget=cfg.budget)
        elif cfg.variant in RETRIEVAL_VARIANTS:
            self.retriever = build_retriever(cfg.variant, store=store)

    def select(self, segment: Segment) -> list[RetrievedRef]:
        if self.selector is not None:
            return self.selector.select(segment_id=segment.index)
        if self.retriever is None or not self.store.segments:
            return []
        return self.retriever.retrieve(segment.prompt, as_of_seconds=self.store.clock, budget=self.budget)


def usable_references(refs: list[RetrievedRef]) -> list[RetrievedRef]:
    """Existing image stills, first occurrence of each path, in rank order."""
    seen: set[str] = set()
    out: list[RetrievedRef] = []
    for ref in refs:
        image = ref.image_ref
        if not image or image in seen:
            continue
        path = Path(str(image))
        if not (path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES):
            continue
        seen.add(image)
        out.append(ref)
    return out


def check_causal(refs: list[RetrievedRef], *, current: int, run_dir: Path) -> list[int]:
    root = run_dir.resolve()
    sources = []
    for ref in refs:
        m = _REP_RE.match(str(ref.representation_id or ""))
        if m is None:
            raise CausalityError(f"segment {current}: reference without a source segment id: {ref!r}")
        source = int(m.group(1))
        if not 0 <= source < current:
            raise CausalityError(f"segment {current}: reference from segment {source} is not strictly earlier")
        if root not in Path(str(ref.image_ref)).resolve().parents:
            raise CausalityError(f"segment {current}: reference {ref.image_ref} is outside this run ({root})")
        sources.append(source)
    return sources


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _stitch(review: Path) -> Path | None:
    segs = sorted((review / "segments").glob("seg_*.mp4"))
    if not segs:
        return None
    listing = review / "_concat.txt"
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    out = review / "long_video.mp4"
    proc = subprocess.run([resolve_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                           "-an", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if proc.returncode == 0 and out.is_file() else None


def _closed_prefix(run_dir: Path) -> list[dict]:
    progress = run_dir / "progress.json"
    if not progress.is_file():
        return []
    records = json.loads(progress.read_text(encoding="utf-8")).get("segments") or []
    closed = []
    for i, rec in enumerate(records):
        if rec.get("segment_id") != i or rec.get("status") != "ok" or not Path(str(rec.get("video"))).is_file():
            break
        closed.append(rec)
    return closed


def run_story(cfg: RunConfig, stream: PromptStream, backend: RealizationBackend, *, log=print) -> dict[str, Any]:
    """Run one story end to end; returns the run manifest."""
    run_dir = run_dir_for(cfg, stream)
    if run_dir.exists() and cfg.overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not cfg.resume:
        raise FileExistsError(f"run dir exists; use --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "input" / "prompts.json", json.loads(stream.path.read_text(encoding="utf-8")))

    segments = stream.segments[: cfg.segments] if cfg.segments else stream.segments
    plan = None
    gold_read = False
    if cfg.variant == LAST_SEEN_VARIANT:
        gold = json.loads(gold_path(BENCH_ROOT, stream.story_id).read_text(encoding="utf-8"))
        plan_doc, audit = build_last_seen_plan(gold, [s.segment_id for s in stream.segments])
        plan_path = write_json(run_dir / "input" / "last_seen_plan.json", plan_doc)
        write_json(run_dir / "input" / "last_seen_audit.json", audit)
        plan = load_plan(plan_path)
        gold_read = True

    store = TrackBGeneratedStore(
        RetrieverConfig(text_provider=cfg.text_provider, frame_provider=cfg.frame_provider,
                        keyframe_provider=cfg.keyframe_provider),
        work_dir=run_dir / "retrieval_read",
    )
    records: list[dict] = _closed_prefix(run_dir) if cfg.resume else []
    for rec in records:
        store.observe_generated_segment(segment_id=rec["segment_id"], prompt_text=rec["prompt"],
                                        segment_video=rec["video"], description=rec["prompt"])
    if records:
        log(f"[testbed] {stream.story_id}: resume after {len(records)} closed segments")
    policy = ReferencePolicy(cfg, store, plan=plan, run_dir=run_dir)
    review = run_dir / "review"
    failures = consecutive = 0
    t_run = time.perf_counter()

    def dump_progress() -> None:
        write_json(run_dir / "progress.json", {
            "story_id": stream.story_id, "variant": cfg.variant, "budget": cfg.budget, "backend": backend.name,
            "seed": cfg.seed, "run_tag": cfg.run_tag, "run_dir": str(run_dir),
            "done": sum(r.get("status") == "ok" for r in records), "total": len(segments), "segments": records,
        })

    for seg in segments[len(records):]:
        t0 = time.perf_counter()
        refs = usable_references(policy.select(seg))
        sources = check_causal(refs, current=seg.index, run_dir=run_dir)
        t1 = time.perf_counter()
        ref_rows = [{"id": r.representation_id, "source_segment": s, "image": str(Path(r.image_ref).resolve()),
                     "score": float(r.score), "arm": r.arm} for r, s in zip(refs, sources)]
        log(f"[testbed] {stream.story_id} segment {seg.index}: refs {[r['id'] for r in ref_rows]}")
        record: dict[str, Any] = {
            "segment_id": seg.index, "gt_segment_id": seg.segment_id, "prompt": seg.prompt,
            "transition": seg.transition, "references": ref_rows, "seed": cfg.seed, "backend": backend.name,
        }
        request = RealizationRequest(segment_id=seg.index, prompt=seg.prompt,
                                     references=tuple(Path(r["image"]) for r in ref_rows), seed=cfg.seed,
                                     out_dir=run_dir / "segments" / f"seg_{seg.index:03d}")
        try:
            result = backend.realize(request)
        except Exception as exc:  # noqa: BLE001 - an isolated generator failure skips the segment
            failures += 1
            consecutive += 1
            record.update(status="failed", error=repr(exc),
                          timing_s={"select": round(t1 - t0, 3), "realize": round(time.perf_counter() - t1, 3)})
            records.append(record)
            dump_progress()
            log(f"[testbed] {stream.story_id} segment {seg.index}: FAILED {exc!r}")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{consecutive} consecutive segments failed (last: {exc!r})") from exc
            continue
        consecutive = 0
        t2 = time.perf_counter()
        video = review / "segments" / f"seg_{seg.index:03d}.mp4"
        _link_or_copy(Path(result.clip), video)
        keyframe = None
        if result.keyframe is not None and Path(result.keyframe).is_file():
            keyframe = review / "keyframes" / f"seg_{seg.index:03d}.png"
            _link_or_copy(Path(result.keyframe), keyframe)
        store.observe_generated_segment(segment_id=seg.index, prompt_text=seg.prompt, segment_video=str(video),
                                        description=seg.prompt)
        t3 = time.perf_counter()
        record.update(status="ok", video=str(video), keyframe=str(keyframe) if keyframe else None,
                      keyframe_mode=result.meta.get("keyframe_mode"), backend_meta=dict(result.meta),
                      timing_s={"select": round(t1 - t0, 3), "realize": round(t2 - t1, 3),
                                "observe": round(t3 - t2, 3)})
        records.append(record)
        dump_progress()

    dump_progress()
    long_video = _stitch(review)
    manifest = {
        "schema": "vmem_bench.testbed.run/1",
        "story_id": stream.story_id, "register": stream.register, "variant": cfg.variant, "budget": cfg.budget,
        "backend": backend.name, "seed": cfg.seed, "run_tag": cfg.run_tag, "run_dir": str(run_dir),
        "prompts": str(stream.path), "segments_requested": len(segments),
        "segments_ok": sum(r.get("status") == "ok" for r in records), "segments_failed": failures,
        "gold_read": gold_read,
        "gold_use": ("last_seen control: gold selects which own earlier shots supply stills (id-only plan)"
                     if gold_read else "none"),
        "retrieval": {"text_provider": cfg.text_provider, "frame_provider": cfg.frame_provider,
                      "keyframe_provider": cfg.keyframe_provider} if cfg.variant in RETRIEVAL_VARIANTS else None,
        "long_video": str(long_video) if long_video else None,
        "wall_s": round(time.perf_counter() - t_run, 3), "host": socket.gethostname(),
        "vmem_bench": _git_head(BENCH_ROOT), "config": {k: (str(v) if isinstance(v, Path) else v)
                                                       for k, v in asdict(cfg).items()},
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def run(cfg: RunConfig, stories: list[str], backend: RealizationBackend, *, prompts_dir: Path = PROMPTS_DIR,
        keep_services: bool = False, log=print) -> list[dict[str, Any]]:
    backend.start()
    try:
        return [run_story(cfg, load_prompt_stream(story_prompt_path(s, prompts_dir)), backend, log=log)
                for s in stories]
    finally:
        if not keep_services:
            backend.stop()
