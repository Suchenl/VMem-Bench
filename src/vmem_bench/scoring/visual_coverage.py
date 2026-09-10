"""Visual-coverage scoring (v3 default; legacy v2.2 available).

WHY THIS EXISTS
---------------
The old scorer (``metrics.py`` / ``visual.py``) scored a system by intersecting
ID-sets between the system's selection and the gold annotation. That is (a) opaque,
(b) unfair to systems that do not emit IDs, and (c) not what a *visual* memory
benchmark should measure. This module replaces it with a purely visual, VLM-judged
protocol that needs only text ground-truth (roster + per-segment present set) — NO
gold crops and NO state annotation.

WHAT IT MEASURES (Track A: the system emits a *context* = reference images)
----------------------------------------------------------------------------
The v3 default judges each unlabeled reference independently against the frozen
gold-present roster. It sends neither target video nor segment prompt, returns a
strict multi-label ``entity_ids`` list, and derives ``missing`` from the union.
This isolates reference-selection coverage and removes cross-reference order
effects. Legacy v2.2 remains selectable and sends batched refs plus target video.

From that we compute, per segment, four transparent metrics in [0,1]:

  precision  = (# reference images matching >=1 gold-present entity) / (# refs)
               -> penalises over-retrieval / hallucinated references.
  recall     = (|continuity| - |missing ∩ continuity|) / |continuity|   [HEADLINE]
               -> memory recall: only over CONTINUITY entities (seen before, must be
                  recalled from memory). First-appearance entities are excluded — a
                  memory system cannot recall something never seen.
  recall_all = (|present| - |missing|) / |present|                       [diagnostic]
               -> coverage over all present entities incl. first appearances.
  f1         = harmonic mean(precision, recall[continuity])              [HEADLINE]
  redundancy is null in v3 because one multi-label ref has no unique per-entity
  partition. Legacy v2.2 reports TWO complementary variants (both PER-ENTITY;
  complementary
  views at different angle/appearance are NEVER penalised, only near-identical ones):
    redundancy_vlm = (# on-screen refs that are per-entity near-duplicates)
                     / (# on-screen refs)      -- VLM counts distinct views per group.
    redundancy_sim = pair-weighted mean of the per-entity DINOv3 cosine self-similarity
                     (Σ cos(i,j)/#pairs, threshold-free). 1.0 => identical crops,
                     lower => more diverse views. Deterministic, no threshold.
  selection_efficiency = useful_refs / (# reference images)   [renamed from `efficiency`]
               where useful_ref = on-screen AND not a per-entity near-duplicate
               (uses the VLM count, the only variant giving an integer redundant count).
               -> "how much of the context you spent was relevant and non-wasteful";
               NOTE: this is a SELECTION-quality ratio. Wall-clock TIME efficiency is a
               SEPARATE family: retrieval_ms (compose) / write_ms (observe) / score_ms.
                  does NOT reward minimalism and does NOT punish complementary views.

Budget (context size) is reported as a plain descriptive stat (avg refs / segment),
NOT folded into a score; compare systems at matched budget when fairness matters.

Empty selection (system returned no images for a segment) is scored WITHOUT a VLM
call: precision undefined (excluded), recall = 0 if any continuity entity was present.

The "what should be present" side is FROZEN gold, so scoring is reproducible;
only per-image visual matching uses the pinned VLM. Structured JSON, bounded
concurrency, and validated content-addressed caching are part of v3.

INPUTS (all already produced by the existing pipeline — this module reads them)
  gold/chunk_annotations.json: per-segment {present, first_appearances, prompt, seconds_span}
  gold/entity_registry.json  : entity_id -> {name, kind, description}
  benchmark_run/visual_selections/<system>.json : per-segment selected reference crops
  <source video>             : per-segment clips are cut on demand with ffmpeg

CLI
  python -m vmem_bench.scoring.visual_coverage \
      --movie   data/BlenderOpenMovies/big_buck_bunny \
      --system  memstrata_memstrata-fast \
      --video   /.../big_buck_bunny_720p_h264.mp4 \
      --out     data/BlenderOpenMovies/big_buck_bunny/benchmark_run/_visual_score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from vmem_bench.common.media import ffmpeg_bin
from vmem_bench.scoring.judge_service import (
    DEFAULT_API,
    DEFAULT_MODEL,
    PooledJudgeCaller,
    build_judge_api as _build_judge_api,
    call_judge,
)

DEFAULT_FFMPEG = ffmpeg_bin()
_BENCH_ROOT = Path(__file__).resolve().parents[3]
_TRACKA_OUTPUT_ROOT = Path(
    os.environ.get(
        "VMEM_TRACKA_OUTPUT_ROOT",
        str(_BENCH_ROOT / "outputs" / "evaluation" / "trackA"),
    )
).expanduser().resolve()
# Per-image downscale (longest side, px) sent to the judge, applied UNIFORMLY to
# every system so the comparison is fair. It bounds per-image token cost so
# large-memory systems (e.g. MemFlow keeps a wide sink+local+bank footprint = tens
# of refs/segment) still fit in the judge's context window. Done client-side (deterministic,
# independent of server-side mm_processor_kwargs, which Qwen3-VL/vLLM ignored here).
# The judge only decides entity presence/identity, for which 384px is ample.
# Keep this constant fixed across the release.
JUDGE_IMG_MAX_SIDE = 384
JUDGE_MAX_IMAGES_PER_PROMPT = int(os.environ.get("MAVE_JUDGE_MAX_IMAGES_PER_PROMPT", "24"))

# Judge video clips are downscaled to the SUT's native observation resolution
# (Wan2.1-T2V-1.3B: 832x480, 16:9). The SUT only ever perceived the film at 480p,
# so scoring at the same resolution is both fair and far cheaper/faster than
# feeding the judge full-res (e.g. 1080p) clips. Keep fixed across the release.
JUDGE_CLIP_W = 832
JUDGE_CLIP_H = 480
LEGACY_METRIC_VERSION = "visual-coverage-2.2"
PER_REF_METRIC_VERSION = "visual-coverage-3.0"
DEFAULT_METRIC_VERSION = PER_REF_METRIC_VERSION
SUPPORTED_METRIC_VERSIONS = {
    LEGACY_METRIC_VERSION,
    PER_REF_METRIC_VERSION,
}
V3_CONTRACT_VERSION = "refs-only-per-reference-multilabel-v1"


def build_judge_api(**kwargs):
    kwargs.setdefault("stage", "trackA_stage2_visual_coverage")
    return _build_judge_api(**kwargs)


# --------------------------------------------------------------------------- IO
def _load_gold(movie: Path):
    # Rich per-segment GT (present/prompt/first_appearances) lives in chunk_annotations.json;
    # chunk_index.json is the thin layout file and is intentionally NOT read here.
    ca = json.loads((movie / "gold/chunk_annotations.json").read_text())
    er = json.loads((movie / "gold/entity_registry.json").read_text())
    ents = {e["entity_id"]: e for e in er["entities"]}
    segments = {}
    for c in ca["chunks"]:
        present = [str(x) for x in (c.get("present") or [])]
        first = {str(x) for x in (c.get("first_appearances") or [])}
        segments[int(c["chunk_id"])] = {
            "present": present,
            "continuity": [e for e in present if e not in first],  # memory-testable set
            "prompt": c.get("prompt", ""),
            "seconds_span": c.get("seconds_span"),
        }
    return segments, ents


def _tracka_run_dir(movie: Path, system: str) -> Path:
    """Stage-1/2 run dir under outputs/evaluation/trackA for this movie+system."""
    dataset = movie.parent.name
    return _TRACKA_OUTPUT_ROOT / system / dataset / movie.name


def _materialize_source_frame(
    movie: Path,
    system: str,
    chunk_id: int,
    source_seconds: float,
    video: Path | None,
    ffmpeg: str,
) -> str:
    """Cut a timestamp reference into the image file consumed by the VLM scorer."""
    if video is None or not video.is_file():
        raise ValueError(
            f"selection for chunk {chunk_id} uses source_seconds={source_seconds}, "
            "but scoring has no readable --video"
        )
    out = (
        _tracka_run_dir(movie, system)
        / "_ref_frames"
        / system
        / f"chunk_{chunk_id:05d}_t{source_seconds:.3f}.jpg"
    )
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{float(source_seconds):.3f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(out),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as error:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"ffmpeg could not materialize source_seconds={source_seconds} "
                f"for chunk {chunk_id}"
            ) from error
    return str(out.resolve())


def _load_selection(
    movie: Path,
    system: str,
    *,
    video: Path | None = None,
    ffmpeg: str = "ffmpeg",
):
    """Return ({segment_id: [materialized image, ...]}, {segment_id: timing}).

    The second dict carries the Stage-1 retrieval/write latencies the runner
    recorded per segment (``retrieval_timing`` in the manifest); empty when absent
    (older manifests) so scoring still works and just leaves those columns null.
    ``source_seconds`` references are cut from ``video`` into scorer-readable
    frames; omitting ``video`` for such a reference raises instead of dropping it.
    """
    # New protocol outputs live under outputs/evaluation/trackA/<system>/<dataset>/<movie>/.
    # Keep the legacy movie/benchmark_run fallback so old smoke artifacts remain readable.
    new_path = _tracka_run_dir(movie, system) / "visual_selections" / f"{system}.json"
    legacy_path = movie / "benchmark_run/visual_selections" / f"{system}.json"
    sel_path = new_path if new_path.is_file() else legacy_path
    vs = json.loads(sel_path.read_text())
    out = {}
    timing = {}
    for c in vs["chunks"]:
        cid = int(c["chunk_id"])
        imgs = []
        for sel in (c.get("selected") or []):
            for rep in (sel.get("representations") or []):
                p = rep.get("crop_abspath") or rep.get("crop_path")
                if p is None and rep.get("source_seconds") is not None:
                    p = _materialize_source_frame(
                        movie,
                        system,
                        cid,
                        float(rep["source_seconds"]),
                        video,
                        ffmpeg,
                    )
                if p:
                    imgs.append(str(p))
        out[cid] = imgs
        timing[cid] = c.get("retrieval_timing") or {}
    return out, timing


# ------------------------------------------------------------------------- clip
def _shared_segment(movie: Path, chunk_id: int) -> Path | None:
    seg = (
        _TRACKA_OUTPUT_ROOT
        / "_shared_segments"
        / movie.parent.name
        / movie.name
        / f"chunk_{int(chunk_id):05d}.mp4"
    )
    return seg if seg.is_file() and seg.stat().st_size > 0 else None


def _cut_clip(ffmpeg: str, src: Path, out: Path, s0: float, s1: float, *, movie: Path | None = None,
              chunk_id: int | None = None) -> Path:
    if movie is not None and chunk_id is not None:
        shared = _shared_segment(movie, chunk_id)
        if shared is not None:
            return shared
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, float(s1) - float(s0))
    # Downscale to the SUT's native 832x480 (see JUDGE_CLIP_*): fair (matches what
    # the SUT observed) and much faster than judging full-res source clips.
    subprocess.run([ffmpeg, "-y", "-ss", f"{float(s0):.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                    "-an", "-threads", os.environ.get("MAVE_FFMPEG_THREADS", "1"),
                    "-vf", f"scale={JUDGE_CLIP_W}:{JUDGE_CLIP_H}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _call(api: str | PooledJudgeCaller, model: str, content: list, max_tokens: int = 2048) -> str:
    return call_judge(api, model, content, temperature=0.0, max_tokens=max_tokens)


def _parse(s: str):
    items = []
    for o in re.findall(r'\{[^{}]*"i"\s*:\s*\d+[^{}]*\}', s):
        for cand in (o, o.replace("'", '"')):
            try:
                items.append(json.loads(cand)); break
            except Exception:
                pass
    missing = []
    m = re.search(r'"missing"\s*:\s*\[([^\]]*)\]', s)
    if m:
        missing = [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]
    return items, missing


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v3_response_format(allowed: set[str]) -> dict[str, Any]:
    entity_schema: dict[str, Any] = {"type": "string"}
    if allowed:
        entity_schema["enum"] = sorted(allowed)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_coverage_v3_reference",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": entity_schema,
                    }
                },
                "required": ["entity_ids"],
                "additionalProperties": False,
            },
        },
    }


def _parse_v3_reference(raw: str, allowed: set[str]) -> list[str]:
    """Strictly validate one structured per-reference multi-label response."""
    payload = json.loads(str(raw).strip())
    if not isinstance(payload, dict) or set(payload) != {"entity_ids"}:
        raise ValueError("v3 response must contain exactly entity_ids")
    entity_ids = payload["entity_ids"]
    if (
        not isinstance(entity_ids, list)
        or any(not isinstance(value, str) for value in entity_ids)
        or len(entity_ids) != len(set(entity_ids))
        or any(value not in allowed for value in entity_ids)
    ):
        raise ValueError("v3 entity_ids must be a unique subset of the roster")
    return sorted(entity_ids)


class _V3JudgeCache:
    """Optional content-addressed cache for validated v3 judge responses."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def key_lock(self, key: str) -> threading.Lock:
        """Return the process-local single-flight lock for one payload."""
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def get(self, key: str, allowed: set[str]) -> tuple[list[str], str] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "vmem-bench.v3-judge-cache.v1"
            or payload.get("key") != key
        ):
            raise ValueError(f"invalid v3 judge cache record: {path}")
        raw = str(payload["raw"])
        entity_ids = _parse_v3_reference(raw, allowed)
        if entity_ids != payload.get("entity_ids"):
            raise ValueError(f"v3 judge cache labels changed: {path}")
        return entity_ids, raw

    def put(self, key: str, entity_ids: list[str], raw: str) -> None:
        path = self._path(key)
        record = {
            "schema": "vmem-bench.v3-judge-cache.v1",
            "key": key,
            "entity_ids": entity_ids,
            "raw": raw,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self._lock:
            if not path.exists():
                temporary.replace(path)
            elif temporary.exists():
                temporary.unlink()


def _v3_prompt(roster: list[dict[str, str]]) -> str:
    roster_text = "\n".join(
        (
            f"- {row['entity_id']} | {row['name']} ({row['kind']}): "
            f"{row['description']}"
        )
        for row in roster
    ) or "(无)"
    return (
        "以下清单是当前片段中权威标注为在场的实体。随后只给出一张历史参考图，"
        "不会给出当前视频，也不要重新判断清单实体是否在场。\n"
        f"{roster_text}\n\n"
        "返回参考图中能由视觉内容直接支持的所有清单实体 ID。"
        "同一图可以同时覆盖人物、物体和地点；人物位于清晰可辨场景中时，"
        "可以同时返回人物和地点。不要由人物身份推断不可见地点。"
        "不得输出清单外 ID。\n"
        '严格只输出 JSON：{"entity_ids":["char_001","loc_001"]}。'
    )


def _judge_v3_reference(
    ref: str,
    roster: list[dict[str, str]],
    api: str | PooledJudgeCaller,
    model: str,
    cache: _V3JudgeCache | None,
) -> dict[str, Any]:
    allowed = {row["entity_id"] for row in roster}
    content = [_txt(_v3_prompt(roster)), _img(ref)]
    response_format = _v3_response_format(allowed)
    key = _json_sha256(
        {
            "contract": V3_CONTRACT_VERSION,
            "model": model,
            "content": content,
            "response_format": response_format,
            "temperature": 0.0,
        }
    )
    if cache is not None:
        cached = cache.get(key, allowed)
        if cached is not None:
            entity_ids, raw = cached
            return {
                "entity_ids": entity_ids,
                "raw": raw,
                "cache_key": key,
                "cache_hit": True,
                "judge_called": False,
            }
    with cache.key_lock(key) if cache is not None else nullcontext():
        if cache is not None:
            cached = cache.get(key, allowed)
            if cached is not None:
                entity_ids, raw = cached
                return {
                    "entity_ids": entity_ids,
                    "raw": raw,
                    "cache_key": key,
                    "cache_hit": True,
                    "judge_called": False,
                }
        raw = call_judge(
            api,
            model,
            content,
            temperature=0.0,
            max_tokens=256,
            response_format=response_format,
        )
        entity_ids = _parse_v3_reference(raw, allowed)
        if cache is not None:
            cache.put(key, entity_ids, raw)
    return {
        "entity_ids": entity_ids,
        "raw": raw,
        "cache_key": key,
        "cache_hit": False,
        "judge_called": True,
    }


def _img(p):
    # Downscale client-side to JUDGE_IMG_MAX_SIDE and inline as base64 so per-image
    # token cost is bounded and uniform (lets large-footprint systems fit context).
    from PIL import Image
    import base64, io
    im = Image.open(p).convert("RGB")
    im.thumbnail((JUDGE_IMG_MAX_SIDE, JUDGE_IMG_MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
def _vid(p): return {"type": "video_url", "video_url": {"url": f"file://{p}"}}
def _txt(t): return {"type": "text", "text": t}


def _distinct_views(imgs, api, model):
    """VLM variant: # of visually distinct views among crops of the SAME entity.

    The VLM merges near-duplicates and returns an integer count. This is the count
    backend for ``redundancy_vlm`` and for ``efficiency`` (efficiency needs an integer
    redundant-image count, which only this variant yields).
    """
    k = len(imgs)
    if k <= 1:
        return k
    max_imgs = max(1, JUDGE_MAX_IMAGES_PER_PROMPT)
    judged = imgs[:max_imgs]
    # vLLM enforces a hard per-prompt image limit. This distinct-count call returns
    # only an integer, not representatives, so cross-batch merging would invent a
    # protocol the judge did not run. For rare over-limit same-entity groups, judge
    # the first service-compatible batch and treat the overflow as distinct. This
    # avoids fabricated duplicate penalties while keeping all per-reference presence
    # scoring in score_segment().
    overflow = max(0, k - len(judged))
    content = [_txt(f"以下 {len(judged)} 张图都是**同一个对象**的参考图。其中有几张是**视觉上明显不同**的"
                    "视角/外观?(近乎重复的多张只算一张。)只输出 JSON:{\"distinct\": N}。")]
    content += [_img(p) for p in judged]
    raw = _call(api, model, content, max_tokens=64)
    m = re.search(r'"distinct"\s*:\s*(\d+)', raw)
    n = int(m.group(1)) if m else len(judged)
    return max(1, min(k, n + overflow))


# A segment routinely has 2-3 same-entity groups, and each one is an independent
# VLM call (2-4 tiny crops, <=64 output tokens) that used to be issued strictly
# one after another, each waiting its turn for a free endpoint. Fanning them out
# overlaps that queue wait. This is EXACTLY semantics-preserving: every call gets
# the identical image list, and the caller still accumulates results in the
# original group order, so even float summation order is unchanged.
# Set MAVE_GROUP_PARALLEL=0 to fall back to the strictly serial path.
_MAX_GROUP_PARALLEL = 4


def _distinct_views_many(jobs, api, model):
    """``{group_key: distinct_views}`` for several same-entity groups at once.

    A single group is called inline so the common case keeps its exact original
    single-threaded behaviour and spawns no threads.
    """
    if not jobs:
        return {}
    if len(jobs) == 1 or os.environ.get("MAVE_GROUP_PARALLEL", "1") == "0":
        return {key: _distinct_views(imgs, api, model) for key, imgs in jobs}
    # Workload tags are thread-local, so a helper thread would report an
    # unlabelled busy endpoint; re-enter the snapshot inside each thread to keep
    # fleet-console attribution identical to the serial path.
    snapshot = api.current_workload() if hasattr(api, "current_workload") else None

    def _one(imgs):
        with (api.workload(**snapshot) if snapshot else nullcontext()):
            return _distinct_views(imgs, api, model)

    out: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_GROUP_PARALLEL, len(jobs))) as pool:
        futures = {pool.submit(_one, imgs): key for key, imgs in jobs}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


# ---- redundancy variant 2: threshold-free DINO self-similarity -------------------
_EMB: dict[str, object] = {"m": "unset"}
_EMB_CACHE: dict[str, list] = {}
_EMB_INIT_LOCK = threading.Lock()
_EMB_INFER_LOCK = threading.Lock()
_EMB_CACHE_LOCK = threading.Lock()


def _get_embedder():
    """Build + eagerly warm up the pinned DINOv3 scoring embedder; None if torch/weights
    absent. We warm up here (not lazily on first crop) so that a load failure is logged ONCE
    with its reason, instead of being swallowed per-image in ``_mean_pairwise_sim`` and
    silently emptying the ``redundancy_sim`` column for the whole run."""
    import sys as _sys
    if _EMB["m"] == "unset":
        with _EMB_INIT_LOCK:
            if _EMB["m"] == "unset":
                try:
                    from vmem_bench.scoring.embedder import (
                        build_scoring_embedder, resolve_pinned_weights)
                    emb = build_scoring_embedder("dinov3")
                    # Force the weights to load now so failures surface immediately.
                    ensure = getattr(emb, "_ensure_loaded", None)
                    if callable(ensure):
                        ensure()
                    _EMB["m"] = emb
                except Exception as exc:  # noqa: BLE001 - CPU/offline hosts have no torch/weights
                    print(f"[visual_coverage] WARNING: DINOv3 scoring embedder unavailable, "
                          f"redundancy_sim will be NULL for this run. resolved_weights="
                          f"{resolve_pinned_weights()!r} PUBLIC_MODELS_ROOT="
                          f"{os.environ.get('PUBLIC_MODELS_ROOT')!r} err={type(exc).__name__}: {exc}",
                          file=_sys.stderr, flush=True)
                    _EMB["m"] = None
    return _EMB["m"]


def _mean_pairwise_sim(imgs, emb):
    """Threshold-free redundancy signal for one same-entity group.

    = mean of the off-diagonal cells of the DINOv3 cosine-similarity matrix
    (Σ cos(i,j) over i<j) / (#pairs). Embeddings are unit-norm so cos = dot.
    1.0 => the crops are visually identical (fully redundant); lower => more diverse
    views. Deterministic given the pinned embedder; NO threshold. Returns None when
    the group has <2 images or the embedder is unavailable.
    """
    if emb is None or len(imgs) < 2:
        return None
    vs = []
    for p in imgs:
        with _EMB_CACHE_LOCK:
            v = _EMB_CACHE.get(p)
        if v is None:
            try:
                # The same scorer process may run many segment threads. Keep the
                # pinned local embedder serialized; VLM calls still occupy the
                # endpoint pool concurrently and dominate wall time.
                with _EMB_INFER_LOCK:
                    v = emb.embed_image(p)
            except Exception:  # noqa: BLE001 - unreadable crop -> skip
                v = None
            with _EMB_CACHE_LOCK:
                _EMB_CACHE[p] = v
        vs.append(v)
    vs = [v for v in vs if v]
    if len(vs) < 2:
        return None
    tot, cnt = 0.0, 0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            tot += sum(a * b for a, b in zip(vs[i], vs[j]))
            cnt += 1
    return tot / cnt if cnt else None


# ---------------------------------------------------------------------- scoring
@dataclass
class SegmentScore:
    chunk_id: int
    n_refs: int
    n_present_gold: int
    n_continuity: int
    precision: float | None
    recall: float | None        # HEADLINE: memory recall over continuity entities
    recall_all: float | None    # diagnostic: over all present entities
    f1: float | None            # harmonic(precision, recall[continuity])
    redundancy_vlm: float | None  # VLM variant: per-entity near-dup rate among on-screen refs
    redundancy_sim: float | None  # DINO variant: mean same-entity pairwise cosine (threshold-free)
    # SELECTION efficiency (quality of the chosen set), NOT time: useful
    # (on-screen & non-dup, VLM count) / total refs. Renamed from ``efficiency``
    # to avoid confusion with the wall-clock time-efficiency metrics below.
    selection_efficiency: float | None
    # --- time-efficiency metrics (wall-clock, per segment) ---
    retrieval_ms: float | None = None  # SUT compose() latency (Stage-1, from manifest)
    write_ms: float | None = None      # SUT observe_segment() latency (Stage-1, from manifest)
    score_ms: float | None = None      # this segment's VLM scoring latency (Stage-2)
    dur: float | None = None           # segment duration in seconds (for duration-weighted aggregation)

    @property
    def segment_id(self) -> int:
        return self.chunk_id


def _f1(p, r):
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else round(2 * p * r / (p + r), 4)


ChunkScore = SegmentScore  # backward-compatible type alias


def _score_segment_v2(
    cid,
    refs,
    present,
    continuity,
    roster_txt,
    prompt,
    clip,
    api,
    model,
    emb=None,
) -> tuple[SegmentScore, dict]:
    P = set(present)
    C = set(continuity)
    n = len(refs)
    # empty selection: no VLM call needed
    if n == 0:
        rec = 1.0 if not C else 0.0
        rec_all = 1.0 if not P else 0.0
        return (SegmentScore(cid, 0, len(P), len(C), None, rec, rec_all,
                           _f1(None, rec) if C else None, None, None, None),
                {"segment_id": cid, "chunk_id": cid, "note": "empty_selection", "refs": [], "missing_pred": sorted(P)})

    max_imgs = max(1, JUDGE_MAX_IMAGES_PER_PROMPT)
    all_items = []
    missing_sets = []
    raw_parts = []
    for start in range(0, n, max_imgs):
        batch = refs[start:start + max_imgs]
        end = start + len(batch) - 1
        if n <= max_imgs:
            index_note = "逐图判断(i=0..N-1):\n"
            count_note = f"随后给出 {n} 张**无标签参考图**(某记忆系统为本片段选出的),以及该片段的一段视频。"
        else:
            index_note = (
                f"本批给出全局参考图 i={start}..{end}; 输出中的 i 必须使用这些全局编号。\n"
                f"该片段总共有 {n} 张参考图,本批只评其中 {len(batch)} 张。\n"
            )
            count_note = (
                f"随后给出该片段参考图的一个批次: {len(batch)} 张**无标签参考图**,以及该片段的一段视频。"
            )
        content = [_txt(
            "下面是本片段 gold 在场实体清单(带描述):\n" + roster_txt +
            f"\n\n{count_note}"
            f"\n该片段提示词:{prompt}\n\n"
            f"{index_note}"
            "1) present:该图代表的对象是否**出现在视频**中(true/false)。"
            "若为场景/地点,present 指视频**发生在该场景**里。\n"
            "2) entity_id:它对应清单里的哪个 id;都不对应填 \"none\"。\n"
            "另给出 missing:清单里在视频中明显出现、但这些参考图**一张都没覆盖**到的 entity_id 列表。\n"
            '严格只输出 JSON:{"items":[{"i":0,"present":true,"entity_id":"char_001"}, ...],"missing":["..."]}。')]
        content += [_img(p) for p in batch] + [_vid(clip)]
        raw_part = _call(api, model, content)
        batch_items, batch_missing = _parse(raw_part)
        for item in batch_items:
            if "i" not in item:
                all_items.append(item)
                continue
            try:
                idx = int(item["i"])
            except Exception:
                all_items.append(item)
                continue
            if not (start <= idx <= end) and 0 <= idx < len(batch):
                item = {**item, "i": start + idx}
            all_items.append(item)
        missing_sets.append(set(batch_missing))
        raw_parts.append(raw_part[:600])

    items = all_items
    # For split batches, an entity is globally missing only if every reference
    # batch judged it missing. Single-batch behavior is unchanged.
    missing = sorted(set.intersection(*missing_sets)) if missing_sets else []
    raw = "\n---BATCH---\n".join(raw_parts)
    by_i = {int(it["i"]): it for it in items if "i" in it}

    rows = []
    for i, p in enumerate(refs):
        it = by_i.get(i, {})
        rows.append({"i": i, "crop": p, "pred_present": it.get("present"),
                     "pred_entity": it.get("entity_id")})

    # -- precision: fraction of selected images actually on-screen
    present_rows = [r for r in rows if r["pred_present"] is True]
    n_present = len(present_rows)
    precision = round(n_present / n, 4)

    # -- per-entity redundancy: near-duplicates WITHIN the same entity group
    groups: dict[str, list[str]] = {}
    for idx, r in enumerate(present_rows):
        e = r["pred_entity"]
        key = e if (e and e != "none") else f"__none_{idx}"  # unclassifiable -> singleton
        groups.setdefault(key, []).append(r["crop"])
    total_redundant = 0
    group_info = []
    sim_num, sim_pairs = 0.0, 0
    # Issue the independent per-group VLM calls together, then fold the results in
    # the ORIGINAL group order below so every accumulation (including the float
    # sim_num sum) stays bit-identical to the serial version.
    dv_by_key = _distinct_views_many(
        [(k, v) for k, v in groups.items() if len(v) >= 2 and not k.startswith("__none_")],
        api, model,
    )
    for key, imgs in groups.items():
        if key in dv_by_key:
            dv = dv_by_key[key]                             # variant 1 (VLM count)
            msim = _mean_pairwise_sim(imgs, emb)            # variant 2 (DINO, threshold-free)
        else:
            dv, msim = len(imgs), None
        red = len(imgs) - dv
        total_redundant += red
        if msim is not None:
            pairs = len(imgs) * (len(imgs) - 1) // 2
            sim_num += msim * pairs
            sim_pairs += pairs
        group_info.append({"entity": key, "n": len(imgs), "distinct_views": dv,
                           "redundant": red, "mean_sim": round(msim, 4) if msim is not None else None})
    redundancy_vlm = round(total_redundant / n_present, 4) if n_present else None
    # pair-weighted mean of per-group DINO self-similarity across the segment
    redundancy_sim = round(sim_num / sim_pairs, 4) if sim_pairs else None

    # -- selection efficiency: useful (on-screen AND non-duplicate) / total selected.
    #    Uses the VLM redundant-count (only variant that yields an integer count).
    #    This is a SELECTION-quality ratio, NOT a wall-clock time metric.
    useful = n_present - total_redundant
    selection_efficiency = round(useful / n, 4)

    # -- recall: headline over continuity (memory) entities; recall_all diagnostic
    missing_in_P = {e for e in missing if e in P}
    missing_C = missing_in_P & C
    recall = round((len(C) - len(missing_C)) / len(C), 4) if C else 1.0
    recall_all = round((len(P) - len(missing_in_P)) / len(P), 4) if P else 1.0

    sc = SegmentScore(cid, n, len(P), len(C), precision, recall, recall_all,
                      _f1(precision, recall), redundancy_vlm, redundancy_sim, selection_efficiency)
    detail = {"segment_id": cid, "chunk_id": cid, "refs": rows, "groups": group_info,
              "missing_pred": sorted(missing_in_P), "raw": raw[:600]}
    return sc, detail


def _score_segment_v3(
    cid: int,
    refs: list[str],
    present: list[str],
    continuity: list[str],
    roster: list[dict[str, str]],
    api: str | PooledJudgeCaller,
    model: str,
    *,
    ref_workers: int = 1,
    judge_cache: _V3JudgeCache | None = None,
) -> tuple[SegmentScore, dict]:
    """Score each reference independently against the gold-present roster."""
    present_set = set(present)
    continuity_set = set(continuity)
    n = len(refs)
    if n == 0:
        recall = 1.0 if not continuity_set else 0.0
        recall_all = 1.0 if not present_set else 0.0
        return (
            SegmentScore(
                cid,
                0,
                len(present_set),
                len(continuity_set),
                None,
                recall,
                recall_all,
                _f1(None, recall) if continuity_set else None,
                None,
                None,
                None,
            ),
            {
                "segment_id": cid,
                "chunk_id": cid,
                "note": "empty_selection",
                "refs": [],
                "missing_pred": sorted(present_set),
                "judge_contract": V3_CONTRACT_VERSION,
                "judge_requests": 0,
                "judge_cache_hits": 0,
            },
        )
    if not roster:
        judged = [
            {
                "entity_ids": [],
                "raw": '{"entity_ids":[]}',
                "cache_key": None,
                "cache_hit": False,
                "judge_called": False,
            }
            for _ in refs
        ]
    else:
        workers = max(1, min(int(ref_workers), n))
        snapshot = (
            api.current_workload()
            if hasattr(api, "current_workload")
            else None
        )

        def _one(ref: str) -> dict[str, Any]:
            with (
                api.workload(**snapshot)
                if snapshot
                else nullcontext()
            ):
                return _judge_v3_reference(
                    ref,
                    roster,
                    api,
                    model,
                    judge_cache,
                )

        if workers == 1:
            judged = [_one(ref) for ref in refs]
        else:
            judged_by_index: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_one, ref): index
                    for index, ref in enumerate(refs)
                }
                for future in as_completed(futures):
                    judged_by_index[futures[future]] = future.result()
            judged = [judged_by_index[index] for index in range(n)]

    covered = {
        entity_id
        for row in judged
        for entity_id in row["entity_ids"]
    } & present_set
    rows = [
        {
            "i": index,
            "crop": ref,
            "pred_present": bool(judgement["entity_ids"]),
            "pred_entity": (
                judgement["entity_ids"][0]
                if len(judgement["entity_ids"]) == 1
                else None
            ),
            "pred_entities": judgement["entity_ids"],
            "cache_key": judgement["cache_key"],
            "cache_hit": judgement["cache_hit"],
        }
        for index, (ref, judgement) in enumerate(zip(refs, judged, strict=True))
    ]
    n_present = sum(row["pred_present"] for row in rows)
    precision = round(n_present / n, 4)
    missing = present_set - covered
    missing_continuity = missing & continuity_set
    recall = (
        round(
            (len(continuity_set) - len(missing_continuity))
            / len(continuity_set),
            4,
        )
        if continuity_set
        else 1.0
    )
    recall_all = (
        round((len(present_set) - len(missing)) / len(present_set), 4)
        if present_set
        else 1.0
    )
    score = SegmentScore(
        cid,
        n,
        len(present_set),
        len(continuity_set),
        precision,
        recall,
        recall_all,
        _f1(precision, recall),
        None,
        None,
        None,
    )
    detail = {
        "segment_id": cid,
        "chunk_id": cid,
        "refs": rows,
        "groups": [],
        "covered_pred": sorted(covered),
        "missing_pred": sorted(missing),
        "judge_contract": V3_CONTRACT_VERSION,
        "judge_requests": sum(row["judge_called"] for row in judged),
        "judge_cache_hits": sum(row["cache_hit"] for row in judged),
        "judge_responses": [
            {
                "i": index,
                "raw": row["raw"],
                "cache_key": row["cache_key"],
                "cache_hit": row["cache_hit"],
                "judge_called": row["judge_called"],
            }
            for index, row in enumerate(judged)
        ],
        "redundancy_note": (
            "Not defined for multi-label references in visual-coverage-3.0."
        ),
    }
    return score, detail


def score_segment(
    cid,
    refs,
    present,
    continuity,
    roster_txt,
    prompt,
    clip,
    api,
    model,
    emb=None,
    *,
    metric_version: str = LEGACY_METRIC_VERSION,
    roster: list[dict[str, str]] | None = None,
    ref_workers: int = 1,
    judge_cache: _V3JudgeCache | None = None,
) -> tuple[SegmentScore, dict]:
    if metric_version == LEGACY_METRIC_VERSION:
        return _score_segment_v2(
            cid,
            refs,
            present,
            continuity,
            roster_txt,
            prompt,
            clip,
            api,
            model,
            emb,
        )
    if metric_version == PER_REF_METRIC_VERSION:
        if roster is None:
            raise ValueError("visual-coverage-3.0 requires structured roster rows")
        return _score_segment_v3(
            cid,
            refs,
            present,
            continuity,
            roster,
            api,
            model,
            ref_workers=ref_workers,
            judge_cache=judge_cache,
        )
    raise ValueError(f"unsupported metric_version: {metric_version}")


score_chunk = score_segment  # backward-compatible function alias


def run(
    movie: Path,
    system: str,
    video: Path,
    out_dir: Path,
    api: str | PooledJudgeCaller,
    model: str,
    ffmpeg: str,
    limit: int | None = None,
    workers: int = 1,
    *,
    metric_version: str = DEFAULT_METRIC_VERSION,
    ref_workers: int = 1,
    judge_cache_dir: Path | None = None,
):
    if metric_version not in SUPPORTED_METRIC_VERSIONS:
        raise ValueError(f"unsupported metric_version: {metric_version}")
    # Resolve to absolute so the clip file:// URLs sent to the VLM server are absolute
    # (the server has a different cwd; relative paths 500 with "No such file").
    movie = movie.resolve()
    video = video.resolve()
    segments, ents = _load_gold(movie)
    sel, sel_timing = _load_selection(movie, system, video=video, ffmpeg=ffmpeg)
    run_dir = _tracka_run_dir(movie, system)
    clip_dir = run_dir / "_clips"
    # v3 intentionally leaves redundancy undefined until multi-label grouping has
    # a non-ambiguous contract, so it neither loads nor runs DINO.
    emb = _get_embedder() if metric_version == LEGACY_METRIC_VERSION else None
    if metric_version == PER_REF_METRIC_VERSION and judge_cache_dir is None:
        judge_cache_dir = out_dir / "_judge_cache" / PER_REF_METRIC_VERSION
    judge_cache = (
        _V3JudgeCache(judge_cache_dir)
        if metric_version == PER_REF_METRIC_VERSION and judge_cache_dir
        else None
    )

    cids = sorted(segments)
    if limit:
        cids = cids[:limit]

    def _score_one(cid: int) -> tuple[int, SegmentScore, dict]:
        meta = segments[cid]
        present = meta["present"]
        continuity = meta["continuity"]
        roster = [
            {
                "entity_id": e,
                "name": str(ents[e]["name"]),
                "kind": str(ents[e]["kind"]),
                "description": str(ents[e].get("description") or "")[:70],
            }
            for e in present
            if e in ents
        ]
        roster_txt = "\n".join(
            (
                f"- {row['entity_id']} | {row['name']} ({row['kind']}): "
                f"{row['description']}"
            )
            for row in roster
        ) or "(无)"
        refs = sel.get(cid, [])
        span = meta.get("seconds_span")
        clip = None
        if refs and metric_version == LEGACY_METRIC_VERSION:
            s0, s1 = span
            clip = _cut_clip(ffmpeg, video, clip_dir / f"chunk_{cid:03d}.mp4", s0, s1,
                             movie=movie, chunk_id=cid)
        workload_ctx = (
            api.workload(
                job_id=f"{system}:{movie.parent.name}/{movie.name}",
                movie_id=movie.name,
                dataset=movie.parent.name,
                segment_id=str(cid),
                system=system,
            )
            if hasattr(api, "workload")
            else nullcontext()
        )
        with workload_ctx:
            _t = time.perf_counter()
            sc, det = score_segment(
                cid,
                refs,
                present,
                continuity,
                roster_txt,
                meta["prompt"],
                clip,
                api,
                model,
                emb,
                metric_version=metric_version,
                roster=roster,
                ref_workers=ref_workers,
                judge_cache=judge_cache,
            )
        sc.score_ms = round((time.perf_counter() - _t) * 1000.0, 2)  # Stage-2 latency
        # Segment duration -> used for duration-weighted aggregation (segments are NOT
        # equal length: observed 5-27s, so equal-weight per-segment mean over/under-weights)
        sc.dur = round(float(span[1] - span[0]), 3) if span and len(span) == 2 else None
        tm = sel_timing.get(cid, {})  # Stage-1 latencies passed through from the runner
        sc.retrieval_ms = tm.get("compose_ms")
        sc.write_ms = tm.get("observe_ms")
        return cid, sc, det

    def _print_segment(sc: SegmentScore) -> None:
        print(f"segment {sc.chunk_id:3d}: refs={sc.n_refs} P={sc.n_present_gold} C={sc.n_continuity} "
              f"prec={sc.precision} rec_c={sc.recall} rec_all={sc.recall_all} f1={sc.f1} "
              f"redun_vlm={sc.redundancy_vlm} redun_sim={sc.redundancy_sim} sel_eff={sc.selection_efficiency} "
              f"retr_ms={sc.retrieval_ms} score_ms={sc.score_ms}", flush=True)

    if workers <= 0:
        workers = int(getattr(api, "size", 1) or 1)
    workers = max(1, min(int(workers), max(1, len(cids))))
    scored: dict[int, tuple[SegmentScore, dict]] = {}
    if workers == 1:
        for cid in cids:
            _, sc, det = _score_one(cid)
            scored[cid] = (sc, det)
            _print_segment(sc)
    else:
        print(f"[visual_coverage] scoring {len(cids)} segments with workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_cid = {pool.submit(_score_one, cid): cid for cid in cids}
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                _, sc, det = future.result()
                scored[cid] = (sc, det)
                _print_segment(sc)

    scores: list[SegmentScore] = [scored[cid][0] for cid in cids]
    details = [scored[cid][1] for cid in cids]

    def agg(attr, cond=lambda s: True):
        vals = [getattr(s, attr) for s in scores if getattr(s, attr) is not None and cond(s)]
        return round(mean(vals), 4) if vals else None

    def wagg(attr, cond=lambda s: True):
        # duration-weighted mean: segments vary 5-27s, so weight each segment's metric by
        # its own seconds. Falls back to equal weight if a segment has no duration.
        pairs = [(getattr(s, attr), (s.dur if s.dur else 1.0))
                 for s in scores if getattr(s, attr) is not None and cond(s)]
        if not pairs:
            return None
        num = sum(v * w for v, w in pairs); den = sum(w for _, w in pairs)
        return round(num / den, 4) if den else None

    with_refs = [s for s in scores if s.n_refs > 0]
    summary = {
        "movie": movie.name, "system": system, "model": model,
        "metric_version": metric_version,
        "judge_contract": (
            V3_CONTRACT_VERSION
            if metric_version == PER_REF_METRIC_VERSION
            else "batch-reference-plus-target-video-single-label-v2"
        ),
        "judge_requests": (
            sum(int(detail.get("judge_requests", 0)) for detail in details)
            if metric_version == PER_REF_METRIC_VERSION
            else None
        ),
        "judge_cache_hits": (
            sum(int(detail.get("judge_cache_hits", 0)) for detail in details)
            if metric_version == PER_REF_METRIC_VERSION
            else None
        ),
        "judge_cache_dir": (
            str(judge_cache_dir.resolve())
            if metric_version == PER_REF_METRIC_VERSION
            and judge_cache_dir is not None
            else None
        ),
        "n_segments": len(scores),
        "n_segments_with_refs": len(with_refs),
        # Backward-compatible aliases for older aggregators/artifacts.
        "n_chunks": len(scores),
        "n_chunks_with_refs": len(with_refs),
        "precision_mean": agg("precision"),
        # recall is ALWAYS over continuity (memory) entities — that is THE recall.
        # recall_all (incl. first appearances) is kept per-segment in details as a diagnostic only.
        "recall_mean": agg("recall", lambda s: s.n_continuity > 0),
        "f1_mean": agg("f1", lambda s: s.n_continuity > 0),
        # redundancy reported as TWO variants side by side (VLM count-based vs DINO similarity)
        "redundancy_vlm_mean": agg("redundancy_vlm", lambda s: s.n_refs > 0),
        "redundancy_sim_mean": agg("redundancy_sim", lambda s: s.n_refs > 0),
        "selection_efficiency_mean": agg("selection_efficiency", lambda s: s.n_refs > 0),
        # budget = descriptive context-size stat, NOT a score
        "budget_avg_refs_per_segment_with_refs": round(mean([s.n_refs for s in with_refs]), 3) if with_refs else 0.0,
        "budget_avg_refs_per_segment_all": round(mean([s.n_refs for s in scores]), 3) if scores else 0.0,
        # Backward-compatible aliases for older aggregators/artifacts.
        "budget_avg_refs_per_chunk_with_refs": round(mean([s.n_refs for s in with_refs]), 3) if with_refs else 0.0,
        "budget_avg_refs_per_chunk_all": round(mean([s.n_refs for s in scores]), 3) if scores else 0.0,
        # time-efficiency (wall-clock means); retrieval/write are null for pre-timing runs
        "retrieval_ms_mean": agg("retrieval_ms"),   # Stage-1 SUT compose() latency
        "write_ms_mean": agg("write_ms"),           # Stage-1 SUT observe_segment() latency
        "score_ms_mean": agg("score_ms"),           # Stage-2 VLM scoring latency
        # duration-weighted variants (headline candidate): segments are NOT equal length,
        # so weight each segment by its seconds. See scoring_v2.md 4.8.
        "precision_wmean": wagg("precision"),
        "recall_wmean": wagg("recall", lambda s: s.n_continuity > 0),
        "f1_wmean": wagg("f1", lambda s: s.n_continuity > 0),
        "redundancy_vlm_wmean": wagg("redundancy_vlm", lambda s: s.n_refs > 0),
        "redundancy_sim_wmean": wagg("redundancy_sim", lambda s: s.n_refs > 0),
        "selection_efficiency_wmean": wagg("selection_efficiency", lambda s: s.n_refs > 0),
        "total_duration_s": round(sum(s.dur for s in scores if s.dur), 2),
    }

    # long-horizon buckets: split segments into early/mid/late thirds by TIME position
    # (cumulative duration), to quantify "does performance/latency degrade as the movie
    # runs longer" WITHOUT baking time into the headline scalar. Full curve in per_segment.csv.
    ordered = sorted(scores, key=lambda s: s.chunk_id)
    tot = sum(s.dur for s in ordered if s.dur) or float(len(ordered) or 1)
    acc = 0.0
    buckets = {"early": [], "mid": [], "late": []}
    for s in ordered:
        frac = acc / tot
        buckets["early" if frac < 1 / 3 else "mid" if frac < 2 / 3 else "late"].append(s)
        acc += (s.dur or (tot / max(len(ordered), 1)))

    def _wm(subset, attr, cond=lambda s: True):
        pairs = [(getattr(s, attr), (s.dur or 1.0)) for s in subset
                 if getattr(s, attr) is not None and cond(s)]
        if not pairs:
            return None
        return round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)

    summary["horizon"] = {
        k: {"n": len(v),
            "recall_wmean": _wm(v, "recall", lambda s: s.n_continuity > 0),
            "f1_wmean": _wm(v, "f1", lambda s: s.n_continuity > 0),
            "precision_wmean": _wm(v, "precision", lambda s: s.n_refs > 0),
            "retrieval_ms_mean": _wm(v, "retrieval_ms")}
        for k, v in buckets.items()
    }

    out = out_dir / system
    out.mkdir(parents=True, exist_ok=True)
    per_segment_rows = [dict(asdict(s), segment_id=s.chunk_id) for s in scores]
    (out / "score.json").write_text(json.dumps(
        {"summary": summary, "per_segment": per_segment_rows, "per_chunk": per_segment_rows}, ensure_ascii=False, indent=2))
    (out / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    # per-SEGMENT CSV: one row per segment, ordered by time, for plotting metric-vs-time
    # curves (does long-horizon hurt recall/latency?). Terminology = SEGMENT (gold unit);
    # `sec_start` doubles as the segment's time position in the movie.
    import csv as _csv
    csv_cols = ["segment_id", "sec_start", "sec_end", "dur_s", "n_refs", "n_present",
                "n_continuity", "precision", "recall", "recall_all", "f1",
                "redun_vlm", "redun_sim", "sel_eff", "retrieval_ms", "write_ms", "score_ms"]
    _t0 = None
    with (out / "per_segment.csv").open("w", newline="", encoding="utf-8") as _fh:
        w = _csv.writer(_fh); w.writerow(["movie", "system"] + csv_cols)
        for s in scores:
            span = segments[s.chunk_id].get("seconds_span") if s.chunk_id in segments else None
            ss, se = (span[0], span[1]) if span and len(span) == 2 else (None, None)
            w.writerow([movie.name, system, s.chunk_id, ss, se, s.dur, s.n_refs,
                        s.n_present_gold, s.n_continuity, s.precision, s.recall,
                        s.recall_all, s.f1, s.redundancy_vlm, s.redundancy_sim,
                        s.selection_efficiency, s.retrieval_ms, s.write_ms, s.score_ms])
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSCORE:   {out}/score.json")
    print(f"DETAILS: {out}/details.json")
    print(f"CLIPS:   {clip_dir}/  (legacy filenames chunk_XXX.mp4, one per segment)")
    print("VISUAL_COVERAGE_DONE")
    return summary


def main():
    ap = argparse.ArgumentParser(description="VMem-Bench Track A visual-coverage scorer (VLM judge)")
    ap.add_argument("--movie", required=True, type=Path)
    ap.add_argument("--system", required=True, help="visual_selections/<system>.json basename")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--api-list", default="", help="comma/whitespace separated /v1 base URLs or chat endpoints")
    ap.add_argument(
        "--endpoint-slots",
        type=int,
        default=1,
        help="concurrent judge requests per endpoint (default 1). Raise ONLY when the "
             "vLLM replicas run a matching MAX_NUM_SEQS; scheduling-only, does not "
             "change prompts, sampling, or metrics",
    )
    ap.add_argument("--fleet", action="store_true", help="resolve online VLM endpoints from the annotation fleet registry")
    ap.add_argument("--fleet-root", type=Path, default=None)
    ap.add_argument("--fleet-role", default="reviewer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    ap.add_argument("--limit", type=int, default=None, help="score only first N segments (smoke)")
    ap.add_argument(
        "--metric-version",
        choices=sorted(SUPPORTED_METRIC_VERSIONS),
        default=DEFAULT_METRIC_VERSION,
        help=(
            "visual-coverage-3.0 (default) uses refs-only per-reference "
            "multi-label judging; visual-coverage-2.2 preserves the legacy "
            "batch+video scorer"
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="segment workers; 0 = endpoint-pool size when pooled, otherwise 1",
    )
    ap.add_argument(
        "--ref-workers",
        type=int,
        default=1,
        help=(
            "bounded per-segment reference workers for visual-coverage-3.0; "
            "global in-flight calls remain capped by the endpoint pool"
        ),
    )
    ap.add_argument(
        "--judge-cache",
        type=Path,
        default=None,
        help=(
            "content-addressed cache for validated visual-coverage-3.0 "
            "responses; defaults to <out>/_judge_cache/visual-coverage-3.0"
        ),
    )
    a = ap.parse_args()
    out_dir = a.out or (_tracka_run_dir(a.movie.resolve(), a.system) / "_visual_score")
    api = build_judge_api(
        api=a.api,
        api_list=a.api_list,
        use_fleet=a.fleet,
        fleet_root=a.fleet_root,
        fleet_role=a.fleet_role,
        model=a.model,
        endpoint_slots=a.endpoint_slots,
    )
    run(
        a.movie,
        a.system,
        a.video,
        out_dir,
        api,
        a.model,
        a.ffmpeg,
        a.limit,
        workers=a.workers,
        metric_version=a.metric_version,
        ref_workers=a.ref_workers,
        judge_cache_dir=a.judge_cache,
    )


if __name__ == "__main__":
    main()
