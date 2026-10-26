"""IAMFlow causal adapter (new protocol, real segments) — faithful reuse of the
proven Track-A driver.

This adapter does NOT re-invent IAMFlow's memory. It reuses IAMFlow's native
pipeline + memory methods, restructured for the bench's causal per-segment order
(``compose`` before ``observe_segment``), with all Track-A glue kept in this file.

Native memory: an entity-aware active-memory of retained frames. Each retained
frame carries a self-attention KV slice + associated entity ids; its selection
score fuses an ``entity_score`` (text-query aggregated over the DiT self-attn KV
of the evicted segment) with a 0.3-weighted VLM visual score. On a prompt boundary
``retrieve_initial_frames`` recalls historical frames by entity coverage. The
retrieved "memory" for a segment is the active-memory frame set, mapped to absolute
source seconds via each frame's temporal id (``p{pid}_c{cid}_f{f}`` -> global
source latent -> seconds).

Causal fidelity vs. the offline driver: the driver pre-extracts entities for ALL
prompts up front (``_precompute_prompt_entities``). That peeks at future prompts,
which the bench forbids, so we DO NOT precompute. ``_process_prompt_start`` then
falls through to on-the-fly ``llm_agent.process_prompt`` per segment (see
agent_causal_inference.py:930-940) — same code path, only fed one prompt at a
time. The LLM is kept preloaded for the whole run.

We drive IAMFlow on the REAL segment: VAE-encode it and run the clean-context DiT
forward (``context_noise`` timestep, ``update_bank=False``, ``q_bank=False``) block
by block so ``kv_cache1`` / ``crossattn_cache`` are populated from real frames (no
denoising / generation). Eviction (lag>=4) / archival / VLM scoring mirror the
driver verbatim. The vendored repo stays pristine; all glue lives here.

Env: torch 2.5 + flash-attn 2.8. The fp8 checkpoint is dequantized to
bf16 so it runs on Ampere/SM80 without fp8 tensor cores (same as the driver).

**HANG FIX (2026-07-25) — offload LLM/VLM to vLLM, do NOT run them in-process.**
Running the Qwen3-VL-2B VLM's HF ``generate`` in-process, on the SAME GPU/stream as
the DiT forward, intermittently deadlocks at the CUDA level: the process goes silent
for 15-60 min with the DiT gpu pinned but no progress (observed on both Reservoir_Dogs
name_anchored and BBB description_provided). The published IAMFlow backend already
serves LLM+VLM from dedicated vLLM servers; use that. Set both endpoints and give the
DiT its OWN gpu; the servers (both small) co-locate on one card:
  1. start Qwen3-4B and Qwen3-VL-2B OpenAI-compatible vLLM servers on the run node.
  2. run this adapter with ``IAMFLOW_LLM_ENDPOINT=http://127.0.0.1:8100/v1`` and
     ``IAMFLOW_VLM_ENDPOINT=http://127.0.0.1:8101/v1`` on a different gpu of the SAME node.
With the offload, per-block VLM scoring becomes steady HTTP traffic (verified: DiT gpu
100% util, LLM/VLM POST counts climb monotonically) and the run COMPLETES. In-process
HF is retained only as a no-server fallback for tiny (<=3 chunk) smoke, never full runs.

**HOST-MEMORY SCALING (2026-07-28) — why long movies died with exit 137.**
IAMFlow's archive is O(timeline). On every archived frame ``MemoryBank`` moves a
30-block KV slice to host RAM (``memory_bank._extract_frame_kv_all_blocks`` ->
``.cpu()``) and keeps it forever in BOTH ``frame_archive[fid].kv_cache`` and
``_frame_kv_store[fid]``; ``max_memory_frames`` bounds only the *active* set, not
the archive. One slice is ``30 blocks x 2 (k,v) x 1560 tokens x 1536 dim x 2 B``
= **274 MiB**, and one bench segment archives ``(dur / 0.25) // 3`` frames, so a
15 s segment costs ~5 GiB of host RAM. Measured on live runs: 2.4-5.5 GiB per
segment of ``RssAnon`` growth, and ``dmesg`` shows cgroup OOM kills at
``anon-rss:545 GB`` against an 800 GiB (A800) / 900 GiB (H800) pod limit. That is
the whole story behind exit 137 -- it is a memory-scaling property of the method
as published, not preemption and not a scheduling bug.

This file therefore adds host-memory *instrumentation and guards* (see
``docs/baselines/tracka_iamflow_host_memory.md``). Two reclaims are numerics-exact
and always on: ``FrameInfo.pixel_frame`` is write-only dead weight that pins a
whole decoded pixel block, and ``pipe._sync_pixel_store`` is only ever read for
the last few chunk keys. They do not touch the KV archive, which is genuinely
load-bearing (any historical frame may be recalled by entity coverage), so they
buy ~10% and no more. The real protection is a measured watchdog that aborts
cleanly before the kernel SIGKILLs us, plus a per-segment RSS log so the growth
is auditable. Bounding the KV archive *does* change IAMFlow's recall pool, so it
is opt-in only and must be reported as a labelled variant, never as "IAMFlow".

Weights (symlinked under the vendored repo, see WEIGHTS.md):
  baselines/Causal/IAMFlow/pretrained/Wan2.1-T2V-1.3B
  baselines/Causal/IAMFlow/pretrained/iamflow_models/iamflow_fp8.safetensors
  baselines/Causal/IAMFlow/pretrained/{Qwen3-4B-Instruct-2507,Qwen3-VL-2B-Instruct}
Run via runner.py --adapter iamflow.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _video_io import latent_local_seconds, read_segment_pixels

_REPO = Path(__file__).resolve().parents[7] / "baselines" / "Causal" / "IAMFlow"
# Flat module loaded by ``runner.py --adapter iamflow``. Keep helper glue here so the
# adapter has no sibling scripts that can be confused with the vendored IAMFlow package.
_SCRIPTS = Path(__file__).resolve().parent
_PRETRAINED = _REPO / "pretrained"
_CONFIG = str(_REPO / "configs" / "iamflow.yaml")
_CKPT = str(_PRETRAINED / "iamflow_models" / "iamflow_fp8.safetensors")
_LLM = str(_PRETRAINED / "Qwen3-4B-Instruct-2507")
_VLM = str(_PRETRAINED / "Qwen3-VL-2B-Instruct")

_FRAME_SEQ = 1560  # tokens per latent frame (Wan 1.3B)
_SECONDS_PER_LATENT = 0.25  # VAE temporal stride 4 / 16 fps
_FRAME_ID_RE = re.compile(r"p(\d+)_c(\d+)_f(\d+)")
_VLM_READY_SENTINEL = object()


def _dequantize_iamflow_state_dict(path: str):
    """Load IAMFlow's fp8 checkpoint and recover bf16 tensors for Ampere GPUs."""
    import torch
    from safetensors import safe_open

    raw = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            raw[key] = handle.get_tensor(key)

    scale_keys = {k for k in raw if k.endswith(".weight_scale")}
    out = {}
    n_dequant = 0
    for key, val in raw.items():
        if key in scale_keys:
            continue
        scale_key = key + "_scale"
        if key.endswith(".weight") and scale_key in raw:
            out[key] = (val.float() * raw[scale_key].float()).to(torch.bfloat16)
            n_dequant += 1
        else:
            out[key] = val.to(torch.bfloat16) if val.is_floating_point() else val
    print(
        f"[iamflow] dequantized {n_dequant} fp8 linears; "
        f"{len(out)} tensors total, {len(scale_keys)} scales dropped",
        flush=True,
    )
    return out


def _build_transformers_vlm(vlm_agent_cls, model_path: str, device: str, enabled: bool):
    """Transformers fallback for tiny IAMFlow smokes; full runs should use vLLM HTTP."""

    class _TransformersVLM(vlm_agent_cls):
        def __init__(self) -> None:
            super().__init__(model_path=model_path, enabled=enabled)
            self._proc = None
            self._device = device

        def preload(self) -> None:
            if not self.enabled:
                return
            import torch
            import transformers
            from transformers import AutoProcessor

            model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if model_cls is None:
                model_cls = transformers.AutoModelForImageTextToText
            self._llm = model_cls.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).eval().to(self._device)
            self._proc = AutoProcessor.from_pretrained(self.model_path)
            print("[iamflow] VLM (transformers Qwen3-VL) ready", flush=True)

        def _score_frames_vllm(self, images, score_instruction: str) -> str:
            import torch

            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": score_instruction})
            messages = [{"role": "user", "content": content}]
            text = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._proc(
                text=[text], images=list(images), return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                out = self._llm.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self._proc.decode(gen, skip_special_tokens=True).strip()

        def shutdown(self) -> None:
            self._llm = None
            self._proc = None

    return _TransformersVLM()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[iamflow][warn] invalid {name}={raw!r}; using {default}", flush=True)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[iamflow][warn] invalid {name}={raw!r}; using {default}", flush=True)
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------
# Host-memory instrumentation / guards (see module docstring)
# --------------------------------------------------------------------------
_GIB = float(1024 ** 3)
# 30 transformer blocks x {k,v} x 1560 tokens/frame x 1536 dim x 2 B (bf16).
# Only a fallback: the live pipeline is measured instead whenever it exists.
_KV_BYTES_PER_FRAME_FALLBACK = 30 * 2 * _FRAME_SEQ * 1536 * 2


class HostMemoryBudgetExceeded(RuntimeError):
    """IAMFlow's archive is about to exceed the pod's host-memory budget.

    Raised deliberately so the run dies with a diagnosable reason and a flushed
    incremental checkpoint, instead of being SIGKILLed (exit 137) by the cgroup
    OOM killer, which leaves no evidence at all.
    """


def _rss_anon_bytes() -> int:
    """Anonymous resident host memory of this process, or 0 if unavailable.

    ``RssAnon`` (not ``VmRSS``) is the figure the cgroup OOM killer accounts:
    file-backed pages are reclaimable, the archive's ``.cpu()`` tensors are not.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cgroup_mem_limit_bytes() -> int:
    """Pod memory ceiling in bytes, or 0 when it cannot be determined."""
    for path in ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.max"):
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # Unlimited cgroups report a sentinel close to 2**63.
        if 0 < value < (1 << 62):
            return value
    return 0


def _host_memory_budget_bytes() -> int:
    """Bytes of RssAnon this process may reach before we abort on purpose."""
    override = _env_float("MAVE_IAMFLOW_MAX_RSS_GB", 0.0)
    if override > 0:
        return int(override * _GIB)
    limit = _cgroup_mem_limit_bytes()
    if limit <= 0:
        return 0
    # Headroom for the co-located vLLM servers (~10-20 GB of host RSS for the
    # Qwen3-4B + Qwen3-VL-2B pair) and page cache. Do not over-reserve: at 0.85 a
    # 800 GiB pod would refuse 0001_American_Beauty, which peaks near 700 GiB and
    # does complete in production.
    return int(limit * _env_float("MAVE_IAMFLOW_RSS_BUDGET_FRACTION", 0.92))


def _kv_bytes_per_archived_frame(pipe) -> int:
    """Host bytes one archived frame costs, measured off the live pipeline."""
    try:
        blocks = len(pipe.kv_cache1) if getattr(pipe, "kv_cache1", None) else 0
        dim = int(pipe.generator.model.dim)
        itemsize = int(pipe.kv_cache1[0]["k"].element_size())
        if blocks > 0 and dim > 0 and itemsize > 0:
            return blocks * 2 * _FRAME_SEQ * dim * itemsize
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass
    return _KV_BYTES_PER_FRAME_FALLBACK


def _archived_frames_for_movie(movie: MovieContext, npb: int) -> int:
    """Upper bound on frames IAMFlow will archive for this movie.

    One frame is archived per committed ``npb``-latent block, and the adapter
    commits ``(segment_latents // npb)`` blocks per segment. Archival is skipped
    when a segment has no extracted entities, so this over-counts; that is the
    safe direction for a warning but the reason the preflight gate does not
    refuse by default (0001_American_Beauty projects 919 GiB yet completed).
    """
    total = 0
    for span in movie.seconds_span_by_chunk.values():
        seconds = max(0.0, float(span[1]) - float(span[0]))
        latents = int(seconds / _SECONDS_PER_LATENT)
        total += max(0, latents // max(1, npb))
    return total


class _HostMemoryGuard:
    """Per-segment RssAnon telemetry plus a measured abort before the OOM killer.

    The projection deliberately uses the *observed* slope rather than the static
    estimate: the static bound over-counts (see ``_archived_frames_for_movie``),
    and only a measured slope can decide whether a given movie will really fit.
    """

    def __init__(self, *, movie_id: str, total_segments: int, kv_bytes_per_frame: int) -> None:
        self.movie_id = movie_id
        self.total_segments = max(1, int(total_segments))
        self.kv_bytes_per_frame = int(kv_bytes_per_frame)
        self.budget_bytes = _host_memory_budget_bytes()
        self.cgroup_limit_bytes = _cgroup_mem_limit_bytes()
        self.log_enabled = _env_flag("MAVE_IAMFLOW_RSS_LOG", True)
        self.enforce = _env_flag("MAVE_IAMFLOW_RSS_WATCHDOG", True)
        # Ignore the first segments: model load and CUDA host buffers dominate
        # there and would badly skew the slope.
        self.warmup_segments = _env_int("MAVE_IAMFLOW_RSS_WARMUP_SEGMENTS", 3)
        self.max_slope = _env_float("MAVE_IAMFLOW_MAX_RSS_SLOPE_GB_PER_CHUNK", 0.0)
        self.baseline_bytes = _rss_anon_bytes()
        self.peak_bytes = self.baseline_bytes
        self._warm_index: int | None = None
        self._warm_bytes: int | None = None
        self.last_slope_gb = 0.0
        self.last_projection_gb = 0.0
        self.abort_reason: str | None = None

    # -- helpers ---------------------------------------------------------
    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / _GIB

    def _projection_bytes(self, index: int, rss: int) -> tuple[float, float]:
        """(slope bytes/segment, projected RssAnon at the final segment)."""
        if self._warm_index is None or index <= self._warm_index:
            return 0.0, float(rss)
        slope = (rss - self._warm_bytes) / float(index - self._warm_index)
        remaining = max(0, self.total_segments - index)
        return slope, float(rss) + slope * remaining

    # -- main hook -------------------------------------------------------
    def observe(self, index: int) -> None:
        """Record RssAnon after bench segment ``index`` (1-based) and guard it."""
        rss = _rss_anon_bytes()
        if rss <= 0:
            return
        self.peak_bytes = max(self.peak_bytes, rss)
        if index >= self.warmup_segments and self._warm_index is None:
            self._warm_index, self._warm_bytes = index, rss
        slope, projected = self._projection_bytes(index, rss)
        self.last_slope_gb = slope / _GIB
        self.last_projection_gb = projected / _GIB
        if self.log_enabled:
            print(
                f"[iamflow][rss] {self.movie_id} segment={index}/{self.total_segments} "
                f"anon_gb={rss / _GIB:.1f} slope_gb_per_seg={self.last_slope_gb:.2f} "
                f"proj_final_gb={self.last_projection_gb:.1f} budget_gb={self.budget_gb:.1f}",
                file=sys.stderr,
                flush=True,
            )
        if not self.enforce:
            return
        if self.max_slope > 0 and self.last_slope_gb > self.max_slope:
            self._abort(
                index,
                rss,
                f"observed host-memory slope {self.last_slope_gb:.2f} GB/segment exceeds "
                f"MAVE_IAMFLOW_MAX_RSS_SLOPE_GB_PER_CHUNK={self.max_slope:.2f}",
            )
        if self.budget_bytes <= 0:
            return
        if rss >= self.budget_bytes:
            self._abort(
                index,
                rss,
                f"RssAnon {rss / _GIB:.1f} GB reached the host-memory budget "
                f"{self.budget_gb:.1f} GB (cgroup limit "
                f"{self.cgroup_limit_bytes / _GIB:.0f} GB)",
            )
        # Stop early when the measured slope says the remaining segments cannot
        # fit: burning another two hours only to be SIGKILLed helps nobody. Two
        # calibrations keep this from firing on runs that would have finished:
        #   * the RSS floor ignores short smoke runs, which see a full-movie
        #     segment count but only execute a handful of segments;
        #   * the margin absorbs projection error, since the slope is fitted early
        #     and real growth decelerates (segments with no extracted entities
        #     skip archival entirely).
        projection_floor = self.budget_bytes * _env_float(
            "MAVE_IAMFLOW_PROJECTION_FLOOR_FRACTION", 0.5
        )
        projection_limit = self.budget_bytes * _env_float(
            "MAVE_IAMFLOW_PROJECTION_MARGIN", 1.15
        )
        if self._warm_index is not None and rss >= projection_floor and projected > projection_limit:
            self._abort(
                index,
                rss,
                f"projected RssAnon at segment {self.total_segments} is "
                f"{self.last_projection_gb:.0f} GB from a measured slope of "
                f"{self.last_slope_gb:.2f} GB/segment, over the "
                f"{projection_limit / _GIB:.0f} GB abort threshold "
                f"({self.budget_gb:.0f} GB budget x margin)",
            )

    def _abort(self, index: int, rss: int, detail: str) -> None:
        self.abort_reason = detail
        raise HostMemoryBudgetExceeded(
            f"iamflow_host_memory_budget_exceeded: {self.movie_id} aborted at segment "
            f"{index}/{self.total_segments}: {detail}. IAMFlow archives a "
            f"{self.kv_bytes_per_frame / (1024 ** 2):.0f} MiB KV slice per archived frame "
            f"and never evicts it; see docs/baselines/tracka_iamflow_host_memory.md."
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "rss_anon_baseline_gb": round(self.baseline_bytes / _GIB, 2),
            "rss_anon_peak_gb": round(self.peak_bytes / _GIB, 2),
            "rss_anon_slope_gb_per_segment": round(self.last_slope_gb, 3),
            "rss_anon_projected_final_gb": round(self.last_projection_gb, 1),
            "host_memory_budget_gb": round(self.budget_gb, 1),
            "cgroup_memory_limit_gb": round(self.cgroup_limit_bytes / _GIB, 1),
            "kv_bytes_per_archived_frame": self.kv_bytes_per_frame,
            "aborted": self.abort_reason,
        }


def _install_host_memory_guards(pipe) -> None:
    """Patch the live pipeline/bank instances to drop provably-dead host memory.

    Instance-level only: the vendored IAMFlow checkout stays pristine. Both
    reclaims below are numerics-exact -- neither field is ever read on a path
    that influences retrieval, scoring or generation.
    """
    bank = getattr(pipe, "agent_memory_bank", None)
    if bank is not None and not getattr(bank, "_mave_host_guarded", False):
        original_select = bank.select_frame_from_chunk

        def select_frame_from_chunk(*args, **kwargs):
            frame_id, score = original_select(*args, **kwargs)
            info = bank.frame_archive.get(frame_id)
            # FrameInfo.pixel_frame is write-only in the whole vendored repo
            # (declared + assigned, never read, absent from to_dict()), and it
            # is a *view* into the decoded pixel block, so keeping it pins the
            # entire block for the rest of the movie.
            if info is not None and getattr(info, "pixel_frame", None) is not None:
                info.pixel_frame = None
            _trim_kv_archive(bank)
            return frame_id, score

        bank.select_frame_from_chunk = select_frame_from_chunk
        bank._mave_host_guarded = True

    if not getattr(pipe, "_mave_host_guarded", False) and hasattr(pipe, "_run_sync_vae_vlm"):
        original_sync = pipe._run_sync_vae_vlm

        def _run_sync_vae_vlm(*args, **kwargs):
            result = original_sync(*args, **kwargs)
            # _sync_pixel_store is only read back through _get_chunk_pixels for
            # the eviction lag (3 chunks) and the current chunk; the vendored
            # code appends to _sync_pixel_order but never pops it.
            keep = max(4, _env_int("MAVE_IAMFLOW_PIXEL_STORE_KEEP", 8))
            order = getattr(pipe, "_sync_pixel_order", None)
            store = getattr(pipe, "_sync_pixel_store", None)
            if isinstance(order, list) and isinstance(store, dict):
                while len(order) > keep:
                    store.pop(order.pop(0), None)
            return result

        pipe._run_sync_vae_vlm = _run_sync_vae_vlm
        pipe._mave_host_guarded = True


def _trim_kv_archive(bank) -> int:
    """Opt-in cap on how many archived frames keep their KV slice in host RAM.

    OFF by default, and it is NOT a bug fix: dropping a slice makes
    ``get_memory_kv`` silently skip that frame (it tolerates ``None``), so the
    recalled memory set shrinks and the numbers stop being IAMFlow's. Enable it
    only to obtain results for movies that cannot otherwise run, and report them
    as ``IAMFlow-boundedKV(K)``, never as ``IAMFlow``.
    """
    cap = _env_int("MAVE_IAMFLOW_MAX_ARCHIVE_KV_FRAMES", 0)
    if cap <= 0:
        return 0
    store = getattr(bank, "_frame_kv_store", None)
    if not isinstance(store, dict) or len(store) <= cap:
        return 0
    protected = set(bank.frame_active_memory)
    droppable = [fid for fid in store if fid not in protected]
    # Newest last: keep the most recent `cap` slices, drop the oldest.
    droppable.sort(key=bank._frame_sort_key)
    n_drop = max(0, len(store) - cap)
    dropped = 0
    for fid in droppable[:n_drop]:
        store.pop(fid, None)
        info = bank.frame_archive.get(fid)
        if info is not None:
            info.kv_cache = None  # metadata stays, so entity recall still ranks it
        dropped += 1
    if dropped:
        bank._memory_kv_cache = None
        bank._memory_kv_cache_key = None
        bank._memory_kv_cache_device = None
    return dropped


def _http_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("IAMFLOW_HTTP_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_check_ready(base_url: str, model: str) -> None:
    req = urllib_request.Request(
        base_url.rstrip("/") + "/models",
        headers=_http_headers(),
        method="GET",
    )
    try:
        with urllib_request.urlopen(req, timeout=min(_env_float("IAMFLOW_HTTP_TIMEOUT", 900.0), 30.0)) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface endpoint failures early
        raise RuntimeError(f"IAMFlow HTTP endpoint is not ready: {base_url}") from exc

    served = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        served = [str(item.get("id", "")) for item in data["data"] if isinstance(item, dict)]
    if served and model not in served:
        print(
            f"[iamflow][warn] requested served model {model!r} not listed by {base_url}; "
            f"available={served}",
            flush=True,
        )


def _http_chat(base_url: str, model: str, messages: list, *,
               max_tokens: int, temperature: float, timeout: float | None = None) -> str:
    """Call an OpenAI-compatible chat endpoint without adding runtime deps."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if os.environ.get("IAMFLOW_DISABLE_THINKING", "1") != "0":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers=_http_headers(),
        method="POST",
    )
    timeout_s = _env_float("IAMFLOW_HTTP_TIMEOUT", 900.0) if timeout is None else timeout
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - user-supplied endpoint
            data = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"IAMFlow HTTP chat failed: url={base_url.rstrip('/')}/chat/completions "
            f"status={exc.code} body={detail}"
        ) from exc
    return (data["choices"][0]["message"]["content"] or "").strip()


def _build_http_llm(llm_wrapper_cls, base_url: str, model: str):
    """HTTP-backed drop-in for IAMFlow's text-only Qwen LLM wrapper."""

    class _HttpLLM(llm_wrapper_cls):
        def __init__(self) -> None:
            super().__init__(model_path=model, use_vllm=False)
            self._base_url = base_url
            self._served = model

        def preload(self) -> None:
            _http_check_ready(self._base_url, self._served)
            print(f"[iamflow] LLM (vLLM HTTP) ready @ {self._base_url} model={self._served}",
                  flush=True)
            return

        def _load_model(self) -> None:
            return

        def generate(self, system_prompt: str, user_prompt: str,
                     max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
            return _http_chat(
                self._base_url,
                self._served,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(int(max_new_tokens), 256),
                temperature=float(temperature),
            )

        def unload(self) -> None:
            return

    return _HttpLLM()


def _build_http_vlm(vlm_agent_cls, base_url: str, model: str, enabled: bool,
                    cache_path: "Path | None" = None):
    """HTTP-backed drop-in for IAMFlow's Qwen3-VL frame scorer."""
    import base64
    import hashlib
    import io
    import threading
    from concurrent.futures import ThreadPoolExecutor

    class _HttpVLM(vlm_agent_cls):
        def __init__(self) -> None:
            super().__init__(model_path=model, enabled=enabled, backend="vllm")
            self._base_url = base_url
            self._served = model
            self._cache_path = Path(cache_path) if cache_path else None
            self._disk_cache: dict[str, str] = {}
            self._disk_lock = threading.Lock()
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self._cache_path and self._cache_path.is_file():
                try:
                    self._disk_cache = json.loads(
                        self._cache_path.read_text(encoding="utf-8")
                    )
                except Exception:  # noqa: BLE001 - corrupt cache is non-fatal
                    self._disk_cache = {}

        def preload(self) -> None:
            if not self.enabled:
                return
            _http_check_ready(self._base_url, self._served)
            self._llm = _VLM_READY_SENTINEL
            self._executor = ThreadPoolExecutor(max_workers=1)
            print(
                f"[iamflow] VLM (vLLM HTTP) ready @ {self._base_url} "
                f"cache={'on' if self._cache_path else 'off'}",
                flush=True,
            )

        def _score_frames_vllm(self, images, score_instruction: str) -> str:
            key = None
            if self._cache_path is not None:
                h = hashlib.sha1()
                h.update(score_instruction.encode("utf-8"))
                for img in images:
                    h.update(img.tobytes())
                key = h.hexdigest()
                with self._disk_lock:
                    hit = self._disk_cache.get(key)
                if hit is not None:
                    return hit
            content = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )
            content.append({"type": "text", "text": score_instruction})
            raw = _http_chat(
                self._base_url,
                self._served,
                [{"role": "user", "content": content}],
                max_tokens=64,
                temperature=0.0,
            )
            if key is not None:
                with self._disk_lock:
                    self._disk_cache[key] = raw
                    tmp = self._cache_path.with_suffix(self._cache_path.suffix + f".tmp.{os.getpid()}")
                    tmp.write_text(json.dumps(self._disk_cache), encoding="utf-8")
                    os.replace(tmp, self._cache_path)
            return raw

        def shutdown(self) -> None:
            self._llm = None
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    return _HttpVLM()


@contextlib.contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class IAMFlowAdapter:
    name = "iamflow"

    def __init__(self, *, config_path: str = _CONFIG, ckpt_path: str = _CKPT,
                 llm_path: str = _LLM, vlm_path: str = _VLM,
                 llm_endpoint: str = "", vlm_endpoint: str = "",
                 ffmpeg: str = "ffmpeg") -> None:
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.llm_path = llm_path
        self.vlm_path = vlm_path
        # vLLM offload optional; empty => in-process HF (default, no servers needed).
        self.llm_endpoint = llm_endpoint or os.environ.get("IAMFLOW_LLM_ENDPOINT", "")
        self.vlm_endpoint = vlm_endpoint or os.environ.get("IAMFLOW_VLM_ENDPOINT", "")
        self.ffmpeg = ffmpeg
        self._pipe = None
        self._cfg = None
        self._device = None
        self._nfb = 3
        self._context_noise = 0
        self._vlm_weight = 0.3
        # global latent index -> absolute source seconds
        self._obs_seconds: list[float] = []
        self._global_lat = 0
        # (prompt_id, chunk_id) -> global start latent index (for frame_id -> src)
        self._pcid_to_start: dict[tuple[int, int], int] = {}
        self._prompt_id = 0
        self._first_prompt = True
        self._cur_cond = None
        self._llm_backend = "hf"
        self._vlm_backend = None
        self._llm_served_model = Path(self.llm_path).name
        self._vlm_served_model = Path(self.vlm_path).name
        # Host-memory guard state (see module docstring).
        self._mem_guard: _HostMemoryGuard | None = None
        self._seg_index = 0

    # ---- host-memory admission check --------------------------------------
    def preflight(self, movie: MovieContext) -> str | None:
        """Warn (or refuse) before spending hours on a movie that cannot fit.

        Returns a skip reason when refusal is enabled, else ``None``. Refusal is
        opt-in (``MAVE_IAMFLOW_PREFLIGHT_ENFORCE=1``) because the static estimate
        is an upper bound and would wrongly reject movies that do complete; the
        measured watchdog in ``observe_segment`` is the reliable gate.
        """
        n_segments = len(movie.seconds_span_by_chunk)
        if n_segments <= 0:
            return None
        kv_bytes = _kv_bytes_per_archived_frame(self._pipe)
        frames = _archived_frames_for_movie(movie, self._nfb)
        calibration = _env_float("MAVE_IAMFLOW_PREFLIGHT_CALIBRATION", 0.75)
        projected_gb = (frames * kv_bytes * calibration + _rss_anon_bytes()) / _GIB
        budget_gb = _host_memory_budget_bytes() / _GIB
        fits = budget_gb <= 0 or projected_gb <= budget_gb
        print(
            f"[iamflow][preflight] {movie.movie_id} segments={n_segments} "
            f"archived_frames<={frames} kv_mib_per_frame={kv_bytes / (1024 ** 2):.0f} "
            f"projected_anon_gb~{projected_gb:.0f} budget_gb={budget_gb:.0f} "
            f"verdict={'fits' if fits else 'OVER_BUDGET'}",
            file=sys.stderr,
            flush=True,
        )
        if fits or not _env_flag("MAVE_IAMFLOW_PREFLIGHT_ENFORCE", False):
            return None
        return (
            f"iamflow_host_memory_budget_exceeded: {movie.movie_id} needs about "
            f"{projected_gb:.0f} GB of host RAM for its KV archive "
            f"({frames} archived frames x {kv_bytes / (1024 ** 2):.0f} MiB) but only "
            f"{budget_gb:.0f} GB is available; see "
            f"docs/baselines/tracka_iamflow_host_memory.md"
        )

    # ---- load -------------------------------------------------------------
    def reset(self, movie: MovieContext) -> None:
        import torch
        from omegaconf import OmegaConf

        if not _REPO.is_dir():
            raise FileNotFoundError(f"IAMFlow checkout missing: {_REPO}")
        # Name collision: this adapter file is ``iamflow.py`` and the vendored repo
        # ships an ``iamflow`` PACKAGE. The runner imported us as sys.modules[
        # "iamflow"], which shadows the package. Yield the name to the package and
        # put the repo first on sys.path so ``import iamflow.*`` hits the real code.
        sys.modules.pop("iamflow", None)
        for p in (str(_SCRIPTS), str(_REPO)):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        os.environ.setdefault("WAN_MODEL_PATH", str(_PRETRAINED / "Wan2.1-T2V-1.3B"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(self._device)
        self._movie = movie
        self._obs_seconds = []
        self._global_lat = 0
        self._pcid_to_start = {}
        self._prompt_id = 0
        self._first_prompt = True
        self._cur_cond = None
        self._seg_index = 0
        self._mem_guard = None

        if self._pipe is not None:
            pipe = self._pipe
            self._maybe_extend_rope(movie, dev)
            pipe._reset_agent_state()
            pipe._precomputed_prompt_entities = {}
            local_attn = int(pipe.local_attn_size)
            total_lat = self._total_latents(movie)
            kv_cache_size = local_attn * _FRAME_SEQ if local_attn != -1 else total_lat * _FRAME_SEQ
            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev,
                                      kv_cache_size_override=kv_cache_size)
            pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
            pipe._initialize_kv_bank(batch_size=1, dtype=torch.bfloat16, device=dev,
                                     kv_bank1_size=pipe.bank_size * _FRAME_SEQ)
            pipe.generator.model.local_attn_size = local_attn
            pipe._set_all_modules_max_attention_size(local_attn)
            self._start_movie_guard(movie)
            return

        with _cwd(_REPO):
            from iamflow.pipelines.agent_causal_inference import AgentCausalInferencePipeline
            from iamflow.agents.vlm_agent import VLMAgent

            cfg = OmegaConf.load(self.config_path)
            OmegaConf.set_struct(cfg, False)
            # Track-A config: no fp8/vLLM runtime, no SPT blending, hard prompt
            # switch (invalidate cross-attn + reset kv-bank) — mirror the driver.
            cfg.dit_quantized = False
            cfg.dit_quantized_ckpt = None
            cfg.dit_quant_scheme = "none"
            cfg.spt_enabled = False
            cfg.transition_strategy = "no_transition"
            cfg.use_tinyvae = False
            self._cfg = cfg
            self._nfb = int(cfg.get("num_frame_per_block", 3))
            self._context_noise = int(cfg.get("context_noise", 0))
            self._vlm_weight = float(cfg.get("vlm_score_weight", 0.3))
            max_mem = int(cfg.get("max_memory_frames", 3))

            pipe = AgentCausalInferencePipeline(
                args=cfg, device=dev, llm_model_path=self.llm_path,
                max_memory_frames=max_mem,
                save_dir=str(Path(movie.work_dir) / ".iamflow_agent_frames"),
                save_frames_to_disk=False, use_vllm=False,
                vlm_model_path=self.vlm_path, vlm_enabled=False,
                async_vae_enabled=False,
            )

            # Overlay the published DiT (fp8 -> bf16). Keys are ``model.``-prefixed
            # (built from WanDiffusionWrapper.state_dict) -> load into the wrapper.
            state = _dequantize_iamflow_state_dict(self.ckpt_path)
            missing, unexpected = pipe.generator.load_state_dict(state, strict=False)
            real_unexpected = [k for k in unexpected if not k.endswith(".weight_scale")]
            real_missing = [k for k in missing if "freqs" not in k]
            if real_unexpected:
                print(f"[iamflow][warn] {len(real_unexpected)} unexpected keys "
                      f"e.g. {real_unexpected[:3]}", flush=True)
            if real_missing:
                print(f"[iamflow][warn] {len(real_missing)} missing keys "
                      f"e.g. {real_missing[:3]}", flush=True)
            pipe.generator = pipe.generator.to(dtype=torch.bfloat16)
            pipe.generator.to(device=dev)
            pipe.vae.model = pipe.vae.model.to(device=dev, dtype=torch.bfloat16)
            self._pipe = pipe

            self._maybe_extend_rope(movie, dev)

            # LLM/VLM backends. Default: in-process HF (no servers). Offload to
            # vLLM HTTP if endpoints are set (IAMFlow's published backend).
            if self.llm_endpoint:
                from iamflow.agents.llm_agent import LLMWrapper as _LLMWrapper
                served = os.environ.get("IAMFLOW_LLM_MODEL") or Path(self.llm_path).name
                self._llm_served_model = served
                http_llm = _build_http_llm(_LLMWrapper, self.llm_endpoint, served)
                pipe.llm_agent.llm = http_llm
                pipe.llm_agent.extractor.llm = http_llm
                pipe.llm_agent.id_manager.llm = http_llm
                http_llm.preload()
                self._llm_backend = "vllm_http"
            else:
                pipe.llm_agent.preload()  # keep resident for on-the-fly extraction
                self._llm_backend = "hf"

            if self.vlm_endpoint:
                served = os.environ.get("IAMFLOW_VLM_MODEL") or Path(self.vlm_path).name
                cache_env = os.environ.get("IAMFLOW_VLM_CACHE", "")
                cache_path = Path(cache_env) if cache_env else None
                self._vlm_served_model = served
                pipe.vlm_agent = _build_http_vlm(VLMAgent, self.vlm_endpoint,
                                                 served, True, cache_path=cache_path)
                self._vlm_backend = "vllm_http"
            else:
                pipe.vlm_agent = _build_transformers_vlm(VLMAgent, self.vlm_path, self._device, True)
                self._vlm_backend = "hf"
            pipe.vlm_agent.preload()
            pipe.vlm_score_weight = self._vlm_weight

            # Fresh agent state; skip precompute so entity extraction stays causal.
            pipe._reset_agent_state()
            pipe._precomputed_prompt_entities = {}

            local_attn = int(pipe.local_attn_size)
            total_lat = self._total_latents(movie)
            kv_cache_size = local_attn * _FRAME_SEQ if local_attn != -1 else total_lat * _FRAME_SEQ
            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev,
                                      kv_cache_size_override=kv_cache_size)
            pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
            pipe._initialize_kv_bank(batch_size=1, dtype=torch.bfloat16, device=dev,
                                     kv_bank1_size=pipe.bank_size * _FRAME_SEQ)
            pipe.generator.model.local_attn_size = local_attn
            pipe._set_all_modules_max_attention_size(local_attn)
            self._start_movie_guard(movie)

    def _start_movie_guard(self, movie: MovieContext) -> None:
        """Arm the host-memory guards for one movie (safe to call on every reset)."""
        pipe = self._pipe
        if pipe is None:
            return
        _install_host_memory_guards(pipe)
        self._mem_guard = _HostMemoryGuard(
            movie_id=movie.movie_id,
            total_segments=len(movie.seconds_span_by_chunk),
            kv_bytes_per_frame=_kv_bytes_per_archived_frame(pipe),
        )

    def _total_latents(self, movie: MovieContext) -> int:
        max_end = max((float(s[1]) for s in movie.seconds_span_by_chunk.values()), default=0.0)
        return int(max_end / _SECONDS_PER_LATENT) + self._nfb + 16

    def _maybe_extend_rope(self, movie: MovieContext, dev) -> None:
        import torch
        total_lat = self._total_latents(movie)
        if total_lat <= 1024:
            return
        from iamflow.vendor.wan.modules.model import rope_params
        m = self._pipe.generator.model
        d = m.dim // m.num_heads
        m.freqs = torch.cat([
            rope_params(total_lat, d - 4 * (d // 6)),
            rope_params(total_lat, 2 * (d // 6)),
            rope_params(total_lat, 2 * (d // 6)),
        ], dim=1).to(device=dev)
        print(f"[iamflow] extended RoPE table to {total_lat} positions", flush=True)

    # ---- compose (prompt switch + retrieval, BEFORE observing this chunk) ---
    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        import torch
        pipe = self._pipe
        rec = RetrievedMemory(chunk_id=req.chunk_id)
        if pipe is None:
            return rec
        self._prompt_id += 1
        pid = self._prompt_id

        if not self._first_prompt:
            # Hard prompt switch (no_transition): invalidate cross-attn + reset bank.
            for blk in pipe.crossattn_cache:
                blk["is_init"] = False
            if pipe.kv_bank1 is not None:
                for blk in pipe.kv_bank1:
                    blk["local_end_index"].zero_()
                    blk["global_end_index"].zero_()
            pipe._iam_bank_length = 0
            pipe._last_injected_memory_key = None

        # Entity extraction (on-the-fly, causal) + retrieve_initial_frames + inject.
        with _cwd(_REPO):
            pipe._process_prompt_start(prompt_text=req.prompt_text or "", prompt_id=pid,
                                       is_first_prompt=self._first_prompt)
            cond = pipe.text_encoder(text_prompts=[req.prompt_text or ""])
            cond["prompt_embeds"] = cond["prompt_embeds"].to(torch.bfloat16)
        self._cur_cond = cond
        self._first_prompt = False

        # Snapshot the active memory IAMFlow holds as of now (history only).
        bank = pipe.agent_memory_bank
        for fid in list(bank.frame_active_memory):
            src = self._frameid_to_srclatent(fid)
            if src is None or not (0 <= src < len(self._obs_seconds)):
                continue
            fi = bank.frame_archive.get(fid)
            rec.items.append(RetrievedItem(
                evidence_kind="frame",
                source_seconds=self._obs_seconds[src],
                score=(float(fi.score) if fi and fi.score is not None else None),
                raw_ref=f"iamflow:{fid}",
            ))
        return rec

    def _frameid_to_srclatent(self, frame_id: str) -> int | None:
        m = _FRAME_ID_RE.match(frame_id)
        if not m:
            return None
        pid, cid, f = int(m.group(1)), int(m.group(2)), int(m.group(3))
        start = self._pcid_to_start.get((pid, cid))
        return None if start is None else start + f

    # ---- observe (write this segment's real video into memory) --------------
    def observe_segment(self, obs: SegmentObservation) -> None:
        import torch
        pipe = self._pipe
        if self._cur_cond is None:
            # compose() must run first (runner guarantees this); defensive no-op.
            return
        pixel, _n = read_segment_pixels(obs.segment_video, ffmpeg=self.ffmpeg)
        pixel = pixel.to(self._device)
        t0 = float(obs.seconds_span[0])
        dev = torch.device(self._device)
        npb = self._nfb
        base_global = self._global_lat

        with _cwd(_REPO), torch.no_grad():
            latents = pipe.vae.encode_to_latent(pixel.to(torch.bfloat16))
            if latents.dim() == 5:
                latents = latents[0]  # [T,C,H,W]
            T = int(latents.shape[0])
            # Only whole npb-blocks are committed to the KV cache; the <npb tail is
            # dropped (mirrors the driver). ``_global_lat`` must count ONLY committed
            # frames so ``current_start`` stays contiguous with the cache — otherwise
            # a chunk whose T is not a multiple of npb leaves a gap and the next
            # chunk's write misaligns (1560-vs-4680 KV size error).
            n_blocks = T // npb
            committed = n_blocks * npb
            for li in range(committed):
                self._obs_seconds.append(t0 + latent_local_seconds(li, obs.fps))

            for b in range(n_blocks):
                start = b * npb
                block = latents[start:start + npb].unsqueeze(0).to(device=dev, dtype=torch.bfloat16)
                current_start_frame = base_global + start

                # VLM submit (sync): decode block -> pixels -> score chunk c+1.
                if getattr(pipe, "vlm_agent", None) is not None:
                    chunk_key = f"p{pipe.current_prompt_id}_c{pipe.current_chunk_id + 1}"
                    pipe._run_sync_vae_vlm(
                        denoised_pred=block, chunk_key=chunk_key,
                        prompt_text=pipe.current_prompt_text,
                        prompt_id=pipe.current_prompt_id,
                        chunk_id=pipe.current_chunk_id + 1,
                        entities=pipe.current_entities,
                        is_first_chunk=(pipe.current_chunk_id == 0),
                    )

                pipe.current_chunk_id += 1
                self._pcid_to_start[(pipe.current_prompt_id, pipe.current_chunk_id)] = current_start_frame

                # Eviction (lag>=4) BEFORE the clean commit, mirroring inference().
                did_evict = False
                if pipe.current_chunk_id >= 4 and pipe.current_entities:
                    pipe._process_chunk_eviction(current_start_frame=current_start_frame,
                                                 current_num_frames=npb)
                    pipe._inject_iam_memory_to_bank()
                    did_evict = True

                context_ts = torch.ones([1, npb], device=dev, dtype=torch.int64) * self._context_noise
                pipe.generator(
                    noisy_image_or_video=block,
                    conditional_dict=self._cur_cond,
                    timestep=context_ts,
                    kv_cache=pipe.kv_cache1,
                    kv_bank=pipe.kv_bank1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=current_start_frame * _FRAME_SEQ,
                    update_bank=False,
                    q_bank=False,
                    update_cache=True,
                    iam_bank_length=pipe._iam_bank_length,
                    prev_crossattn_cache=pipe.prev_crossattn_cache,
                    transition_alpha=None,
                )

                # Archival AFTER the commit (only if we did not evict).
                if not did_evict and pipe.current_chunk_id >= 1 and pipe.current_entities:
                    pipe._process_chunk_archival(current_start_frame)
                    pipe._inject_iam_memory_to_bank()
                    if pipe.vlm_agent is not None and pipe.current_chunk_id == 2:
                        corr = pipe.vlm_agent.get_attribute_corrections(pipe.current_prompt_id)
                        if corr:
                            pipe.agent_memory_bank.apply_attribute_corrections(
                                pipe.current_prompt_id, corr)

        self._global_lat += committed
        # Release this segment's decoded pixels before the guard samples RssAnon,
        # so the reading reflects retained memory rather than in-flight buffers.
        del pixel, latents
        self._seg_index += 1
        if self._mem_guard is not None:
            self._mem_guard.observe(self._seg_index)

    def finalize(self) -> dict[str, Any]:
        pipe = self._pipe
        bank = getattr(pipe, "agent_memory_bank", None) if pipe else None
        host_memory: dict[str, Any] = {
            "n_archived_frames": len(getattr(bank, "frame_archive", {}) or {}) if bank else 0,
            "n_kv_slices_resident": len(getattr(bank, "_frame_kv_store", {}) or {}) if bank else 0,
            "max_archive_kv_frames_cap": _env_int("MAVE_IAMFLOW_MAX_ARCHIVE_KV_FRAMES", 0),
        }
        if self._mem_guard is not None:
            host_memory.update(self._mem_guard.telemetry())
        return {
            "host_memory": host_memory,
            "retrieval": "iamflow_entity_active_memory_on_self_encoded_real_segments",
            "vlm_score_weight": self._vlm_weight,
            "num_frame_per_block": self._nfb,
            "local_attn_size": int(pipe.local_attn_size) if pipe else None,
            "bank_size": int(pipe.bank_size) if pipe else None,
            "max_memory_frames": int(pipe.max_memory_frames) if pipe else None,
            "llm_backend": self._llm_backend,
            "vlm_backend": self._vlm_backend,
            "llm_endpoint": self.llm_endpoint or None,
            "vlm_endpoint": self.vlm_endpoint or None,
            "llm_served_model": self._llm_served_model,
            "vlm_served_model": self._vlm_served_model,
            "n_observed_latents": len(self._obs_seconds),
        }


def build_adapter() -> IAMFlowAdapter:
    return IAMFlowAdapter()
