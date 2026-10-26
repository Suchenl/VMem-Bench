"""MemFlow causal adapter (new protocol, real segments) — faithful KV-bank trace.

Two published-config variants share this file (``sma`` flag), so the baseline table
gets two rows, MemFlow (w/o SMA) and MemFlow (with SMA), matching MemFlow's own
``SMA`` switch (``causal_model.py:462``):

* **w/o SMA** (``sma=False``, the shipped ``configs/inference.yaml`` default):
  attention pool per step = ``[sink | bank | local]`` and the WHOLE bank is used.
  Native memory = ``sink`` (first ``sink_size`` latent frames) + ``local window``
  (last ``local_attn_size`` frames, recency) + ``KV bank`` (<= ``bank_size``
  historical blocks chosen by MemFlow's per-layer text-saliency top-k in
  ``compress_kv_bank``). ``compress_kv_bank`` copies KV blocks **verbatim**, so
  every surviving 1560-token bank block is bit-identical to the ``k_new`` block
  committed for a specific source latent frame; we fingerprint each committed
  frame's per-layer key and match live bank blocks back to their source latent
  index (per-layer + vote-aggregated). Not recency: MemFlow's real text-saliency.

* **with SMA** (``sma=True``): attention pool per step = ``[φ-routed top-k of
  (sink∪bank) | local]``. ``dynamic_topk_routing_attention`` mean-pools each
  candidate chunk into a **compact descriptor φ** and selects top-k by ``φ_q·φ_k``
  (``causal_model.py:104-187``). We turn SMA on for every self-attn module and
  hook that routing during the real forward, fingerprinting the φ-selected chunks
  back to their source latent index (the same verbatim-KV fingerprint trace). The
  routing query is the real chunk being observed (the bench substitutes the real
  segment for the generator output), and we keep only sources strictly older than
  the current chunk, so the retrieved set stays causal.

We drive MemFlow on the **real segment**: VAE-encode it and run the clean-context
forward (``context_noise=0``, ``q_bank=True``) block by block so the KV cache +
bank are populated from real frames (no denoising / generation), conditioned on
each chunk's own prompt (text saliency is prompt-conditioned). The vendored repo
stays pristine; all glue lives here. Retrieval maps sink/local/bank/routed source
latents to absolute source seconds for frame materialization.

Env: torch 2.6 + flash-attn 2.6 — verified to run the forward.
  baselines/Causal/MemFlow/wan_models/Wan2.1-T2V-1.3B -> Wan-AI/Wan2.1-T2V-1.3B
  baselines/Causal/MemFlow/checkpoints/{base.pt,lora.pt} -> KlingTeam/MemFlow
Run via runner.py --adapter memflow.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _video_io import latent_local_seconds

_REPO = Path(__file__).resolve().parents[7] / "baselines" / "Causal" / "MemFlow"
_CONFIG = "configs/inference.yaml"
_FRAME_SEQ = 1560  # tokens per latent frame (Wan 1.3B)
_SECONDS_PER_LATENT = 0.25  # VAE temporal stride 4 / 16 fps


@contextlib.contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _fingerprint(block) -> tuple:
    """Collision-safe fingerprint of a [1560, H, D] key block (verbatim-copied)."""
    b = block.float()
    L, H, D = b.shape
    return (
        round(float(b.sum()), 3),
        round(float((b * b).sum()), 3),
        round(float(b[0, 0, 0]), 5),
        round(float(b[L // 2, H // 2, D // 2]), 5),
        round(float(b[L - 1, H - 1, D - 1]), 5),
    )


def _find_causal(m):
    q = collections.deque([m])
    seen: set[int] = set()
    while q:
        o = q.popleft()
        if type(o).__name__ == "CausalWanModel":
            return o
        if id(o) in seen:
            continue
        seen.add(id(o))
        import torch
        for attr in ("base_model", "model"):
            child = getattr(o, attr, None)
            if isinstance(child, torch.nn.Module):
                q.append(child)
    raise RuntimeError("could not locate underlying CausalWanModel under peft wrapping")


class MemFlowAdapter:
    def __init__(self, *, sma: bool = False, config_path: str = _CONFIG,
                 ffmpeg: str = "ffmpeg") -> None:
        self.sma = bool(sma)
        self.name = "memflow_sma" if self.sma else "memflow"
        self.config_path = config_path
        self.ffmpeg = ffmpeg
        self._pipe = None
        self._cfg = None
        self._device = None
        self._nfb = 3
        self._local = 12
        self._sink = 3
        self._bank = 3
        self._n_layers = 0
        # global latent index -> absolute source seconds
        self._lat_seconds: list[float] = []
        self._global_lat = 0
        # per-layer fingerprint(k_new) -> global source latent index
        self._committed_fp: list[dict] = []
        # --- SMA-only routing-capture state (unused when sma=False) ---
        import collections as _c
        self._patched = False
        self._capture_on = False
        self._routing_capture = _c.Counter()
        self._merged_fp: dict = {}
        self._chunk_start_global = 0
        self._pending = None  # RetrievedMemory filled during observe (SMA path)
        self._sma_modules = 0

    # ---- load -------------------------------------------------------------
    def reset(self, movie: MovieContext) -> None:
        import torch
        from omegaconf import OmegaConf

        if not _REPO.is_dir():
            raise FileNotFoundError(f"MemFlow checkout missing: {_REPO}")
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._movie = movie
        self._lat_seconds = []
        self._global_lat = 0

        with _cwd(_REPO):
            if self._pipe is None:
                cfg = OmegaConf.load(self.config_path)
                self._cfg = cfg
                self._nfb = int(cfg.get("num_frame_per_block", 3))
                mk = cfg.get("model_kwargs", {})
                self._local = int(mk.get("local_attn_size", 12))
                self._sink = int(mk.get("sink_size", 3))
                self._bank = int(mk.get("bank_size", 3))
                self._context_noise = int(cfg.get("context_noise", 0))

                pipe = self._build_pipeline(cfg)
                self._pipe = pipe
                self._n_layers = int(pipe.num_transformer_blocks)
                if self.sma:
                    self._enable_sma()
                    self._install_routing_hook()

            self._committed_fp = [dict() for _ in range(self._n_layers)]
            self._maybe_extend_rope(movie)
            self._init_caches()

    def _enable_sma(self) -> None:
        """Flip every self-attn module to the φ-routing (SMA) read path."""
        real_model = _find_causal(self._pipe.generator.model)
        cnt = 0
        for m in real_model.modules():
            if hasattr(m, "SMA"):
                m.SMA = True
                cnt += 1
        self._sma_modules = cnt
        if cnt == 0:
            raise RuntimeError("SMA requested but no module exposes an `SMA` flag")

    def _install_routing_hook(self) -> None:
        """Class-level monkeypatch that records φ-routed chunk selections.

        The vendored ``dynamic_topk_routing_attention`` still runs verbatim; we only
        observe its inputs. The active adapter is stored on the class so a fresh run
        (new adapter instance) rebinds cleanly without stacking wrappers.
        """
        import wan.modules.causal_model as cm

        attn_cls = cm.CausalWanSelfAttention
        attn_cls._memflow_adapter = self
        if getattr(attn_cls.dynamic_topk_routing_attention, "_memflow_wrapped", False):
            self._patched = True
            return
        orig = attn_cls.dynamic_topk_routing_attention

        def patched(self_attn, query, key, value, chunk_size, top_k):
            import torch

            # Vendored MemFlow comments say this path should "handle padding",
            # but the implementation asserts when sink+bank is not an exact
            # multiple of a latent-frame chunk (1560 tokens). Real Track-A
            # segments can hit that partial global pool, so pad only inside the
            # adapter wrapper and leave the vendored checkout untouched.
            rem = int(key.shape[1]) % int(chunk_size)
            if rem:
                pad = int(chunk_size) - rem
                key_for_route = torch.cat(
                    [key, key.new_zeros((key.shape[0], pad, key.shape[2], key.shape[3]))],
                    dim=1,
                )
                value_for_route = torch.cat(
                    [value, value.new_zeros((value.shape[0], pad, value.shape[2], value.shape[3]))],
                    dim=1,
                )
            else:
                key_for_route = key
                value_for_route = value
            out = orig(self_attn, query, key_for_route, value_for_route, chunk_size, top_k)
            ad = getattr(cm.CausalWanSelfAttention, "_memflow_adapter", None)
            if ad is not None and ad._capture_on:
                try:
                    ad._record_routing(query, key_for_route, chunk_size, top_k)
                except Exception:  # noqa: BLE001 - capture must never break the forward
                    pass
            return out

        patched._memflow_wrapped = True
        attn_cls.dynamic_topk_routing_attention = patched
        self._patched = True

    def _record_routing(self, query, key, chunk_size, top_k) -> None:
        """Replicate the φ selection and fingerprint the chosen chunks -> source."""
        import torch

        with torch.no_grad():
            B, Lq, H, D = query.shape
            num_chunks = key.shape[1] // chunk_size
            if num_chunks == 0:
                return
            kc = key.view(B, num_chunks, chunk_size, H, D)
            phi_q = query.mean(dim=1, keepdim=True).permute(0, 2, 1, 3)      # [B,H,1,D]
            phi_k = kc.mean(dim=2).permute(0, 2, 1, 3)                        # [B,H,nc,D]
            rel = torch.matmul(phi_q, phi_k.transpose(-2, -1)).squeeze(2)     # [B,H,nc]
            ksel = min(int(top_k), num_chunks)
            _, idx = torch.topk(rel, k=ksel, dim=-1)                         # [B,H,ksel]
            for c in torch.unique(idx).tolist():
                blk = key[0, c * chunk_size:(c + 1) * chunk_size]           # [1560,H,D]
                if float(blk.abs().sum()) == 0.0:
                    continue
                src = self._merged_fp.get(_fingerprint(blk))
                if src is not None and 0 <= int(src) < self._chunk_start_global:
                    self._routing_capture[int(src)] += 1

    def _build_pipeline(self, cfg):
        import peft
        import torch
        from pipeline import CausalInferencePipeline
        from utils.lora_utils import configure_lora_for_model

        pipe = CausalInferencePipeline(cfg, device=torch.device(self._device))
        sd = torch.load(cfg.generator_ckpt, map_location="cpu")
        raw = sd.get("model") or sd.get("generator") or sd
        pipe.generator.load_state_dict(raw, strict=False)
        if getattr(cfg, "adapter", None):
            pipe.generator.model = configure_lora_for_model(
                pipe.generator.model, model_name="generator",
                lora_config=cfg.adapter, is_main_process=True)
            lc = torch.load(cfg.lora_ckpt, map_location="cpu")
            state = lc["generator_lora"] if isinstance(lc, dict) and "generator_lora" in lc else lc
            peft.set_peft_model_state_dict(pipe.generator.model, state)
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.generator.to(self._device)
        pipe.vae.to(self._device)
        return pipe

    def _maybe_extend_rope(self, movie: MovieContext) -> None:
        """Extend the temporal RoPE table to cover the whole film (exact, not approx)."""
        import torch

        max_end = max((float(s[1]) for s in movie.seconds_span_by_chunk.values()), default=0.0)
        total_lat = int(max_end / _SECONDS_PER_LATENT) + self._nfb + 16
        if total_lat <= 1024:
            return
        real_model = _find_causal(self._pipe.generator.model)
        from wan.modules.model import rope_params

        dh = real_model.dim // real_model.num_heads
        real_model.freqs = torch.cat([
            rope_params(total_lat, dh - 4 * (dh // 6)),
            rope_params(total_lat, 2 * (dh // 6)),
            rope_params(total_lat, 2 * (dh // 6)),
        ], dim=1).to(torch.device(self._device))

    def _init_caches(self) -> None:
        import torch

        pipe = self._pipe
        dev = torch.device(self._device)
        kv_cache_size = self._local * _FRAME_SEQ if self._local != -1 else 4096 * _FRAME_SEQ
        pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev,
                                  kv_cache_size_override=kv_cache_size)
        pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
        pipe._initialize_kv_bank(batch_size=1, dtype=torch.bfloat16, device=dev,
                                 kv_bank1_size=pipe.bank_size * _FRAME_SEQ)
        pipe.generator.model.local_attn_size = self._local
        pipe._set_all_modules_max_attention_size(self._local)

    # ---- observe ----------------------------------------------------------
    def observe_segment(self, obs: SegmentObservation) -> None:
        import torch
        from _video_io import read_segment_pixels

        pipe = self._pipe
        pixel, _n = read_segment_pixels(obs.segment_video, ffmpeg=self.ffmpeg)
        pixel = pixel.to(self._device)
        t0 = float(obs.seconds_span[0])
        if self.sma:
            # Capture φ routing over THIS chunk's real forward. Pool = history only:
            # bank/sink hold chunks < current at forward time, and we filter routed
            # sources to < chunk_start below, so the retrieved set stays causal.
            import collections as _c
            self._chunk_start_global = self._global_lat
            self._merged_fp = {}
            for li in range(self._n_layers):
                self._merged_fp.update(self._committed_fp[li])
            self._routing_capture = _c.Counter()
            self._capture_on = True
        with torch.no_grad():
            latents = pipe.vae.encode_to_latent(pixel.to(torch.bfloat16))[0]  # [T,C,H,W]
            T = int(latents.shape[0])
            # The vendored Wan causal model expects full num_frame_per_block chunks.
            # Track-A tail segments can decode to a partial latent block; do not
            # commit that tail into KV/cache state or RoPE dimensions can misalign.
            n_blocks = T // self._nfb
            committed = n_blocks * self._nfb
            # Text-saliency bank compression is prompt-conditioned: use THIS chunk's
            # prompt (the prompt the real generator would have used for this chunk).
            cond = pipe.text_encoder(text_prompts=[obs.prompt_text or ""])
            for blk in pipe.crossattn_cache:
                blk["is_init"] = False
            for start in range(0, committed, self._nfb):
                block = latents[start:start + self._nfb].unsqueeze(0).to(torch.bfloat16)
                nf = int(block.shape[1])
                g_start = self._global_lat + start
                ts = torch.ones([1, nf], dtype=torch.int64, device=self._device) * self._context_noise
                pipe.generator(
                    noisy_image_or_video=block,
                    conditional_dict=cond,
                    timestep=ts,
                    kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=g_start * _FRAME_SEQ,
                    kv_bank=pipe.kv_bank1,
                    update_bank=True,
                    q_bank=True,
                    update_cache=True,
                )
                # Fingerprint the frame this commit pushed toward the bank (k_new =
                # first frame of the block) -> global source latent index.
                for li in range(self._n_layers):
                    knew = pipe.kv_bank1[li]["k_new"][0]  # [1560,H,D]
                    if float(knew.abs().sum()) != 0.0:
                        self._committed_fp[li][_fingerprint(knew)] = g_start
            for li in range(committed):
                self._lat_seconds.append(t0 + latent_local_seconds(li, obs.fps))
        self._global_lat += committed
        if self.sma:
            self._capture_on = False
            self._fill_pending_sma()

    def _fill_pending_sma(self) -> None:
        """Fill the compose() rec (mutated by reference) with SMA retrieval:
        ``[φ-routed historical chunks] ∪ [local recency window before this chunk]``."""
        rec = self._pending
        if rec is None:
            return
        cs = self._chunk_start_global
        n = len(self._lat_seconds)
        chosen: dict[int, str] = {}
        for i in range(max(0, cs - self._local), cs):        # local recency window
            chosen[i] = "local"
        for gi in sorted(self._routing_capture):             # φ-routed historical
            chosen[gi] = "routed"
        for gi, kind in sorted(chosen.items()):
            if 0 <= gi < n:
                rec.items.append(RetrievedItem(
                    evidence_kind="kv", source_seconds=self._lat_seconds[gi],
                    raw_ref=f"memflow_sma:{kind}:lat{gi}"))
        self._pending = None

    def _snapshot_bank(self) -> list[int]:
        """Vote-aggregated source latent indices currently in the KV bank."""
        pipe = self._pipe
        votes: dict[int, int] = {}
        for li in range(self._n_layers):
            bank_k = pipe.kv_bank1[li]["k"]  # [1, bank*1560, H, D]
            fps = self._committed_fp[li]
            n_blocks = bank_k.shape[1] // _FRAME_SEQ
            seen: set[int] = set()
            for bi in range(n_blocks):
                blk = bank_k[0, bi * _FRAME_SEQ:(bi + 1) * _FRAME_SEQ]
                if float(blk.abs().sum()) == 0.0:
                    continue
                src = fps.get(_fingerprint(blk))
                if src is not None:
                    seen.add(int(src))
            for s in seen:
                votes[s] = votes.get(s, 0) + 1
        return sorted(votes)

    # ---- compose ----------------------------------------------------------
    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        rec = RetrievedMemory(chunk_id=req.chunk_id)
        if self.sma:
            # SMA retrieval is query-dependent, so it is captured during the real
            # forward in observe_segment(); we hand back this rec now and fill it
            # by reference there (runner materializes after the whole loop).
            self._pending = rec
            return rec
        n = len(self._lat_seconds)
        if n == 0:
            return rec
        chosen: dict[int, str] = {}
        for i in range(min(self._sink, n)):
            chosen[i] = "sink"
        for i in range(max(0, n - self._local), n):
            chosen.setdefault(i, "local")
        for gi in self._snapshot_bank():
            if 0 <= gi < n:
                chosen.setdefault(gi, "bank")
        for gi, kind in sorted(chosen.items()):
            rec.items.append(RetrievedItem(
                evidence_kind="kv", source_seconds=self._lat_seconds[gi],
                raw_ref=f"memflow:{kind}:lat{gi}"))
        return rec

    def finalize(self) -> dict[str, Any]:
        if self.sma:
            retrieval = ("memflow_SMA_phi_topk_routing(dynamic_topk_routing_attention)"
                         "+local, fingerprint_trace_on_real_latents")
        else:
            retrieval = "memflow_sink+local+kv_bank_fingerprint_trace_on_real_latents"
        return {
            "variant": "with_SMA" if self.sma else "without_SMA",
            "sma": self.sma, "sma_modules_flipped": self._sma_modules,
            "retrieval": retrieval,
            "local_attn_size": self._local, "sink_size": self._sink,
            "bank_size": self._bank, "num_frame_per_block": self._nfb,
            "n_layers": self._n_layers, "n_observed_latents": len(self._lat_seconds),
        }


def build_adapter() -> MemFlowAdapter:
    return MemFlowAdapter(sma=False)
