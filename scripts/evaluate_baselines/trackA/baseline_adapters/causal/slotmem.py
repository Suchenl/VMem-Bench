"""SlotMem causal adapter (new protocol, real segments).

SlotMem's native memory is **role-wise slot memory**: per recurring character it keeps
compact memory slots (``RoleWiseSlotMemoryBank`` in ``infer_slotmem.py``), updated during
autoregressive generation and injected into that character's localized tokens. Under the
causal bench protocol the intended mapping is:

* ``observe_segment`` (memory WRITE): VAE-encode the real segment, run SlotMem's own
  slot extraction to produce per-role slot tokens, and ``add_memory(char_id, tokens,
  source_chunk_idx=cid, ...)`` into ``RoleWiseSlotMemoryBank`` — tagging each write with
  the source chunk / seconds.
* ``compose`` (memory READ): read the bank per role and return each role's contributing
  source chunks as temporal :class:`RetrievedItem` (mapped to source seconds), for the
  bench-side ``frame_materializer``. (Read path below is implemented.)

Status: **PORTED — native Wan2.2 base only, single teacher-forced memory probe**
-------------------------------------------------------------------------------
Earlier notes called this "BLOCKED / protocol-forbidden". That was wrong. Two corrections:

1. **Slot extraction is a single DiT-forward attention probe, not multi-step generation.**
   ``_extract_memory_from_current_step`` (infer_slotmem.py ~L2109) registers attention
   hooks (``AttentionMapExtractorV8`` + feature taps), runs **one** ``run_native_dit_forward``
   at a chosen timestep over ``noisy_latents``, then turns the attention maps into slots
   (``_extract_memory_from_step_maps``). Native code merely *calls* it inside
   ``generate_chunk``'s denoising loop; the extraction itself is a single forward. So it
   teacher-forces exactly like MemFlow/IAMFlow: VAE-encode the real segment, add noise at a
   chosen timestep, run **one** forward with the probe hooks, read out slots — **no
   multi-step denoising / no video generation**. TrackA still uses the native Wan2.2 base;
   the distilled/lightx2v base is forbidden because its SlotMem-LoRA visual smoke is unstable.
2. **Localization is by the character-NAME token in the prompt, not a roster.**
   ``_extract_memory_from_current_step`` returns ``None`` unless ``character_name`` appears
   in the prompt, then uses ``find_token_index_in_prompt(prompt, character_name)`` to pick
   the character's cross-attention region. Our ``name_anchored`` prompt already contains the
   names, so no roster is needed; ``char_latent_boxes`` is an *optional* refinement (default
   ``None``). No protocol conflict.

* **Env (available):** SlotMem uses its **vendored** ``diffsynth`` (via ``sys.path``) +
  ``flash_attn==2.8.0.post2`` (see ``requirements_slotmem.txt``); it does **not** use
  lightx2v. The ``vace`` conda env has ``flash_attn 2.8.3`` + ``torch 2.5.1+cu124`` and is
  the right place to run it (the ``slotmem`` env has torch 2.7.1 but no flash_attn built).
* **Weights (present):** base ``Wan-AI/Wan2.2-I2V-A14B`` + SlotMem LoRA
  ``Causal_Video_Generation/SlotMem/ckpt/{stage1,stage2}/{stage*_high,stage*_low}.pt``.
* **Do not use distilled Wan2.2 for Track A.** A distilled Wan2.2 + SlotMem LoRA smoke
  can load via a symlink layout and emits a video, but visual quality is unstable
  (smearing, blocky/checkerboard background, geometry drift). Formal SlotMem numbers
  must therefore use the native Wan2.2-I2V-A14B base plus SlotMem's own LoRA/encoder.
* **Use 10 denoising steps when invoking SlotMem generation/probe scheduler configs.**
  A native smoke on H800 with stage1+stage2 LoRA showed 10 steps gives stable video,
  while 5 steps is only a plumbing check (blur/smear/detail loss).

Run via runner.py --adapter slotmem in the ``vace`` env.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _video_io import read_segment_pixels

_REPO = Path(__file__).resolve().parents[7] / "baselines" / "Causal" / "SlotMem"
# Wan2.2 VAE temporal stride / native fps (i2v A14B) → seconds per latent frame.
_SECONDS_PER_LATENT = 0.25


class SlotMemRuntimeError(RuntimeError):
    """Raised for SlotMem adapter setup/runtime failures."""


@contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class SlotMemAdapter:
    name = "slotmem"

    def __init__(
        self,
        *,
        ckpt_dir: str = "${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B",
        lora_ckpt: str = "${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/SlotMem/ckpt",
        ffmpeg: str = "ffmpeg",
        max_memory_characters: int = 2,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.lora_ckpt = lora_ckpt
        self.ffmpeg = ffmpeg
        self.max_memory_characters = int(max_memory_characters)
        self._mem_manager = None
        self._engine = None
        self._slotmem_mod = None
        self._movie: MovieContext | None = None
        self._args = None
        self._last_segment_first_frame_pil = None
        # char_id -> {source_chunk_idx: absolute source seconds} (temporal-identity table).
        self._chunk_seconds: dict[int, float] = {}

    def reset(self, movie: MovieContext) -> None:
        if not _REPO.is_dir():
            raise FileNotFoundError(f"SlotMem checkout missing: {_REPO}")
        if "lightx2v" in str(self.ckpt_dir).lower() or "distill" in str(self.ckpt_dir).lower():
            raise SlotMemRuntimeError(
                "SlotMem TrackA adapter must use native Wan2.2-I2V-A14B, not the distilled "
                "Wan2.2/lightx2v layout. The distilled+SlotMem-LoRA smoke loads but has "
                "unusable visual quality."
            )
        self._movie = movie
        self._chunk_seconds = {}

        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        import infer_slotmem as slotmem_mod  # type: ignore

        self._slotmem_mod = slotmem_mod
        self._mem_manager = slotmem_mod.RoleWiseSlotMemoryBank()
        self._load_engine()

    def _load_engine(self) -> None:
        slotmem_mod = self._slotmem_mod
        if slotmem_mod is None:
            raise SlotMemRuntimeError("SlotMem module not imported")
        args = self._build_engine_args()
        self._args = args
        with _cwd(_REPO):
            self._engine = slotmem_mod.SlotMemInferenceEngine(args)

    def _slotmem_ckpt_paths(self) -> dict[str, str]:
        root = Path(self.lora_ckpt)
        if root.name in {"stage1", "stage2"}:
            root = root.parent
        paths = {
            "stage1_low": root / "stage1" / "stage1_low.pt",
            "stage1_high": root / "stage1" / "stage1_high.pt",
            "stage2_low": root / "stage2" / "stage2_low.pt",
            "stage2_high": root / "stage2" / "stage2_high.pt",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing SlotMem checkpoint(s): " + ", ".join(missing))
        return {key: str(path) for key, path in paths.items()}

    def _build_engine_args(self):
        """Build the same SlotMem stage2 inference args used by test_slotmem_stage2.sh."""
        slotmem_mod = self._slotmem_mod
        if slotmem_mod is None:
            raise SlotMemRuntimeError("SlotMem module not imported")

        ckpts = self._slotmem_ckpt_paths()
        argv = [
            "slotmem_tracka_adapter",
            "--ckpt_dir", str(self.ckpt_dir),
            "--json_path", str(_REPO / "dummy_tracka_adapter.json"),
            "--output_path", str(_REPO / "inference_outputs" / "tracka_adapter"),
            "--train_noise_domain", "low_noise",
            "--noise_domain_boundary_ratio", "0.9",
            # H800 smoke: 10 steps is visually stable; 5 steps is blurry/smeared.
            "--num_inference_steps", "10",
            "--cfg_scale", "5.0",
            "--cfg_scale_extraction", "5.0",
            "--height", "480",
            "--width", "832",
            "--context_frames", "81",
            "--num_overlap_frame", "5",
            "--dual_expert_load_mode", os.environ.get("SLOTMEM_DUAL_EXPERT_LOAD_MODE", "active"),
            "--dual_expert_offload_dtype", os.environ.get("SLOTMEM_DUAL_EXPERT_OFFLOAD_DTYPE", "bfloat16"),
            "--dual_expert_vram_limit", os.environ.get("SLOTMEM_DUAL_EXPERT_VRAM_LIMIT", "-1"),
            "--no-dual_expert_manage_aux_models",
            "--high_expert_checkpoint_path", f"{ckpts['stage1_high']},{ckpts['stage2_high']}",
            "--low_expert_checkpoint_path", f"{ckpts['stage1_low']},{ckpts['stage2_low']}",
            "--extract_layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "--role_token_selection_mode", "two_role_diff",
            "--top_visual_tokens", "0.1",
            "--token_weight", "1.0",
            "--suffix_attention_scale", "1.0",
            "--max_memory_tokens_per_character", "512",
            "--use_attn_score_selection",
            "--max_memory_characters", str(max(1, self.max_memory_characters)),
            "--slotmem_memory_bank_mode", "single",
            "--neighbor_filter_kernel", "5",
            "--neighbor_filter_any_window",
            "--lora_rank", "128",
            "--lora_alpha", "128",
            "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
            "--sparse_role_memory_layer_idx", "3",
            "--sparse_role_memory_injection_layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "--memory_layer_binding_mode", "layerwise",
            "--char_attn_noise_scope", "low_noise",
            "--sparse_role_memory_num_heads", "8",
            "--sparse_role_memory_head_dim", "128",
            "--sparse_role_memory_rope_dim", "256",
            "--sparse_role_memory_use_half_role_heads",
            "--sparse_role_memory_feature_source", "attn_out",
            "--sparse_role_memory_init_scale", "0.1",
            "--sparse_role_memory_time_gate",
            "--slotmem_memory_encoder_mode", "on",
            "--slotmem_memory_encoder_layers", "0-15",
            "--slotmem_memory_encoder_layer_groups", "0-4,5-10,11-15",
            "--slotmem_memory_encoder_slots", "64",
            "--slotmem_memory_encoder_dim", "512",
            "--slotmem_memory_encoder_hidden_dim", "1024",
            "--slotmem_memory_encoder_use_t_embed",
            "--slotmem_memory_encoder_use_slot_index_embed",
            "--train_stage", "stage2",
            "--slotmem_memory_writer_mode", "auto",
            "--memory_runtime_log_every", "1",
            "--no-save_denoise_step_edge_viz",
            "--enable_sparse_role_memory_attn",
            "--max_chunks", "-1",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            return slotmem_mod.parse_args()
        finally:
            sys.argv = old_argv

    def observe_segment(self, obs: SegmentObservation) -> None:
        cid = int(obs.chunk_id)
        self._chunk_seconds[cid] = float(obs.seconds_span[0])
        role_slots = self._extract_role_slots(obs)
        for char_id, banks in role_slots.items():
            for bank_idx, payload in banks.items():
                if isinstance(payload, tuple) and len(payload) == 2:
                    tokens, token_meta = payload
                else:
                    tokens, token_meta = payload, []
                stored_mem, stored_meta, _stats = self._stage2_prepare_payload_for_bank(
                    str(char_id),
                    int(bank_idx),
                    tokens,
                    token_meta,
                )
                self._mem_manager.add_memory(
                    str(char_id),
                    stored_mem,
                    bank_idx=int(bank_idx),
                    token_meta=stored_meta,
                    source_chunk_idx=cid,
                )

    def _candidate_role_names(self, prompt: str) -> list[str]:
        """Infer SlotMem role strings from the SUT-visible prompt text only."""
        import re

        override = os.environ.get("SLOTMEM_TRACKA_ROLE_NAMES", "").strip()
        candidates: list[str] = []
        if override:
            candidates.extend([x.strip() for x in override.split(",") if x.strip()])

        # description_provided prompts append "name: appearance"; consume only names present
        # in that prompt text, not any hidden benchmark registry.
        for name in re.findall(r"(?:^|[;\n，。])\s*([^:：;\n，。]{2,48})\s*[:：]", prompt):
            name = name.strip(" -[](){}\"'")
            if name and name.lower() not in {"prompt", "description", "appearance"}:
                candidates.append(name)

        for match in re.findall(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+){0,3}\b", prompt):
            if match.lower() not in {"The".lower(), "A".lower(), "An".lower()}:
                candidates.append(match)

        # Common lower-case descriptive names used in SlotMem-style prompts.
        role_nouns = (
            "man", "woman", "boy", "girl", "person", "child", "keeper", "captain",
            "soldier", "teacher", "doctor", "rabbit", "cat", "dog", "bird", "fox",
        )
        noun_alt = "|".join(role_nouns)
        for match in re.findall(rf"\b(?:[a-z]+[- ]){{0,3}}(?:{noun_alt})\b", prompt, flags=re.I):
            text = match.strip()
            if len(text.split()) <= 4:
                candidates.append(text)

        seen = set()
        out = []
        for cand in candidates:
            norm = " ".join(str(cand).replace("_", " ").split())
            key = norm.lower()
            if not norm or key in seen:
                continue
            seen.add(key)
            out.append(norm)
            if 0 < self.max_memory_characters <= len(out):
                break
        return out

    def _encode_real_segment_latents(self, segment_video: str):
        engine = self._engine
        if engine is None:
            raise SlotMemRuntimeError("SlotMem engine is not loaded")
        import torch

        pixels, _n_frames = read_segment_pixels(
            segment_video,
            ffmpeg=self.ffmpeg,
            height=int(getattr(engine.args, "height", 480)),
            width=int(getattr(engine.args, "width", 832)),
            fps=16.0,
        )
        from PIL import Image

        first = ((pixels[0, :, 0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
        self._last_segment_first_frame_pil = Image.fromarray(first.to(dtype=torch.uint8).cpu().numpy())

        engine.pipe.vae.to(engine.device)
        vae_dtype = getattr(engine, "dtype", torch.bfloat16)
        video_for_vae = pixels[0].to(device=engine.device, dtype=vae_dtype)
        try:
            latents = engine.pipe.vae.encode([video_for_vae], device=engine.device)[0]
        except TypeError:
            latents = engine.pipe.vae.encode([video_for_vae])[0]
        engine.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return latents.unsqueeze(0).to(device=engine.device, dtype=torch.float32)

    def _encode_prompt_and_image_cond(self, prompt: str, clean_latents):
        engine = self._engine
        if engine is None:
            raise SlotMemRuntimeError("SlotMem engine is not loaded")
        import torch

        engine.pipe.text_encoder.to(device=engine.device, dtype=torch.float32)
        if hasattr(engine.pipe, "prompter"):
            engine.pipe.prompter.text_encoder.to(device=engine.device, dtype=torch.float32)
        prompt_emb = engine.pipe.encode_prompt(prompt, positive=True)
        neg_emb = engine.pipe.encode_prompt(engine.args.negative_prompt, positive=False)
        prompt_emb["context"] = prompt_emb["context"].to(dtype=engine.dtype)
        neg_emb["context"] = neg_emb["context"].to(dtype=engine.dtype)
        engine.pipe.text_encoder.to("cpu")
        if hasattr(engine.pipe, "prompter"):
            engine.pipe.prompter.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        image_emb = {}
        if getattr(engine.pipe.dit, "has_image_input", False):
            first_frame = self._last_segment_first_frame_pil
            if first_frame is None:
                raise SlotMemRuntimeError("SlotMem I2V conditioning requires the first real-segment frame")
            engine.pipe.image_encoder.to(engine.device)
            image_emb = engine.pipe.encode_images_adaptive(
                first_frames=[first_frame],
                random_ref_frame=first_frame,
                num_frames=int(getattr(engine.args, "context_frames", 81)),
                height=int(getattr(engine.args, "height", 480)),
                width=int(getattr(engine.args, "width", 832)),
                use_first_aug=bool(getattr(engine.args, "use_first_aug", False)),
                ref_pad_cfg=bool(getattr(engine.args, "ref_pad_cfg", False)),
                ref_pad_num=int(getattr(engine.args, "ref_pad_num", 0)),
            )
            image_emb["num_condition_frames"] = 1
            engine.pipe.image_encoder.to("cpu")
            image_emb = {key: value for key, value in image_emb.items() if key != "num_condition_frames"}
        del clean_latents
        return prompt_emb, neg_emb, image_emb

    def _probe_timestep(self):
        engine = self._engine
        if engine is None:
            raise SlotMemRuntimeError("SlotMem engine is not loaded")
        import torch

        scheduler = engine.pipe.scheduler
        try:
            scheduler.set_timesteps(
                int(getattr(engine.args, "num_inference_steps", 50)),
                shift=float(getattr(engine.args, "sample_shift", 5.0)),
            )
        except TypeError:
            scheduler.set_timesteps(int(getattr(engine.args, "num_inference_steps", 50)))
        timesteps = getattr(scheduler, "timesteps", None)
        if not isinstance(timesteps, torch.Tensor) or int(timesteps.numel()) <= 0:
            raise SlotMemRuntimeError("SlotMem scheduler produced no timesteps")
        bank_percents = engine._single_online_memory_bank_percents()
        target_percent = float(bank_percents[0]) if bank_percents else 0.0
        num_train = float(engine._get_num_train_timesteps())
        step_idx = min(
            range(int(timesteps.numel())),
            key=lambda idx: abs(float(timesteps[idx].detach().float().item()) / num_train - target_percent),
        )
        return timesteps[step_idx].to(engine.device).unsqueeze(0), [target_percent]

    def _extract_role_slots(self, obs: SegmentObservation) -> dict[str, dict[int, Any]]:
        """Return ``{char_id: {bank_idx: (tokens, token_meta)}}`` for the real segment."""
        engine = self._engine
        if engine is None:
            raise SlotMemRuntimeError("SlotMem engine is not loaded")
        import torch

        prompt = str(obs.prompt_text)
        role_names = self._candidate_role_names(prompt)
        if not role_names:
            return {}

        clean_latents = self._encode_real_segment_latents(obs.segment_video)
        prompt_emb, neg_emb, image_emb_for_denoising = self._encode_prompt_and_image_cond(prompt, clean_latents)
        timestep, _bank_percents = self._probe_timestep()
        engine._set_inference_noise_domain_from_timestep(timestep)
        gen = torch.Generator(device=engine.device).manual_seed(int(obs.chunk_id) + 12345)
        noise = torch.randn(clean_latents.shape, generator=gen, device=engine.device, dtype=clean_latents.dtype)
        noisy_latents = engine.pipe.scheduler.add_noise(clean_latents, noise, timestep)

        out: dict[str, dict[int, Any]] = {}
        for role in role_names:
            extracted = engine._extract_memory_from_current_step(
                noisy_latents=noisy_latents,
                timestep=timestep,
                prompt=prompt,
                char_id=str(role).replace(" ", "_"),
                cond_context=prompt_emb["context"],
                uncond_context=neg_emb["context"],
                image_emb_for_denoising=image_emb_for_denoising,
                char_latent_boxes=None,
                return_positions=False,
                return_token_meta=True,
            )
            if extracted is None:
                continue
            if isinstance(extracted, tuple) and len(extracted) == 2:
                mem, token_meta = extracted
            else:
                mem, token_meta = extracted, []
            if mem is None or (isinstance(mem, torch.Tensor) and int(mem.numel()) == 0):
                continue
            out[str(role).replace(" ", "_")] = {0: (mem, token_meta)}
        return out

    def _stage2_prepare_payload_for_bank(self, char: str, bank_id: int, mem: Any, token_meta: Any):
        engine = self._engine
        slotmem_mod = self._slotmem_mod
        if engine is None or slotmem_mod is None:
            raise SlotMemRuntimeError("SlotMem engine is not loaded")

        role_wise_slot_memory_bank_enabled = bool(
            getattr(engine, "memory_writer_enabled", False)
            and getattr(engine, "jigsaw_extra_encoder_enabled", False)
        )
        if not role_wise_slot_memory_bank_enabled:
            return mem, token_meta if isinstance(token_meta, list) else [], {"enabled": 0.0, "mode": "raw"}

        old_payload = self._mem_manager.get_memory_payload(char, bank_id)
        domain = str(getattr(engine.args, "train_noise_domain", "low_noise")).strip().lower()
        if slotmem_mod._is_layerwise_token_payload(mem):
            out_layers = {}
            out_meta_layers = {}
            layer_stats = {}
            old_tokens_payload = old_payload.get("tokens", None) if isinstance(old_payload, dict) else None
            old_meta_payload = old_payload.get("token_meta", None) if isinstance(old_payload, dict) else None
            for layer, update_layer_tokens in slotmem_mod._iter_layerwise_items(mem):
                layer_meta = slotmem_mod._select_layerwise_value(token_meta, layer, default=[])
                old_layer_tokens = slotmem_mod._select_layerwise_value(old_tokens_payload, layer, default=None)
                old_layer_meta = slotmem_mod._select_layerwise_value(old_meta_payload, layer, default=[])
                if old_layer_tokens is not None:
                    stored_tokens, stored_meta, stats = engine.stage2_update_slot_payload(
                        old_layer_tokens,
                        old_layer_meta if isinstance(old_layer_meta, list) else [],
                        update_layer_tokens,
                        layer_meta if isinstance(layer_meta, list) else [],
                        noise_domain=domain,
                        layer_idx=layer,
                    )
                    mode = "writer_update"
                else:
                    stored_tokens, stored_meta, stats = engine._encode_memory_payload_to_stage2_slots(
                        update_layer_tokens,
                        layer_meta if isinstance(layer_meta, list) else [],
                        noise_domain=domain,
                        layer_idx=layer,
                    )
                    mode = "initial_slot_extract"
                out_layers[slotmem_mod._layer_key(layer)] = stored_tokens
                out_meta_layers[slotmem_mod._layer_key(layer)] = stored_meta
                layer_stats[slotmem_mod._layer_key(layer)] = dict(stats, mode=mode)
            return slotmem_mod._make_layerwise_container(out_layers), slotmem_mod._make_layerwise_container(out_meta_layers), {
                "enabled": 1.0,
                "mode": "layerwise",
                "layers": layer_stats,
            }

        old_tokens = old_payload.get("tokens", None) if isinstance(old_payload, dict) else None
        old_meta = old_payload.get("token_meta", []) if isinstance(old_payload, dict) else []
        if old_tokens is not None:
            stored_tokens, stored_meta, stats = engine.stage2_update_slot_payload(
                old_tokens,
                old_meta if isinstance(old_meta, list) else [],
                mem,
                token_meta if isinstance(token_meta, list) else [],
                noise_domain=domain,
                layer_idx=0,
            )
            mode = "writer_update"
        else:
            stored_tokens, stored_meta, stats = engine._encode_memory_payload_to_stage2_slots(
                mem,
                token_meta if isinstance(token_meta, list) else [],
                noise_domain=domain,
                layer_idx=0,
            )
            mode = "initial_slot_extract"
        return stored_tokens, stored_meta, dict(stats, enabled=1.0, mode=mode)

    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        rec = RetrievedMemory(chunk_id=req.chunk_id)
        if self._mem_manager is None:
            return rec
        # Read path: per role, surface each contributing source chunk as a temporal item.
        bank = getattr(self._mem_manager, "memory_meta_bank", {}) or {}
        for _char_id, banks in bank.items():
            metas = banks.values() if isinstance(banks, dict) else []
            for meta_list in metas:
                for item in (meta_list or []):
                    src = item.get("source_chunk_idx") if isinstance(item, dict) else None
                    if src is None:
                        continue
                    sec = self._chunk_seconds.get(int(src))
                    if sec is None or sec >= float(req.seconds_span[0]):
                        continue
                    rec.items.append(RetrievedItem(
                        evidence_kind="slot", source_seconds=sec, source_chunk_id=int(src),
                        raw_ref=f"slotmem:c{src}"))
        return rec

    def finalize(self) -> dict[str, Any]:
        return {
            "system": "slotmem",
            "status": "ported_native_single_forward_probe",
            "memory_space": "RoleWiseSlotMemoryBank (role-wise slot memory)",
            "probe": "VAE-encode real segment -> add noise at SlotMem's single-bank timestep -> "
                     "one native DiT forward with SlotMem attention hooks -> stage2 slot encoder/writer.",
            "distilled_wan2_2": "forbidden_for_trackA: loads with SlotMem LoRA but visual quality is unusable",
            "ckpt_dir": self.ckpt_dir,
            "lora_ckpt": self.lora_ckpt,
        }


def build_adapter() -> SlotMemAdapter:
    return SlotMemAdapter()
