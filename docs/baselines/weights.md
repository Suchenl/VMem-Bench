# Weights for Track A retrieval (GT history) and Track B generation

> Track A still needs **retrieval** weights for methods that use them
> (e.g. LongLive-RAG ``ae_latent_mem.pt``, Wan VAE to encode gold video).
> It does **not** need to run the full generator denoising loop.
> Track B uses each vendor's official generation scripts + full checkpoints.

本机根目录::

  ${PUBLIC_MODELS_ROOT}/

## Track A needs

| Method | Track A assets |
|---|---|
| Helios | none (window policy) |
| LongLive-RAG | `ae_latent_mem.pt` + Wan VAE to encode **gold** video → `gold_latents.pt` |
| MemFlow | selection/topk path on GT-encoded history (pending wiring) |
| DecMem | LTM selection on GT (pending; full LTM may need SM90) |
| IAMFlow | LLM `Qwen3-4B-Instruct-2507` + VLM `Qwen3-VL-2B-Instruct` + IAMFlow DiT `iamflow_fp8.safetensors` (+`tinyvae.pth`) — all present; DiT required because entity_score comes from its forward (dequantized fp8→bf16 for SM80) |

## Local paths (filled)

| 资产 | 路径 | 状态 |
|---|---|---|
| Wan2.1-T2V-1.3B | `.../Wan-AI/Wan2.1-T2V-1.3B` | ✅ (VAE encode gold) |
| LongLive AE | `.../Causal_Video_Generation/LongLive-RAG/checkpoints/ae_latent_mem.pt` | ✅ |
| MemFlow ckpt | `.../KlingTeam/MemFlow` | ✅ Track B / pending Track A wire |
| DecMem | `.../KlingTeam/DecMem` | ✅ |
| Helios-Distilled | `.../BestWishYsh/Helios-Distilled` | ✅ Track B only |
| Qwen3-4B-Instruct-2507 (IAMFlow LLM) | `.../Qwen/Qwen3-4B-Instruct-2507` | ✅ |
| Qwen3-VL-2B-Instruct (IAMFlow VLM) | `.../Qwen/Qwen3-VL-2B-Instruct` | ✅ |
| IAMFlow DiT (`iamflow_fp8.safetensors`, `tinyvae.pth`) | `.../Causal_Video_Generation/IAMFlow/` | ✅ (dequantized fp8→bf16 at load) |

## Track B

Use vendor official scripts under `baselines/Causal/*/`; we do not wrap generation
as the Track A path. See `docs/TRACK_A.md`.
