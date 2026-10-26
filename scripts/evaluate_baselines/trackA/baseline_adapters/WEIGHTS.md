# baseline_adapters — baseline weights and environment reference

> This file records the **absolute weight paths** for each causal baseline, the shared backbone, and the recommended Python dependency stack.
> Last updated 2026-07-24. Update this file when a weight location changes; do not hardcode scattered paths in adapters.

## Shared backbone / auxiliary models

| Purpose | Path |
|---|---|
| Wan2.1-T2V-1.3B (DiT+VAE+T5, shared backbone for MemFlow/IAMFlow/LongLive) | `${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.1-T2V-1.3B` |
| Wan2.1-T2V-1.3B (equivalent copy already downloaded and vendored with LongLive) | `baselines/Causal/LongLive-RAG/wan_models/Wan2.1-T2V-1.3B` |
| Qwen3-4B-Instruct-2507 (IAMFlow LLM) | `${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-4B-Instruct-2507` |
| Qwen3-VL-2B-Instruct (IAMFlow frame-selection VLM) | `${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL-32B (scoring judge, see §Scoring) | see §Scoring below |

## Per-baseline weights

| baseline | weight root | key files |
|---|---|---|
| LongLive-RAG | `.../Causal_Video_Generation/LongLive-RAG/checkpoints` | `longlive_base.pt` / `longlive_lora.pt` / `ae_latent_mem.pt` (already in place in the vendored repo) |
| MemFlow | `${PUBLIC_MODELS_ROOT}/KlingTeam/MemFlow` | `base.pt` / `lora.pt` |
| IAMFlow | `${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/IAMFlow` | `iamflow_fp8.safetensors` / `tinyvae.pth` |
| SlotMem | LoRA/encoder: `${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/SlotMem/ckpt/{stage1,stage2}`; base: `${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B` | TrackA only uses native Wan2.2-I2V-A14B + `ckpt/{stage1,stage2}/{stage*_low.pt,stage*_high.pt}`. Do **not** use distilled/lightx2v Wan2.2 for SlotMem formal runs; the smoke can load LoRA but visual quality is unusable. |
| MemStrata (this system, perception weights) | `${PUBLIC_MODELS_ROOT}` | Default grounder: **WeDetect-Ref** (a describe→bbox service, `MEMSTRATA_WEDETECT_URL`, deployed separately); DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`; SAM3 (fallback only when WeDetect is unavailable): `facebook/sam3`. `name_source=mllm` additionally needs `Qwen/Qwen3.5-9B-Instruct`. For download and layout, see `MODELS.md` in the MemStrata repo. |
| (optional) DecMem | `${PUBLIC_MODELS_ROOT}/KlingTeam/DecMem` | needs H100/WorldMem, not integrated for now |

## Python / library mapping

| baseline | stack | notes |
|---|---|---|
| LongLive-RAG | torch 2.5 + einops | only uses VAE+AE, bypasses DiT/flash_attn |
| MemFlow | torch 2.6 + flash-attn 2.6 | forward verified |
| IAMFlow | torch 2.5 + flash-attn; optional standalone vLLM process for Qwen | `iamflow_fp8.safetensors` is dequantized to bf16 inside the adapter, so it runs on SM80 |
| SlotMem | torch 2.5 + flash-attn 2.8 (vendored diffsynth, not lightx2v) | formal TrackA experiments forbid distilled/lightx2v Wan2.2 |
| MemStrata | CPython 3.11 + torch; SAM3 vendored at `models/vendor/sam3_transformers59` | GPU-verified: BBB name_anchored limit=4 |

## How backbone/ckpt are wired (keeping the vendored repo untouched)

Each vendored repo expects weights at a **relative path under its own repo root** (e.g. MemFlow expects `wan_models/Wan2.1-T2V-1.3B/` and `checkpoints/base.pt`). Wire them in uniformly via **symlink** from the absolute paths in the table above — no copying, no source changes:

- MemFlow: `baselines/Causal/MemFlow/wan_models/Wan2.1-T2V-1.3B` → Wan backbone; `.../MemFlow/checkpoints/{base.pt,lora.pt}` → KlingTeam/MemFlow
- IAMFlow: `baselines/Causal/IAMFlow/pretrained/{Wan2.1-T2V-1.3B, iamflow_models/iamflow_fp8.safetensors, Qwen3-4B-Instruct-2507, Qwen3-VL-2B-Instruct}` → the paths in the table above
- SlotMem: `causal/slotmem.py` wires the absolute paths for native Wan2.2-I2V-A14B + SlotMem `ckpt/{stage1,stage2}` directly; the vendored repo is untouched. Formal TrackA does not use distilled/lightx2v Wan2.2.
- MemStrata: no vendored symlink needed; perception weights are resolved via `PUBLIC_MODELS_ROOT`. To run (local gpu0):

```bash
SAM3=./models/vendor/sam3_transformers59
SRC=./src
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal   # from the VMem-Bench repo root
PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$SAM3:$SRC:$(pwd)" \
python3 -u runner.py \
  --adapter memstrata --movie-dir ../../../data/BlenderOpenMovies/big_buck_bunny \
  --limit 4 --input-mode name_anchored
# Output: <movie>/benchmark_run/visual_selections/memstrata.json (+ _ref_frames/memstrata/*.png)
```

> The SAM3 bundle is prepended in PYTHONPATH to obtain transformers 5.9; the adapter `reset` promotes `src` to the front and lifts `causal/memstrata.py`'s shadowing of the `memstrata` package (otherwise `import memstrata.bank` would resolve to the adapter file).

## Scoring judge (Qwen3-VL-32B) — wired up and verified

Weights: `${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-32B-Instruct`
Stack: torch + vLLM. `scoring.visual_coverage` defaults to `127.0.0.1:8110`, model `qwen3-vl-32b`.

**Start the service** (on a shared node, loading the 64GB weights can OOM if an external job grabs the card; just retry, and use `nvidia-smi` to pick a free card):

```bash
NVJIT=$(python3 -c "import pathlib,sys;p=pathlib.Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'/'nvidia'/'nvjitlink'/'lib';print(p if p.is_dir() else '')")
CUDA_VISIBLE_DEVICES=<free_gpu> LD_LIBRARY_PATH=$NVJIT NO_PROXY=localhost,127.0.0.1,0.0.0.0 PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 -m vllm.entrypoints.openai.api_server --model $Qwen3-VL-32B-Instruct \
  --served-model-name qwen3-vl-32b --host 0.0.0.0 --port 8110 --tensor-parallel-size 1 \
  --max-model-len 28672 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":96,"video":1}' --allowed-local-media-path /data --trust-remote-code
```

- `--limit-mm-per-prompt image=96`: large-footprint systems (MemFlow can reach 40+ reference images per segment) need a higher per-request image cap (the default 24 is not enough).
- `--max-model-len 28672`: a safe upper bound the KV cache can hold at 0.92 memory utilization (32768 leaves insufficient KV).

**Run scoring** (`NO_PROXY` connects directly to the local judge):

```bash
cd VMem-Bench
NO_PROXY=localhost,127.0.0.1 CUDA_VISIBLE_DEVICES=<free_gpu> PYTHONPATH=src \
python3 -m vmem_bench.scoring.visual_coverage \
  --movie data/BlenderOpenMovies/big_buck_bunny --system <memflow|longlive_rag|...> \
  --video <source_video.mp4> [--limit N]
```

Scorer robustness notes (`src/vmem_bench/scoring/visual_coverage.py`):
1. `run()` resolves `--movie/--video` to absolute paths — otherwise a clip's relative `file://` path cannot be resolved server-side (500).
2. `_call` carries the server-side error body into the exception message — easier to diagnose (it used to report only a bare `HTTP 500`).
3. `_img` downsamples client-side to `JUDGE_IMG_MAX_SIDE=384` and embeds as base64 — a uniform, deterministic way to compress tokens so that large-footprint systems like MemFlow (40+ reference images) fit in context (the server-side `mm_processor_kwargs.max_pixels` has no effect on Qwen3-VL).

**Verified results** (big_buck_bunny, judge=qwen3-vl-32b, video/reference images uniformly at 480p, see running_eval §3.1):
- LongLive-RAG (8 segments): prec 0.74 / rec 0.72 / f1 0.69 / redun_vlm 0.58 / redun_sim 0.92 / eff 0.24 / budget 6.0
- MemFlow (w/o SMA) (5 segments): prec 0.44 / rec 0.67 / f1 0.62 / redun_vlm 0.62 / redun_sim 0.79 / eff 0.04 / budget 31.5
- MemFlow (with SMA) (5 segments): prec 0.56 / rec 1.00 / f1 0.84 / redun_vlm 0.88 / redun_sim 0.83 / eff 0.06 / budget 29.5
  - **The two variants write identically** (`compress_kv_bank` text-saliency top-k), and **differ only in the read**: w/o SMA uses the whole bank; with SMA uses φ compact-descriptor routing over the top-3 chunks (`dynamic_topk_routing_attention`, the `sma` switch in `memflow.py`). φ routing focuses on the more relevant historical blocks → higher precision/recall/F1 with slightly fewer frames. Adapter names: `memflow` / `memflow_sma`.
- IAMFlow (15 segments, 6 with recall): prec 1.00 / rec 0.22 / f1 0.65 / redun_vlm 0.04 / redun_sim 0.72 / eff 0.96 / budget 3.0 (mean over segments with recall) / 1.2 (mean over all segments)
  - The low rec is because the first 9 chunks have no character reappearance and their recall=0 is counted into the 15-block mean; looking only at blocks with recall, rec is clearly higher. IAMFlow's character: **few but accurate, almost zero redundancy** (the opposite of MemFlow's large-footprint high redundancy).

**Two redundancy measures (redun_vlm vs redun_sim)**:
- `redundancy_vlm`: the VLM measure. For each "same-entity" reference-image group, the judge is asked how many visually distinct views there are; `redundancy = group size − distinct_views`, then normalized over all segments. It produces an **integer count**, which `efficiency` uses.
- `redundancy_sim`: a **non-VLM measure**. Within the same group, pairwise DINOv3 CLS cosine is computed and the off-diagonal mean is taken (threshold-free; under unit-norm, cos=dot), then averaged with log weighting. 1.0 = selected frames are almost identical (fully redundant); lower = more diverse views.
- The two use identical grouping (both the judge-determined present + entity_id); only the "degree of redundancy" metric differs: one uses VLM semantic dedup, the other uses DINOv3 feature similarity. redun_sim is computed offline by `scripts/baselines/recompute_redundancy_sim.py` (reusing the grouping in details.json, needing only DINOv3, without re-running the judge); it needs `PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}`.
- ⚠️ Before comparing headline numbers directly across baselines, note that the segment counts differ (8/5/15); a formal release should re-run over a unified segment range.
