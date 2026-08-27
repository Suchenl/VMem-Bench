# baseline_adapters/causal — new-protocol causal baseline adapter layer

This directory holds **the bench adapter code for each causal baseline**. The vendored upstream repositories (`baselines/Causal/<name>/`) remain **untouched**: all bench glue is written here and never stuffed into third-party code, which keeps review and re-pulling easy.

> For the authoritative protocol, see [`docs/benchmark/running_eval.md`](../../../docs/benchmark/running_eval.md) §0 (the three iron rules) + §1 (the SUT contract).
> This layer is the implementation of that protocol for the causal baselines.

## Why this layer exists (the old route was removed)

The old `src/vmem_bench/baseline_adapters/` used **gold-replay / online-gold / gold-id mapping**: it fed the SUT gold crop pixels or gold entity ids, then mapped the retrieval results back to gold entities for ID-level scoring. This violates iron rules 2/3 (the bench must not hand out images and must not leak answers), and **has been removed entirely per the user's decision**.

The new protocol: for each segment the bench gives the SUT only two things — the **prompt text + the real segment video** (the real segment is used to **replace the SUT generator's output** and eliminate generation noise). The SUT observes the real segment with its own perception/memory mechanism, builds its own memory, and retrieves on its own. Retrieved memory items carry a **temporal identity** (absolute seconds in the source video / source chunk id); on the bench side, [`frame_materializer.py`](frame_materializer.py) **cuts the corresponding frames out of the real source video** as reference images and hands them to `vmem_bench.scoring.visual_coverage` for VLM visual-coverage scoring. This is what "mapping retrieved memory into frames via temporal consistency" means — method-neutral, no gold injected, no labels on the reference images.

## Data flow

```
runner.py  ──per-segment time order──▶  adapter.compose(prompt)      → RetrievedMemory(with temporal identity)
                                  adapter.observe_segment(real clip) → update native memory (compose first, observe after)
      │
      └▶ frame_materializer  ──temporal identity→cut real frames──▶ outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/<name>.json
                                                     │
                                                     └▶ scoring.visual_coverage (VLM scoring)
```

- **Causal guardrail**: at materialization time, items with "source time ≥ the current chunk's start" are dropped (a causal SUT can only draw on the past), and the count is recorded in the manifest.
- **The bench side only**: cuts the real segment, drives compose/observe, cuts reference frames, and writes the manifest. It does not perceive, crop, hand out images, or provide gold.

## What each baseline must implement (`CausalMemoryAdapter`, see `contract.py`)

Because these causal baselines (LongLive-RAG / MemFlow / IAMFlow, etc.) are essentially **video generators**, whose native loop is "generate a chunk → extract memory from generated frames → inject into the next chunk," the new protocol replaces the generated output with the real segment, so each adapter's core hooks are:

1. `reset(movie)`: load the baseline's model/VAE/memory manager (in **its own conda environment**).
2. `observe_segment(obs)`: run the **real segment** through the baseline's own encoder/VAE and its **native memory write** (latent / KV bank / role-wise slot / frame buffer), tagging each memory item with the source second / source chunk temporal label. **No generation** — only borrowing its memory-write path.
3. `compose(req)`: use its **native retrieval algorithm** (AE cosine / attention relevance / role-slot hits, …) to select several items from the current memory and return `RetrievedItem(source_seconds/source_chunk_id, evidence_kind, ...)`.
4. `finalize()`: optional, returns run-level metadata such as config/memory size/retrieval mode.

Each adapter module must expose a `build_adapter()` factory for `runner.py --adapter <module>` to call.

## Running

```bash
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal
python runner.py --adapter memstrata \
  --movie-dir ../../../../../assets/trackA/BlenderOpenMovies/big_buck_bunny --limit 5
```

Or from the repo root: `bash scripts/run_tracka_smoke.sh` (runs `scripts/doctor.py` first).
MemStrata is resolved from `../MemStrata/src` or `MEMSTRATA_SRC`.

## runner batch protection (shared by all causal baselines)

A `--movie-list` is a batch of **mutually independent** jobs, and the long runtimes (memflow_sma is 11–165 s per segment, 8 h+ for a full film) make the following properties necessary; otherwise you waste hours of compute:

- **One failed film does not take down the others**: `main()` wraps each movie in its own try/except; a failure only records a single `failed` summary and continues to the next, still aggregating with exit 31 at the end. Previously, an exception on one movie would drop the entire remaining list behind it.
- **The lock self-heals and does not steal by mistake**: `.stage1.lock` now writes `pid=... host=... start=...` and heartbeats **after each segment** (`_touch_job_lock`). Before grabbing the lock, it checks liveness: same host → probe the pid; different host → check whether the heartbeat has exceeded `MAVE_STAGE1_LOCK_STALE_MINUTES` (default 45 min, far larger than the slowest single segment). A **legacy-format lock without a `host=` field is always treated as alive** and never stolen — there really are legitimate memflow_sma processes holding the lock 8 h+ in production, and stealing it would produce two runners working the same film. This also makes manual "stale lock cleanup" batches unnecessary, which was exactly what caused duplicate runners before.
- **Progress lines carry an ETA**: `[stage1] ... segment=i/N ... eta_min=...`. Per-segment time swings between 11–165 s with node contention, so you cannot eyeball whether a long run is worth keeping — so we just print the answer.
- **Adapters may decline a job**: if an adapter implements `preflight(movie) -> str | None`, the runner calls it before `reset()`; returning a string skips that film and records the reason. IAMFlow uses it for a host-memory budget check, see [`docs/baselines/tracka_iamflow_host_memory.md`](../../../../../docs/baselines/tracka_iamflow_host_memory.md) (Chinese).

Regression tests: `benchmarks/VMem-Bench/tests/test_trackA_stage1_job_lock.py`, `test_iamflow_host_memory_guard.py` (no GPU, no weights, <1 s).

## Current status

| baseline | Python / libs | adapter | native memory / retrieval | status |
|---|---|---|---|---|
| **MemStrata (this system)** | torch + transformers (SAM3 vendored bundle prepended to PYTHONPATH) | `memstrata.py` | layered AssetBank (SAM3-concept+DINOv3 perception write) / IntentInterpreter name-anchor + model-free compose | **TrackA minismoke PASS**: BlenderOpenMovies:`big_buck_bunny` + LSMDC:`0001_American_Beauty`, limit=6 each, both produce `visual_selections`. |
| SlotMem | torch 2.5 + flash-attn 2.8 (diagnostic only) | `slotmem.py` | character slots (RoleWiseSlotMemoryBank) | **does not enter the oracle-free main table**: its released interface needs external/scripted `role_names` to locate slots stably; this is an oracle-role / Scripted diagnostic condition and does not fit TrackA/B prompt-only causal production evaluation. The runner skips it quickly on the mainline via `scripts/evaluate_baselines/trackA/.disable_slotmem_mainline`. |
| LongLive-RAG | torch (VAE + AE; DiT need not run) | `longlive_rag.py` | self-encoded latent descriptors + AE cosine top-k | **TrackA minismoke PASS**: 1 sample per dataset at limit=6 both pass; pure descriptor computation, no generator forward needed. |
| MemFlow (w/o SMA) | torch 2.6 + flash-attn 2.6 | `memflow.py` | sink + local window + KV bank (text-saliency top-k) | **TrackA minismoke PASS**: 1 sample per dataset at limit=6 both pass. |
| MemFlow (with SMA) | same as above | `memflow.py` (`sma=True`/`memflow_sma`) | same as above; read uses `dynamic_topk_routing_attention` φ top-3 chunk routing | **TrackA minismoke PASS**: 1 sample per dataset at limit=6 both pass. |
| IAMFlow | torch 2.5 + flash-attn; fp8→bf16 dequantization | `iamflow.py` | entity-aware active-memory frames (entity+VLM fused frame selection) | **TrackA minismoke PASS**: 1 sample per dataset at limit=6 both pass. |

> The three share `_video_io.py` to decode the real segment into Wan pixel tensors (480×832, [-1,1], 16 fps); this is pure IO, not perception/memory — each baseline still uses **its own VAE + native memory/retrieval**.
>
> For full-scale TrackA, IAMFlow should put the LLM/VLM behind a long-lived OpenAI-compatible vLLM service, to avoid each worker loading Qwen3-4B / Qwen3-VL-2B on its own. Once `IAMFLOW_LLM_ENDPOINT` and `IAMFLOW_VLM_ENDPOINT` are set, `iamflow.py` goes over HTTP; if unset, the original in-process HuggingFace fallback is kept. The current Stage-1 startup instructions and helper script are in `experiments/results/e2e/tracka_full_20260726/IAMFLOW_SERVICE.md`.
>
> Completeness notes:
> - **LongLive-RAG** retrieval is pure descriptor computation (no generator forward needed), fully implemented per the released `latentmem` rules (`memory_size=6/recent_exclude=5/sink_size=1`), and **has passed smoke** (12 reference frames).
> - **MemFlow / IAMFlow** memory writes happen inside the generator forward, filling native memory with a single teacher-forced real-latent forward; the minismoke on both datasets confirmed these write/read paths can produce materializable `visual_selections`. SlotMem is kept only as an oracle-role diagnostic and does not participate in the mainline batch.
> - **Retrieval families (real encoders)**: the text Qwen3-Embedding + frame `seg_uniform` (uniform sampling) variant previously passed smoke (27 reference frames, future_dropped=0). The default `seg_framererank` (SigLIP2 frame-text re-ranking) crashes under **transformers 5.x** because `get_text_features(...)` returns `BaseModelOutputWithPooling` instead of a tensor (the `.float()` call fails). Use **transformers 4.57.x** (verified 4.57.1–4.57.6). SigLIP2 weights: `${PUBLIC_MODELS_ROOT}/google/siglip2-base-patch16-512`.
> - **Track A batch-readiness status**: **PASS**. BlenderOpenMovies:`big_buck_bunny` and LSMDC:`0001_American_Beauty`, 1 sample each, cover the current oracle-free main-table rows (MemStrata, LongLive-RAG, MemFlow, MemFlow-SMA, IAMFlow, frame_text, seg_uniform, seg_dinokey, seg_framererank). SlotMem has been moved to the Scripted/oracle-role diagnostic class.
> - Full-corpus scoring is not yet done: Stage 2 VLM scoring + `aggregate_trackA_outputs.py` aggregation still need to be started.
>
> The weights each baseline needs are in the header comment of the corresponding adapter module (all need the Wan2.1-T2V-1.3B backbone + VAE + their own ckpt, placed under the vendored repo root).
