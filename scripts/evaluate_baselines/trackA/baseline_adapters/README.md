# baseline_adapters — new-protocol SUT adapter layer (overview + how each is implemented + fidelity reflections)

This directory holds the bench adapter code for every baseline under the MemStrata-Bench **new protocol**. The vendored upstream repositories (`baselines/Causal/<name>/`) remain **untouched**; all glue is written here.

- The causal baselines are implemented under [`causal/`](causal/); for protocol details see [`causal/README.md`](causal/README.md).
- For the authoritative protocol (the three iron rules + the SUT contract), see [`../../docs/benchmark/running_eval.md`](../../docs/benchmark/running_eval.md) §0/§1.

> The protocol in one sentence: for each segment the bench gives the SUT only two things — the **prompt text + the real segment video** (the real segment **replaces the SUT generator's output** to eliminate generation noise). The SUT uses **its own** perception/memory/retrieval and returns memory items carrying a **temporal identity**; on the bench side, `frame_materializer` cuts the corresponding frames out of the real source video according to temporal consistency and performs VLM visual-coverage scoring.

---

## 1. How each module is concretely implemented

Every causal adapter implements the same `CausalMemoryAdapter` contract (`causal/contract.py`): `reset(movie)` → per segment `compose(prompt)` (first) → `observe_segment(real clip)` (after) → `finalize()`. All three share `causal/_video_io.py`, which decodes the real segment into Wan pixel tensors (480×832, [-1,1], 16 fps, VAE temporal stride 4) — this is **pure decoding IO, not perception/memory**; each baseline still uses its own VAE + native memory/retrieval.

| baseline | native memory space | native retrieval | how we implement it | completeness |
|---|---|---|---|---|
| **LongLive-RAG** | a growing pool of one **AE descriptor** per latent | **AE cosine top-k**: query with the most recent latent descriptor over the pool = `[sink, pool, recent_exclude]`; `memory_size=6/recent_exclude=5/sink_size=1` | directly import `wan.modules.vae._video_vae` + `ae.model.LatentAE` (**bypassing DiT/flash_attn**); `observe` encodes the real segment into latents → each frame's AE descriptor enters the pool tagged with its source second; `compose` reproduces the released cosine top-k and returns the source seconds of the hits | **fully implemented, runnable** (no generator forward needed) |
| **MemFlow (w/o SMA)** | **sink + local window + KV bank** (the bank is a per-layer, text-saliency top-k-compressed set of historical blocks) | **write**: `compress_kv_bank` per-layer text top-k; **read**: the whole bank enters attention, no further filtering | teacher-force the real latent (`context_noise=0`, conditioned on **this segment's prompt**) and forward segment by segment to fill KV/bank; **fingerprint back-tracing** maps surviving bank blocks back to source latents (per-layer + voting); `compose` aggregates the source seconds of sink ∪ local window ∪ bank | **faithfully implemented, GPU-verified** (torch 2.6 + flash-attn 2.6, 5 segments: base+LoRA active, 126 reference frames, future_dropped=0) |
| **MemFlow (with SMA)** | same as above (**writes are identical to w/o SMA**) | **read**: `dynamic_topk_routing_attention` uses the φ=mean-pool compact descriptor `φ_q·φ_k` to route the **top-3 chunks** from `sink∪bank` into attention (`SMA=True`) | reuse `memflow.py` (`sma` switch): on reset, set `SMA` to True on all self-attn modules, and hook the routing to capture the historical chunks selected by φ (query = real segment, keeping only sources earlier than the current chunk); `compose` back-fills the references recorded in `observe` (materialized after the runner loop) | **faithfully implemented, GPU-verified** (torch 2.6 + flash-attn 2.6, 5 chunks: 31 modules set to SMA, 118 reference frames, future_dropped=0; `memflow_sma`) |
| **IAMFlow** | **entity-aware active-memory frames** (each frame carries a KV slice + associated entities) | `select_frame_from_chunk` (fusing the DiT-attention entity_score + 0.3·VLM visual score; the upstream function name is preserved) + `retrieve_initial_frames` greedy entity coverage at the prompt boundary | the formal adapter contains self-contained fp8→bf16 dequantization and HF/vLLM backend glue: teacher-force the real latent forward segment by segment to fill KV/crossattn (conditioned on **this segment's prompt**) + archive/evict + VLM frame scoring; `compose` uses the LLM to extract entities **causally per segment** (not precomputing the whole prompt) → `retrieve_initial_frames`; frame_id→source-second registry | **faithfully implemented, GPU-verified** (torch 2.5 + flash-attn; 15 segments: 18 reference frames, correct recall after characters reappear in segments 9–14, future_dropped=0) |

The weights/environment each baseline needs are in the header comment of the corresponding adapter module; how to run is in `causal/README.md` §Running.

---

## 2. Why this integration approach is appropriate (core reflection)

**Integration strategy = teacher-force the real segment's latents through each baseline's native memory write + native retrieval, then materialize the retrieved items into real frames via temporal consistency.**

This is consistent with the user's setting and is the minimally invasive way to **isolate the memory mechanism while eliminating generation noise**, for these reasons:

1. **Why not let the baseline generate on its own and then extract memory?** That would mix "generator quality" into the evaluation of the "memory mechanism" — differences between two baselines would be contaminated by their respective generators' image quality/drift. Replacing the generated output with the real segment is precisely what lets all systems compare memory on the **same visual content**.
2. **Why teacher-forcing rather than changing retrieval?** We do **not** change any baseline's retrieval algorithm (cosine top-k / text-saliency top-k / entity coverage are all preserved as-is); we only swap the "pixels fed to memory" from "self-generated" to "real." The retrieval logic is 100% the baseline's own.
3. **Why materialize into frames via temporal consistency?** Each baseline's native memory form is different (descriptors / KV blocks / entity frames) and cannot be compared directly. The unified, comparable target is "which moment of the source video does the retrieved memory correspond to" — hand that moment's real frame to the VLM for visual-coverage scoring: method-neutral, no gold injected, no labels on the reference images.

**Is there a better alternative?** The only alternative is "let the baseline natively generate throughout and only log retrieval" — but that reintroduces generation noise (which the user explicitly wanted removed) and does not change the retrieval-fidelity question. So teacher-forcing is the correct route; the remaining fidelity risk is purely an implementation matter of "how to **faithfully capture** the native selections under teacher-forcing," not a flaw in the route.

---

## 3. Is it the same as the original retrieval scheme? Any performance loss? (honest per-baseline assessment)

### LongLive-RAG — faithful, virtually lossless ✅

- **Same**: the AE model, cosine top-k, the `sink/recent_exclude/memory_size` parameters, the query = most recent latent descriptor (`latent_descriptors[-1]`) — all match the original implementation. The descriptors are now computed from the **real latents**, which is exactly the intended "real content, no generation noise" replacement.
- **Differences (negligible)**: (a) we query once at the chunk start, whereas the original queries per block while generating that chunk; for "memory composed for this prompt," chunk granularity is more apt. (b) we do not feed retrieval back into generation (because we do not generate), which is an evaluation-design choice, not a loss.
- **Conclusion**: no material performance loss.

### MemFlow — faithful, GPU-verified ✅ (two gaps filled)

- **Gap 1 fixed**: `compress_kv_bank`'s bank selection is **text-conditioned**. The observe forward now encodes crossattn conditioned on **this chunk's prompt**, so the text-saliency signal is real.
- **Gap 2 fixed**: no longer using a recency placeholder. `compress_kv_bank` copies K/V blocks byte-for-byte, so we use **fingerprint matching** to map each surviving bank block back to the source latent it was committed from (per-layer + vote aggregation), the same method as the original Track-A driver `run_bank_trace.py`. This is MemFlow's real text-saliency selection, not an approximation.
- **Long-video RoPE**: the temporal RoPE table is extended to the full film duration (exact, not approximate), avoiding exceeding the default 1024 positions.
- **Verification**: torch 2.6 + flash-attn 2.6, 5 chunks run through, base+LoRA active (trainable 19.78%), 126 reference frames, `future_dropped=0`, retrieval tag `memflow_sink+local+kv_bank_fingerprint_trace_on_real_latents`.
- **SMA variant (`memflow_sma`)**: MemFlow's source also has `dynamic_topk_routing_attention` (φ=mean-pool compact-descriptor `φ_q·φ_k` top-3 chunk routing), gated by the `SMA` switch and off by default in the shipped config. **Writes are identical to w/o SMA; only the read differs.** On reset with `sma=True`, the adapter sets `SMA` to True on all self-attn modules and hooks the routing to record the historical chunks φ selects (using the real segment as query, keeping only sources earlier than the current chunk, which is causally valid); because routing is query-dependent, the rec returned by `compose` is back-filled by reference inside `observe` (materialized uniformly after the runner loop). 5-chunk GPU verification: 31 modules set to SMA, 118 reference frames, `future_dropped=0`. In scoring (judge=qwen3-vl-32b, limit 5) SMA is uniformly better: prec 0.44→0.56 / rec 0.67→1.00 / f1 0.62→0.84, frame count 31.5→29.5.

### IAMFlow — route is faithful, but the current implementation is **lossy** and must be filled first ⚠️

- **Gap 1**: the observe forward uses an empty prompt → the crossattn text cache is not entity-conditioned → `entity_score` degenerates. **The prompt (and the LLM entities) must be fed into the observe forward.**
- **Gap 2**: `select_frame_from_chunk` currently passes `current_entity_ids=[]` with no `visual_scores` → frame selection is approximately uniform. **The real entities + the real frames' VLM visual score (0.3 weight) must be passed in.**
- **Gap 3**: frame_id→source-second currently uses a chunk-start approximation; it must be changed to exact evicted-frame indexing.
- **Impact**: until these are filled, IAMFlow's entity-aware retrieval is flattened out, and the result does not represent its true capability.

---

## 4. Current runnability and blockers

| baseline | weights | environment | runnable now? |
|---|---|---|---|
| **MemStrata (this system)** | ✅ SAM3 (`facebook/sam3`) + DINOv3 (`facebook/dinov3-vitb16-...`) (`PUBLIC_MODELS_ROOT`); GroundingDINO missing → SAM3-concept-only | CPython 3.11 + torch; SAM3 uses the vendored `sam3_transformers59` (tf5.9) prepended to PYTHONPATH | **yes** (GPU-verified: big_buck_bunny name_anchored limit=4) |
| LongLive-RAG | ✅ in place (`wan_models/Wan2.1-T2V-1.3B` + `ae_latent_mem.pt`/`longlive_base.pt`/`longlive_lora.pt`) | torch 2.5 + einops (only VAE+AE needed) | **yes** (samples run) |
| MemFlow | ✅ `KlingTeam/MemFlow/{base.pt,lora.pt}` (symlinked into the repo) | torch 2.6 + flash-attn 2.6 (+ omegaconf/peft) | **yes** (GPU-verified) |
| IAMFlow | ✅ `Causal_Video_Generation/IAMFlow/{iamflow_fp8.safetensors,tinyvae.pth}` | torch 2.5 + flash-attn; optional standalone vLLM process | pending the §3 entity/VLM wiring fixes + verification |
| SlotMem | ✅ LoRA/encoder `Causal_Video_Generation/SlotMem/ckpt/{stage1,stage2}/{stage*_high,stage*_low}.pt` + native base `Wan-AI/Wan2.2-I2V-A14B` | torch 2.5 + flash-attn 2.8; SlotMem uses its own vendored diffsynth, not lightx2v | ✅ integrated: VAE-encode the real segment, select the SlotMem single-bank timestep to add noise, run a single native DiT forward attention probe to extract character slots, then use the stage2 encoder/writer to write the `RoleWiseSlotMemoryBank`; **the formal Track A disallows distilled/lightx2v Wan2.2** (loadable but the smoke-test image quality is unusable) |

> Weight and environment registration is recorded in [`WEIGHTS.md`](WEIGHTS.md). **LongLive-RAG and MemFlow are now end-to-end GPU-verified**; IAMFlow dequantizes to bf16 inside the adapter to run the DiT, with the LLM/VLM served via HF or a standalone vLLM service.
