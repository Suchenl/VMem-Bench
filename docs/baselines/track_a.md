# Track A = real retrieval / memory on GT visuals

> **Archived protocol note.** The historical gold-replay discussion below is
> not the public evaluation contract. For the runnable release protocol use
> [`docs/benchmark/running_eval.md`](../benchmark/running_eval.md) and
> [`REPRODUCE.md`](../../REPRODUCE.md), which drive each SUT with real video
> observations and self-produced memory.

> The historical terminology and examples below are retained for provenance
> only. They must not be used to generate new benchmark numbers: the release
> contract is real-video causal evaluation, with SUT-owned memory and no gold
> observation handout.
## What is swapped vs what is kept

| Component | Track A | Track B |
|---|---|---|
| Memory / retrieval algorithm | **Keep** (AE top-K, KV top-K, entity buffer, …) | Keep |
| History bank content | **GT** frames / crops / latents from gold | Self-generated latents / frames |
| Next-chunk video synthesis | **Not run** (or stubbed); score memory decisions only | Official generator scripts |

```
for chunk t in gold:
  bank  = GT visuals observed so far          # replaces self-generated history
  query = prompt_t (+ GT cue at t if needed)
  M_t   = baseline.retrieve(query, bank)      # REAL retrieval
  score(M_t → ComposedContextRecord, gold)
  bank ← bank ∪ GT observation_t              # post-score oracle / gold ingest
```

## Per-baseline Track A (what “real retrieval” means)

| Baseline | Track A (retrieval on GT) | Track B |
|---|---|---|
| **Helios** | Fixed 73-frame window + first-frame on GT timeline — the published memory *is* this policy | Full Helios generate |
| **LongLive-RAG** | Encode **gold** video → Wan latents → **AE cosine top-K** (`memory_size` / `recent_exclude` / `sink`) | `inference.sh longlive latentmem` |
| **MemFlow** | Gold-encoded history → **real per-layer KV-bank top-K + local window**, driven by the published base+LoRA (see below) | Official MemFlow infer |
| **DecMem** | Gold history → LTM top-K + anchored local — **not runnable here** (LTM kernel needs SM90a; needs WorldMem action/pose inputs) | Official DecMem infer |
| **IAMFlow** | Gold latents/frames/prompts → **real entity/active-memory** (DiT KV entity_score + VLM visual score + `retrieve_initial_frames`), driven by the published DiT+LLM+VLM (see below) | Full IAMFlow generate |

### Implementation status (BBB freeze)

| Baseline | Status | Notes |
|---|---|---|
| Helios | ✅ real | window policy = method |
| LongLive-RAG | ✅ real | AE cosine top-K on `gold/gold_latents.pt` |
| MemFlow | ✅ real | `scripts/baselines/memflow/run_bank_trace.py` → `gold/memflow_bank_trace.json` → `MemFlowAdapter` |
| DecMem | ⛔ blocked | LTM Video-Sparse-Attn kernel is `sm_90a`-only (host is A800/SM80); needs WorldMem action/pose control track absent from a passive film. `decmem.pt` is downloaded. |
| IAMFlow | ✅ real | `scripts/baselines/iamflow/run_agent_trace.py` → `gold/iamflow_agent_trace.json` → `IAMFlowAdapter`. Published DiT `iamflow_fp8.safetensors` (dequantized fp8→bf16 for SM80) + LLM `Qwen3-4B-Instruct-2507` (HF) + VLM `Qwen3-VL-2B-Instruct` (transformers, no vLLM). |

DecMem (the one blocked row) raises `NotImplementedError` with the exact reason;
its `*_budget_proxy` remains an ablation only and never enters the method table.

### Per-sample speed (BBB freeze, single A800, 52 chunks)

Cost to produce a memory decision for **one** bench sample (one chunk intent). Two
distinct numbers — do not conflate them:

- **read-path**: per-prompt latency the harness records while scoring the composed
  context (`efficiency.mean_composition_ms` in the leaderboard). For the
  generator-coupled methods this is tiny *only because* their decisions are
  precomputed into a replay trace and replayed here.
- **mechanism compute**: honest end-to-end cost of running the method's own memory
  mechanism per sample. For MemStrata/Helios this equals the read-path (decided
  online, no model forward). For MemFlow/IAMFlow it is the vendor model forward over
  the gold history, measured as `trace wall_seconds / 52`.

| System | Memory mechanism | read-path (ms/sample) | mechanism compute (per sample) | model forward? |
|---|---|---:|---:|---|
| **MemStrata (fast)** | name-match + dict deref | 0.13 | 0.13 ms | none |
| **Helios** | closed-form recency window | 0.03 | 0.03 ms | none |
| **LongLive-RAG** | `LatentAE` cosine top-k | 7.0 (p95 67.7) | ~7.0 ms | AE encode |
| **MemFlow** | base+LoRA per-layer KV bank | 0.19† | ~5.5 s | DiT base+LoRA |
| **IAMFlow** | entity/active-memory agent | 0.03† | ~32.5 s | DiT + LLM + VLM |

† decided online almost instantly because the decision is replayed from a precomputed
trace; the honest cost is the mechanism-compute column. Sources:
`iamflow_agent_trace.json` `wall_seconds=1687.4` (÷52 ≈ 32.5 s),
`memflow_bank_trace.json` `wall_seconds=285.5` (÷52 ≈ 5.5 s), read-path from
`data/_runs/bbb_track_a_iamflow_20260721/leaderboard.json`. Full-benchmark wall for a
BBB-scale film ≈ 24 min (MemFlow) / ≈ 28 min (IAMFlow) vs seconds for MemStrata. This
table is mirrored in the paper appendix (`app:per-sample-speed`).

### MemFlow real KV-bank driver

MemFlow's memory = a local attention window (recency) + a compressed long-term
**KV bank** whose historical blocks are chosen by MemFlow's own text-saliency
`compress_kv_bank` top-k, *independently per transformer layer*.
`scripts/baselines/memflow/run_bank_trace.py` loads the published `base.pt` + `lora.pt`,
streams the GT latents (`memflow_latents.pt`, native Wan-VAE) through the vendor
clean-context commit path with `q_bank=True`, and recovers each surviving bank
block's **source latent frame by bit-exact fingerprint** (compression only
`gather`/concatenates blocks, so retained blocks are byte-identical to the
committed `k_new` of a specific source frame). The per-chunk snapshot records
`bank_source_latents` (+ per-layer + vote), and `local_window_source_latents`.
`MemFlowAdapter` then projects those *real* selections to entities via frozen
gold presence (source latent → seconds → frozen chunk → gold entities). The
retrieval is 100% MemFlow's; only the frame→entity step uses frozen gold truth.
For a full film the 1024-position temporal RoPE table is rebuilt to the timeline
length (RoPE angle = p·θ, so this is the exact embedding at higher positions).

**Frame→entity representation (causal nearest-rep):** a retrieved latent belongs to
a source chunk where an entity is *present*, but the entity's gold crop may live at a
different chunk. The projection resolves each selected entity to the exact crop at the
source chunk, else its most recent crop at a chunk ≤ source (strict causality), else its
earliest crop — never an empty representation set. (An earlier version left ~13% of
selections with no representation, which inflated Compactness above its 1.0 ceiling.)
MemFlow's low Parsimony (~0.29) is the honest, expected tradeoff: bank_size=3 + a
12-frame local window across 30 layers unions to ~11 selected entities/chunk vs ~3
returning-present, i.e. broad recall / low restraint — not a projection error.

### IAMFlow real entity/active-memory driver

IAMFlow's memory is an MLLM-driven **entity/active-memory** path, not a standalone
retriever. `scripts/baselines/iamflow/run_agent_trace.py` runs IAMFlow's own agent stack over
the frozen GT history: it loads the published DiT `iamflow_fp8.safetensors`
(**dequantized fp8→bf16**: the checkpoint stores per-channel `weight` + `weight_scale`,
so `w_bf16 = w_fp8·scale` recovers the published weights exactly and runs on Ampere/SM80
without fp8 tensor cores), the LLM `Qwen3-4B-Instruct-2507` (HF backend), and the VLM
`Qwen3-VL-2B-Instruct` (transformers backend — same weights vLLM would serve, just an HF
forward; a separate vLLM process is optional). Per npb-block it (1) commits the GT latent
block through the DiT clean-context path so the `kv_cache`/`crossattn_cache` the bank reads
are the model's real self-/cross-attention keys over the **gold** visuals, (2) VAE-decodes
the block to pixels and scores them with the VLM (0.3-weighted visual score), and (3) runs
`AgentMemoryBank.select_frame_from_chunk`, which fuses the DiT-derived `entity_score` with
the visual score; on each prompt (gold-chunk) boundary `retrieve_initial_frames` recalls
historical frames by entity coverage. Only the *generated pixels / self-generated history*
are replaced by gold; the retrieval/selection is 100% IAMFlow's. Per gold chunk we snapshot
the active memory as of that chunk and map each retained frame `p{pid}_c{cid}_f{f}` back to
its source latent (block start + f). `IAMFlowAdapter` projects `retrieved_source_latents`
to entities via the same frozen-gold, strictly-historical, causal-nearest-rep projection as
MemFlow. The 1024-position temporal RoPE table is rebuilt to the full timeline (RoPE angle
= p·θ, exact at higher positions). On BBB, IAMFlow's score is low (~0.44): its entity
extraction is human/person-centric while BBB is an anthropomorphic-animal film, early
nature-only chunks yield empty memory, and `max_memory_frames=3` (+id_memory 4) is a narrow
buffer — an honest reflection of the method on this content, not a projection error.

Diagnostics (`sliding_window`, `full_history`, `recency`, `selection_oracle`) remain analysis-only.

### Embedder scope (MemStrata SUT only)

`score_memstrata.py --embedder {hash,dinov3,...}` selects the crop embedder for the
**MemStrata SUT only**; it does not touch any Track-A baseline (longlive_rag/memflow/
helios each carry their own retrieval). With the default name-match SUT variant
(`planner=None`), prompt-time asset selection is embedder-free, so on the frozen BBB
gold `hash` and `dinov3` score **identically** (verified delta 0.0). The embedder only
changes the asset bank's internal crop dedup/diversity, which does not alter the
composed context here. Use `--embedder-weights <local dinov3 dir>` to load a local
snapshot (e.g. `${PUBLIC_MODELS_ROOT}/facebook/dinov3-vitb16-pretrain-lvd1689m`) and
avoid a network fetch; the harness now resolves gold-relative `crop_path` to the crop
files under `<movie>/gold/` so real image embedders can actually load them (the `hash`
embedder tolerated missing files, which hid this).

## Low-level memory → context projection (required for KV / latent methods)

KV and latent selections are valid Track-A evidence: they are not discarded
merely because they are not entity IDs.  The adapter must retain the native
provenance at cache write time and carry it through every gather/compression:

```
native slot → source latent index → source seconds → frozen source chunk
```

`baseline_adapters.common.latent_projection.LatentTimeline` performs the
frozen latent-time → chunk conversion and rejects current/future evidence.  A
result must preserve raw layer/head/slot identifiers in its evidence sidecar;
the final entity/crop context is then a deterministic projection through the
frozen source-chunk presence and representation index.  Post-hoc semantic
guessing or an unlogged aggregation across layers is not a valid projection.

## What we do *not* do in Track A

- Claim that a name-match / budget-only CPU proxy *is* LongLive-RAG / MemFlow / …
- Require (or wrap) full vendor video generation to fill the main table
- Score final mp4 quality (that is Track B)

Budget-only proxies may exist as **ablations / scaffolding** under names like
`*_budget_proxy`; they are **not** the method rows in Table 1.

## Weight sources and placement

> Merged from the old `weights.md` (2026-07-22). Track A (**quantitative** gold-replay) only needs each method's
> **retrieval / memory** weights and does **not** run the full generation denoising loop; Track B (**qualitative** rollout)
> is what uses each vendor's official generation scripts + full checkpoint. The local public weights root is
> `${PUBLIC_MODELS_ROOT}/` (abbreviated as `.../` in the table below).

### Weights each method needs for Track A

| Method | Track A assets needed |
|---|---|
| Helios | none (closed-form window policy, loads no weights) |
| LongLive-RAG | `ae_latent_mem.pt` + Wan VAE to encode the **gold** video into `gold_latents.pt` |
| MemFlow | the released `base.pt` + `lora.pt`, running the real per-layer KV-bank top-k + local window on GT-encoded history |
| IAMFlow | LLM `Qwen3-4B-Instruct-2507` + VLM `Qwen3-VL-2B-Instruct` + DiT `iamflow_fp8.safetensors` (+`tinyvae.pth`); the DiT is required because entity_score comes from its forward (fp8→bf16 dequantization to run on SM80) |
| DecMem | LTM selection on GT — **not runnable on this node** (the LTM kernel only supports SM90a; and it needs WorldMem action/pose inputs); `decmem.pt` is downloaded but serves only as the ablation `*_budget_proxy` |

### Local paths (in place)

| Asset | Path | Status |
|---|---|---|
| Wan2.1-T2V-1.3B | `.../Wan-AI/Wan2.1-T2V-1.3B` | ✅ (VAE encode gold) |
| LongLive AE | `.../Causal_Video_Generation/LongLive-RAG/checkpoints/ae_latent_mem.pt` | ✅ |
| MemFlow ckpt | `.../KlingTeam/MemFlow` | ✅ Track A real trace (`base.pt`+`lora.pt`) + Track B |
| DecMem | `.../KlingTeam/DecMem` | ⛔ Track A blocked on this node by the SM90a kernel; Track B / ablation proxy only |
| Helios-Distilled | `.../BestWishYsh/Helios-Distilled` | ✅ Track B (Track A needs no weights) |
| Qwen3-4B-Instruct-2507 (IAMFlow LLM) | `.../Qwen/Qwen3-4B-Instruct-2507` | ✅ |
| Qwen3-VL-2B-Instruct (IAMFlow VLM) | `.../Qwen/Qwen3-VL-2B-Instruct` | ✅ |
| IAMFlow DiT (`iamflow_fp8.safetensors`, `tinyvae.pth`) | `.../Causal_Video_Generation/IAMFlow/` | ✅ (fp8→bf16 dequantization at load) |

Track B uses each vendor's official scripts under `baselines/Causal/*/`; we do not wrap generation the way the Track A path does.

## CLI (as adapters mature)

```bash
PYTHONPATH=src \
python3 -m vmem_bench.baseline_adapters.run_gold_replay --list
```

Helios, LongLive-RAG, MemFlow and IAMFlow run as real online gold-replay Track-A
rows. DecMem is hard-blocked on this node (see the status table); its budget
proxy stays ablation-only and is never promoted to a method row.
