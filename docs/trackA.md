# Track A protocol (new causal protocol)

Track A evaluates "the retrieval quality of visual memory." **The old gold-replay protocol has been removed entirely**, and Track A now runs only the causal protocol below. The old protocol (`run_gold_replay.py` / `registry` / `convert` / `common/` / `diagnostics/` / `external/causal/` / `external/scripted/` under `src/vmem_bench/baseline_adapters/`, plus the gold-replay orchestration and the various gold-trace/latent production scripts under `scripts/get_trackA_assets/compare/`) has been removed from the repository; it is no longer kept and no longer imported.

## The three iron rules

1. **The bench does no perception**: no cropping, no clustering, no presence judgment.
2. **The bench hands no images to the SUT**: the SUT receives only the prompt text + the real segment video, and never gold crops / gold entity ids.
3. **The bench leaks no answers**: no `present`/roster provided, no labels on reference images.

## Data flow for each chunk

In time order, for each chunk `t`:

1. **Cut the real segment**: from the source video, the bench cuts the chunk's real clip according to `seconds_span` (pure IO).
2. **`compose(prompt)`**: hand the **prompt text** to the SUT. The SUT retrieves from its **current memory** (built only from history chunks `< t`) and returns memory items carrying a **temporal identity** (`source_seconds` / `source_chunk_id` + `evidence_kind`).
3. **`observe_segment(real clip)`**: hand this chunk's **real segment** to the SUT so it can update memory through its own native memory-write path.

`compose` **strictly precedes** the `observe_segment` of the same chunk, so when the SUT composes context for chunk `t`, it can never peek at chunk `t`'s video.

**The real segment replaces the SUT's generated output** to eliminate generation noise. In principle, **skip generation whenever possible**:

- MemStrata (this system) and the retrieval families (see below) are perception/retrieval computations and **do not need** any generator forward.
- LongLive-RAG's retrieval is pure self-encoded descriptor computation and likewise **does not need** a generator forward.
- MemFlow / IAMFlow / SlotMem write memory inside the generator forward, but all only need a **single teacher-forced real-latent forward** for extraction (`context_noise=0`), **not** multi-step denoising generation: MemFlow/IAMFlow fill KV / memory frames; SlotMem's character-slot extraction (`_extract_memory_from_current_step`) is essentially an **attention probe of a single DiT forward** — register a hook, run one forward to get the attention map, and characters are located by the character-name tokens in the prompt (`find_token_index_in_prompt`; `name_anchored` already provides names, so no roster is needed, and `char_latent_boxes` is only an optional refinement). Therefore **no chunk needs true multi-step generation**.

> **SlotMem status: integrated into the TrackA adapter.** The adapter uses native `Wan2.2-I2V-A14B`
> + SlotMem's own stage1/stage2 LoRA/encoder, running under torch 2.5 + flash-attn 2.8:
> VAE-encode the real segment → select the SlotMem single-bank timestep to add noise → a single native DiT forward with an attention probe
> → the stage2 slot encoder/writer writes the `RoleWiseSlotMemoryBank`. **Formal TrackA experiments forbid the distilled
> Wan2.2/lightx2v version**: distilled + SlotMem LoRA can load and generate video, but the smoke-test visual quality is unstable (smearing,
> blocky background, geometric drift) and cannot serve as a fair formal result.

## Causal guardrail

When `frame_materializer.py` materializes a temporal identity into a reference frame, it drops items with "source time ≥ the current chunk's start" (a causal SUT can only draw on the past), and the drop count is recorded in the manifest (`future_dropped`).

## Artifacts and persistence

All artifacts are written under `<movie>/benchmark_run/`, and **the context composed by compose is not deleted** (used both for scoring and for picking qualitative comparison figures for the paper):

| Path | Content |
|---|---|
| `benchmark_run/visual_selections/<run_name>.json` | the context composed for each chunk (the selected memory items + the resolved reference frames + this chunk's prompt), **kept permanently** |
| `benchmark_run/_ref_frames/<run_name>/` | reference frames cut from the real source video (for scoring + finding figures for the paper) |
| `benchmark_run/_segments/chunk_NNNNN.mp4` | the real segment cut for each chunk |
| `benchmark_run/_adapter_work/<run_name>/finalize.json` | run-level metadata (input_mode / budget / memory size / retrieval mode, etc.) |

`<run_name>` naming rule: `name_anchored` uses `adapter.name`; `description_provided` uses `<adapter.name>__descprov`; if `--budget B` is specified, `__B<B>` is appended. Artifacts of the two input modes therefore never overwrite each other.

## Input modes

Track A has **only two** canonical input modes: `name_anchored` and `description_provided`, reported side by side (the fairness axis).

- **`name_anchored` (default, main table)**: the prompt is the verbatim S4 screenplay prose, where recurring entities are referred to by their natural names. Systems that index memory by name (such as MemStrata's name-anchoring, and text-conditioned baselines) get a strong textual handle here.
- **`description_provided`**: on top of the `name_anchored` prompt, for entities whose **name already appears in that chunk's prompt**, a deterministic appearance-description suffix is appended (of the form `[Entity appearance reference] Name: appearance…; …`). It **only adds appearance text and never removes names**, so that systems that "match against their own visual memory via appearance descriptions" also get a fair textual handle. Leak-safe: only entities already named in the prompt are described (just those few per chunk), and no `present`/roster is ever exposed; the same deterministic rule applies equally to all systems. All entity kinds (character / prop / location) can be described — props and locations are also first-class recurring visual identities.

> The runner `runner.py` also accepts a diagnostic `description_only` mode (which replaces registered names already present in the prompt with neutral referents + appends appearance text, likewise never exposing the roster). It is **not part of the canonical main table** and is used only for stress/ablation purposes.

In either mode, the bench deterministically rewrites the prompt text; the entity metadata in `gold/entity_registry.json` is used **only** to generate the appearance-description suffix and is **never** handed to the SUT as a present/roster list.

## Scoring

Reference frames are scored by `vmem_bench.scoring.visual_coverage` for VLM visual coverage (method-neutral, no gold injected, no labels on reference images); the leaderboard is aggregated by `scripts/get_trackA_assets/compare/build_leaderboard_v2.py`, and qualitative comparison figures can be parsed from `visual_selections/` by `scripts/get_trackA_assets/compare/export_visual_selections.py`.

## Baselines

The bench adapter code for the causal baselines is in [`scripts/evaluate_baselines/trackA/baseline_adapters/causal/`](../scripts/evaluate_baselines/trackA/baseline_adapters/causal/README.md) (one `build_adapter()` factory per baseline, with the vendored upstream repos untouched).

The retrieval families (`text_frame_retrieval` / `text_segment_retrieval_then_uniform_sampling` / `text_segment_retrieval_then_dino_keyframe_sampling` / `text_segment_retrieval_then_frame_retrieval` + recency / bm25 / random diagnostic controls) are a **self-contained in-repo implementation**, located at `src/vmem_bench/baseline_adapters/external/retrieval/`, which **does not import the SUT package `memstrata`** (the encoder base is in the same directory, `_retrieval_encoders.py`) and is hooked into the causal protocol via `baseline_adapters/causal/retrieval_family.py`.

## Running

```bash
# Use a Python with the appropriate dependencies for each baseline (see the header comment of each adapter module)
PY=python3
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal

# Stage 1: drive one baseline through a whole movie (name_anchored main table)
$PY runner.py --adapter longlive_rag \
  --movie-dir <movie_dir> --input-mode name_anchored

# description_provided mode (artifacts land in <name>__descprov, not overwriting the main table)
$PY runner.py --adapter longlive_rag \
  --movie-dir <movie_dir> --input-mode description_provided

# Stage 2: VLM visual-coverage scoring
$PY -m vmem_bench.scoring.visual_coverage \
  --movie <movie_dir> --system <run_name> --video <source_video>

# Stage 3: aggregate the per-movie benchmark_run/ results by baseline/dataset/sample
PYTHONPATH=src python scripts/evaluate_baselines/trackA/aggregate_trackA_outputs.py
# -> outputs/evaluation/trackA/<baseline>/<dataset>/<sample>/<input_mode>[/B<budget>]/
#    {score.json, visual_selections.json, finalize.json, meta.json}
#    + per-baseline aggregate.{json,md} + top-level leaderboard.{json,md}
```
