# MemStrata-Bench operating manual (end-to-end evaluation flow)

> Status: **the current authoritative operating manual**. For scoring formulas / data shapes, see [`scoring_v2.md`](scoring_v2.md) (Chinese).
> This document explains **how to take a movie from S4 gold to a score**, and **exactly what this evaluation measures and what must never be done**.
> Anyone — human or agent — about to touch this benchmark should **read §0 (the three iron rules) + §1 (the SUT contract) first**, before the operational steps that follow.

---

## 0. The three iron rules (hard constraints; violating any invalidates the evaluation — do not ask twice)

1. **Gold = S4 human-reviewed annotation, and text only.**
   Gold is always produced by [`build_gold_from_s4_review.py`](../../scripts/get_trackA_assets/maintenance/build_gold_from_s4_review.py)
   from `tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json`.
   Gold **contains only text** (roster + per-chunk present / first_appearances / prompt / seconds_span),
   and **contains no crop pixels** (every representation's `crop_path` is empty, deliberately).
   **Do not** use any old gold from a non-S4 source.

2. **Reference images (context) are always produced by the SUT itself; the bench hands out no images.**
   The SUT observes the real chunk video, does its own perception/decomposition, and stores **its own crops** in **its own memory bank**.
   Scoring scores exactly these SUT-produced images. **Do not** feed gold crops (or any bench-provided pixels) as the SUT's memory/context —
   that is the deprecated, incorrect protocol (see the warning in §5).

3. **The SUT can only see the prompt + video, and never sees the "answer."**
   For each chunk, the bench gives the SUT only: ① that chunk's **prompt text**; ② that chunk's **real video clip**.
   **The prompt = the S4 human-reviewed screenplay `action` (+ sound effects) verbatim**, where entities are mentioned only as the screenplay's natural narration names them (who is present is naturally named by the prose, which is exactly the source of the name-anchor).
   **Never inject** a suffix like `Canonical entities in this chunk: ...` that explicitly lists the full present roster
   (this was deleted on 2026-07-24 along with `ensure_prompt_entity_coverage`; prompt completeness is only "measured," not "rewritten," see §4).
   **It is absolutely forbidden** to feed the SUT gold's `present` / `first_appearances` / roster entity_id lists —
   that would directly tell the system under test "who should appear in this segment." "Who should be there" is used **only on the bench scoring side** and is invisible to the SUT.

> These three are the foundation of this benchmark. gold = S4 text, images produced by the SUT, the SUT never sees the answer — if any one of these is missing, the resulting scores are meaningless.

**Input registers (the fairness axis; both must be run and reported side by side)** — `bench_adapters/causal/runner.py --input-mode`:
- `name_anchored` (default, main setting): prompt = the verbatim S4 screenplay prose, with entities naturally named by the narration.
- `description_provided`: on top of name-anchored, a deterministic rule appends an appearance description of "what this entity looks like" at the end of the prompt,
  **describing only objects already named in that prompt and that are identity entities (`kind==character`)** (not leaking present/roster).
  **Props (prop) / locations (location) are not described** — adding an appearance to an "open meadow" or an "apple" is pure noise for entity retrieval, and it also dilutes the character signal of baselines that rely on text
  saliency (MemFlow) and pushes the prompt over umT5's 512-token limit; this was exactly why the early "describe everything"
  made descprov worse than name_anchored (corrected on 2026-07-25 to describe characters only). It gives systems that match memory by
  appearance/description a fair textual handle, avoiding a name_anchored-only setting that favors name-based retrieval systems. Output name `<adapter>__descprov`,
  with both registers coexisting and scored separately. For details and reporting, see [`../experiments/fairness_experiment_plan.md`](../experiments/fairness_experiment_plan.md) (Chinese).

---

## 1. The SUT contract (one table for the whole boundary)

**SUT = any system that consumes this benchmark** (`memstrata` is one such SUT, evaluated on the same S4 gold on equal footing with all baselines;
"SUT" does not specifically mean memstrata). Driven chunk by chunk in time order:

> **Dual evaluation registers (important, do not misread again)**: `memstrata` and all baselines (longlive_rag / memflow /
> memflow_sma / iamflow / slotmem …) are **completely equal in status**, all just SUTs. They are measured under **two evaluations**:
> ① **this memory benchmark** (the current document) — replace the generator output with the real segment and measure only the "memory mechanism," producing
> precision/recall/F1/redundancy/selection_efficiency + time-efficiency metrics;
> ② **the real MemStrata production pipeline** (`scripts/memstrata/run_production.sh`,
> `python3 -m memstrata.production.run`) — plug the same SUT into
> the end-to-end "script → generate → look back to build memory → generate again" loop, and measure its **output quality and stability** on real long-form video with generator noise.
> So memstrata is **not** the "judge/bench side," and the baselines are **not** present only inside the bench; in both evaluations they are the objects under test,
> with a consistent interface (prompt+video in, memory/context out). When writing code/docs, do not special-case memstrata as part of the bench.

| Step | bench → SUT (input) | SUT → bench (output) | The SUT never sees |
|---|---|---|---|
| Compose context | that chunk's **prompt text** | a set of **reference images it stored itself** (context) | `present` / `first_appearances` / roster ids |
| Observe memory | that chunk's **real video clip** | (nothing; only updates its own memory bank) | gold crops / any bench pixels |

- Temporal discipline: first `handle_prompt` (compose context using the **already-built** memory), **then** `handle_observation`
  (hand this chunk's video to the SUT to update its memory). When composing context for chunk t, the SUT **cannot** have already seen chunk t's video observation.
- Carrying entity names in the prompt is **allowed** (this is exactly the "name-cued memory recall" mechanism); carrying structured present/first_appearances is **not allowed**.
- **All SUTs (including all baselines: helios / retrieval / causal, etc.) are treated identically**: all build memory and produce images using only the prompt + the real video,
  and **no system may do gold-crop replay**.
- **The bench hands the SUT only two things, and never hands out images**: ① that chunk's **prompt text**; ② that chunk's **original segment video**.
  This segment is **used to replace the SUT generator's output** — in a real system, the SUT would first generate a clip from context and then look back to build memory,
  and here we just hand it the real segment directly, **eliminating generator noise**, so the evaluation focuses only on the "memory mechanism."
  The bench **does no perception, produces no crops, hands out no images or ID answers**.
- **Perception + memory always live on the "method side (SUT)," not the bench**: after receiving the segment, the SUT decides on its own how to turn it into memory —
  detection / cropping / encoding / storage / recall all done by that SUT's (baseline's) adapter. Multiple baselines may **share one method-side perception frontend**
  (the same detect→crop→embed), which is a "method-side tool," a separate matter from "the bench handing out images"; their real difference is only in **memory / selection strategy**.
- **The adapter must not maintain or back-fill memory/retrieval results on the SUT's behalf**: each SUT's memory read/write must come from its own native mechanism
  (e.g. MemStrata's `AssetBank`/`MemoryUpdater`/`IntentInterpreter+compose`, SlotMem's `RoleWiseSlotMemoryBank`, MemFlow's KV bank/SMA routing, IAMFlow's `agent_memory_bank`,
  LongLive-RAG's latent descriptor pool). The bench adapter is only allowed two kinds of glue:
  ① wire up `prompt -> compose` and `real segment -> observe/write` per this protocol;
  ② project the memory representation the SUT has already selected/retained into **temporal refs with `source_seconds`**, for the bench
  to materialize reference frames from the real video. **It is forbidden** to add "if the SUT did not recall, back-fill the most recent frame/gold/roster" fallbacks in the adapter;
  if a real SUT returns empty, record it as empty, and the score naturally reflects that read path's capability.
- Scoring (Stage 2) looks only at the context — this set of SUT-produced images: does it cover the continuity entities, does it over-recall, is it redundant;
  "who should be there" comes from the S4 gold text and is used only on the bench side.

---

## 2. Terminology

- **frozen gold (= S4 human-reviewed)**: `<movie>/gold/`, `human_reviewed=true`, text GT only, read-only for evaluation.
- **context / visual selection**: a set of **self-produced** reference images the SUT composes for each chunk from **its own memory bank**.
- **continuity entities**: `present \ first_appearances`, i.e. "entities seen before, that should be recalled from memory this time"; recall is computed only over these.

---

## 3. Three-stage overview

| Stage | What it does | Needs GPU/VLM? | Artifacts |
|---|---|---|---|
| **Stage 0** | Convert S4 human-reviewed annotation into `gold/` (text GT) | No | `<movie>/gold/{chunk_index,entity_registry,chunk_annotations}.json` |
| **Stage 1** | Each SUT observes the real chunk video, **builds its own memory bank**, and produces **self-produced** context chunk by chunk | **Yes** (perception/decomposition: GroundingDINO/SAM3/DINOv3, etc.) | `outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/<system>.json` (pointing to SUT-stored crops/temporal refs) |
| **Stage 2** | VLM visual-coverage scoring | **Yes** (`qwen3-vl-32b` judge; DINOv3 only for the `redundancy_sim` diagnostic column) | `outputs/evaluation/trackA/<system>/<dataset>/<movie>/_visual_score/<system>/score.json` |

---

## 3.1 Resolution convention (uniformly preprocess to 480p / 832×480)

**The rule: all video pixels fed to models/judge are uniformly preprocessed to 832 wide × 480 high** (Wan2.1-T2V-1.3B's native size, 16:9),
not the source resolution (source films are often 720p/1080p, which is slow and expensive and unnecessary for judging). Three points enforce this uniformly, all already in code:

| Step | Location | Handling |
|---|---|---|
| Stage 1 segment cutting | `baselines/bench_adapters/causal/runner.py:_cut_segment` | ffmpeg `scale=832:480` to disk |
| Stage 1 SUT observation decoding | `baselines/bench_adapters/causal/_video_io.py` (`WAN_W=832,WAN_H=480`) | scale to 832×480 on decode |
| Stage 1 reference-frame extraction | `baselines/bench_adapters/causal/frame_materializer.py:_cut_frame` | ffmpeg `scale=832:480` |
| Stage 2 judge video clip | `src/vmem_bench/scoring/visual_coverage.py:_cut_clip` (`JUDGE_CLIP_W/H`) | ffmpeg `scale=832:480` |
| Stage 2 reference-image embedding | same `_img` (`JUDGE_IMG_MAX_SIDE=384`) | downscale to max side 384px before sending |

**Why 480p is fair**: the SUT perceives only at 480p throughout, and judging at the same resolution is both fair and fast; judge reference images are further compressed to
384px to control tokens (a large-footprint system can reach 40+ images per chunk). The release fixes these constants; do not raise them ad hoc.

> Existing old artifacts (`_segments/*.mp4`, `_ref_frames/*`, `_clips/*.mp4`) **will not be re-cut automatically** because of the `if out.is_file()` cache;
> to let old clips enjoy the 480p speedup/consistency, delete the corresponding cache directory and re-run (the decode side `read_segment_pixels` already re-scales to 832×480,
> so tensors are unchanged and only speed/disk are affected).

---

## 4. Stage 0 — convert S4 human-reviewed annotation into gold (text GT)

```bash
cd VMem-Bench
PYTHONPATH=src $PY scripts/get_trackA_assets/maintenance/build_gold_from_s4_review.py \
    --movie-dir data/BlenderOpenMovies/big_buck_bunny
```

Reads `tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json`
and converts it into the standard 3-file gold (`chunk_annotations.json` with present/first_appearances/prompt/seconds_span,
and every representation's `crop_path` empty). **The S4 gold is the final gold standard to be released.**

- **The three files have distinct roles, no duplication**: `chunk_index.json` = thin layout (chunk_id + shot/frame/seconds spans + `layout_hash`);
  `chunk_annotations.json` = per-chunk rich GT (present/first_appearances/prompt/..., **the scorer reads from here**);
  `entity_registry.json` = the entity roster. They are read by the scorer / harness+freeze+publish / roster respectively, all indispensable.
  (Since 2026-07-24, `chunk_index` no longer redundantly carries rich annotations.)
- **prompt = the S4 screenplay `action` (+ sound effects) verbatim**, with **no canonical-entity suffix injected** during conversion
  (`ensure_prompt_entity_coverage` has been deleted, see iron rule 3). The field name in gold is uniformly `prompt`, with no `action` wording retained.
- **Gold is minimal**: each movie has only the three JSON files above, no crop pixels, no embeddings, no tmp.

## 5. Stage 1 — each SUT builds its own memory bank and produces self-produced context (needs perception services)

**The core of this benchmark, and the step most easily done wrong.** The correct protocol (strictly follow §0 / §1): the bench does two things per chunk in time order,
and everything else is on the method side:

1. **Hand the prompt to the SUT** → the SUT composes context using its **already-built memory** and stores it (`handle_prompt`). This context is the scoring target.
2. **Hand that chunk's original segment video to the SUT** (replacing the generator output, eliminating generation noise) → the SUT's adapter does
   perception + memory on this real video (detect→crop→embed→store, `handle_observation`), updating its own memory bank for later chunks to recall.

The `crop_abspath` in `visual_selections/<system>.json` **must come from real frames/images materialized via temporal refs out of the SUT's own memory representation**,
never `gold/crops/...`, and there will be no bench-provided images; if a SUT's internal memory is not explicit crops (e.g. KV, latent, slot),
the adapter may only project the source time it has already retained/recalled into `source_seconds`, and the bench then uniformly cuts the real reference frames.

- **The bench side** is only responsible for: parsing the source video, cutting each chunk's **original segment** according to `seconds_span`, driving the two-step loop above, collecting and saving
  each SUT's context, and finally scoring. **The bench does no perception, no cropping, no image handout.**
- **The method side (each baseline's adapter)** is responsible for: perception (detect/crop/embed) + the memory mechanism (store/dedup/recall/lifecycle) + compose.
  Multiple baselines may share the same **method-side perception frontend** (ensuring fairness and burning GPU only once); their difference is only in memory/selection strategy — this is still a method-side tool,
  not "the bench handing out images."

> ⚠️ **Do not use the retired gold-replay runner as Stage 1.** Feeding gold
> observation crops or labels into a SUT violates the no-image-handout and
> no-gold-leakage rules. That retired path is kept only as historical context
> and cannot produce a valid public score. Use
> `scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py`.
>
> The correct Stage 1 driver is
> `scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py`.
> It cuts the real segment, drives `compose → observe`, and writes the
> `visual_selections` manifest used by Stage 2. See
> `REPRODUCE.md` and `scripts/evaluate_baselines/your_method/README.md` for
> copy-pasteable commands.

## 6. Stage 2 — VLM visual-coverage scoring

Prerequisites (all indispensable):

1. **VLM judge resident**: `qwen3-vl-32b`, OpenAI-compatible `POST /v1/chat/completions`, default `http://127.0.0.1:8110`
   (`--api` to change). `temperature=0`, `fps=2.0` are pinned. **Without the service running, Stage 2 cannot run.**
2. **Source video**: resolved via `data/dataset_dirs.txt` (see §7). The scorer cuts chunk clips on the fly with ffmpeg.
3. **ffmpeg**: default `ffmpeg` (`--ffmpeg` to change).
4. **DINOv3** (optional): used for the `redundancy_sim` column; when torch/weights are unavailable this column is `null` and does not block the headline.

```bash
cd VMem-Bench
# Smoke: score only the first 5 chunks to verify the pipeline; drop --limit for a full run
PYTHONPATH=src $PY -m vmem_bench.scoring.visual_coverage \
    --movie  data/BlenderOpenMovies/big_buck_bunny \
    --system <system> \
    --video  ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4 \
    --limit  5
```

`<system>` = the filename under `outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/` with `.json` removed
(run once for each SUT you want to compare). Artifacts:
`outputs/evaluation/trackA/<system>/<dataset>/<movie>/_visual_score/<system>/{score.json,details.json}`;
chunk clips are cached in the same run directory's `_clips/`.

## 7. How source video paths are resolved

`data/dataset_dirs.txt` is the list of dataset root directories (registered inside the benchmark, with paths that may point to a large disk):

```
BlenderOpenMovies: ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos
LSMDC: ${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched
```

A movie's video = `<that dataset's root>/<movie_id>/<video_file>`, e.g.:
`big_buck_bunny` → `.../BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4`.

## 8. Movie-level → corpus-level

Single-movie per-chunk → movie-level mean (`score.json.summary`); across corpora, take the macro-average over movies (see [`scoring_v2.md`](scoring_v2.md) (Chinese) §4.8).
The release includes a noise floor (`scoring_v2.md` §5).

## 9. Common pitfalls

- **Used non-S4 old gold** → violates iron rule 1. Gold must be converted from S4 by Stage 0.
- **`gold/crops/...` appears in context** → violates iron rule 2; images must be SUT-produced, evaluation invalid.
- **Fed present/roster to the SUT** → violates iron rule 3, equivalent to leaking the answer, evaluation invalid.
- **Stage 2 cannot connect to 8110 / times out** → the `qwen3-vl-32b` judge is not running or is unhealthy.
- **`redundancy_sim` is all `null`** → the scoring machine lacks torch/DINOv3 weights; does not affect the headline.

---

## Appendix A: the interpreter

Use any Python with the dependencies installed (the same interpreter as the scorer is fine, with ffmpeg on PATH):

```bash
PY=python3
# All python3 below uses $PY; the orchestrator starts subprocesses via sys.executable, inheriting the same interpreter.
```
