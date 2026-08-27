# Scoring modules — VLM visual-coverage (new causal protocol)

> Authoritative Track A protocol (including the new protocol's data flow, input modes, and artifact paths): [`docs/trackA.md`](../../../docs/trackA.md).
>
> The old v1 scorers (`metrics.py` ID-set intersection + `visual.py` gold-crop embedding nearest-neighbor + the `runner.py`/`__main__` gold-replay harness, plus the `vmem_bench.benchmark_run` v1 orchestration) **were removed entirely along with the gold-replay protocol**; they are no longer kept and no longer imported.

## Active (release): `visual_coverage.py` — VLM-based visual coverage, v2

The public MemStrata benchmark scorer. Track A: the system emits a *context* (a set
of reference images); the scorer judges them **visually** against the segment video,
using only **frozen text ground-truth** (roster + per-segment present set). No gold
crops, no state annotation.

Per-segment metrics (all in `[0,1]`, defined in the module docstring):

| metric | meaning |
|---|---|
| `precision` | fraction of the system's selected images that are actually on-screen (penalises over-retrieval) |
| `recall` **[headline, default]** | memory recall over **continuity** entities only (seen before → must be recalled); first-appearance entities excluded (cannot be recalled). This is THE recall. |
| `recall_all` | per-segment **details-only** diagnostic (incl. first appearances); not in the summary |
| `f1` **[headline]** | harmonic mean(precision, recall[continuity]) |
| `redundancy_vlm` | **per-entity** near-dup rate among on-screen refs, VLM counts distinct views; complementary views NOT penalised |
| `redundancy_sim` | **per-entity** DINOv3 (ViT-B/16, CLS token) self-similarity (mean upper-triangular cosine, threshold-free); 1.0=identical, lower=diverse |
| `efficiency` | useful (on-screen AND non-duplicate) refs / total refs — relevance-and-non-waste, does not reward minimalism |
| `budget_*` | descriptive context-size stats (avg refs/segment), **not** a score; compare at matched budget |

Reproducibility: "what should be present" is deterministic gold; only the per-image
visual judgement uses a **pinned** open VLM (`qwen3-vl-32b`). Always report alongside
a measured human-agreement / noise-floor number.

Run:

```bash
PYTHONPATH=src python -m vmem_bench.scoring.visual_coverage \
  --movie  data/BlenderOpenMovies/big_buck_bunny \
  --system memstrata_memstrata-fast \
  --video  <source_video.mp4>
```

Inputs are read from artifacts the causal runner already produces
(`gold/chunk_index.json`, `gold/entity_registry.json`,
`benchmark_run/visual_selections/<system>.json`). Per-segment clips are cut on demand.
The file/key names still contain `chunk` for backward compatibility with existing
gold and Stage-1 manifests; new scorer outputs use segment terminology while
keeping legacy aliases.

## Track B: `end2end_coverage.py` — end-to-end generated-video judge

Judges what the SUT **rendered** (not what it selected). Reads the per-segment GT
(`gt_version: trackB-gt-2.0`) authored under `assets/trackB/gt_source/` and compiled by
`assets/trackB/complete_gt.py`; the SUT-facing prompts come from
`assets/trackB/get_sut_prompts.py`. Uses the same pinned VLM (`qwen3-vl-32b`) on a **blinded
mixed roster** per segment = present(cast) ∪ forbidden ∪ false-friend targets ∪ deterministic
decoys.

Scores **per memory capability** (each with enough opportunities via the builder's balance
check) plus a **gap-stratified recall decay curve**; the headline recall is unweighted
(pooled micro over character+prop present):

| group | metrics |
|---|---|
| headline | `recall_char_prop`, `precision`, `f1`, `avoidance_ok` |
| recall abilities | `first_appearance`, `continuity`, `long_gap_reappearance`, `temporal_reference` (name-free intent recall) |
| state | `state_change` (transform moment), `persist_state` (state kept across time/gap → attribute-negation & cross-gap consistency) |
| identity | `lookalike_disambiguation` (correct/wrong instance), `false_friend` (`correct_id_rate` + `confusion_rate`, confusion scored only when the look-alike target is absent) |
| avoidance | `deprecation_avoidance`, `reference_indirect`, `lookalike_absent_avoidance` (Wilson 95% CIs) |
| quantity | `count_memory` (`exact_rate`, `off_by_one_rate`) |
| decay | `recall_by_gap` over strata `0 / 1-4 / 5-29 / >=30` |
| noise floor | `decoy_fpr`, `vote_self_consistency`, `abstain_rate` |

```bash
PYTHONPATH=src python -m vmem_bench.scoring.end2end_coverage \
  --gt      assets/trackB/gt/0001_lighthouse_keeper.json \
  --run     <sut_run_dir> \
  --prompts assets/trackB/sut_prompts/0001_lighthouse_keeper_name_anchored.json
```

## Other active modules

- `embedder.py` — pinned image embedders (DINOv3 ViT-B/16 CLS, ArcFace, MegaLoc); used by
  `redundancy_sim` here and reused by the self-contained retrieval baselines
  (`src/vmem_bench/baseline_adapters/external/retrieval/_retrieval_encoders.py`).

## Retired

- `_archive/trackb_gt.py` — the old screenplay-derived, per-shot Track B GT exporter. Track B
  GT is now hand-authored (`assets/trackB/gt_source/`) and compiled by `complete_gt.py`.
