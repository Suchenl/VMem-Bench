# `vmem_bench` package architecture

`vmem_bench` is MemStrata's package for benchmarking, annotation, frozen releases, and deterministic scoring. It does not import the `memstrata` SUT; the SUT interacts with the benchmark only through the JSON contracts defined in `common.schemas`.

This file describes only package-level responsibilities, public boundaries, and migration rules. For the stages, artifacts, and human-review details of the VLM annotation pipeline, see the consolidated doc [`docs/benchmark/annotation_pipeline.md`](../../docs/benchmark/annotation_pipeline.md) (Chinese).

## Track A protocol (new causal protocol)

The old **gold-replay / ID-fidelity** protocol (the adapter machinery under `baseline_adapters/`, the v1 scoring harness in `scoring` — `runner`/`metrics`/`visual`/`__main__` — and the v1 embedder orchestration under `benchmark_run/`) **has been removed entirely**. Track A now runs only the causal protocol: for each chunk the bench gives the SUT a prompt plus the real segment; the SUT first calls `compose` to assemble context (persisted, for scoring and for finding figures for the paper) and then `observe_segment` to build memory; retrieved items are materialized into real frames according to their temporal identity, and `scoring.visual_coverage` performs VLM visual-coverage scoring. See [`docs/trackA.md`](../../docs/trackA.md) for the authoritative description.

## Directory responsibilities

```text
vmem_bench/
├── README.md                    # this file: package architecture and migration boundaries
├── __init__.py                  # package initialization and runtime environment constraints
├── common/                      # cross-module public contracts and deterministic utilities
├── baseline_adapters/           # ★ self-contained in-repo retrieval-family baselines (zero import of memstrata)
│   └── external/retrieval/      # four retrieval families + control conditions + self-contained encoder base
├── annotation/
│   ├── pipeline/                # the only maintained annotation production pipeline
│   ├── chunking.py              # shared chunk/layout utilities
│   ├── pipeline_track_first/    # legacy during migration, kept only as a copy/compat source
│   └── pipeline_vlm_dominant/   # legacy during migration, kept only as a copy/compat source
├── scoring/                     # SUT-agnostic VLM visual-coverage scoring + pinned embedders
├── publish.py                   # freeze movie gold → release package
├── judger/                      # legacy annotation-time VLM client; kept until migration completes
├── services/                    # legacy/optional resident model services
├── skills/                      # reusable algorithmic components, e.g. SBD
└── docs/                        # protocol, schema, design, and historical-decision docs
```

## Public boundary: must be preserved

The following directories/files are the public surface depended on across pipelines, across baselines, or by the release package. They are not deleted or turned into SUT-specific logic during the annotation refactor:

- `common/schemas.py`: the bench ↔ SUT JSON contract;
- `common/paths.py`: the movie-directory and asset-path contract;
- `common/gold_lint.py`: the strict gate for candidate/freeze/publish;
- `common/media.py`, `common/vecmath.py`, `common/model_weights.py`: stable base utilities;
- `baseline_adapters/external/retrieval/`: self-contained in-repo retrieval-family baselines (do not import the SUT);
- `scoring/`: `visual_coverage`/`end2end_coverage` VLM visual-coverage scoring + the `embedder` pinned embedders;
- `publish.py`: frozen-gold release;
- `common/schemas.py`: the authoritative definition of fields and protocol (`docs/trackA.md` is the prose description of the Track A protocol).

`scoring/` never constructs, imports, or special-cases any concrete SUT. The construction and running of causal baselines lives outside the package in `scripts/evaluate_baselines/trackA/baseline_adapters/causal/` (runner + per-baseline adapters + scoring driver).

## Annotation production boundary

[`annotation/pipeline/`](annotation/pipeline/) is the only production annotation implementation:

- all new VLM prompts, post-processing, automatic segment review, crop acquisition, web-based human review, freeze artifacts, and annotation batch orchestration go in this directory;
- it does not import or modify `pipeline_track_first/` or `pipeline_vlm_dominant/` at runtime;
- the old implementations may serve only as a one-time copy source; once copied, `pipeline/` maintains its own copy;
- `pipeline/` may use `common/`'s public contracts and base utilities read-only.

`annotation/chunking.py` is kept as a shared layout utility. The current v5 annotation uses `visual_segments` as the chunk layout and does not require the new pipeline to run SBD; the old pipelines and historical tests can still use SBD.

## Responsibilities of Track A evaluation orchestration

The dataset-level, SUT-aware causal evaluation orchestration lives outside the package under `scripts/evaluate_baselines/trackA/`: `baseline_adapters/causal/runner.py` drives a single baseline chunk by chunk (Stage 1), `scoring.visual_coverage` scores it (Stage 2), and `overnight_two_movie_run.sh` orchestrates two movies. For full runs, use your own job scheduler to invoke `baseline_adapters/causal/runner.py` repeatedly, and `scripts/get_trackA_assets/compare/build_leaderboard_v2.py` to aggregate the leaderboard. Per-movie evaluation artifacts are written to:

```text
<movie>/benchmark_run/{visual_selections,_visual_score,_ref_frames,_segments}/
```

Dataset/sample-level aggregation is collected by `scripts/evaluate_baselines/trackA/aggregate_trackA_outputs.py` into `outputs/evaluation/trackA/<baseline>/<dataset>/<sample>/`.

## Migration and archival rules

### Current migration period

- `annotation/pipeline_track_first/` and `annotation/pipeline_vlm_dominant/` are still referenced by old scripts, tests, historical experiments, and the current v5 gold entry point, and **must not be deleted immediately**.
- For each stage completed in the new `annotation/pipeline/`, it must have its own tests and CLI before the corresponding callers are migrated.
- No new prompts, sample JSON, or probe outputs are added to the old pipelines; new production assets are written only to `annotation/pipeline/`, `data/`, or `experiments/`.

### Preconditions for deletion or archival

Before any legacy file is deleted, all of the following must hold simultaneously:

1. All repo-wide imports and CLI references have been eliminated, or migrated to the new `pipeline/` / `benchmark_run/` entry points;
2. The corresponding public-contract tests, annotation-stage tests, and scoring tests pass;
3. At least BBB, one additional BlenderOpenMovies sample, and one LSMDC aggregate sample have passed end-to-end acceptance through the new pipeline;
4. Historical code is first moved into `benchmarks/MemStrata/_archive/annotation_legacy/` or marked `LEGACY` before physical deletion is considered.

The explicit first batch of archival candidates is:

- in-source-tree experiment artifacts such as `pipeline_vlm_dominant/web_vlm/**/outputs/*.json` (move to `experiments/` or the corresponding `data/`);
- multiple old prompts already superseded by the pinned v5;
- tests and scripts that serve only the old VLM-first `annotate_movie()`.

`common/`, `scoring/`, `publish.py`, `docs/schemas_and_contracts.md`, and `annotation/chunking.py` are not deletion candidates.

## Known entry-point debt

During migration, prefer fixing rather than working around the following mismatches:

- the `vmem_bench.annotation.pipeline_track_first.run` referenced by old scripts/docs does not match the actual legacy CLI path;
- the compat entry point for `vmem_bench.web.server` does not match the actual legacy web location;
- the old in-process scoring CLI docs/tests do not match the current records-dir CLI arguments;
- the adapter package claimed by the `baselines/` docs is not the currently runnable implementation.

These are migration tasks and should not be "fixed" by stuffing SUT construction logic into `scoring/`.

## Authoritative documentation hierarchy

1. `common/schemas.py` + [`docs/benchmark/schemas_and_contracts.md`](../../docs/benchmark/schemas_and_contracts.md) (Chinese): the public data and evaluation contracts;
2. [`docs/benchmark/annotation_pipeline.md`](../../docs/benchmark/annotation_pipeline.md) (Chinese): the annotation production stage details (S1–S7);
3. this file: package-level directories, boundaries, and migration rules;
4. [`docs/benchmark/annotation_tracking_internals.md`](../../docs/benchmark/annotation_tracking_internals.md) (Chinese), [`docs/benchmark/pitfalls.md`](../../docs/benchmark/pitfalls.md) (Chinese), etc.: historical/mechanism design records, which cannot override the three above.
