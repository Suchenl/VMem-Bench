# Reproducing paper numbers

`pytest` does **not** reproduce paper tables. Default collection is `tests/` only (bench self-tests). Cross-package adapters under `scripts/evaluate_baselines/tests/` are excluded on purpose.

## Current CVPR 2027 manuscript

Start with [the audited reproduction checklist](docs/CVPR27_REPRODUCTION_CHECKLIST.md).
The main shared-generator results use `testbed_rerun_20260927`, **seed 2026 only**,
30 stories, and VMem-Bench commit `0b41d3df8d5c2b53ecc491ad00c769db2fa24a8b`.
The older `prod_retr_*` runs below are historical; they are not the current
shared-generator table. A three-seed command is a prospective protocol,
not a description of the paper's measured seed coverage.

## Choose the measurement you need

| Your question | Input / output boundary | Start here |
|---|---|---|
| What visual evidence does my memory select? | Track A: select references before reveal; observe the real segment afterward; emit `visual_selections` | [Track-A protocol](docs/trackA.md), [adapter/evaluation flow](docs/benchmark/running_eval.md) |
| What does my policy actually render? | Track B: chronological prompts, your generated history and clips; score description-matching presence, state, exclusion, and conditional identity proxies separately | [Track-B scoring](docs/benchmark/trackB_scoring.md) |
| How were the current paper's tables produced? | Frozen analysis snapshots and deterministic renderers; statistical reconstruction and new video generation are separate operations | [CVPR reproduction checklist](docs/CVPR27_REPRODUCTION_CHECKLIST.md) |

The two tracks use different instances; do not correlate their stories or treat
Track-A recall as a prediction of Track-B recall. For a new paired analysis,
retain reference selections and generated outcomes on the same inputs. Native
systems use their own generators; Last-seen is a privileged diagnostic control.
The paper supplement opens with an evaluation checklist explaining how these
choices affect interpretation.

## Code revisions and public availability

The public `src/vmem_bench` is the current implementation, including the main
FLUX.2 + Wan2.2 testbed and later H3/judge compatibility work. The paper-generation
commit identifies the code that produced the already-measured rollouts; it is
not evidence that the current implementation is missing those features. Use
current `main` for new evaluations. The archived source is optional provenance
for reconstructing the historical software environment, not the default install.

`main` is the moving working implementation. The initial public export starts a
fresh Git history; it does not contain the historical `paper-reproduction`
Track-A branch. That branch must not be used to claim reproduction of the current
Track-B tables.

For the recorded Track-B generation revision, unpack the bundled source archive:

```bash
mkdir -p /tmp/vmem-paper-generation
tar -xzf reproduction/generation-source.tar.gz -C /tmp/vmem-paper-generation
```

Use the outer release's cohort list and setup documentation with that snapshot.
The archived source has the original code and Track-B inputs; it does not include
model weights, the original Git history or raw generated videos/scores.
See [release contents](docs/PUBLIC_RELEASE.md).

## Track A (retrieval / visual coverage)

Protocol: [`docs/trackA.md`](docs/trackA.md), [`docs/benchmark/running_eval.md`](docs/benchmark/running_eval.md).

1. Obtain source videos — exact URLs, application page, and on-disk layout: [`docs/DATA.md`](docs/DATA.md). Smoke: `bash scripts/prepare_blender.sh`. List missing files: `python3 scripts/check_source_videos.py`.
2. Download gold JSON from [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench) (`trackA/`).
3. Run a SUT adapter under `scripts/evaluate_baselines/trackA/` so it emits `visual_selections`.
   For the MemStrata adapter, set **`MEMSTRATA_TRACKA_NAME_SOURCE=mllm`** to reproduce the paper mechanism; the default `perception` names entities with generic SAM3/DINOv3 concepts and under-recalls location/continuity entities. Requires `transformers>=5.9` (SAM3) and `ffmpeg` on PATH.
4. Score with `python3 -m vmem_bench.scoring.visual_coverage` (needs a pinned VLM judge).

The SUT sees prompt text + the **real** segment video; gold never goes to the SUT.

## Track B (generated pixels)

1. Download `trackB/` from the same Hugging Face dataset.
2. Run a generator SUT on `sut_prompts/` only.
3. Score with `python3 -m vmem_bench.scoring.end2end_coverage` against `gt/`, and the identity
   metrics (`identity_consistency`, `embedding_consistency`); protocol in
   [`docs/benchmark/trackB_scoring.md`](docs/benchmark/trackB_scoring.md).

The shared-generator rows (text-frame, seg-uniform, seg-DINO-key, seg-frame, no memory, last-seen)
run on the benchmark's own testbed, `python -m vmem_bench.testbed.run --variant <condition>`; flags,
seeds, run tags and the frozen provenance tags are in
[`docs/TRACKB_PAPER_CONDITIONS.md`](docs/TRACKB_PAPER_CONDITIONS.md).

## What this snapshot does not claim

Full Track-A and 30-story Track-B GPU rollouts are **not** executed in CI.
The current manuscript reports 92 films; the historical Stage-1 branch has a
different frozen scope. Table rendering uses saved analysis snapshots; a new
generation run requires weights, videos, and GPUs, and does not guarantee
byte-identical pixels. Local result symlinks are not a public artifact bundle;
see the checklist for what has and has not been verified.

## Citation

See [`CITATION.cff`](CITATION.cff).
