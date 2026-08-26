# Reproducing paper numbers

`pytest` does **not** reproduce paper tables. Default collection is `tests/` only (bench self-tests). Cross-package adapters under `scripts/evaluate_baselines/tests/` are excluded on purpose.

## Branches

| Branch | Meaning |
|---|---|
| `main` | Production harness. May move. |
| `paper-reproduction` | Intended freeze for paper metrics. **Currently the same commit as `main`** (first public snapshot). |

## Track A (retrieval / visual coverage)

Protocol: [`docs/trackA.md`](docs/trackA.md), [`docs/benchmark/running_eval.md`](docs/benchmark/running_eval.md).

1. Obtain source videos yourself (Blender / LSMDC).
2. Download gold JSON from [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench) (`trackA/`).
3. Run a SUT adapter under `scripts/evaluate_baselines/trackA/` so it emits `visual_selections`.
4. Score with `python -m vmem_bench.scoring.visual_coverage` (needs a pinned VLM judge).

The SUT sees prompt text + the **real** segment video; gold never goes to the SUT.

## Track B (generated pixels)

1. Download `trackB/` from the same Hugging Face dataset.
2. Run a generator SUT on `sut_prompts/` only.
3. Score with `python -m vmem_bench.scoring.end2end_coverage` against `gt/`.

## What this snapshot does not claim

Full 91-movie Track A Stage 1 and 30-story Track B GPU rollouts are **not** executed in CI. Matching the paper tables is Phase 3 work: weights, videos, and GPUs must be supplied by the reader.
