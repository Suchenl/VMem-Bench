# Reproducing the paper protocol

This branch is the frozen `paper-reproduction` VMem-Bench checkout. It preserves the
Track A Stage-1 protocol used for paper comparison. `pytest` checks the contracts and
local plumbing; it does not reproduce the full paper tables.

## Branch roles

| Branch | Role |
|---|---|
| `main` | Moving production and benchmark harness |
| `paper-reproduction` | Frozen Track A Stage-1 protocol and compatible scoring code |

## Track A: causal retrieval evaluation

Each SUT receives the prompt text and the real video segment. It must run its own
`compose -> observe` loop and write `visual_selections`; gold labels never enter the
SUT memory.

```bash
# Run from the VMem-Bench paper-reproduction checkout.
export MEMSTRATA_SRC="../MemStrata-paper/src"
export MEMSTRATA_TRACKA_NAME_SOURCE=mllm
python3 scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py \
  --adapter memstrata \
  --movie-dir assets/trackA/BlenderOpenMovies/big_buck_bunny \
  --budget 16 \
  --limit 2
```

Use [`docs/DATA.md`](docs/DATA.md) for source-video acquisition and layout. Full
91-movie runs require the source videos, a pinned VLM judge, GPU model weights,
`PUBLIC_MODELS_ROOT`, and the matching paper-reproduction MemStrata checkout.
Score the emitted selections with `vmem_bench.scoring.visual_coverage`.

## Track B: evaluate a generated method

This branch can score Track B submissions, but it does not claim to reproduce the
paper's later Track B tables. Generate one video per segment, write the standard
`progress.json` plus `review/segments/<segment_id>.mp4`, then run:

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.end2end_coverage \
  --gt assets/trackB/en/gt/0001_lighthouse_keeper.json \
  --prompts assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
  --run /path/to/your_run_dir \
  --api http://127.0.0.1:8110
```

## Bring your own method and annotation

Copy `scripts/evaluate_baselines/your_method/`, implement `compose` and `observe`,
and run its CPU dry run before using a real video. The S1-S7 annotation pipeline is
available through `scripts/get_trackA_assets/core/run_annotation.sh`; proposal-only
runs are diagnostics, while publishable gold requires a human-confirmed roster and
review/freeze gates.

## CPU self-check

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage --help
PYTHONPATH=src python3 -m vmem_bench.scoring.end2end_coverage --help
python3 scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py --help
PYTHONPATH=src python3 -m pytest -q
```

The checks above do not download videos or weights. See MemStrata's
[`MODELS.md`](https://github.com/Suchenl/MemStrata/blob/paper-reproduction/MODELS.md)
for GPU prerequisites and [`REPRODUCE.zh.md`](REPRODUCE.zh.md) for the Chinese
version of this guide.

## Scope and citation

Do not report a full paper-table reproduction until the external videos, pinned judge,
weights, and all required runs have completed. Code is Apache-2.0; self-authored gold
annotations are CC BY 4.0; third-party videos and model weights retain their original
terms. See [`CITATION.cff`](CITATION.cff).
