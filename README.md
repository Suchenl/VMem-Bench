# VMem-Bench

Memory-aware benchmark for **causal long-video generation**.

- Code: this repository (`vmem_bench`)
- Gold / prompts (no source videos): [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench)
- Method under test (one SUT among others): [github.com/Suchenl/MemStrata](https://github.com/Suchenl/MemStrata)

`main` is the production tree. `paper-reproduction` freezes the paper-metric snapshot (currently the same commit as the first public `main`).

## What it measures

| Track | Question | SUT input | Scoring gold |
|---|---|---|---|
| **A** | Can the memory mechanism retrieve the right visual identity over a movie timeline? | per-chunk **prompt text + real video segment** (the real clip *stands in for* generator pixels) | frozen text gold (`entity_registry`, `chunk_annotations`); **no gold crops** |
| **B** | Does visual memory survive all the way into **generated** pixels? | chronological **prompt stream only** | authored stories + `memory_probes`; SUT never sees GT |

Protocol details: [`docs/trackA.md`](docs/trackA.md), [`assets/trackB/README.md`](assets/trackB/README.md), [`docs/benchmark/running_eval.md`](docs/benchmark/running_eval.md).

## Fairness contract

Copied from [`AGENTS.md`](AGENTS.md). A violation invalidates the numbers.

1. Identical inputs to every SUT (including MemStrata).
2. No test-set fitting (no eval-derived lexicons or per-case branches).
3. Adapters must not add capabilities the method does not have.
4. Symmetric preprocessing — or none.
5. No gold leakage to any SUT.

`src/vmem_bench` imports only `vmem_bench`. Cross-package adapters live only under `scripts/evaluate_baselines/`.

## Layout

```
src/vmem_bench/          # protocol, scoring, annotation (self-contained)
scripts/evaluate_baselines/   # SUT adapters (MemStrata, LongLive-RAG, MemFlow, IAMFlow, …)
assets/trackA|trackB/    # small gold samples for tests; full gold is on Hugging Face
docs/                    # protocol and fairness notes
tests/                   # assert-based; no GPU required
```

## Data

Source videos are **not** in git or on Hugging Face.

- Track A videos: [Blender Open Movies](https://studio.blender.org/) (per-film licenses) and [LSMDC](https://sites.google.com/site/describingmovies/download) (apply for access). Point `dataset_dirs.txt` / `VMEM_DATASETS_ROOT` at your copies.
- Track A / Track B **text gold**: `huggingface.co/datasets/Suchenl/VMem-Bench` (CC BY 4.0). Cite Rohrbach et al., IJCV 2017 when using LSMDC titles.

```bash
huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data
```

## Install and unit tests

There is no `pip` package yet. From this directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`pytest.ini` sets `pythonpath=src`. These tests do **not** download movies, load Wan weights, or reproduce paper tables.

## Reproducing paper tables

That is **not** `pytest`. It needs GPUs, generator / encoder weights, and the source videos. See [`REPRODUCE.md`](REPRODUCE.md).

Use git branch `paper-reproduction` (tag `paper-reproduction-v1` when frozen). `main` may move.

## License

Apache-2.0 for code (`LICENSE`). Dataset annotations on Hugging Face are CC BY 4.0. Third-party videos and model weights keep their original terms.
