# VMem-Bench

A benchmark for visual memory in causal long-video generation. Track A evaluates
reference selection before generation; Track B evaluates the entities and states
that appear in generated video.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-007ec6?style=flat)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2610.35001-b31b1b?style=flat)](https://arxiv.org/abs/2610.35001) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-FFD21E?style=flat)](https://huggingface.co/datasets/Suchenl/VMem-Bench) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=flat)](https://huggingface.co/papers/2610.35001) [![Project Page](https://img.shields.io/badge/Project-Page-E6550D?style=flat&logo=googlechrome&logoColor=white)](https://suchenl.github.io/VMem-Bench/)

[中文说明](README.zh.md)

## Install and try the interface

```bash
git clone https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e .
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
  --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

This CPU example uses placeholder references to demonstrate the interface. It
requires no video, weights or MemStrata, and does not produce benchmark scores.
Implement your method's `compose` and `observe` hooks using the
[adapter template](scripts/evaluate_baselines/your_method/README.md).

## Data and models

Git includes only input fixtures needed by examples and tests: two Track-A films'
text annotations and the evaluated 30 English Track-B prompt/gold pairs. The full
data is on Hugging Face:

```bash
python3 -m pip install huggingface_hub
hf download Suchenl/VMem-Bench --repo-type dataset --local-dir assets \
  --include 'trackA/**' 'trackB/**'
```

Source videos and model weights are separate downloads. Follow [DATA.md](docs/DATA.md)
for video access and directory layouts; use
[scripts/install_scoring_models.sh](scripts/install_scoring_models.sh) for scoring
models. Track B needs no original films. The MemStrata adapter optionally needs a
sibling [MemStrata](https://github.com/Suchenl/MemStrata) checkout.

## Evaluation

- **Track A:** [protocol](docs/trackA.md) and [running evaluation](docs/benchmark/running_eval.md).
- **Track B:** [shared-generator testbed](src/vmem_bench/testbed/README.md) and
  [scoring instructions](docs/benchmark/trackB_scoring.md).
- **Annotation:** [pipeline guide](docs/benchmark/annotation_pipeline.md).

Select references before revealing the current segment. Keep inputs and shared
preprocessing identical across methods; gold is scoring-only. Last-seen's explicit
gold-ID access is a diagnostic-control exception, not a deployable method.

The paper's main Track-B comparison uses one rollout per story/condition, seed
2026. New runs must record their own code revision, cohort, models and seeds.
Original rollout videos and per-segment score bundles are not included here.

## Repository layout

```text
src/vmem_bench/  benchmark, annotation, scoring and shared-generator testbed
scripts/        method adapters and utilities
tests/          code tests
assets/         lightweight input fixtures and authoring scripts
docs/           setup and evaluation instructions
site/           GitHub Pages project site
```

## Citation

If you use VMem-Bench in your research, please cite the accompanying arXiv preprint. GitHub also provides **Cite this repository** using [CITATION.cff](CITATION.cff).

```bibtex
@misc{chen2026vmembench,
  title={More Memory, Better Videos? A Controlled Study of Visual Memory in Long-Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Xu, Yulong and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua and Zhang, Weiming},
  year={2026},
  eprint={2610.35001},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.35001}
}
```

## License

Code: [Apache-2.0](LICENSE). Self-authored gold/prompts: CC BY 4.0.
Source films, official LSMDC CSV/AD, and model weights are not redistributed.
See [NOTICE](NOTICE).
