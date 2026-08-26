# VMem-Bench

# Getting started

Clone **next to** [MemStrata](https://github.com/Suchenl/MemStrata) (the adapter looks in `../MemStrata/src`):

```bash
git clone https://github.com/Suchenl/MemStrata.git
git clone https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python -m pip install -e ".[dev]"
python scripts/doctor.py
python -m pytest -q
```

`doctor.py` prints the exact next command for anything missing (ffmpeg, sibling MemStrata, BBB video, SAM3).

```bash
bash scripts/prepare_blender.sh    # official BBB 720p; gold is already in assets/trackA/
python scripts/check_source_videos.py
# Track A Stage-1 smoke (needs PUBLIC_MODELS_ROOT + GPU perception):
bash scripts/run_tracka_smoke.sh
```

**How to get source videos** (BBB one-command, other Blender/CC films, LSMDC application + stitch layout): [`docs/DATA.md`](docs/DATA.md).

Full gold/prompts (no videos): `huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data`

`main` is production. `paper-reproduction` is the Track A Stage-1 freeze. See [`REPRODUCE.md`](REPRODUCE.md).

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

Source videos are **not** in git or on Hugging Face. Step-by-step obtain + directory layout: [`docs/DATA.md`](docs/DATA.md).

- Track A videos: [Blender Open Movies](https://studio.blender.org/) (per-film licenses) and [LSMDC](https://sites.google.com/site/describingmovies/download) (apply for access). Point `dataset_dirs.txt` / `VMEM_DATASETS_ROOT` at your copies.
- Track A / Track B **text gold**: `huggingface.co/datasets/Suchenl/VMem-Bench` (CC BY 4.0). Cite Rohrbach et al., IJCV 2017 when using LSMDC titles.

```bash
huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data
```

## Install and unit tests

There is no PyPI release yet. From this directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

`pytest.ini` sets `pythonpath=src`. These tests do **not** download movies, load Wan weights, or reproduce paper tables.

## Reproducing paper tables

That is **not** `pytest`. It needs GPUs, generator / encoder weights, and the source videos. See [`REPRODUCE.md`](REPRODUCE.md).

Use git branch `paper-reproduction` (tag `paper-reproduction-v1` when frozen). `main` may move.

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

When using LSMDC titles, also cite Rohrbach et al., IJCV 2017 (see [`docs/DATA.md`](docs/DATA.md)).

See [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 for code (`LICENSE`). Dataset annotations on Hugging Face are CC BY 4.0. Third-party videos and model weights keep their original terms.
