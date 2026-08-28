# VMem-Bench

A benchmark for long-range visual memory in causal long video generation:
Track A measures identity retrieval over a movie timeline, while Track B
measures whether memory survives into generated pixels.

> **This is the `paper-reproduction` branch.** It freezes the Track A Stage-1
> protocol used for the paper. `main` is the moving production harness.
> Reproduction steps are in [`REPRODUCE.md`](REPRODUCE.md).

> Chinese documentation: [`README.zh.md`](README.zh.md).
> Chinese reproduction guide: [`REPRODUCE.zh.md`](REPRODUCE.zh.md).
> Dataset card and downloadable gold: [Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench).

## Install and self-check

Clone this repository next to [MemStrata](https://github.com/Suchenl/MemStrata):

```bash
git clone --branch paper-reproduction https://github.com/Suchenl/MemStrata.git
git clone --branch paper-reproduction https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e ".[dev]"
python3 scripts/doctor.py
python3 -m pytest -q
```

The tests and CLI help checks do not download videos or model weights. Source
videos and model weights are external; see [`docs/DATA.md`](docs/DATA.md) and
MemStrata's [`MODELS.md`](https://github.com/Suchenl/MemStrata/blob/paper-reproduction/MODELS.md).

## Evaluate your own method

Copy [`scripts/evaluate_baselines/your_method/`](scripts/evaluate_baselines/your_method/).
It is a standard-library-only bring-your-own-method template. Implement
`compose` and `observe`, then run the CPU wiring check:

```bash
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
    --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

With a real source video and a pinned VLM judge, score the emitted Track A
selections:

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage \
    --movie assets/trackA/BlenderOpenMovies/charge \
    --system your_method-recency \
    --video /path/to/charge_source.mp4 \
    --api http://127.0.0.1:8110 \
    --limit 5
```

The full contract and fairness checklist are in
[`scripts/evaluate_baselines/your_method/README.md`](scripts/evaluate_baselines/your_method/README.md).
References must come from your own prior observations; the scorer never gives
your method the gold roster or current-chunk labels.

## Track B submissions

For Track B, generate one real video per story segment and use
`write_trackb_run(...)` from [`your_method/sut_interface.py`](scripts/evaluate_baselines/your_method/sut_interface.py)
to create `progress.json` and `review/segments/<segment_id>.mp4`. Then score:

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.end2end_coverage \
    --gt assets/trackB/en/gt/0001_lighthouse_keeper.json \
    --prompts assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
    --run /path/to/your_run_dir \
    --api http://127.0.0.1:8110
```

The Track B scorer sees only your generated pixels and the scorer-side gold.

## Create annotations for your own video

The repository includes the resumable S1–S7 annotation pipeline. It accepts
one continuous source video and OpenAI-compatible VLM endpoints:

```bash
PYTHONPATH=src python3 -m vmem_bench.annotation.pipeline_track_first.run --help
```

Use [`scripts/get_trackA_assets/core/run_annotation.sh`](scripts/get_trackA_assets/core/run_annotation.sh)
for a service-backed run. `PROPOSAL_ONLY=1` is diagnostic only. A publishable
gold package requires a human-confirmed `ROSTER_SEED` and the review/freeze
gates. See [`docs/DATA.md`](docs/DATA.md) for video layout and service setup.

## Paper scope and protocol

Track A Stage 1 feeds each SUT the prompt text and the real video segment; the
segment stands in for generated pixels. Track B uses prompt streams and
generated videos. The frozen branch can run the Track B scorer, but it does
not claim to reproduce the paper's later Track B tables.

For the 91-movie Track A run, use the causal runner with `--budget 16`, set
`MEMSTRATA_TRACKA_NAME_SOURCE=mllm` for the paper MemStrata path, and follow
[`REPRODUCE.md`](REPRODUCE.md). Full runs require the source videos, pinned VLM
judge, GPU, and model weights; they are not part of CI.

## License and citation

Code is Apache-2.0. Self-authored gold annotations are CC BY 4.0. Third-party
videos and model weights retain their original terms. See [`LICENSE`](LICENSE)
and [`CITATION.cff`](CITATION.cff).

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```
