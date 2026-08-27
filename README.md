# VMem-Bench: Memory-aware causal long video generation

> **This is the `paper-reproduction` branch** — the frozen Track A Stage-1 core used for the paper tables. For the production harness and full documentation, use the `main` branch. Reproduction steps: [`REPRODUCE.md`](REPRODUCE.md).

> 中文文档见 [`README.zh.md`](README.zh.md).

## Evaluate your own method

The branch includes the same stdlib-only BYOM template as `main`:
[`scripts/evaluate_baselines/your_method/README.md`](scripts/evaluate_baselines/your_method/README.md).
Use it to validate your `compose`/`observe` contract on the frozen Track A inputs,
then score the emitted selection with the paper-branch scorer.

The annotation pipeline is also available in this frozen checkout. Inspect
`PYTHONPATH=src python -m vmem_bench.annotation.pipeline_track_first.run --help`
and use `scripts/get_trackA_assets/core/run_annotation.sh` for a service-backed
S1–S7 annotation run; production gold still requires a human-confirmed roster
and the review/freeze gates.

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff).
