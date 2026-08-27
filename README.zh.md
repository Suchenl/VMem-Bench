# VMem-Bench · 中文文档（paper-reproduction 分支）

> 英文主文档见 [`README.md`](README.md)。本文件与之一一对应，不做中英混排。

> **本分支是 `paper-reproduction`** —— 用于论文表的 Track A Stage-1 冻结核。生产 harness 与完整文档请看 `main` 分支。复现步骤见 [`REPRODUCE.md`](REPRODUCE.md)。

本冻结分支也保留标注流水线。可先运行
`PYTHONPATH=src python -m vmem_bench.annotation.pipeline_track_first.run --help`
查看 CLI，再用 `scripts/get_trackA_assets/core/run_annotation.sh` 启动依赖服务的
S1–S7 标注；正式 gold 仍需人工确认 roster，并通过 review/freeze 门禁。

## 引用

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

另见 [`CITATION.cff`](CITATION.cff)。
