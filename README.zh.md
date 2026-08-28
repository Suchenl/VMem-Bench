# VMem-Bench · 中文文档（paper-reproduction 分支）

> 英文主文档见 [`README.md`](README.md)。本文件与之一一对应，不做中英混排。
> HF 数据集与可下载 gold：[Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench)。

> **本分支是 `paper-reproduction`** —— 用于论文表的 Track A Stage-1 冻结核。生产 harness 与完整文档请看 `main` 分支。复现步骤见 [`REPRODUCE.md`](REPRODUCE.md)。

面向**因果长视频生成**的长程视觉记忆基准：Track A 考察时间线上的身份检索，
Track B 考察视觉记忆能否一路存活到生成像素。

## 安装与自检

将本仓库与 [MemStrata](https://github.com/Suchenl/MemStrata) 的
`paper-reproduction` 分支并列放置：

```bash
git clone --branch paper-reproduction https://github.com/Suchenl/MemStrata.git
git clone --branch paper-reproduction https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e ".[dev]"
python3 scripts/doctor.py
python3 -m pytest -q
```

单测与 CLI 检查不会下载影片或模型权重。源视频获取与目录布局见
[`docs/DATA.md`](docs/DATA.md)，权重见 MemStrata 的
[`MODELS.md`](https://github.com/Suchenl/MemStrata/blob/paper-reproduction/MODELS.md)。

## 评测你自己的方法

复制 [`scripts/evaluate_baselines/your_method/`](scripts/evaluate_baselines/your_method/)，
实现 `compose` 与 `observe` 两个钩子，先做 CPU 接线检查：

```bash
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
    --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

取得真实源视频并启动固定版本的 VLM judge 后，运行 Track A 打分：

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage \
    --movie assets/trackA/BlenderOpenMovies/charge \
    --system your_method-recency \
    --video /path/to/charge_source.mp4 \
    --api http://127.0.0.1:8110 \
    --limit 5
```

完整契约、公平性清单与 `source_seconds` 用法见
[`scripts/evaluate_baselines/your_method/README.md`](scripts/evaluate_baselines/your_method/README.md)。
方法只能使用自己在此前 chunk 观察得到的参考图，不能读取 gold roster 或当前 chunk 标签。

## Track B 提交

Track B 需要每个故事片段生成一个真实视频。使用
[`your_method/sut_interface.py`](scripts/evaluate_baselines/your_method/sut_interface.py)
中的 `write_trackb_run(...)` 写出 `progress.json` 与
`review/segments/<segment_id>.mp4`，然后运行：

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.end2end_coverage \
    --gt assets/trackB/en/gt/0001_lighthouse_keeper.json \
    --prompts assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
    --run /path/to/your_run_dir \
    --api http://127.0.0.1:8110
```

打分器只读取方法生成的像素和 scorer 侧 gold。

## 给自己的视频制作标注

本冻结分支保留可恢复的 S1–S7 标注流水线。无需启动服务即可查看 CLI：

```bash
PYTHONPATH=src python3 -m vmem_bench.annotation.pipeline_track_first.run --help
```

服务化运行使用 [`scripts/get_trackA_assets/core/run_annotation.sh`](scripts/get_trackA_assets/core/run_annotation.sh)。
`PROPOSAL_ONLY=1` 只用于诊断；正式 gold 必须提供人工确认的 `ROSTER_SEED`，
并通过 review/freeze 门禁。源视频布局与服务启动方式见 [`docs/DATA.md`](docs/DATA.md)。

## 论文复现范围

论文 Track A Stage 1 给每个 SUT 的输入是提示词文本与真实视频片段，真实片段
代替生成像素；Track B 使用提示词流与方法生成的视频。本冻结分支可以运行 Track B
打分器，但不声称能复现论文后续版本的 Track B 表。

91 部 Track A 运行请使用 causal runner、`--budget 16`，并设置
`MEMSTRATA_TRACKA_NAME_SOURCE=mllm` 以复现论文中的 MemStrata 路径；
完整步骤见 [`REPRODUCE.md`](REPRODUCE.md)。全量运行仍需要源视频、固定 VLM judge、
GPU 与模型权重，CI 不执行这些重任务。

## 数据、许可与引用

源视频不入 git 或 Hugging Face。完整 gold/prompts 位于
`huggingface.co/datasets/Suchenl/VMem-Bench`；逐步获取源视频见
[`docs/DATA.md`](docs/DATA.md)。代码使用 Apache-2.0，自撰 gold 标注使用
CC BY 4.0，第三方视频与模型权重沿用原始条款。详见 [`LICENSE`](LICENSE)
与 [`CITATION.cff`](CITATION.cff)。

## 引用

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

另见 [`CITATION.cff`](CITATION.cff)。
