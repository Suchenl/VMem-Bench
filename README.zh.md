# VMem-Bench · 中文文档

> 英文主文档见 [`README.md`](README.md)。本文件是与之对应的中文说明，与英文 README 一一对应，不做中英混排。
> 深层设计文档（`docs/benchmark/*`、`docs/baselines/*`、`docs/design/*` 等）本身即为中文文档，本文末尾给出索引。
> HF 数据集与可下载 gold：[Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench)。

面向**因果长视频生成**的长程视觉记忆基准：Track A 考察一部影片时间线上的身份检索，Track B 考察视觉记忆能否一路存活到**生成**的像素里。

## 快速开始

克隆时与 [MemStrata](https://github.com/Suchenl/MemStrata) **并列放置**（adapter 会去 `../MemStrata/src` 找方法包）：

```bash
git clone https://github.com/Suchenl/MemStrata.git
git clone https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e ".[dev]"
python3 scripts/doctor.py
python3 -m pytest -q
```

`doctor.py` 会为每个缺失项（ffmpeg、并列的 MemStrata、BBB 视频、SAM3）打印出确切的下一步命令。

```bash
bash scripts/prepare_blender.sh    # 官方 BBB 720p；gold 已在 assets/trackA/
python3 scripts/check_source_videos.py
# Track A Stage-1 冒烟（需 PUBLIC_MODELS_ROOT + GPU 感知）：
bash scripts/run_tracka_smoke.sh
```

**源视频如何获取**（BBB 一条命令、其余 Blender/CC 影片、LSMDC 申请 + 拼接布局）见 [`docs/DATA.md`](docs/DATA.md)。

完整 gold/prompts（不含视频）：`huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data`

`main` 为生产分支；`paper-reproduction` 为 Track A Stage-1 冻结分支，见 [`REPRODUCE.md`](REPRODUCE.md)。

## 评测你自己的方法

复制 [`scripts/evaluate_baselines/your_method/`](scripts/evaluate_baselines/your_method/) —— 一个仅依赖标准库的「自带方法」模板。实现两个钩子（`compose`、`observe`），先跑 CPU 干跑在自带 gold 上验证接线，再用两条 `vmem_bench.scoring` 命令打分。完整流程 + 公平性清单见其 [`README.md`](scripts/evaluate_baselines/your_method/README.md)。

```bash
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
    --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

## 给自己的视频制作标注

VMem-Bench 同时包含可复现的 S1–S7 标注流水线。它接收一部连续的源视频，
通过 OpenAI-compatible 的 VLM 服务完成起草与审核，并写出可恢复的中间产物
以及可以经过人工冻结的 `gold/` 数据包。源视频目录布局见 [`docs/DATA.md`](docs/DATA.md)，
服务化运行命令见 [`scripts/get_trackA_assets/core/run_annotation.sh`](scripts/get_trackA_assets/core/run_annotation.sh)。
`PROPOSAL_ONLY=1` 只用于诊断；正式 gold 必须提供人工确认的 `ROSTER_SEED`，
并通过审核 / freeze 门禁。无需启动服务即可查看流水线 CLI：

```bash
PYTHONPATH=src python3 -m vmem_bench.annotation.pipeline_track_first.run --help
```

## 两个 Track 测什么

| Track | 问题 | SUT 输入 | 打分 gold |
|---|---|---|---|
| **A** | 记忆机制能否在整部影片时间线上检索到正确的视觉身份？ | 每个 chunk 的**提示词文本 + 真实视频片段**（真实片段**代替**生成器像素） | 冻结的文本 gold（`entity_registry`、`chunk_annotations`）；**无 gold 抠图** |
| **B** | 视觉记忆能否一路存活进**生成**的像素里？ | 时序的**提示词流** | 自撰故事 + `memory_probes`；SUT 永不接触 GT |

协议细节见 [`docs/trackA.md`](docs/trackA.md)、[`assets/trackB/README.md`](assets/trackB/README.md)、[`docs/benchmark/running_eval.md`](docs/benchmark/running_eval.md)。

## 公平性契约

摘自 [`AGENTS.md`](AGENTS.md)。任何一条被违反都会使数字失效。

1. 对每个 SUT（含 MemStrata）输入完全一致。
2. 不拟合测试集（不使用从评测导出的词表或按样例分支）。
3. adapter 不得给方法添加它本身没有的能力。
4. 预处理对称——或干脆都不做。
5. 不向任何 SUT 泄露 gold。

`src/vmem_bench` 只 import `vmem_bench`。跨包 adapter 只住在 `scripts/evaluate_baselines/` 下。

## 目录布局

```
src/vmem_bench/          # 协议、打分、标注（自包含）
scripts/evaluate_baselines/   # SUT adapter（MemStrata、LongLive-RAG、MemFlow、IAMFlow…）
assets/trackA|trackB/    # 供测试用的小份 gold；完整 gold 在 Hugging Face
docs/                    # 协议与公平性说明
tests/                   # assert 型；无需 GPU
```

## 数据

源视频**不**入 git、也**不**放 Hugging Face。逐步获取 + 目录布局见 [`docs/DATA.md`](docs/DATA.md)。

- Track A 视频：[Blender Open Movies](https://studio.blender.org/)（各片各自许可）与 [LSMDC](https://sites.google.com/site/describingmovies/download)（需申请）。把 `dataset_dirs.txt` / `VMEM_DATASETS_ROOT` 指向你的副本。
- Track A / Track B **文本 gold**：`huggingface.co/datasets/Suchenl/VMem-Bench`（CC BY 4.0）。使用 LSMDC 标题时请引用 Rohrbach et al., IJCV 2017。

```bash
huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data
```

## 安装与单元测试

暂无 PyPI 发布。在本目录下：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

`pytest.ini` 设置了 `pythonpath=src`。这些测试**不会**下载影片、加载 Wan 权重或复现论文表。

## 复现论文表

这**不是** `pytest`。它需要 GPU、生成器/编码器权重与源视频，见 [`REPRODUCE.md`](REPRODUCE.md)。使用 git 分支 `paper-reproduction`（冻结时打 tag `paper-reproduction-v1`）；`main` 可能会动。

## 引用

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

使用 LSMDC 标题时另请引用 Rohrbach et al., IJCV 2017（见 [`docs/DATA.md`](docs/DATA.md)）。另见 [`CITATION.cff`](CITATION.cff)。

## License

代码 Apache-2.0（`LICENSE`）。Hugging Face 上的数据集标注为 CC BY 4.0。第三方视频与模型权重沿用其原始条款。

---

## 中文设计文档索引

以下为保留为中文的深层设计/协议文档（面向贡献者与深入使用者）：

- 基准协议 / 打分：[`docs/benchmark/`](docs/benchmark/)（`scoring.md`、`scoring_v2.md`、`schemas_and_contracts.md`、`annotation_pipeline.md`、`pitfalls.md` 等）
- baseline 与公平性：[`docs/baselines/`](docs/baselines/)（`strategy.md`、`fairness_decisions.md`、`external_baseline_audit.md`、`tracka_iamflow_host_memory.md` 等）
- 基准设计：[`docs/design/bench/`](docs/design/bench/)
- 标注流水线运行时提示词：`src/vmem_bench/annotation/`（运行时资产，保持中文原样）
