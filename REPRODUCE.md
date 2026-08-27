# paper-reproduction：VMem-Bench 冻结核

本分支是公开的 `paper-reproduction` 冻结分支，保留论文 Track A
Stage 1 的协议、gold JSON、adapter 与打分实现。包名和目录均使用当前
公开名称 `vmem_bench` / VMem-Bench；不需要任何内部仓库或路径。

| 分支 | 职责 |
|---|---|
| `main` | 生产 harness，会继续改，不保证对齐论文表 |
| `paper-reproduction`（本分支） | Track A Stage 1 当时的协议、gold JSON、adapter、打分代码 |

论文数字只认本分支（建议再打 tag `paper-reproduction-v1`）。`pytest` **不能**复现论文表。

## Track A（本冻结核能对上的协议）

对应论文 Track A Stage 1：SUT 看到 prompt 文本 + **真实片段视频**（真实 clip 代替生成像素），写 `visual_selections`，再用 VLM 做 visual coverage。

**Stage 1 真实入口**（`--help` 必须能起来）：

```bash
python scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py --help
python scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py \
  --adapter memstrata \
  --movie-dir assets/trackA/BlenderOpenMovies/big_buck_bunny \
  --limit 2
```

跑 MemStrata adapter 还需要：

- **`MEMSTRATA_TRACKA_NAME_SOURCE=mllm`（复现论文机制必须）**：adapter 默认 `perception`，用 SAM3/DINOv3 的通用名，对“地点/连续性”类实体会 under-recall（实测 BBB 前两段 location-only 时 recall=0）。论文表走 `mllm`——由 MemStrata 自己的 VLM 绑定可见 prompt 名。跑复现请务必 `export MEMSTRATA_TRACKA_NAME_SOURCE=mllm`。

- `MEMSTRATA_SRC` 指向 MemStrata `paper-reproduction` 仓的 `src/`（两仓零互相 import，adapter 在本仓黑盒加载方法）
- 源视频：逐步获取（BBB 一键、其他 CC 片、LSMDC 申请与拼接目录）见 [`docs/DATA.md`](docs/DATA.md)；`python scripts/check_source_videos.py` 列出缺哪些 id
- `PUBLIC_MODELS_ROOT`（SAM3 / DINOv3 等）
- GPU

> **实测前置补记**（真机 GPU 跑通验证）：SAM3 需 `transformers>=5.9`（仓库自带 vendored 适配可放上 `PYTHONPATH`）；`ffmpeg` 必须在 PATH。二者缺一 Stage 1 会失败但可 100% 复现地补齐，非源码问题。

91 部全量：用你自己的作业调度器反复调用 `causal/runner.py`（预算 `__B16` = `--budget 16`）。不要依赖本仓里的集群投放脚本。

Stage 2 打分：

```bash
python -m vmem_bench.scoring.visual_coverage --help
```

需要 pinned 的 VLM judge（论文用的 Qwen2.5-VL / Qwen3-VL 路径以当时配置为准），以及 Stage 1 产出的 `visual_selections`。本快照 **没有** 把 91 部 GPU 产物放进 git。

Gold JSON（含 LSMDC 自建标注）在本冻结核的 `assets/trackA/`。源视频仍须自备。LSMDC 官方 csv/AD/像素不转授。

## Track B（不要假装本冻结核能复现论文表）

论文 Track B 表是 **30 stories / system**，在 **比本快照更晚的代码** 上跑的。

本冻结核 **已经带有** Track B 打分器：

- `src/vmem_bench/scoring/end2end_coverage.py`（来自 freeze `51be2914`，不是后来从 `main` 回拷）
- `assets/trackB/`（50 个故事的 gt / sut_prompts 也在 freeze 里）

因此：

- 可以用 freeze 里的打分 CLI 对 **某次你自己跑的生成结果** 打分；
- **不能**声称 `51be2914` 这棵树已经对上论文 Track B 表。

```bash
python -m vmem_bench.scoring.end2end_coverage --help
python -m vmem_bench.scoring.end2end_coverage \
  --gt assets/trackB/zh/gt/0001_lighthouse_keeper.json \
  --run <your_generated_run_dir>
```

要对 Track B 表，还缺：各系统在后来代码上的生成视频、当时的 VLM judge、以及 30 条故事的那次正式 run。这些都不在本分支。

## 无 GPU 冒烟

本分支是冻结核，同时保留完整的 CPU CI 门禁。可靠的无 GPU 信号是：
装得上 + 三个 `--help` 入口起得来 + 全部收集到的单测通过。

```bash
export CUDA_VISIBLE_DEVICES=
export PUBLIC_MODELS_ROOT=/tmp/dummy_models   # 让 embedder __init__ 能解析路径（不加载权重）
python -m pip install -e ".[dev]"             # 现在会装齐 opencv/scipy/safetensors
PYTHONPATH=src python -m vmem_bench.scoring.visual_coverage --help
PYTHONPATH=src python -m vmem_bench.scoring.end2end_coverage --help
python scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py --help
PYTHONPATH=src python -m pytest -q
```

`tests/` 只测 bench 自身。`scripts/evaluate_baselines/tests/` 会碰到方法包，默认不收集。

当前冻结分支已验证：`367 passed`。其中修复项只涉及缓存失效、
兼容的公开路径、标注审核队列和 legacy gold 的迁移兼容性，不改变
论文 Track A 的输入、记忆协议或评分公式。

`pytest.ini` 忽略了 freeze 测试里引用但 freeze `src/` 并不存在的两个模块：
`tests/test_pipeline_vlm_dominant.py`（`postprocess` vs `postprocess_segments`）、
`tests/test_services_placement.py`（没有 `vmem_bench.services`）。没有补写这些模块。

`test_model_weights_root.py` 使用独立仓断言，不要求任何外部源码树。

## 环境变量

| 变量 | 用途 |
|---|---|
| `PUBLIC_MODELS_ROOT` | 编码器 / VLM 权重根。未设置时 CPU import / `--help` 不得崩；真正加载权重才报错 |
| `VMEM_DATASETS_ROOT` | Blender / LSMDC 源视频根 |
| `MEMSTRATA_SRC` | MemStrata 仓的 `src/`（仅 adapter 需要） |
| `MEMSTRATA_TRACKA_NAME_SOURCE` | Track A 命名来源：`mllm`（论文复现路径）/ `perception`（默认，通用感知名，会 under-recall） |
| `FFMPEG` | 可选，默认 `ffmpeg` |
| `VMEM_KEEPALIVE_STATUS_DIR` | 仅内部 GPU 保活；公开复现不要设 |

未设置 `PUBLIC_MODELS_ROOT` 时不要跑需要 SAM3/VLM 的 adapter。

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff).

