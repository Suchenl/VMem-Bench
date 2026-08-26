# `vmem_bench` 包级架构

`vmem_bench` 是 MemStrata 的 benchmark、标注、冻结发布和确定性评分包。它不导入 `memstrata` SUT；SUT 只通过 `common.schemas` 定义的 JSON 契约与 benchmark 交互。

本文件只描述包级职责、公共边界和迁移规则。VLM 标注管线的阶段、artifact 和人审细节见统一文档 [`docs/benchmark/annotation_pipeline.md`](../../docs/benchmark/annotation_pipeline.md)。

## Track A 协议（新因果协议）

旧的 **gold-replay / ID-fidelity** 协议（`baseline_adapters/` 的适配机器、`scoring` 的
`runner`/`metrics`/`visual`/`__main__` v1 打分 harness、`benchmark_run/` v1 嵌入器编排）
**已整体删除**。Track A 现在只跑因果协议：bench 每 chunk 给 SUT prompt + 真实 segment，SUT
先 `compose` 组合上下文（持久化，供评分+论文找图）再 `observe_segment` 建记忆，检索项按时序身份
物化成真帧，由 `scoring.visual_coverage` 做 VLM 视觉覆盖打分。权威说明见
[`docs/trackA.md`](../../docs/trackA.md)。

## 目录职责

```text
vmem_bench/
├── README.md                    # 本文件：包级架构与迁移边界
├── __init__.py                  # 包初始化与运行时环境约束
├── common/                      # 跨模块公开契约与确定性工具
├── baseline_adapters/           # ★ 仓内自包含检索族 baseline（零 import memstrata）
│   └── external/retrieval/      # 四个检索族 + 控制项 + 自包含编码器基座
├── annotation/
│   ├── pipeline/                # 唯一维护的标注生产管线
│   ├── chunking.py              # 公共 chunk/layout 工具
│   ├── pipeline_track_first/    # 迁移期 legacy，仅作复制/兼容来源
│   └── pipeline_vlm_dominant/   # 迁移期 legacy，仅作复制/兼容来源
├── scoring/                     # SUT 无关的 VLM 视觉覆盖打分 + 固定嵌入器
├── publish.py                   # 冻结 movie gold → 发布包
├── judger/                      # legacy 标注期 VLM 客户端；迁移完成前保留
├── services/                    # legacy/可选常驻模型服务
├── skills/                      # 可复用算法组件，例如 SBD
└── docs/                        # 协议、schema、设计与历史决策文档
```

## 公共边界：必须保留

以下目录/文件是跨管线、跨 baseline 或发布包依赖的公开面，不随 annotation 重构删除或改为 SUT 专用逻辑：

- `common/schemas.py`：bench ↔ SUT JSON 契约；
- `common/paths.py`：movie 目录与资产路径契约；
- `common/gold_lint.py`：candidate/freeze/publish 的严格门；
- `common/media.py`、`common/vecmath.py`、`common/model_weights.py`：稳定基础工具；
- `baseline_adapters/external/retrieval/`：仓内自包含检索族 baseline（不 import SUT）；
- `scoring/`：`visual_coverage`/`end2end_coverage` VLM 视觉覆盖打分 + `embedder` 固定嵌入器；
- `publish.py`：冻结 gold 发布；
- `common/schemas.py`：字段与协议的权威定义（`docs/trackA.md` 为 Track A 协议文字说明）。

`scoring/` 不构造、import 或特殊分支任何具体 SUT。因果 baseline 的构造与运行在包外的
`scripts/evaluate_baselines/trackA/baseline_adapters/causal/`（runner + 各 baseline 适配 + 打分驱动）。

## 标注生产边界

[`annotation/pipeline/`](annotation/pipeline/) 是唯一的生产标注实现：

- 所有新 VLM prompt、后处理、segment 自动审核、crop 获取、Web 人审、freeze artifact 和标注 batch 编排均放在此目录；
- 它不运行时 import 或修改 `pipeline_track_first/`、`pipeline_vlm_dominant/`；
- 旧实现只能作为一次性复制来源；复制后由 `pipeline/` 自己维护；
- `pipeline/` 可只读使用 `common/` 的公开契约和基础工具。

`annotation/chunking.py` 保留为公共 layout 工具。当前 v5 标注以 `visual_segments` 作为 chunk 布局，不要求新管线运行 SBD；旧管线和历史测试仍可使用 SBD。

## Track A 评测编排的职责

数据集级、SUT-aware 的因果评测编排在包外
`scripts/evaluate_baselines/trackA/`：`baseline_adapters/causal/runner.py` 逐 chunk 驱动单个
baseline（Stage 1），`scoring.visual_coverage` 打分（Stage 2），`overnight_two_movie_run.sh` 做两部影片编排；全量请用你自己的作业调度器反复调用
`baseline_adapters/causal/runner.py`，
`scripts/get_trackA_assets/compare/build_leaderboard_v2.py` 汇总榜单。每部影片的评测 artifact 写到：

```text
<movie>/benchmark_run/{visual_selections,_visual_score,_ref_frames,_segments}/
```

数据集/样本级汇总由 `scripts/evaluate_baselines/trackA/aggregate_trackA_outputs.py` 收进
`outputs/evaluation/trackA/<baseline>/<dataset>/<sample>/`。

## 迁移与归档规则

### 当前迁移期

- `annotation/pipeline_track_first/` 和 `annotation/pipeline_vlm_dominant/` 仍被旧脚本、测试、历史实验和当前 v5 gold 入口引用，**不得立即删除**。
- 新 `annotation/pipeline/` 每完成一个阶段，必须有自己的测试和 CLI，再迁移对应调用方。
- 旧管线中的 prompts、样例 JSON、probe outputs 不再新增；新的生产资产只写入 `annotation/pipeline/`、`data/` 或 `experiments/`。

### 删除或归档的前置条件

任何 legacy 文件删除前必须同时满足：

1. 全仓 import 与 CLI 引用已清零，或已改为新的 `pipeline/` / `benchmark_run/` 入口；
2. 对应的公共契约测试、标注阶段测试和评分测试通过；
3. 至少 BBB、一个额外 BlenderOpenMovies 样本、一个 LSMDC 聚合样本已通过新管线端到端验收；
4. 历史代码先迁入 `benchmarks/MemStrata/_archive/annotation_legacy/` 或标记 `LEGACY`，再考虑物理删除。

明确的首批归档候选是：

- `pipeline_vlm_dominant/web_vlm/**/outputs/*.json` 等源码树内实验产物（迁到 `experiments/` 或对应 `data/`）；
- 多份已被固定 v5 替代的旧 prompt；
- 仅服务于旧 VLM-first `annotate_movie()` 的测试和脚本。

`common/`、`scoring/`、`publish.py`、`docs/schemas_and_contracts.md`、`annotation/chunking.py` 不属于删除候选。

## 已知入口债务

迁移期间应优先修复而不是绕过以下失配：

- 旧脚本/文档引用的 `vmem_bench.annotation.pipeline_track_first.run` 与实际 legacy CLI 路径不一致；
- `vmem_bench.web.server` 的兼容入口与实际 legacy Web 位置不一致；
- 旧的 in-process scoring CLI 文档/测试与当前 records-dir CLI 参数不一致；
- `baselines/` 文档所声称的 adapter 包并非当前可运行实现。

这些是迁移任务，不应通过把 SUT 构造逻辑塞进 `scoring/` 来“修复”。

## 权威文档层级

1. `common/schemas.py` + [`docs/benchmark/schemas_and_contracts.md`](../../docs/benchmark/schemas_and_contracts.md)：公开数据与评测契约；
2. [`docs/benchmark/annotation_pipeline.md`](../../docs/benchmark/annotation_pipeline.md)：标注生产阶段细节（S1–S7）；
3. 本文件：包级目录、边界和迁移规则；
4. [`docs/benchmark/annotation_tracking_internals.md`](../../docs/benchmark/annotation_tracking_internals.md)、[`docs/benchmark/pitfalls.md`](../../docs/benchmark/pitfalls.md) 等：历史/机制设计记录，不能覆盖以上三项。
