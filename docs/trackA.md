# Track A 协议（新因果协议）

Track A 评测「视觉记忆的检索质量」。**旧的 gold-replay 协议已整体删除**，Track A 现在只跑
下面这一套因果协议。旧协议（`src/vmem_bench/baseline_adapters/` 里的 `run_gold_replay.py` /
`registry` / `convert` / `common/` / `diagnostics/` / `external/causal/` / `external/scripted/`，
以及 `scripts/get_trackA_assets/compare/` 下的 gold-replay 编排、各 gold 轨迹/潜变量生产脚本）
已从仓库移除，不再保留、不再导入。

## 三条铁律

1. **bench 不做感知**：不抠图、不聚类、不判存在。
2. **bench 不发图给 SUT**：SUT 只拿到 prompt 文本 + 真实 segment 视频，绝不拿 gold crop / gold 实体 id。
3. **bench 不泄答案**：不给 `present`/roster、不给参考图贴标签。

## 每个 chunk 的数据流

按时间序，对每个 chunk `t`：

1. **切真实 segment**：bench 从源视频按 `seconds_span` 切出该 chunk 的真实片段（纯 IO）。
2. **`compose(prompt)`**：把 **prompt 文本**交给 SUT。SUT 从**当前记忆**（只由 `< t` 的历史 chunk 建成）
   里检索，返回带**时序身份**的记忆项（`source_seconds` / `source_chunk_id` + `evidence_kind`）。
3. **`observe_segment(真实 clip)`**：把该 chunk 的**真实 segment** 交给 SUT，让它走自己原生的记忆写入
   路径更新记忆。

`compose` **严格先于**同一 chunk 的 `observe_segment`，所以 SUT 在为 chunk `t` 组合上下文时，绝不可能
偷看 chunk `t` 的视频。

**真实 segment 替代 SUT 的生成产出**，以消除生成噪声。原则上**能跳过生成就跳过**：

- MemStrata（本系统）、检索族（见下）是感知/检索运算，**不需要**任何生成器前向。
- LongLive-RAG 的检索是纯自编码描述子运算，同样**不需要**跑生成器前向。
- MemFlow / IAMFlow / SlotMem 的记忆写入发生在生成器前向内部，但都只需 **teacher-force 真 latent 做单次前向**
  抽取（`context_noise=0`），**不是**多步去噪生成：MemFlow/IAMFlow 填 KV / 记忆帧；SlotMem 的角色 slot 抽取
  （`_extract_memory_from_current_step`）本质是**单次 DiT 前向的注意力探针**——注册 hook 跑一次前向拿注意力图即可，
  角色靠 prompt 里的角色名 token 定位（`find_token_index_in_prompt`；`name_anchored` 已给名字，无需 roster，
  `char_latent_boxes` 仅为可选精修）。因此**没有任何 chunk 需要真正的多步生成**。

> **SlotMem 现状：已接入 TrackA adapter。** adapter 使用 native `Wan2.2-I2V-A14B`
> + SlotMem 自己的 stage1/stage2 LoRA/encoder，在 torch 2.5 + flash-attn 2.8
> 里执行：VAE 编码真 segment → 选 SlotMem 单 bank timestep 加噪 → 单次 native DiT 前向带注意力探针
> → stage2 slot encoder/writer 写 `RoleWiseSlotMemoryBank`。**TrackA 正式实验禁止使用 distilled
> Wan2.2/lightx2v 版本**：distilled + SlotMem LoRA 虽可加载并生成视频，但烟测视觉质量不稳定（涂抹、
> 块状背景、几何漂移），不可作为公平正式结果。

## 因果护栏

`frame_materializer.py` 在把时序身份物化成参考帧时，会丢弃「源时间 ≥ 当前 chunk 起点」的项（因果 SUT
只能取过去），丢弃计数记进 manifest（`future_dropped`）。

## 产物与持久化

所有产物写在 `<movie>/benchmark_run/` 下，**compose 组合出的上下文结果不删除**（既用于评分，也用于给论文
挑定性对比图）：

| 路径 | 内容 |
|---|---|
| `benchmark_run/visual_selections/<run_name>.json` | 每 chunk 组合出的上下文（选中的记忆项 + 解析出的参考帧 + 该 chunk 的 prompt），**持久保留** |
| `benchmark_run/_ref_frames/<run_name>/` | 从真实源视频切出的参考帧（评分 + 论文找图用） |
| `benchmark_run/_segments/chunk_NNNNN.mp4` | 每 chunk 切出的真实 segment |
| `benchmark_run/_adapter_work/<run_name>/finalize.json` | run 级元数据（input_mode / budget / 记忆规模 / 检索模式等） |

`<run_name>` 命名规则：`name_anchored` 用 `adapter.name`；`description_provided` 用
`<adapter.name>__descprov`；若指定了 `--budget B` 再追加 `__B<B>`。两种输入模式的产物因此不会互相覆盖。

## 输入模式

Track A 的规范输入模式**只有两个**：`name_anchored` 与 `description_provided`，两者并排上报（公平性轴）。

- **`name_anchored`（默认，主表）**：prompt 就是 S4 剧本散文原文，复现实体用它们的自然名字指代。按名字索引
  记忆的系统（如 MemStrata 的名锚，以及文本条件的 baseline）在这里拿到强文本抓手。
- **`description_provided`**：在 `name_anchored` 的 prompt 之上，为**名字已经出现在该 chunk prompt 里**的
  实体，确定性地追加一段外观描述后缀（形如 `[实体外观参考] 名字：外观…；…`）。它**只增加外观文本、不删名字**，
  让「靠外观描述去匹配自己视觉记忆」的系统也拿到公平的文本抓手。泄漏安全：只描述 prompt 已经点到名的实体
  （每 chunk 就那几个），不暴露任何 `present`/roster；同一条确定性规则对所有系统一视同仁。所有实体种类
  （character / prop / location）都可被描述——道具、场景也是一等的复现视觉身份。

> 运行器 `runner.py` 另外还接受一个诊断用的 `description_only` 模式（把 prompt 中已出现的注册名替换成中性
> 指代 + 追加外观文本，同样不暴露 roster）。它**不属于规范主表**，只作压力/消融用途。

任一模式都由 bench 侧确定性地改写 prompt 文本，`gold/entity_registry.json` 里的实体元数据**只**用来生成
外观描述后缀，**绝不**作为 present/roster 列表交给 SUT。

## 打分

参考帧由 `vmem_bench.scoring.visual_coverage` 做 VLM 视觉覆盖打分（方法中立，不注入 gold、不给参考图贴标签）；
榜单由 `scripts/get_trackA_assets/compare/build_leaderboard_v2.py` 汇总，定性对比图可用
`scripts/get_trackA_assets/compare/export_visual_selections.py` 从 `visual_selections/` 解析。

## Baselines

因果 baseline 的 bench 适配代码在
[`scripts/evaluate_baselines/trackA/baseline_adapters/causal/`](../scripts/evaluate_baselines/trackA/baseline_adapters/causal/README.md)（每个 baseline 一个 `build_adapter()` 工厂，vendored 原始仓库零改动）。

检索族（`text_frame_retrieval` / `text_segment_retrieval_then_uniform_sampling` /
`text_segment_retrieval_then_dino_keyframe_sampling` / `text_segment_retrieval_then_frame_retrieval`
+ recency / bm25 / random 诊断控制项）是**仓内自包含实现**，位于
`src/vmem_bench/baseline_adapters/external/retrieval/`，**不导入 SUT 包 `memstrata`**（编码器基座在
同目录 `_retrieval_encoders.py`），经 `baseline_adapters/causal/retrieval_family.py` 挂到因果协议下运行。

## 运行

```bash
# 每个 baseline 用装好对应依赖的 Python（见各 adapter 模块头注释）
PY=python3
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal

# Stage 1：驱动一个 baseline 跑完一部电影（name_anchored 主表）
$PY runner.py --adapter longlive_rag \
  --movie-dir <movie_dir> --input-mode name_anchored

# description_provided 模式（产物落到 <name>__descprov，不覆盖主表）
$PY runner.py --adapter longlive_rag \
  --movie-dir <movie_dir> --input-mode description_provided

# Stage 2：VLM 视觉覆盖打分
$PY -m vmem_bench.scoring.visual_coverage \
  --movie <movie_dir> --system <run_name> --video <source_video>

# Stage 3：把 per-movie benchmark_run/ 结果按 baseline/dataset/sample 汇总
PYTHONPATH=src python scripts/evaluate_baselines/trackA/aggregate_trackA_outputs.py
# -> outputs/evaluation/trackA/<baseline>/<dataset>/<sample>/<input_mode>[/B<budget>]/
#    {score.json, visual_selections.json, finalize.json, meta.json}
#    + 每 baseline aggregate.{json,md} + 顶层 leaderboard.{json,md}
```
