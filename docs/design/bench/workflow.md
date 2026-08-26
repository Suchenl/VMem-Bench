# MemStrata-Bench 评测工作流（v2 · 离线标注 + 确定性回放）

> 状态：**协议草案，待批准冻结**（2026-07-08）。批准后才允许编写实现代码（SDD）。
> 本版取代旧的「闭环在线流式标注」工作流；与 v1 的差异见文末「v1 → v2 变更记录」。
> 数据契约与指标公式的权威定义见 [`schemas_and_contracts.md`](../../benchmark/schemas_and_contracts.md)。

## 设计决策（已由用户拍板，2026-07-08）

| # | 决策 |
|---|---|
| D1 | 评测数据集：BlenderOpenMovies（首部打样 `big_buck_bunny_720p_h264.mp4`），后续视效果扩展 LSMDC。 |
| D2 | chunk 粒度：**先切镜头、再拼接**——边界必须落在镜头边界上，拼接后每个 chunk 帧数落在 **[min_frames, max_frames]** 区间（默认 [120, 360] 帧 @24fps ≈ 5–15s，可配置）；超长镜头在镜头内均分，欠长碎段并入相邻 chunk（并入不超上限时）。 |
| D3 | prompt 可使用权威实体名（剧本式）；**实体首次出现的 chunk，prompt 必须携带其外观描述**，之后仅用名字。**prompt 绝不包含资产来源提示**（如“取 chunk 0 的资产 A”），即隐式提示原则（设计原则 #3）。 |
| D3b | **提示词即完整生成源**（设计原则 #9）：GT 画面中出现的实体与状态变化，必须都能从 prompt + 组合上下文推出；状态事件在其发生的 chunk 的 prompt 中叙述。SUT（System Under Test，被测系统）通过 prompt 叙述与 observation 反馈获知世界变化，绝不接触评分用的 forbidden 物化表。 |
| D4 | Avoidance 负例来源：**剧情状态事件**（如苹果被吃掉、蝴蝶被拍死）→ 事件发生后，该实体旧状态的 representation 对后续 chunk 即为过期引用，选中要扣分。无状态事件的影片该指标记 **N/A**，绝不默认满分。 |
| D8 | 所有 embedding 一律 `.safetensors` 旁挂存储，JSON 只存 `embedding_key` 引用（设计原则 #10）。 |
| D9 | 标注质检以 **VLM 自检闭环**为主（独立校验角色 + checklist 二分校验 + prompt 优化器重标，≤K 轮），人工只审 flagged 项与抽检样本（设计原则 #11）。 |
| D5 | 标注必须**离线**完成、人工审查后**冻结落盘**；评测回放阶段**零 VLM 调用**，指标全部为确定性集合运算。 |
| D6 | `vmem_bench` 与 `memstrata` **零相互导入**；SUT 通过 JSON 契约对接，adapter 放在 SUT 侧。 |
| D7 | 本轮只建 vmem_bench 本体，不做 baselines。 |
| D10 | location 仅作 prompt/scene context；不进入 SUT asset selection 或 headline metrics。 |
| D11 | 生产 gold 使用**人工一次确认的 canonical roster + exemplar**；自动 roster discovery 仅生成 proposal，不能冻结。实体范围只覆盖持续角色与 benchmark 消费的叙事关键 prop。 |

---

## 阶段一：离线标注管线（每部影片一次）

```
[原始长视频]
   │ 1. SBD（TransNetV2 + DINOv3 refine，已有 skill）→ shot_boundaries.csv
   │ 2. Chunking：镜头聚合成 chunk（帧数落在 [min,max]；单镜头超限则镜头内均分）
   │    → gold/chunk_index.json + gold/shot_boundaries.csv（只出切分逻辑，不落视频切片）
   │ 3. Proposal（可选）：VLM 只提出候选 roster；人一次性确认 benchmark-relevant 的
   │    canonical roster、identity_scope、aliases/grounding phrase 与 3–5 个 exemplar
   │ 4. 逐 shot 感知：专职 detector/segmenter + tracker 产生 tracklet/crop/embedding
   │ 5. 封闭集身份分配：individual 与所有同 kind seed exemplar 比较；category 按
   │    canonical phrase 收拢；低相似/低 margin → unknown/reject，禁止强分类/新建 entity
   │ 6. presence / first appearance：canonical tracklet span ∩ chunk（确定性）
   │ 7. 状态事件：VLM 只提议有限 ontology；确定性校验 entity policy + 描述，reject
   │    visible/in-focus/camera/location 等伪事件
   │ 8. prompt：VLM 自然化；canonical name 与已接受事件由确定性后处理补全并校验
   │ 9. ★ 阻断 QA：unknown、seed 漏证据、prompt omission、alias split、无效事件、
   │    missing crop 均 flag；strict lint 非零时禁止 freeze
   │ 10. 人审：按 canonical entity 一问一卡 + 按 entity 聚合 state timeline；
   │     → review_patch → apply_patch → freeze（所有 blocking 项归零）→ human_reviewed: true
   ▼
[冻结的评测实例（gold）]
```

- VLM 只做 proposal、文本起草和低置信语义判断；稳定 ID 来自人工确认的 seed，定位交给专职感知模型，最终标注以冻结 JSON 为准。
- 独立可并行的 VLM/embedding/检测调用必须批量/并行执行（设计原则 #8）。
- 人工工作量与 canonical 实体/真实状态事件近似线性，而不是与 tracklet pair 数增长。

### 数据目录（发布形态）

发布只版本化「切分逻辑 + 标注结果 + 下载方式」；视频、切片、抽帧、crop 一律本地按需从源视频重建（gitignore）。

```
benchmarks/VMem-Bench/data/blender_open_movies/big_buck_bunny/
├── manifest.json             # ★ 源视频下载方式 + sha256 + fps/时长 + license + layout_hash + 管线出处
├── gold/                     # ★ 冻结标注 + 切分逻辑（human_reviewed: true 后不可变，小、可发布）
│   ├── chunk_index.json      #   chunk ↔ shot/frame 映射 + layout_hash（legacy: layout/chunk_index.json）
│   ├── shot_boundaries.csv   #   SBD 边界（legacy: layout/boundaries.csv）
│   ├── entity_registry.json  #   每个表征带 frame_index+bbox → crop 可重生
│   ├── chunk_annotations.json
│   └── embeddings.safetensors  # 不可重生（依赖具体 embedder）→ 必须随库
├── assets/{characters,props,locations}/<entity_id>/
│                              # ★ 发布资产库（crop + cover.jpg；旧 assets/<entity_id>/
│                              #   与 legacy derived/assets/ 仍接受）
├── review.html               # 人审页面（不进发布包）
└── tmp/                      # ☓ 过程文件、gitignore、可从源视频重建（legacy: build/ + derived/）
    ├── clips/chunk_XXX.mp4    #   按需切片（评测 oracle），不随库
    ├── frames/               #   抽帧缓存
    ├── candidates/           #   QA 落败分支 crop（调试）
    ├── auto_review.json      #   机器审核报告（灰区合并提案 + must_review 队列）
    └── checkpoint/ events.jsonl annotation_qa.json identity_candidates.json services.json …
# tmp/、logs/、review.html / review_patch.json 均不进发布包
```

---

## 阶段二：评测回放（每个 SUT 跑一遍，确定性）

对每个 chunk t（t = 0, 1, …，严格因果，见设计原则 #1）：

| 步骤 | 信息 | 流向 | 时机 |
|---|---|---|---|
| A | `PromptPacket`（prompt 文本，无来源提示） | bench → SUT | 评分前 |
| B | `ComposedContextRecord`（选中实体/representation + 指令 + 排除项） | SUT → bench | 评分前 |
| C | 五指标计算（纯集合运算，对照 gold；**不调 VLM**） | bench 内部 | — |
| D | `ObservationPacket`（GT chunk 视频 + per-entity crop + 权威命名/ID + 首现描述 + 状态事件） | bench → SUT ingester | 评分**后**（generator-oracle 反馈） |

- gold 的出现清单、forbidden 清单、scenario 标签、embeddings **永不可见于 SUT**（设计原则 #5）。
- 步骤 D 的权威 ID 传播（设计原则 #4）使 SUT 记忆键与 gold 天然对齐，评分退化为确定性 ID 集合匹配——这是回放阶段能做到零 VLM 的前提。
- SUT 被测的能力：给定剧本式 prompt，从自己的记忆库中**选出正确的资产子集与 representation**（含拒绝干扰项、拒绝过期状态、控制上下文大小）。
- 每 chunk 评分明细与最终报告落盘；报告 `versions` 必须记录 gold 版本、指标版本、标注模型指纹。

指标定义（Sufficiency / Parsimony / Compactness / Fidelity / Avoidance）及空输出/N-A 边界规则见 [`schemas_and_contracts.md`](../../benchmark/schemas_and_contracts.md)；总原则：**缺失输出按 gold 要求罚分或记 N/A，绝不送满分**。

---

## v1 → v2 变更记录

1. **删除在线动态标注评测**：v1 在评测时用 VLM 从 *SUT 自己的记忆库* 候选中判定视频出现实体（GT 不独立、可被 gaming）；v2 标注全部离线冻结，回放零 VLM。
2. **裁决显式/隐式提示矛盾**：v1 workflow 曾要求 prompt 显式声明资产来源，与设计原则 #3 冲突；v2 以隐式提示为准，显式检索指令仅作为 gold 内部字段供 Fidelity 评分，绝不进 prompt。
3. **评分锚点从 SUT asset space 换到冻结 gold**（设计原则 #5 归位）。
4. **修复空输出退化值**：v1 中空 selected/instructions/exclusions 均得 1.0（躺平可刷分）；v2 按 gold 要求罚分或 N/A。
5. **Avoidance 语义修正**：v1 检查 forbidden 资产是否出现在 GT 视频里（评的是视频不是 SUT）；v2 检查 SUT 组合的上下文是否引用了过期/禁止资产。
6. **解耦**：移除 `from memstrata import ...`；契约类型全部在 `vmem_bench.common.schemas` 自包含定义。
