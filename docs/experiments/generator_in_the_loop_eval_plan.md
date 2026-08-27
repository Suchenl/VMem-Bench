# Track B：超长程记忆寻址 + 读效率（generator-in-the-loop）评测设计

> 状态：**定稿中（v2，2026-07-23）**。本版把评测重构为「两个问题 + 两张图」，并锚定一个关键事实：
> **一致性类的定量 headline 是 generation-free 的**（复用 Track A 的确定性组合层打分，跑在超长受控
> 场景上），真实生成器只用于**少数 hard-case 的定性/像素级佐证**。术语以
> [`glossary.md`](https://github.com/Suchenl/MemStrata/blob/main/docs/glossary.md) 为准；与冻结的 Track A 主表（[`../benchmark/scoring.md`](../benchmark/scoring.md)）
> **互补、不替代**。

## 0. 为什么这样重构（算力硬约束）

真实生成 1 chunk ≈ 11min（Wan2.2-I2V-A14B+SVI，fast 4×A800）。**200 chunk × 11min ≈ 37h/场景/系统**，
3 系统 × 3 场景不可行。因此 **80–200 chunk 的定量 headline 必须不依赖逐 chunk 生成**。幸运的是，
本 track 想测的两件事本质都不需要像素，见 §1。

## 1. 两个问题（本 track 的定位）

MemStrata 的主张：**意图对齐的具名实体组合，在超长程上优于被动检索 dump**。这拆成两个可测问题：

1. **一致性能否撑到超长距离？** 第 5 chunk 出现的角色/场景/道具，在第 80/120/200 chunk 被重新
   点名（controlled return）时，系统是否**把正确的那个 asset 取回来**（而不是相似但错的实例、
   或已废弃的旧状态）。→ 这是**记忆寻址正确性**问题，对着场景 ground-truth 判定，**确定性、零生成、
   零 VLM**。
2. **读路径是否保持快？** flat-history retrieval / full-context packing / VLM-read 的读开销随历史
   增长；MemStrata 默认读路径 = name/alias 匹配 + identifier 解引用，理论上应**近似平坦**。→ 纯计时 +
   模型调用计数，**零生成**。

两张 headline 图：**① Consistency vs memory-distance**、**② Read-latency vs chunk-index**。

## 2. 现状盘点：≈80% 已在（Track A 打分器复用）

`vmem_bench/scoring/{metrics,runner}.py` 已产出（所有系统经同一 `ComposedContextRecord` 接口，
`timing_ms`/`model_calls` 内建）：

| 用户想要的 Track B 指标 | 已实现的对应物 | 位置 |
|---|---|---|
| Return Success / Long-gap Recall | `MemRecall`（返场实体距离加权召回）+ `memdist_curve`（x=memory_length,y=recall） | `metrics.py: returning_distances / memdist_curve / memdist_auc` |
| Avoidance Violation | `Avoidance = 1-|R∩F_active|/|F_active|` | `metrics.py`, `F_active` 由 `state_events` 推导 |
| State Correctness / deprecated-evidence | `Fidelity`（指令/需求匹配）+ `F_active`（废弃表征） | `metrics.py` |
| Read Latency / Latency Slope | `horizon_curve[].timing_ms` + `efficiency.{mean,median,p95,max}_composition_ms` | `runner.py:444-462` |
| Model Calls on Read | `horizon_curve[].model_calls` + `efficiency.total_model_calls` | 同上 |
| multi-instance 场景 | `scenario_tags` 自动标 `multi-instance`，`per_scenario_tag_mean` | `runner.py` |

**所以本 track 的定量部分不是从零造，而是：把已有确定性打分器跑在新的超长受控场景上 + 补 3 个小指标 +
画 2 张图。**

## 3. 指标（确定性；新增项标 ★）

**一致性类（headline 图 ①）**
- `MemRecall` / `memdist_curve` / `memdist_auc`：返场召回随记忆距离的曲线与归一化 AUC（越平越好）。
- `Avoidance`：是否复用已废弃/失败证据。
- `Fidelity`：是否用**当前**状态而非 deprecated（State Correctness）。
- ★ **Wrong-instance Rate**：在 `multi-instance` 标记的 chunk 里，SUT 选了**相似但错**的具名实体的比例。
  （现有 `Sufficiency` 只看「选没选中对的」；错拿相似实例需单列，因为这是 name-anchor/type-routing
  消融的直接卖点。）判定：`selected ∩ (confusable_wrong_set)` 非空。
- 报告 **memstrata vs baselines 的曲线/AUC 对比**，并按 near/mid/far 与 absence-length bucket 分层。

**速度类（headline 图 ②）**
- Read Latency：`efficiency.*_composition_ms`（+ 逐 chunk `timing_ms` 做斜率/散点）。
- Model Calls on Read：`efficiency.total_model_calls`（MemStrata 默认路径应 = 0；VLM-read baseline > 0）。
- Latency Slope：`timing_ms` 对 chunk index 的回归斜率（MemStrata 应 ≈ 0；full-context/flat 应 > 0）。
- ★ **Context Size**：送给 generator 的 reference 数 / 图像数（由 `ComposedContextRecord.selected` 派生）
  / 估算 token；随 chunk 记录。
- ★ **Memory Growth**：记忆库条目数随 chunk 增长（需在读路径旁记 per-chunk memory size）。

**方法学红线（写进文中）**
1. 一致性判定在**组合层**（asset_id 集合 vs 场景 ground-truth），不在生成像素上——避免感知噪声污染
   headline；生成像素只进定性/佐证（§5）。
2. 读延迟对比要求**每个 baseline adapter 都真实记录 `timing_ms`**（同 `handle_prompt` span）；铺开前
   先抽查（承接 `run_all_movies_handoff.md` 的「先验证再信任」）。
3. Wrong-instance / State 的 confusable/deprecated 集合来自场景 ground-truth，非模型判断。

## 4. 测试样本：超长受控场景 = 「视觉长程 needle-in-haystack」

**样本的输入 = 一段段提示词**（对所有系统唯一且公平的输入）；**评分 key（present/first_appearances/
state_events/gold_instructions + confusable 组 + 期望返场 asset）是隐藏的 ground-truth**，系统看不到。

- **少而长**：3–5 个 long-rollout 场景，每个 **80–200 chunk**。
- **controlled return prompts**：按**递增 absence-length**（gap=5/20/50/120…）周期性点名早期实体，直接
  喂满 `memdist_curve` 的 x 轴。
- **5 类 hard case**（每类都要有）：re-appearance、multi-instance（两个相似角色/道具）、scene-return
  （离开再回到某地点）、state-change（实体外观/状态改变，旧表征应被废弃）、deprecated-evidence
  （曾失败/被否的证据不得复用）。
- **合成生成器（推荐）**：写一个确定性场景合成器 `make_long_rollout_scenario(seed, n_chunks,
  n_entities, return_schedule, hardcase_mix)`，同时产出 `prompts[]` 与匹配的 gold `ChunkAnnotation[]`。
  透明声明为**构造化压力测试**（等价于 LLM 长上下文的 NIAH），可复现、可无痛铺到 200 chunk。
- 可选：再挂 1 部 **BBB frozen gold 派生**场景作「连 Track A」的次级对照（内容是动画，但能对齐主表口径）。

## 5. 生成器的角色：定性画廊（少数 hard-case）

真实 FLUX→Wan2.2+SVI 管线（已在 A800 跑通，见 `experiments/results/micro_bench/svi_wan_natural_long_video`）
**不进定量 headline**，只做：
- 挑 **2–3 个代表性 hard-case 段**（如「相似双角色 + 远距返场」）在 memstrata vs 一个 dump baseline 下各
  渲染 ~15–20 chunk，**并排展示像素级后果**（身份串味 / 场景漂移）。
- 可选在这些短段上做像素级 re-ID（pinned ArcFace/DINOv3/MegaLoc）做**小样本佐证**，声明为定性证据、
  非 headline。

## 6. 落地顺序（smoke first，已选）

1. **场景合成器 + gold key**：先出 1 个场景 ~40–60 chunk（generation-free，便宜），覆盖 5 类 hard case +
   递增 return gap。
2. **跑既有打分器**：MemStrata-fast vs 2–3 强对比（`full_history` = dump 上界、`frame_retrieval@budget`、
   一个有 read-time model call 的 VLM-read/`recency`）。核对：`memdist_curve` 有值、`efficiency.timing_ms`
   各系统非零且可比、multi-instance 标记出现。
3. **补 3 小指标 + 画 2 图**：Wrong-instance Rate / Context Size / Memory Growth；Consistency-vs-distance、
   Latency-vs-chunk。
4. 全绿后：铺 3–5 场景 × 80–200 chunk（仍 generation-free）+ 渲染 2–3 段定性画廊。

## 7. 待你拍板的剩余开放点

1. **场景合成器 vs 手写剧本**：推荐确定性合成器（可复现、可到 200 chunk、NIAH 叙事好卖）。
2. **对比系统集**（速度图想凸显 read-path 差异，建议至少含一个 VLM-read / full-context 的「慢读」系统）。
3. 是否要 BBB-gold 派生场景做 Track A 连接（次级）。

## 8. 与代码/其他文档的关系
- 打分器：`vmem_bench/scoring/{metrics,runner}.py`（已含 memdist/efficiency/horizon）。
- 生成侧接线：[`method/generator_wiring.md`](https://github.com/Suchenl/MemStrata/blob/main/docs/method/generator_wiring.md)（video + image_backends 已 vendored）。
- 公平性：[`../baselines/fairness_decisions.md`](../baselines/fairness_decisions.md)——同后端/同 seed/同 encoder，只换记忆模块。
- 运行纪律：[`run_all_movies_handoff.md`](https://github.com/Suchenl/MemStrata/blob/main/docs/experiments/run_all_movies_handoff.md)（先验证再信任、静默降级门槛）。
