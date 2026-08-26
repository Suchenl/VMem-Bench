# MemStrata-Bench 评分规范 v2（视觉覆盖 · Visual-Coverage）

> 状态：**当前权威评分规范**。取代 [`scoring.md`](scoring.md)（v1，ID 交集 + embedder 最近邻，已废弃）。
> 实现见 [`../../src/vmem_bench/scoring/visual_coverage.py`](../../src/vmem_bench/scoring/visual_coverage.py)
> 与 [`../../src/vmem_bench/scoring/README.md`](../../src/vmem_bench/scoring/README.md)。
> `metric_version = visual-coverage-2.1`。
>
> 本文是**评分规范**（公式 + 数据形态）；要"跑一部影片"的端到端流程（Stage 1 产 context → Stage 2 打分、
> 命令、依赖、坑）见运行手册 [`running_eval.md`](running_eval.md)。
> Stage 2 judge 服务加速与替代方案分析见
> [`tracka_stage2_scoring_acceleration_opus5_20260728.md`](tracka_stage2_scoring_acceleration_opus5_20260728.md)。

## 0. 一句话

benchmark 评的是**长程视觉实体记忆**:一个记忆系统在超长视频里,为每个片段选出的**参考图(context)**,是否
**覆盖了该出现的实体、没有乱召回、不冗余**。判定全部由**钉死版开源 VLM 对着视频**做,GT 只提供"该有谁"的
**冻结文本清单**——不依赖 gold crop,不评状态。

作用域(scope):**视觉实体记忆**(存在性 / 记忆覆盖 / 冗余)。**动作/事件/交互记忆、外观状态一致性 out of scope**
(后者作为 v2+ 的可选一致性轴,见 §6)。

---

## 1. benchmark 数据形态(v1 发布内容)

每部影片包含,且**仅包含**:

| 数据 | 说明 | 来源 |
|---|---|---|
| 超长视频 | 原始影片 | 公开数据集 |
| 逐 chunk 提示词 | S4 剧本 `action`(+音效)原文,实体按 prose 自然点名(强制系统依赖记忆);不注入 present 名单 | `chunk_annotations.json.prompt` |
| 冻结 roster | 每个实体:`entity_id` / `kind`(character/prop/location)/ **唯一稳定的 name** / `description` | `gold/entity_registry.json` |
| 逐 chunk `present` | 该片段在场的实体 id 集合(记忆覆盖的 ground truth) | `chunk_annotations.json.present` |
| 逐 chunk `first_appearances` | 该片段首次出现的实体(用于区分 continuity) | `chunk_annotations.json.first_appearances` |

**不包含**(相对旧版):gold crop、状态/state_events、gold latents。**gold(文本 GT)以 S4 人核标注为准**
(S1 VLM 标注 → S2 后处理 → S3 逐段自动复审 → **S4 采样人核**),由
[`build_gold_from_s4_review.py`](../../scripts/vmem_bench/maintenance/build_gold_from_s4_review.py)
从 `tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json` 转出 `gold/`;
**S5–S7(crop 采集 / crop 人核 / crop 冻结)不需要——参考图由 SUT 自产,不落 gold**。运行流程见
[`running_eval.md`](running_eval.md)。

**命名硬约束**:任何会被后续提示词 name-anchored 引用的**复现实体**,必须有**唯一、跨段稳定**的 name;否则系统无法被
"点名"唤起记忆。一次性实体可命名可不命名,但需有 id。

### 关键定义

- **continuity 实体**(记忆可测集):`C = present \ first_appearances`。首现实体没有历史可回忆,不计入记忆召回。

---

## 2. SUT 契约(两个 Track)

- **Track A（本规范主评）**:SUT 每个 chunk 输出一个 **context = 一组参考图**(它从记忆里选出的实体图)。
  bench 对这组图评存在性/覆盖/冗余。系统输出落在
  `benchmark_run/visual_selections/<system>.json`(每 chunk 的 `selected[].representations[].crop_abspath`)。
- **Track B**:SUT 输出真实生成的**视频**;同样的指标改在生成视频上评(把"参考图"换成"生成片段中的实体呈现")。

> 参考图由 **SUT 自己**产出(它从历史视频建的记忆),**不是** benchmark 标注的 gold crop。

---

## 3. 评测协议(VLM judge)

对每个 chunk,做**一次** VLM 调用,输入:

1. **roster 文本**:该 chunk `present` 实体的 `entity_id | name (kind): description`;
2. **N 张无标签参考图**(SUT 为该 chunk 选的,不带任何 id/名字标签);
3. **该 chunk 的视频片段**(按 `seconds_span` 现切,`video_url` 直接喂,模型按 `fps=2.0` 自采样)。

VLM 返回严格 JSON:

```json
{"items": [{"i": 0, "present": true, "entity_id": "char_001"}, ...],
 "missing": ["<该出现却无任何参考图覆盖的 entity_id>", ...]}
```

- `present`:该图代表的对象是否**出现在视频**中(场景/地点的 `present` = 视频**发生在该场景**里);
- `entity_id`:该图对应 roster 里的哪个 id,都不对应填 `"none"`;
- `missing`:roster 里在视频中明显出现、但这 N 张图**一张都没覆盖**到的 id。

**钉死项(复现性)**:judge = `qwen3-vl-32b`(开源、可复现),`temperature=0`,`fps=2.0`。"该有谁"来自冻结 gold,
**确定**;只有"像不像/在不在"的逐图判定用 VLM →「确定性 GT + 有界噪声的感知判定」。

---

## 4. 指标定义(逐 chunk 计算,再对 chunk 取平均)

记单个 chunk 为 \(c\):
- \(S_c\) = SUT 选出的参考图集合,\(n = |S_c|\);
- \(P_c\) = gold present 集合;\(C_c = P_c \setminus \text{first\_appearances}_c\) = continuity 集合;
- \(\mathrm{Pr}_c = \{ i \in S_c : \text{present}_i = \text{true} \}\),\(m = |\mathrm{Pr}_c|\)(判为在场的图);
- \(\text{missing}^P_c = \text{missing} \cap P_c\),\(\text{missing}^C_c = \text{missing}^P_c \cap C_c\)。

### 4.0 指标总览（当前发布口径，逐 chunk 算 → 取平均）

分三族：**质量**（选得准不准、全不全、省不省）、**时间效率**（快不快，与质量正交）、**描述量**（只报告不计分）。
`score.json.summary` 里的键名如下，论文表格直接取这些：

| 族 | summary 键 | 指标 | 一句话 | 范围 | 方向 | headline |
|---|---|---|---|---|---|---|
| 质量 | `precision_mean` | precision | 选进 context 的图里真正在场的比例（反乱召回） | 0–1 | ↑ | |
| 质量 | `recall_mean` | recall（continuity） | 该靠记忆调回的实体被覆盖的比例（**只在 continuity 上算**） | 0–1 | ↑ | ✅ |
| 质量 | `f1_mean` | F1 | precision 与 recall 的调和均值 | 0–1 | ↑ | ✅ 主分 |
| 质量 | （per-chunk）`recall_all` | recall_all | 含首现的召回，**仅诊断**、不进 summary | 0–1 | ↑ | |
| 质量 | `redundancy_vlm_mean` | redun_vlm | 同实体近重复占比（VLM 计数） | 0–1 | ↓ | |
| 质量 | `redundancy_sim_mean` | redun_sim | 同实体近重复（DINOv3 自相似，threshold-free） | 0–1 | ↓ | |
| 质量 | `selection_efficiency_mean` | sel_eff | 既在场又非近重复的图占总选图的比例（选择有用率） | 0–1 | ↑ | |
| 时间 | `retrieval_ms_mean` | retrieval_ms | SUT `compose()` 每 chunk 检索延迟（Stage-1） | ms | ↓ | |
| 时间 | `write_ms_mean` | write_ms | SUT `observe_segment()` 每 chunk 写记忆延迟（Stage-1） | ms | ↓ | |
| 时间 | `score_ms_mean` | score_ms | 该 chunk 的 VLM 打分延迟（Stage-2，非 SUT 成本） | ms | ↓ | |
| 描述 | `budget_avg_refs_per_chunk_with_refs` | budget | 平均每 chunk 选图数（context 规模，**只报告不计分**） | ≥0 | — | |

> **质量 vs 时间是两个正交的族**：`sel_eff`（选择有用率）**不是**时间效率；时间效率单独由 `*_ms` 三项刻画。
> `retrieval_ms` / `write_ms` 随片长增长的曲线（每 chunk 都记）用于体现"记忆机制的时间可扩展性"。

### 4.1 precision(相关性 / 反乱召回)

\[ \text{precision}_c = \frac{m}{n}, \quad (n>0) \]

系统选进 context 的图里,真正在画面里的比例。乱召回(放了不在场的图)→ 下降。

### 4.2 recall（记忆覆盖，**headline**）

\[ \text{recall}_c = \frac{|C_c| - |\text{missing}^C_c|}{|C_c|}, \quad (|C_c|>0,\ \text{否则}=1) \]

**只在 continuity 实体上算**:该靠记忆调回的实体里,被参考图覆盖到的比例。首现实体不计。

> 诊断量 `recall_all`(含首现,`(|P_c|-|missing^P_c|)/|P_c|`)只写进 per-chunk details,不进 summary。

### 4.3 f1（headline 主分）

\[ \text{f1}_c = \frac{2\cdot \text{precision}_c \cdot \text{recall}_c}{\text{precision}_c + \text{recall}_c} \]

### 4.4 redundancy（per-entity 近重复,**两个变体并列**)

先把 \(\mathrm{Pr}_c\) 按预测 `entity_id` 分组(`none` 各自成单元素组)。两个变体都是 **per-entity**:
**只有同一实体内部的近重复算浪费**,同实体的不同角度/景别(互补视角)**永不罚**。设计上不用单一定义,因为
"重复"既可从语义(VLM)也可从表征(DINO)看,两者互为交叉验证。

**变体 1 —— `redundancy_vlm`(VLM 计数)**:对每个真实 id 且 \(|g|\ge 2\) 的组 \(g\),VLM 判组内**视觉上不同的视角数**
\(dv_g\)(近重复合并):

\[ \text{redundant}_c = \sum_g \big(|g| - dv_g\big), \qquad
   \text{redundancy\_vlm}_c = \frac{\text{redundant}_c}{m}, \quad (m>0) \]

**变体 2 —— `redundancy_sim`(DINO 自相似,threshold-free)**:对每个真实 id 且 \(|g|\ge 2\) 的组,取组内 DINOv3
余弦相似度矩阵的**上三角均值**(即 你说的"矩阵和 / 格数"),再对全 chunk 按 pair 数加权平均:

\[ \text{msim}_g = \frac{\sum_{i<j}\cos(e_i,e_j)}{\binom{|g|}{2}}, \qquad
   \text{redundancy\_sim}_c = \frac{\sum_g \text{msim}_g \cdot \binom{|g|}{2}}{\sum_g \binom{|g|}{2}} \]

\(e_i\) 为 DINOv3 的 **CLS token** L2 归一化向量(故 \(\cos = \) 点积)。钉死 embedder =
**`facebook/dinov3-vitb16-pretrain-lvd1689m`**(DINOv3 ViT-B/16,约 86M 参数,CLS 维度 768,LVD-1689M 预训练);
取 `last_hidden_state[:,0]` 的 CLS(不用 patch-mean:CLS 全局语义对"同实体近重复"更稳,已实测足够)。
**\(\to 1\) = 选的图几乎相同(高冗余),越低 = 视角越多样**。确定、可复现、**不设阈值**(阈值太脆,弃用)。
若组内相似度普遍偏高,即为选图冗余的直接信号。

> 两变体分工:`redundancy_vlm` 语义层、给整数冗余计数;`redundancy_sim` 表征层、连续、确定。发布时都报,互相印证。

### 4.5 selection_efficiency（选择有用率，`sel_eff`）

\[ \text{selection\_efficiency}_c = \frac{m - \text{redundant}_c}{n}, \quad (n>0) \]

放进 context 的图里,**既在场又非近重复**的比例(近重复计数取 `redundancy_vlm` 的整数 \(\text{redundant}_c\),因为
只有它给整数计数)。**不奖励极简、不惩罚互补多视角**。它同时惩罚"放了不在场的图"(降 precision 分子)和
"放了同实体的重复图"(算进 redundant),所以是一个"这次花的 context 预算里有多少是既相关又不浪费的"的比率。
**举例**:SUT 放了 10 张,7 张在场,这 7 张里有 2 张是同一实体的近重复 → useful=7−2=5 → sel_eff=5/10=0.5。

> **重命名说明(2026-07-25)**:原名 `efficiency`(表头 `eff`)易与"时间效率"混淆,已统一改为
> `selection_efficiency`(表头 `sel_eff`)。它衡量**选择质量**,与下面 4.5b 的**时间效率**是两个正交的族。

### 4.5b time-efficiency（时间效率,墙钟,与选择质量正交)

三项 per-chunk 墙钟延迟,体现"记忆机制快不快、随片长可不可扩展":
- \(\text{retrieval\_ms}_c\) = SUT `compose()`(从当前记忆检索出 context)的耗时,**Stage-1**、算 SUT 成本;
- \(\text{write\_ms}_c\) = SUT `observe_segment()`(把本段真实视频写进记忆)的耗时,**Stage-1**、算 SUT 成本;
- \(\text{score\_ms}_c\) = 该 chunk 的 VLM 打分耗时,**Stage-2**、是**评测侧成本**(不算 SUT,仅供参考)。

`retrieval_ms` / `write_ms` 逐 chunk 记录,画成随片长(累计 chunk 数)增长的曲线 → 时间可扩展性。

### 4.6 budget（描述性,非评分)

\[ \text{budget}_c = n \]

context 规模。**只报告、不计分**;跨系统公平比较时在**同 budget** 下比较。

### 4.7 空 context

\(n=0\):precision/redundancy/selection_efficiency 未定义(聚合时剔除);\(\text{recall}_c = 0\)(若 \(|C_c|>0\),否则 1)。

### 4.8 聚合（**时长加权**,因为 chunk 不等长）

**关键事实**:本 benchmark 的 chunk 是**语义/镜头切分,长度不等**(实测 3–27s,单片内均值 ~11–13s)。
所以"每个 chunk 等权"的简单算术平均会让一个 5s 段和一个 27s 段贡献相同 → 不合理。

**片内聚合(两套并报,`*_wmean` 为 headline 候选)**:
- `*_mean`(等权 chunk 平均):对有定义的 chunk 取算术平均(precision/selection_efficiency 在 \(n>0\);
  recall/f1 在 \(|C_c|>0\);redundancy 在 \(m>0\);时间三项在各自非空)。**作稳健性参考**。
- `*_wmean`(**时长加权**):\(\overline{x}_{\text{dur}} = \frac{\sum_c x_c \cdot \Delta t_c}{\sum_c \Delta t_c}\),
  \(\Delta t_c\) 为 chunk 秒数。长段权重大,反映"每单位视频时长"的表现。summary 里 `total_duration_s` 记录总时长。

> 备注(更严格的可选口径):对 precision/recall 这类**比率**,理论上最规范的是 **micro-average**
> (\(\sum m_c / \sum n_c\)、\(\sum \text{covered}_c / \sum |C_c|\)),按各自分母(参考图数/实体数)加权。
> micro 与时长加权高度相关(长段通常实体/参考更多)。当前先落地时长加权;如需 micro,已预留 per-chunk 计数可加。

**语料级聚合(跨片)**:按**影片总时长**加权合并各片的时长加权值(长片权重大),同时报**宏平均**(每片等权,
防单一超长片主导)。两者并列,headline 取时长加权、宏平均作稳健性对照。

---

## 5. 复现性与 noise floor

- GT(roster + present)**冻结**,scoring 只在感知判定处用**钉死** VLM → 可复现。
- 发布时**随附 noise floor**:judge 的人机一致性 / 内在噪声(由受控探针测得,见
  `data/_runs/vlm_judge_probe/`)。**只声称超过 noise floor 的系统差距。**
- GT 由 S1–S3(含 S3 逐段人核)产出,带**已知且有界**的误差率;误标为二阶噪声,靠聚合稀释 + noise-floor 声明兜底。

---

## 6. 与 v1 的关系 / 未来轴

- v1(`scoring.md`,`metrics.py`/`visual.py`):ID 交集 + embedder 最近邻,依赖 gold crop 与状态,**已废弃**,仅留作复现旧数。
- **一致性(consistency)轴**(参考图 vs 视频里该实体的外观是否一致)为 v2+ 可选扩展;设计上**拿视频当外观事实源**,
  仍不需要 gold crop。当前 v1 发布**不含**该轴。
