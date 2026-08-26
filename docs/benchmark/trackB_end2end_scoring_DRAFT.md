# MoVE-Bench · Track B 打分方案（剧本驱动 · 端到端生产）— DRAFT v0.2

> 状态：**草案 v0.2**，已按 Opus-5 评审收敛到可实现。不改动权威规范 [`scoring_v2.md`](scoring_v2.md)（Track A / gold-replay 权威）。
> 命名：**MoVE-Bench** = *Memory-aware causal long video generation benchmark*。
> 本文只定义 **Track B**（端到端生产）：以**剧本**为输入，对 **SUT 生成出来的视频**打分，**不使用 gold 影片**。
>
> **v0.2 相对 v0.1 的关键修订**（评审 P0/P1）：
> 1. 判官 roster 从「只放 present」改成 **闭世界混合盲化 roster**（present ∪ forbidden ∪ decoys），修复 precision≡1、avoidance 不可测，并顺带得到判官假阳率。
> 2. `run.py` 的两条 GT 泄漏（forbidden 直接 deprecate、referenced_entities 喂 router）由 **`--bench-mode`** 关闭；旧 optCA 结果标注为 **oracle-assisted**。
> 3. prompt 由 **bench 冻结**（不再由 SUT 侧 `adapters/screenplay.py` 渲染），并设 `name_anchored` / `description_provided` 两个**公开档位**（对齐 Track A）。
> 4. 身份漂移**降级为诊断**（medoid 锚 + within-shot 归一 + coverage/frozen_rate 门），**不做 headline**；headline = `f1 + avoidance_ok(bench-mode) + state_correctness`。
> 5. GT 由 **`vmem_bench` 侧独立导出器**产出（不扩 SUT 包），并叠加**手写硬样本标注层**（state 新态文本、lookalike 对、present 完备性）。
> 6. 新增确定性 **stitch_coherence**（拼接跳变）；砍掉 embedding 二次确认 / `redundancy_sim` / 4.10 差值（暂缓）/ MegaLoc·ArcFace 路由（暂缓）。

## 0. 一句话
给定一部**带因果硬样本标注的剧本**，SUT 端到端生成整部长视频；MoVE-Bench 用钉死 VLM 对着**生成视频**判：
该出现的记忆实体是否**真的出现在画面里**、有没有**乱生成**、**该规避的**（已故/损毁/证伪）是否**没出现**、
**状态变化**后是否呈现新态、**同名不同实例**是否选对；身份漂移作为诊断曲线。GT 全部来自剧本冻结文本（+手写硬样本标注）。

作用域：**长程视觉实体记忆的端到端实现**（存在性 / 覆盖 / 反乱生成 / 避让 / 状态 / 实例区分；身份漂移诊断）。
与 Track A 的区别：**Track A 判「SUT 选的参考图」是否覆盖；Track B 判「SUT 生成的视频」里实体是否真被画出来**——测「记忆 → 生成」的落地保真，不只是选择。

---

## 1. SUT 契约（防作弊硬约束）
- **输入**：bench 冻结的**逐 shot prose prompt**（`trackB_prompts/<story>_<register>.json`，带 SHA）。SUT **只能原样消费**，不得自渲染、不得追加 GT。
  - 两个**公开档位**（bench 控制，不是 SUT 自由度，对齐 Track A）：
    - `name_anchored`：prose 里 `(E#)` 标签解析为**规范全名**（`楔形泥碑（E2）→ 楔形泥碑`），实体被点名。
    - `description_provided`：标签解析为**外观短语**而非专名，测「靠描述从历史唤起」。
  - **绝不注入** present / forbidden / state / look-alike / kind 答案——只在 GT，永不进 prompt。
- **输出**：
  1. 逐 chunk **生成视频片段**（生产循环的 `review/segments/seg_NNN.mp4`，与 shot index 1:1）；
  2. 可选**选择轨迹** `progress.json.chunks[].selected_assets`（供诊断，非必需，不进 headline）。
- **运行契约**：Track B 的生产 run 必须以 **`--bench-mode`** 跑（见 §6），启动时断言「本 run 未消费任何 GT 字段」并写入 run manifest；否则该 run 的 avoidance/recall 记为 **oracle-assisted**，不得作为 Track B 结果。

## 2. GT（来自剧本冻结文本 + 手写硬样本标注层）
由 **`vmem_bench` 侧导出器**（`scoring/trackb_gt.py`）从 `production_screenplay` 逐 shot 派生，叠加**手写标注 sidecar**（`trackB_gt/<story>.overrides.json`）。**导出器与 SUT 包零 import**（遵 `benchmarks/VMem-Bench/AGENTS.md` 规则2）。

| GT 字段 | 定义 | 来源 |
|---|---|---|
| `present_required` | 该 shot **必须**在场、计召回 | `active_characters` ∪ planned_assets(`required=true`, op∈{preserve,transform}) |
| `present_allowed` | 允许出现、**不计召回也不算乱生成**（如 POV 主体、场景内自然出现） | sidecar 人核补充 |
| `forbidden` | 必须**不出现** | planned_assets `operation∈{avoid,deprecate}` |
| `first_appearances` | 该 shot 首现实体（前缀扫描 present_required） | 派生 |
| `continuity` = `present_required \ first_appearances` | 记忆可测集（首现不计召回） | 派生 |
| `decoys` | 本 shot 明确**不在场**的实体（其它场景 location / 已退场角色），采样 3–5 | 派生（roster \ present \ forbidden）+ sidecar 可指定 |
| `state_expected[eid]` | `{from_shot, label, desc}`：变化后应呈现的新态，**沿 shot 序单调粘性传播** | sidecar（`operation=transform` 只标发生镜，粘性靠 sidecar+传播规则） |
| `lookalike_pairs` | 需区分的相似实体对及**各自区分特征** | sidecar |
| `gap[eid]` | 久别重逢：与上次 present 相隔 chunk 数 | 派生 |
| `kind[eid]` | `character / location / prop`（`entity_type=object → prop`） | `main_entities[].entity_type` 映射 |

> **为何必须有 sidecar**：实测 `0001` 的 `traceability` 只有散文 `conversion_notes`、无结构化 `hard_cases`；E3 日志「完好→风化」**只写在 `continuity_requirements` 散文里、没有任何 `operation=transform`**；E4 透镜 transform 只标在发生镜 `shot_0010`，之后镜是 `preserve`。所以 `state_expected` 无法从 `operation` 完整机器导出，需要手写标注层 + 粘性传播。工作量：`0001/0002/0003` 共 16+11+11=38 shot。

> 「谁该出现 / 谁要规避 / 该是什么状态 / 哪个实例对」**确定**；只有「在不在 / 什么状态 / 哪个实例 / 像不像」用 VLM/embedding。=「确定性 GT + 有界噪声感知判定」，与 scoring_v2 同哲学。

## 3. 判定协议（VLM judge · 闭世界盲化 roster · 复用 Track A 基建）
对每个生成 chunk，做 **VLM 调用（k=3 投票）**，输入：
1. **混合盲化 roster 文本**：`roster = present_required ∪ present_allowed ∪ forbidden ∪ decoys`，**打乱顺序**，每条给 `entity_id | name (kind): 外观描述`（state-change 实体附「原态/新态两态简述」；lookalike 对附「两实例区分特征」）。**判官不被告知**哪些是 present / forbidden / decoy。
2. **该 chunk 的生成视频片段**（`review/segments/seg_NNN.mp4`，`fps=2.0` 自采样，832×480，与 Track A 同款）。

VLM 逐 roster 条目返回严格 JSON（以 `entity_id` 为键）：
```json
{"items": [
  {"entity_id": "E2", "present": true,  "state": "changed", "instance_match": "E2"},
  {"entity_id": "E5", "present": false, "state": null,       "instance_match": null},
  {"entity_id": "E1", "present": "abstain"}
 ],
 "extra": ["<画面里明显出现、但不在 roster 的显著对象描述>"]}
```
- `present ∈ {true, false, "abstain"}`：该实体是否**出现在生成视频**里；不确定→`abstain`（不计入分子分母）。场景/地点的 present = 视频**发生在该场景**里（照抄 scoring_v2）。
- `state`（仅 state-change 实体）：`default` / `changed`（对照 `state_expected.label`）。
- `instance_match`（仅 lookalike 对成员）：画面里那个实例更像哪个 id（判「选对没」）。
- `extra`：**诊断用**自由文本（凭空生成的显著对象），**不进任何公式**（precision 走闭世界，见 §4.2）。

**判定分工（设计原则）**：
- **存在性 / 实例判别 / 状态 = VLM 为主，不依赖抠图**。VLM 看整段视频、用文字区分特征直接判，绕开「从生成视频抠实体极易抠错」。
- **视觉 embedding 只服务身份漂移诊断（§4.5），且仅在干净 crop 上算**；抠不准就**弃权(N/A)**。

**钉死项**：judge=`qwen3-vl-32b`，`temperature=0`，`k=3` 多数票 + 报 item 级一致率（一致率本身是判官噪声地板的一部分），`fps=2.0`；固化 vLLM 版本与 `mm_processor_kwargs`，**落盘全部原始响应**。

**鲁棒解析（P1-19 修复）**：判官输出走**整体 JSON 解析 + 严格 schema 校验**；失败**重试一次**；再失败该 chunk 记 `parse_error`（剔除并计数），**绝不静默当 false**。

### 3.1 实体定位与弃权（localization & abstention · 仅身份漂移诊断需要）
身份漂移（§4.5）需从生成帧抠出实体 crop。为避免抠错污染：
- **定位**：按 roster **name/外观**用检测/分割（GDINO+SAM，复用 **bench 侧 S5** `annotation/pipeline/stages/s5_*`；**与 SUT 内部无关 → 可公平地测别的端到端管线**，**不引** `memstrata/skills/crop_acquisition/*`）。
- **置信门 + VLM 确认**：crop 只有（检测置信度高）**且**（VLM 确认「该框就是目标实体」）才接受；否则记 **N/A**。
- **弃权而非错分**：坏 crop 一律 N/A；`coverage`（可用样本 / 应有样本）是**一等指标**（见 §4.5、§5），不是附注。

### 3.2 嵌入器路由（v0 单路 + 诊断）
- **v0**：身份漂移只用 **DINOv3** 单路（`facebook/dinov3-vitb16-pretrain-lvd1689m`），无论 kind，并强制报覆盖率。理由：生成分辨率 832×480，人物脸区常 <80px，`ArcFace` 无脸即 raise；`0002` 两快递员同穿红雨衣、`0003` 哈桑与劫匪同披沙色斗篷，DINOv3 二次确认在这些剧本上注定失效——所以**不把 embedding 当实例判别信号**。
- **暂缓**：`(kind × 可用性)` 两级路由、ArcFace（脸）、MegaLoc（地点）作为**诊断列**（报覆盖率），不进 v0 headline。

---

## 4. 指标（逐实体决策 → pooled micro 主口径 → 影片/语料聚合；硬样本子分只在被标注 shot 上聚合）
记单 chunk：`Preq`=present_required，`C`=continuity，`F`=forbidden；判官判 present=true 的 roster 条目集 `A`。

### 4.1 recall[continuity]（**headline**：生成侧记忆覆盖，**按 kind 拆分**）
\[ \text{recall}_c = \frac{|C \cap A|}{|C|}\quad(|C|>0) \]
靠记忆调回的实体**真的被画进视频**的比例。**必报 `recall_by_kind`**（character / prop / location 分别报）：location 的 present 近乎白送（任何室内镜都在该室内），且占 GT 三到四成，混在一起会系统性抬高 headline。headline 用 **character+prop 为主口径**，location 单列。
- 两个可操作档位（判官 prompt 内写明阈值，`fps=2.0`，5s≈10 帧）：`recall_loose` = ≥1 帧可辨；`recall_strict` = ≥25% 采样帧可辨。

### 4.2 precision（反乱生成，**闭世界**）
\[ \text{precision}_c = \frac{|A \cap (Preq \cup Pallow)|}{|A|}\quad(|A|>0) \]
判官从**混合 roster**里判 present 的条目中，真正**该/可在场**的比例。因为 roster 含 forbidden+decoys，`A` 不再构造性 ⊆ present（修复 v0.1 的 precision≡1）。`present_allowed` 计入分子（不罚正确的 POV/场景内自然出现），**location 不进分母**（它不是「乱生成」出来的对象）。`extra` 自由文本仅诊断。

### 4.3 f1
`precision` 与 `recall`(character+prop) 的调和平均（headline 主分）。

### 4.4 avoidance_ok（**废弃证据规避**，硬样本，bench-mode 下才可信）
\[ \text{avoidance\_ok} = 1 - \frac{\#\{e\in F:\ \text{判 present}\}}{\#\{e\in F\}} \]
已故/损毁/证伪实体是否**没出现**。因 forbidden 现在进了盲化 roster，判官被真问过「它在吗」，**可测**。报 **`violations / opportunities` 计数 + Wilson 区间**（机会少，勿用比率宏平均冒充指标）。**必须在 `--bench-mode` 下重跑**才算数（旧 optCA 的 avoid_ok 0.667→1.0 是 oracle 注入的，见 §6）。

### 4.5 identity-drift 身份漂移（**诊断，非 headline**）
对每个复现实体 `e`：经 §3.1 定位+置信门抠出**可用** crop（抠不准弃权），DINOv3 取单位向量 `x_e(t)`。
- **锚点 = 前 K=3 次出现中通过 crop QA 的 crop 的 medoid**（复用 bench S5 `identity_consistency.py` 的「紧簇→medoid，否则 VLM 裁决」逻辑）；簇不紧致 → 该实体整条曲线记 `no_valid_anchor`。
- **within-shot 归一**：同一 shot 内同实体多帧 crop 的 `1-cos` 均值 = 该 SUT 输出上的**定位+embedding 本地噪声底**（同时刻真实漂移≈0）。
\[ \text{drift}_e(k) = \big(1-\cos(anchor_e, x_e(t_k))\big) - \overline{\text{within\_shot\_dissim}}_e \]
- **frozen 守门（防冻帧刷分）**：若跨 shot crop 的 DINOv3 相似度 ≥ 片内上界（近乎同一张图）→ 该样本记 `frozen`，不计入 drift、单列 `frozen_rate`。
- **coverage 门**：可用样本 / 应有样本 <50% → 该实体/该片 drift 记 `insufficient_coverage`，**不出数**（宁可不发，也不发低覆盖率曲线）。
- 报 `drift vs k` 曲线 + `coverage` + `frozen_rate` + `no_valid_anchor` 率。标量用 **identity margin**（跨 embedder 可比、无阈值）：`margin_e(t)=cos(anchor_e,x_e(t)) − mean_{e'≠e,同kind} cos(anchor_e,x_{e'})`（>0 才算认得出是同一个）。

### 4.6 state_correctness（**状态变化**，硬样本）
在 `state_expected` 覆盖（含**粘性传播**）的 shot 上：\(\text{state\_ok} = \frac{\#\{VLM.state=changed\ \text{且}\ e\ \text{present}\}}{\#\{state\text{-}change\ shots\ 且\ e\ present\}}\)。测「变化后**持续**呈现新态而非旧态」——传播是关键（0001 的 E4 在 0010 迸裂，0011/0014/0015 必须仍带裂）。

### 4.7 instance_discrimination（**同名不同实例 / look-alike**，硬样本，**VLM 主判**）
对 lookalike 对里**每个成员独立**判 present + `instance_match`，出**两个**指标：
- `instance_correct` = 该在场的那个被正确画出/认出；
- **`wrong_instance_rate`** = 被画成了对里的**另一个**（真正有信息量的失败模式）。
「pair 同时在场」的 shot（如 0002 的擦肩镜）**单列子集**。**embedding 不参与实例判别**（§3.2 已论证在这些剧本上无效）。

### 4.8 long_gap_reappearance（**久别重逢**，硬样本）
仅在 `gap[e] ≥ K`（如 K=5）的重逢 shot 上，单列 4.5 的诊断（身份是否长间隔后仍稳）+ 4.1 的 recall（久别的是否还被调回）。

### 4.9 stitch_coherence（**拼接/时序连贯**，新增，零模型成本）
逐 chunk 产物拼接成 `long_video.mp4`，chunk 边界跳变是本路线最典型可见缺陷。
\[ \text{stitch\_jump} = d_{boundary}/\overline{d_{intra}} \]
`d_boundary` = 相邻 chunk 边界两帧 DINOv3 距离；`d_intra` = 片内相邻采样帧距离均值。`transition="continue"` 的镜应≈1，`cut` 的镜排除。确定性、无 VLM。

### 4.10 描述量（不计分）
`budget` = |A|；`extra` 计数（诊断）。**砍掉** v0.1 的 `redundancy_sim`（Track B 里同实体近重复非失败模式，且与 §4.5 frozen 守门语义冲突）。

---

## 5. 聚合与复现
- **pooled micro 为主口径**（把全语料的实体级决策汇成一个分母），宏平均（逐 chunk 比率再平均）为**次要**——现在 `|C|=1` 的 chunk 与 `|C|=5` 等权、方差大且可博弈。
- `recall/precision/f1` 报 **bootstrap CI**；`avoidance_ok` 报 `violations/opportunities` + **Wilson 区间**。
- 身份类（4.5/4.8）**只在可用样本上聚合**并**随附 coverage + frozen_rate**；coverage<50% 不出数。
- **三级 noise floor（方差相加取 √，非取最大）**：
  1. **判官噪声** = k=3 自一致率 +（抽样 60 个 `(chunk,entity)` 人核一致性）；
  2. **判官假阳率(FPR)** = decoy 上被判 present 的比率（**免费**，直接来自盲化 roster）；
  3. **定位/embedding 噪声** = §4.5 的 within-shot 本地噪声（**在线估计**，不外推 Track A gold）；
  4.（补充）**生成随机性** = 至少一部剧本 ×3 seed 的 seed 间标准差。
  只声称**超过合成噪声地板**的系统差距。
- GT（present/forbidden/state/lookalike/gap）**冻结**；感知判定用**钉死** VLM + DINOv3 → 可复现。

## 6. 防作弊 / bench-mode（关闭 run.py 的两条 GT 泄漏）
现状 `src/memstrata/production/run.py` 有两条真实泄漏，与「SUT 看不到 GT」矛盾：
1. `for aid in shot.forbidden_ids: bank.update_status(aid, DEPRECATED)` —— 把 GT `forbidden` 直接写进 SUT 记忆库（`OPTIMIZATION_JOURNAL` it6–8 记录 avoid_ok 0.667→1.0 正是此 oracle）。
2. `router.route(..., referenced_entities=shot.referenced_entities, ...)` —— 把 GT `present` 集合交给 SUT 路由。

**`--bench-mode`**：(a) 跳过 forbidden 的 deprecate；(b) 不向 router 传 `referenced_entities`/`onscreen_entities`（router 只靠 prompt+transition+scene_return；记忆选择仍由 prompt 驱动的读路径完成，合法）；(c) 启动断言「未消费任何 GT 字段」并写 `run_manifest.json`。
- **旧 optCA 结果**（morphic backend，非 bench-mode）标注为 **oracle-assisted**，只能作为「上界/内部诊断」，不作 Track B 数字。bench-mode 重跑三片后，把「oracle-assisted vs bench-mode」并列成表（本身是论文里一张好证据）。

## 7. 效度对照（退化 SUT，证明指标能区分）
最便宜的效度验证，跑三个 sanity SUT：
- **frozen SUT**：把首帧复制到所有 chunk（可从现有产物**合成**，无需 GPU）——预期 `drift≈0` 但 `frozen_rate=1`、`recall` 低；验证 §4.5 的 frozen 守门。
- **memoryless T2V**：每 chunk 独立生成、无记忆——预期 `recall[continuity]` 低、`stitch_jump` 高。
- **full-oracle SUT**：GT 全给——上界。
三者若在指标上分不开，则指标是坏的。

## 8. 硬样本 → 指标 覆盖表
| 硬样本 | 剧本例 | 主要指标 |
|---|---|---|
| 废弃证据规避（已故/沉没/证伪） | 伊莱亚斯亡、E5 沉船、伪造地图 | 4.4 avoidance_ok（bench-mode） |
| 状态变化（粘性） | 泥碑 完好→碎裂→拼合、透镜裂、日志风化 | 4.6 state_correctness |
| 同名/相似实例 | 两快递员、劫匪 vs 向导、玛拉 vs 莉娜 | 4.7 instance（correct + wrong_instance_rate） |
| 久别重逢 | 玛拉长间隔返场、向导归来 | 4.8 long_gap + 4.1 recall |
| 记忆覆盖 / 反乱生成 | 全体 | 4.1 recall_by_kind / 4.2 precision / 4.3 f1（headline） |
| 长程身份保持 | 主角贯穿全片 | 4.5 identity-drift（**诊断**，非 headline） |
| 拼接连贯 | continue 镜边界 | 4.9 stitch_coherence |

> **命名**：60–90s / 11–16 shot 撑不起「long-horizon」；v0 把 4.5 称 `mid-horizon identity consistency`，待补一部 40+ shot 剧本再用 long-horizon。

## 9. 落地路线（2 天可出可信首版；利好：三片 optCA 产物已在盘上，v0 判定不需新 GPU 生成）
**Day 1 — GT + 判官（产出 f1 / state / instance）**
1. bench 侧独立 GT 导出器 `scoring/trackb_gt.py` → `trackB_gt/<story>.json`（present_required/allowed、forbidden、first/continuity、decoys、gap、kind、state_expected 粘性、lookalike）。
2. 手写硬样本 sidecar `trackB_gt/<story>.overrides.json`（state 新态文本、lookalike 对、present 完备性）；人核 38 shot。
3. 冻结逐 shot prompt `trackB_prompts/<story>_<register>.json`（含 SHA）。
4. 判官 `scoring/end2end_coverage.py`：混合盲化 roster、逐 `entity_id` JSON、严格解析+重试、k=3 投票、落盘原始响应 → 跑现有三片 optCA 产物 → `recall_by_kind / precision / f1 / state_ok / instance_correct / wrong_instance_rate` + **decoy FPR**。

**Day 2 — 干净 avoidance + 噪声地板 + drift（诊断）**
5. `run.py --bench-mode` 关两条 oracle 通路，重跑三片（约 2h GPU）→ **干净 avoidance_ok**，与 oracle-assisted 并列。
6. 判官噪声地板：抽 60 个 `(chunk,entity)` 人核 + decoy FPR + k=3 自一致率。
7. drift 曲线（**诊断**）：bench S5 抠 crop、DINOv3 单路、medoid 锚、within-shot 归一、报 coverage/frozen_rate；coverage<50% 不发布并写明原因。
8. 退化 SUT 对照（§7）：至少 frozen + memoryless，证明指标可区分。

**明确 defer**：MegaLoc/ArcFace 三路由、embedding 二次确认与 `disputed`、`redundancy_sim`、4.10 决策-落地差值、多 seed 方差全量、全量剧本、count/relation/共现、1h 长片。
**最高风险步骤**：§3.1/§4.5「从生成视频抠干净 crop」（prop 落库偏慢 → 定位召回低）。缓解 = drift 降级为诊断 + coverage 作为发布门槛。
