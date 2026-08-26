# MemStrata-Bench 标注管线：阶段级构造规格（S1–S7 产出 frozen gold）

> **本文是标注管线的阶段级权威（S1–S7 产出 frozen gold）；track-first 的追踪/re-ID 内部机制见
> [`annotation_tracking_internals.md`](annotation_tracking_internals.md)；回放/评分见 [`scoring.md`](scoring.md)。**
>
> 用途：本文件是 **MemStrata-Bench gold 数据"怎么造出来"** 的 benchmark 构造规格——逐阶段（S1–S7）
> 说明每个阶段做什么、由谁做、产出什么派生产物，直到冻结 gold。它是标注管线的**阶段级事实权威**，
> 供实现、复现与核对使用（不是论文章节叙述稿）。当前 Big Buck Bunny gold 已按本规格冻结（52 chunks，
> 末 chunk 为 `chunk_051`），生产走 `pipeline/` 的 S1–S7 orchestrator。
>
> 标注约定：
> - **[代码]** = 已在 `src/vmem_bench/annotation/pipeline/` 落地、可核验的事实（附文件路径）。
> - **[实跑]** = 用户当前实际运行时采用的配置（可能与代码默认参数不同）。
> - **[OPEN]** = 尚未拍板、需继续确认的点，见文末 §7（每条只保留一个明确结论或一个明确 OPEN，不留矛盾表述）。

---

## 0. 总原则（先读这条）

1. **模型输出永远是可替换的 proposal，gold 由确定性契约 + 人工裁决固定。**
   VLM 草稿、reviewer 判定、检测框都不是真值；稳定 ID、prompt-completeness、最终 gold 由确定性
   代码和人工决策锁定。[代码]
2. **质量控制不是"最后统一审一次"，而是分层穿插在多个阶段。** 每一类错误交给"能可靠抓到它、
   且最便宜"的层：可形式化错误 → 确定性 gate；需要视频语义判断 → 更强/专用模型；模型无法可靠
   区分的身份/叙事 → 人工。[代码 + 文档]
3. **人审是"决策面"，不是"日志"。** 人只裁决残余不确定性（必审 + 抽样），每次决策后重跑完整
   确定性复验；人工量随不确定性增长，不随 segment 数增长。[代码 + `staged_pipeline_plan.md`]
4. **原始 S1 起草结果永不被下游改写**，每个阶段写自己的派生产物（resumable，可中途停下人审再续）。[代码 `orchestrator.py`]

---

## 1. 阶段总览

管线为 **S1–S7**，由 `annotation/pipeline/orchestration/orchestrator.py` 统一编排（S2 起），
阶段目录见 `annotation/pipeline/stages/s1_..s7_`。**S1 内含起草 + 起草期审核**（VLM 起草 → 半人工
格式/覆盖审计与目标化重标 → 人工确认实体本体），产出格式干净、实体经人工确认的 `vlm_output.json`
后交给确定性 S2。复现细节：`stages/s1_vlm_annotation/`。

| 阶段 | 名称 | 谁来做 | 做什么 | 审核层 |
|---|---|---|---|---|
| **S1** | `s1_vlm_annotation`（起草 + 起草期审核） | **Qwen3.7 Plus** 起草/重标；**审计 agent** 扫格式与覆盖；**人工** 粘贴结果 + 审实体 | 全片起草 → 确定性扫描失败并按动作类型目标化重标（kit）→ 人工确认**实体本体/roster** | 起草 + 半人工格式审 + 人审（实体） |
| **S2** | `s2_annotation_postprocess` | 确定性代码 | schema/ID/时间/去重/可逆格式归一 + structural lint；S1 presence 仅作**候选** | 确定性 gate |
| **S3** | `s3_segment_auto_review_revise` | **Qwen3-VL-8B-Instruct**（reviewer，图快） | 对**视频切片**建立 PresenceLabel + 修 action 文本（canonical name 齐全）+ 多轮重排队；发 verdict | 模型审 + 修复回环 |
| **S4** | `s4_segment_sampling_human_review` | **Cursor agent 签字**（代替人工） | 审全部 BLOCK + 分层抽样；每次决策重跑完整确定性复验；生产阻塞到 S5 | 人审①（segment 级，已由 agent 代替） |
| **S5** | `s5_entities_visual_crop_acquisition` | GroundingDINO / SAM3 + **Qwen3-VL-8B-Instruct** + DINOv3/SigLIP | 逐实体建视觉库：检测/分割 → 关键帧 → 封闭集选 crop（对人工 exemplar）→ crop QA（含确定性暗门禁）→ **逐实体身份一致性门禁（WHO）** → 覆盖上限 + 因果 ≤t 绑定 | 确定性感知 + 模型（+ 残余人工） |
| **S6** | `s6_entities_visual_crop_human_review` | **人工** | 按 canonical 实体逐个审 crop/身份；高置信自动接受，残余入队 | 人审②（实体级） |
| **S7** | `s7_freeze_publish` | 确定性代码 + 人工冻结门 | 未决 BLOCK 清零 + 两处人审通过才 freeze；build_gold + layout hash + release_manifest | 确定性冻结门 |

> 一句话：**S1 起草+审核（VLM 起草 → 半人工格式/覆盖审与目标化重标 → 人工审实体）→ 确定性归一 →
> 8B 视频复核+修复 → 人审 segment → 感知取 crop → 人审实体 → 冻结**。审核穿插在 S1、S3、S4、S6。
>
> 归属说明：**起草期格式审 + 人工审实体归入 S1**，**不放 S2**——S2 在代码里是纯确定性 gate
> （无模型、无人工；不自动语义拆段/补洞），保持"确定性层"语义干净是本 benchmark 的卖点。

---

## 2. 逐阶段详解

### S1 — 起草 + 起草期审核（Qwen3.7 Plus 起草 + 半人工格式审 + 人工审实体）
S1 不只是"VLM 出草稿"，而是"把草稿打磨到格式干净、实体经人工确认"的作者侧阶段，产出可交给确定性
S2 的 `vlm_output.json`。权威复现文档：
`stages/s1_vlm_annotation/README.md`、`audit_checklist.md`、`prompts/`。[代码 + 实跑]

含三步：

1. **起草（Qwen3.7 Plus）**：模型 **Qwen3.7 Plus**（起草/captioning）。**[实跑基座]**
   `prompt_qwen3_7_plus_quick_v5.md`（`v6` 为对照/实验，非默认）。[代码]
   产出 `movie_dir/vlm_output.json`——按时间切 segment，每段含 action 句、`present_entity_ids`、
   首现标记、有限本体状态事件。action 简短（S3 限 ≤120 字符）；schema 要求
   `duration_seconds ≤ 15` 且段数 ≥ `ceil(T/15)`。
2. **起草期格式/覆盖审计 + 目标化重标（半人工）**：起草结束后对 `vlm_output.json` 做**确定性扫描**
   （覆盖头尾、重叠、内容大洞、超长段、实体时间戳/引用、脏 JSON 等），再按失败类型发起目标化
   重标，而不是一律整片重跑。角色分工（写论文时勿写成「Grok 直接改正式 JSON」）：
   - **审计 agent**（实跑多为 Cursor/Grok）：扫描、分类、生成 kit、合并、清场；
   - **人**：复制 `SEND_TO_VLM.md`、上传**完整原片**、粘贴返回 JSON；
   - **起草 VLM（仍为 Qwen3.7 Plus）**：按 v5 + 补丁（或 C 类精简提示）产出修补 JSON。
   动作类型：`FULL_RERUN` / `CONTINUE_HEAD|TAIL` / `FILL_GAP(S)` / `REPLACE_RANGE` /
   `REVISE_OVERLONG` / `SPLIT_OVERLONG_GT30`；稳定模板在
   `stages/s1_vlm_annotation/prompts/`，当次作业台在 `data/_vlm_rerun_kit_*/`
   （合并后清场，见 `benchmarks/VMem-Bench/AGENTS.md` 规则 3）。[实跑]
   **已拍板策略**：生产基座 v5；**`>30s` 强制 VLM 拆分**；**15–30s 可暂缓**硬切（若日后强制合规，
   优先 S2 确定性硬切而非 S3 重标）；≈1s 小缝可忽略；续标默认不附整份旧 JSON。
   验收清单见 `stages/s1_vlm_annotation/audit_checklist.md`。
3. **人工审实体**：人工确认**实体本体 / canonical roster**（哪些是 benchmark 相关实体、稳定 ID、
   多视角 exemplar、identity scope），作为下游 S5 封闭集分类与命名的**权威 seed**。[实跑 + 机制见
   `annotation_tracking_internals.md` §10 "人工 seed 约束的封闭集分配"]

- 产出：格式干净、实体经人工确认的 `vlm_output.json`（+ 人工 roster/exemplar seed），喂入 S2。
- **归属**：起草期格式审 + 人工实体审归入 S1，**不放 S2**（S2 保持纯确定性）。
- **为何必须审**：起草 VLM 常见截断/重叠/大洞/超长；**S2 只 lint/归一、不语义补洞拆段**，拖到
  S3 成本更高。动机与失败→动作映射见 `s1_vlm_annotation/README.md` + `audit_checklist.md`。
- **原始起草 JSON 语义上是"未经信任的草稿"**；S2 起的编排流水线都写派生文件、不改 S1 产物。
  （起草期修复**可以**改写正式 `vlm_output.json`，那是作者侧定稿。）
  [代码 `vlm_auto_review.py` 注释 "raw S1 JSON is never modified"]
- **固化程度（已核实）**：步骤 2–3 当前为**半人工/外部**（kit + 审计 agent + 起草 VLM + 人工 roster），
  **尚未**挂进 `pipeline/orchestrator`。roster seed 代码线索在
  `pipeline_track_first/roster_seed.py`。论文应如实写「半人工」，不必假装已编排化。

### S2 — 确定性后处理 + structural lint
- 谁：确定性代码，无模型。`stages/s2_annotation_postprocess/`（`normalize.py`、`materialize.py`、
  `entity_checks.py`、`segment_checks.py`、`structural_lint.py`）。[代码]
- 做什么：schema、identifier、timestamp、重复项、可逆格式的归一化 + 结构 lint。
  **明确不做**：按语义自动拆 `>15s` 超长段、自动补时间轴内容洞（这些属 S1 作者侧或日后可选的
  S2 硬切策略，当前未默认开启）。
- 关键：**S1 的 `present_entity_ids` 在 S2 只是候选**（标注来源：seed-claimed / presence-window
  overlap / model-proposed），**不因人工 roster 已确认就把每段 presence 当作可信事实**。
  [代码 + `staged_pipeline_plan.md` §3]
- 产出：`tmp/pipeline/s2_annotation_postprocess/normalized_annotation.json`。

### S3 — 更强/快速模型视频复核 + action 修复（多轮，穿插审核）
- 谁：**Qwen3-VL-8B-Instruct**（reviewer，用户为图快选 8B）。
  **[代码默认]** `vlm_auto_review.py: DEFAULT_MODEL = "qwen3-vl-32b"`、`orchestrator --reviewer-model` 默认
  `qwen3-vl-32b`；**[实跑]** 用 8B。多 endpoint 共享池并发。[代码 + 实跑]
- 做什么（`stages/s3_segment_auto_review_revise/`）：
  1. **视频对齐 PresenceLabel**：把每个 segment **连同其视频切片**发给 reviewer，对每个候选实体
     判 present/absent + 置信度 + 引用帧证据；低置信或与草稿冲突时由**独立第二 endpoint / 不同抽帧复核**。
     [代码 `vlm_auto_review.py`、设计 `staged_pipeline_plan.md` §4.1]
  2. **action 文本修复**：某实体视频确认在场但 action 未逐字提到其 canonical name 时，触发确定性
     action-repair（要求所有 required canonical name 逐字出现、无"实体列表尾巴 coda"、不引入 roster
     外实体），有限次重试（`DEFAULT_MAX_ACTION_REPAIR_ATTEMPTS = 3`）。[代码 `canonical_names.py`]
  3. **多轮重排队**：未 accept 的 segment 重新入共享池，直到 accept 或
     `DEFAULT_MAX_REVIEW_ROUNDS = 2`。[代码]
  4. **typed verdict**（`verdicts.py`）：`PASS` / `WARN` / `BLOCK` / `RETRYABLE_ERROR`。
     - 确定性 blocker（缺 canonical name、entity-list coda、空 canonical name）→ **BLOCK**；
     - 高置信 reviewer 冲突 → **BLOCK**；
     - 基础设施失败（输出截断、JSON 解析失败、上下文溢出、请求失败、崩溃）→ **RETRYABLE_ERROR**，
       自动换 endpoint 重试，**不当作标注判断丢给人**；
     - 上游 presence 标记可信时的低/中置信分歧 → 降为 PASS 的**审计样本**（spot_check）。[代码 `verdicts.py`]
- 产出：`tmp/pipeline/s3_segment_auto_review_revise/auto_revised_annotation.json` + `segment_audit.jsonl`。
- **这一层与 S1 内的起草期审核互补**：S1 的格式/覆盖审 + 人工实体审在**整片草稿/实体本体**层面把关，
  S3 是**逐段视频级**的自动复核 + 多轮修复。

### S4 — 人审 segment（第一处人审，生产阻塞）
- 谁：**人工**（review UI）。`stages/s4_segment_sampling_human_review/`（`sampling.py`、`clips.py`、`decisions.py`）。[代码]
- 队列：**全部 BLOCK + 其余 WARN/PASS 的分层随机审计样本**（`build_sample`，最小 3 条、抽样率 1%；
  抽样率应由控制集错误率校准而非任意设定）。[代码 orchestrator `_S4_SAMPLE_MINIMUM=3`、`_S4_SAMPLE_RATE=0.01`]
- 规则：每次人工决策都**重跑完整 annotation 的确定性复验**（schema、canonical-name、presence-ID、
  prompt-complete、layout gate）；`accept` **不能覆盖任何确定性失败**。[代码 `decisions.apply_s4_decisions`]
- 门控：生产 `s4_mode=blocking`，阻塞 S5 直到所有 blocking 决策解决；非阻塞模式仅用于 automation
  smoke，并在产物打 `automation_smoke_only` 标记。[代码 orchestrator]
- **由 agent 代替人审（2026-07 起，BlenderOpenMovies）**：S4 只是签字门,真正的逐段视觉复审已在 **S3
  用 VLM** 完成;因此 S4 的人工签字由 **Cursor agent** 代替,采纳 S3 的 VLM 复审标注、accept 整条队列
  (`review_queue.json` 的全部 BLOCK + 抽样),将 stage 置 `human_reviewed=true`,并在 `review_audit.json`
  写 `agent_reviewed=true / reviewer=cursor-agent / review_method=s4_agent_review.py / reviewed_at` 溯源。
  工具:`scripts/vmem_bench/maintenance/s4_agent_review.py`(走生产 `decisions.apply_s4_decisions`,
  同样重跑确定性复验;需要时可用 `--decisions` 传入个别 segment 的 present/action 修订)。这样 S4 视为
  "确定完",不再等待人工。[代码 `s4_agent_review.py`]

### S5 — 逐实体视觉库（确定性感知 + 8B VLM）
- 谁/模型：
  - 检测：**GroundingDINO-base**（`grounding_dino.py: IDEA-Research/grounding-dino-base`）；[代码]
  - 分割：**SAM3** concept + box/point refine（`sam3_concept.py`、`sam3_refine.py`）；[代码]
  - embedding / crop QA：**DINOv3**（`embedding.py: facebook/dinov3-vits16-pretrain-lvd1689m`）
    原型余弦，可选 SigLIP2 零样本分类；[代码]
  - VLM（crop picker / 身份审计）：**Qwen3-VL-8B-Instruct**。**[代码默认]** grounder-model 默认
    `qwen3-vl-32b`（`vlm_grounding.py: DEFAULT_MODEL="Qwen3VL-32B-Instruct"`）；**[实跑]** 用 8B，
    **之后可能改**。[实跑]
- 做什么（`stages/s5_entities_visual_crop_acquisition/`）：关键帧选择 → 检测/分割出候选 → **封闭集
  选 crop**（`crop_picker.QwenCropPicker`：对**人工 exemplar**同 kind 封闭集匹配，margin 不足则
  unknown/reject，不强行分类）→ **exemplar 认领带 runner-up margin**（`exemplar_identity.
  exclusive_assign_candidates`：候选对某实体的 DINOv3 相似度须比次高实体高出 `assign_margin=0.06`
  才自动认领，否则留给 VLM，避免相近身份在低阈值附近串号）→ crop QA（几何/清晰度 +
  **确定性暗/低信息门禁**）→ 属性去重（DINOv3/SigLIP2 剔混类桶）→ **逐实体身份一致性门禁**
  （WHO 而非 WHERE）→ 覆盖上限、mask 质量门、**因果 ≤t slot 绑定**。[代码]
- **crop QA 的确定性暗/低信息门禁**（`crop_qa.audit_crop` + `entity_luminance_stats`）：在**蒙版实体
  像素**（`alpha>0`，而非白底合成图）上算亮度均值/方差，近黑且近平（`mean<26 且 std<16`，`location`
  不判）→ `dark_low_information` → `accepted=false`。阈值保守（暗但有结构、std 高的 crop 不误杀），
  在**进 VLM/人工之前**就拦掉黑影 blob，省算力也省人工。[代码]
- **逐实体身份一致性门禁**（`identity_consistency.run_identity_consistency`，`character`/`prop`；
  `location` 不判）：单靠 DINOv3 阈值在不同分布上会崩（Blender 干净可分；LSMDC 真人电影里相似
  外观/光照/景别轻松越过低 re-ID 阈值 → 库内混入别的实体）。故把 DINOv3 当**便宜 triage**，真正
  的"是否同一实体"交给 VLM：(1) 全部接受库 crop 紧贴 medoid（`skip_vlm_medoid_floor=0.55` /
  `skip_vlm_min_pairwise=0.40`）→ 判单一身份、**不调 VLM 不进人工**（人工量下降主来源）；
  (2) 否则把该实体**全部 crop 一次性**发 8B，对每张判 **`identity_visible`**（是否露出可辨识身份
  视图；后脑勺/背影/严重模糊/近黑/大遮挡 → false，直接丢库 `identity_not_visible`，不计为"不同实体"，
  专治 8B 对黑影/背影凭发色轮廓硬判 same 的欠拒）与 **`same_entity`**（先定 dominant 身份再逐张判，
  非本实体且置信度达标 → `identity_gate_reject`）；(3) 无 VLM auditor 时**只标 `needs_human`、绝不凭
  DINOv3 单独拒绝**，仅当无任何可用可见 crop 存活才整实体退人工，防误清空。[代码 + 已接入
  `orchestrator._run_s5` 两条 route]
- 产出：`tmp/pipeline/s5_entities_visual_crop_acquisition/crop_proposals.json` + crop 文件；
  **`identity_audit.json`**（每实体 cohesive/vlm_resolved/needs_human 判定 + rejected/not_visible/
  needs_human crop 计数，量化人工节省）。
- 后端（检测器/分割器）可切换，作**构造期消融**。[代码 `--crop-route`、`--proposer`]
- **[实跑核验 2026-07]** Gran Torino 单片 `s5_only` 重跑：改前 9 实体里 2 个整实体退人工、9 张混杂 crop
  原样留库；改后 8/8 全自动判定、`needs_human=0`、`identity_visible` 丢弃 11 张（含 char_007 后脑勺、
  char_005 三张严重模糊）、暗门禁采集期拦 6 张。见 §7.2 的模型档位讨论。

### S6 — 人审实体/crop（第二处人审，冻结前阻塞）
- 谁：**人工**。`stages/s6_entities_visual_crop_human_review/`（`queue.py`、`auto_accept.py`、
  `patch.py`、`review_apply.py`）。[代码]
- 做什么：按 **canonical 实体逐个**审 crop / 身份（全片 crop grid per entity）；高置信 proposal
  自动接受，只有残余/歧义入队；应用 review patch。[代码]
- **这就是"人工检查实体"的环节。**

### S7 — 冻结 + 发布
- 谁：确定性代码 + 人工冻结门。`stages/s7_freeze_publish/`（`gates.py`、`freeze.py`、`build_gold.py`、`backfill.py`）。[代码]
- freeze gate（`gates.py`）：
  - **未决内容 BLOCK 硬阻塞**（S4 可用 `resolved_verdicts` 覆盖后再判）；
  - `RETRYABLE_ERROR` 属基础设施信号，记为**软残留**，不硬阻塞；
  - 需两处人审均 `human_reviewed`。[代码]
- 产出：`gold/entity_registry.json`、`s7_freeze_publish/release_manifest.json`；pin layout hash。
  layout / excluded-segment 变更 = 新数据版本，强制重算 layout hash。[代码 + `staged_pipeline_plan.md` §8]

---

## 3. 模型档位表（当前实跑 vs 代码默认）

| 用途 | 阶段 | **实跑模型** | 代码默认 | 备注 |
|---|---|---|---|---|
| 起草 / 目标化重标 | S1 | **Qwen3.7 Plus** | prompt **v5**（v6 对照） | 起草 + kit 重标仍用同一起草模型 |
| 格式/覆盖审计编排 | S1 | **审计 agent**（实跑 Cursor/Grok） | —（外部/半人工） | 扫描失败、出 kit、合并清场；不替代起草 VLM |
| 实体审核 | S1 | **人工** | —（roster seed） | 确认实体本体/roster |
| 视频复核 reviewer | S3 | **Qwen3-VL-8B-Instruct** | `qwen3-vl-32b` | 用户为图快选 8B |
| 检测 | S5 | GroundingDINO-base | 同 | 确定性 |
| 分割 | S5 | SAM3 (concept+refine) | 同 | 确定性 |
| embedding / crop QA | S5 | DINOv3 (+可选 SigLIP2) | 同 | 零样本，非 VLM |
| crop picker / 身份审计 VLM | S5 | **Qwen3-VL-8B-Instruct** | `Qwen3VL-32B-Instruct` | 之后可能改 |
| 人审 segment | S4 | 人工 | — | 全 BLOCK + 抽样 |
| 人审实体 | S6 | 人工 | — | 逐 canonical 实体 |

> **[OPEN]** identity-critical 阶段（S5/S6）设计文档提到"可能要求更强 model tier"
> （`staged_pipeline_plan.md` §"模型层级边界"）；当前实跑 S5 用 8B，是否/何时把 identity 判定升 32B 待定（见 §7.2）。

---

## 4. 审核分层与 verdict 裁决（论文卖点）

- **三层错误处置**（[`dashboard_and_review.md`](dashboard_and_review.md) §10）：
  A. 确定性自动 gate（schema/ID/路径/派生一致/prompt 覆盖）——无需模型或人；
  B. 高置信机器建议 + 有限自动修复（保留 diff/证据/阈值，可撤销，需抽样人审校准）；
  C. 人工裁决（多实例身份、通道冲突、不可逆状态、影响大量 chunk 的 merge/split、抽样审计）。
- **穿插位置**：S1（半人工格式/覆盖审 + 人工实体审）、S3（8B 视频复核 + 修复回环）、S4（人审 segment）、
  **S5（视觉库 WHO 一致性：确定性暗/低信息门禁 → DINOv3 cohesion triage → 8B 身份审计 → 残余人工，
  是 A/B/C 三层的又一干净实例）**、S6（人审实体）、S7（确定性冻结门）。
- **verdict**（`verdicts.py`）：`PASS`/`WARN`/`BLOCK`/`RETRYABLE_ERROR`；确定性 blocker 与高置信冲突
  恒为 BLOCK，基础设施失败为 RETRYABLE 自动重试，低置信分歧降为 PASS 审计样本。
- **三元度量**（annotation triad）：速度（wall-clock）、质量（annotation quality）、人工量
  （review-card 数）——每次发布都报告，使标注成本可复现。[docs 原则 12]

---

## 5. 数据契约与产物

- 输入：`movie_dir/vlm_output.json`（S1 起草，并经 S1 内格式/覆盖审 + 人工实体审修复）。
- 中间：`movie_dir/tmp/pipeline/s2..s7/`（各阶段派生 annotation、audit、queue、proposals、state.json）。
- 冻结 gold：`movie_dir/gold/entity_registry.json` 等 + `release_manifest.json`（含 layout hash）。
- embedding 落盘：数值载荷（embedding/feature）存 `.safetensors` sidecar（按 `representation_id` 索引），
  JSON 只存字符串引用。[docs 原则 10]
- gold instance 结构见论文附录 A.6 / `benchmark_data_construction` 图：chunk intents（prompt-complete）
  + typed entity records + 分层视觉表征 + 生命周期事件 + 每 chunk required records + forbidden/deprecated
  （仅用于打分、不给 SUT）+ hard-case 标签。

---

## 6. 下游对应：论文文本待同步项（本节仅为下游指引，权威仍是上面的阶段规格）

- 论文 §5.2（正文，两段）：已按"分层穿插审核 + S3 多轮修复 + 两处人审"改写，但**模型档位当前写的是
  "stronger reviewer (default 32B-tier)"**——**需按本文改为 Qwen3-VL-8B（实跑）**，并把 S1 补成
  "起草 + 半人工格式/覆盖审与目标化重标 + 人工审实体"（勿写成 Grok 单独改 JSON）。
- 论文附录 A.5（`app:bench-making`）：已逐阶段展开 S1–S7 + verdict + freeze gate；**同样需把 S3/S5
  的模型改为 8B，并把 S1 按上条补全**；**S5 需补上"WHO 身份一致性门禁"**（确定性暗门禁 → DINOv3
  cohesion triage → 8B 一次性多图身份审计含 `identity_visible`/`same_entity` → 残余人工），作为
  分层 QC 卖点在 identity 维度的又一实例。
- 图 `benchmark_making_pipeline`：**待重画**（S1–S7 一行主轴 + 审核层色条：确定性 / 半人工起草期审+8B 复核↺
  / 人审①② / 冻结门；去标题、大字、小框）。

---

## 7. 已核实结论与 OPEN 项

> 说明：以下每条要么给出**已核实结论**，要么标 **[OPEN]** 保留一个明确未决问题——不再留矛盾表述。

1. **S1 内审核的固化程度（已核实 2026-07）**：归入 S1、不单列 S1.5；步骤 2–3 为半人工
   （`s1_vlm_annotation/prompts` + `data/_vlm_rerun_kit_*` + 审计 agent + 起草 VLM + 人工 roster），
   **未**挂进 `pipeline/orchestrator`；roster seed 见 `pipeline_track_first/roster_seed.py`。
   如实写明半人工。复现读 `stages/s1_vlm_annotation/README.md`。
2. **S3/S5 模型（已核实实跑 = Qwen3-VL-8B-Instruct）+ [OPEN] identity 是否升 32B**：
   **[实跑核验 2026-07]** S5 身份门禁在 8B 上的行为：`identity_visible` + 暗门禁把后脑勺/模糊/黑影
   crop 稳定丢弃（Gran Torino 丢 11+6），人工量归零；但 8B 不擅长"确认是谁"，`n_crops_rejected`
   （自信判为不同实体）趋于 0——**清晰但混入的同类他人**（如另一红发角色）仍可能被判 same 漏网，
   需 **32B** 或 **reference-anchored 审计**（把最清晰正脸作显式参考图逐张比对）兜底。
   **[OPEN]** 是否把 identity-critical 判定的默认档位升到 32B，及升级时点，待定。
3. **两条代码目录的关系（已核实）**：生产 gold **走 `annotation/pipeline/` 的 S1–S7 orchestrator**；
   `annotation/pipeline_track_first/`（track/re-ID/identity_clustering/roster_seed）不是"另一条竞争
   管线"，而是 S5 视觉库/身份判定所调用的**追踪/re-ID 内部机制**（机制权威见
   `annotation_tracking_internals.md`）。身份判定的**生产默认 = `seeded`（人工 roster 约束的封闭集
   分配）**，`cluster_vlm`/`greedy` 仅作 proposal/消融开关。
4. **[OPEN] 数据集分支**：BlenderOpenMovies 与 LSMDC 是否走**同一条**管线？LSMDC（真人，如 American
   Beauty 300-segment）在 `staged_pipeline_plan.md` 有专门讨论，是否有分支差异待确认。
5. **[OPEN] verdict/轮次/抽样率**：`MAX_REVIEW_ROUNDS=2`、`ACTION_REPAIR_ATTEMPTS=3`、S4 抽样率
   1%/最小 3——这些是否为最终生产值待确认（抽样率应由控制集错误率校准）。
6. **[OPEN] 人审规模**：S4/S6 每片实际人工卡数 / 分钟数（triad 的"人工量"轴）是否有实测数据可引用。
