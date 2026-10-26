# MemStrata Crop 原则（Bench + SUT）

> 统一约定：**如何采 crop、挂什么属性、如何去重 / 绑定、两边如何对齐**。
> 细节实现可改，本文件记录的是已拍板契约。
>
> 相关文档：
> - 属性包字段契约：本文件 `docs/benchmark/crop_contract.md` §3（原 `crop_attributes_contract.md` 已并入）
> - 评分外观定义：`docs/benchmark/scoring.md`（`current_appearance`）
> - SUT 包设计：`docs/method/design.md`

---

## 0. 总原则（两边共用）

1. **评分 / 记忆契约 ≠ 「每个 segment 现采一张」**  
   需要的是：给定 chunk \(t\) 与实体 \(e\)，能解析出「此刻该有的样子」  
   `current_appearance(t, e)` = \(t\) 上的 crop；若无 → **`chunk_id ≤ t` 且同状态的最近一张**。  
   稀疏库 + ≤t 绑定合法；逐槽全量 GPU 采集只是遗留消融。

2. **默认存蒙版（RGBA）**；白底合成只发生在喂模型时（`load_crop_rgb_for_model`）。

3. **属性用 VLM 闭集选择题**，不用亮度等启发式当主路径；失败降级为 `unknown`，**不阻塞流水线**。

4. **`memstrata` ↔ `vmem_bench` 零交叉 import**  
   属性包、去重算法各自镜像一份；枚举 / schema / 桶定义必须对齐（见 §3）。

5. **去重桶**（两边同一公式）：  
   `(spatial_angle, state_angle, shot_size, lighting)`  
   `occlusion` 只落盘，**不进**默认多样性桶。

6. **结构化 JSON 截断**：`JSONDecodeError: Unterminated string …` 大概率是 `max_tokens` 太短  
   （或 Qwen3 thinking 吃预算）——加大预算、关 thinking、对 `finish_reason=length` 重试；  
   不要先怀疑标注文件坏了。

---

## 1. 角色分工

| 包 | 角色 | Crop 在干什么 |
|----|------|----------------|
| **`vmem_bench`** | 造 gold / 评测数据 | S5 **采集**视觉库 → S6 审 → S7 冻金；槽位用 ≤t 绑定补齐 |
| **`memstrata`** | 被测系统（SUT） | 生产闭环里对已有 crop **分类属性** → 写入 `AssetRepresentation` → **属性多样性门控** \(R_j\) |

两边共享「属性包 + 去重桶」语义；**采集调度只在 Bench**，**在线记忆库维护只在 SUT**。

---

## 2. vmem_bench：S5 采集

### 2.1 目标

为每个实体建一份**有限覆盖的视觉库**（视角 / 状态 / 景别 / 光照等），再保证每个
`present(chunk, entity)` 槽位在 S6/S7 上**可解析**（精确 acquire 或 ≤t bind）。

### 2.2 模式

| 模式 | 含义 |
|------|------|
| **`coverage`（默认）** | 按实体 cap 采；首现 + 缺席再现 + 时间补点 → 属性去重 → ≤t 填槽 |
| `per_slot` | 遗留消融：每个 segment × 在场实体各采一次 |

**默认 cap：** character 8 · prop 5 · location 12（偏软；不对全片 location 再加硬税）。

**单个实体 acquire 优先级：**

1. 首次出现（必采）
2. 缺席后再现（在场 chunk 列表中间隔 &gt; 1）
3. 均匀时间补点直到 cap

CLI：`--task-mode`、`--cap-character/prop/location`、`--max-total-acquire`。

### 2.3 蒙版门禁

- 默认 **RGBA 蒙版**落盘；mask 外透明。
- **破碎 mask 不进候选池**（默认不做 bbox 兜底），见 `mask_quality.py`：
  - 最大连通域 &lt; 前景 75% → 拒
  - 孔洞占比 &gt; 12% → 拒
  - 显著碎片 &gt; 2 → 拒

### 2.4 路由

| 路由 | 行为 |
|------|------|
| **`propose_and_pick`（默认）** | 检测器（SAM3 / GDINO / fusion）出候选 → VLM **只选 index**（不做 bbox 回归） |
| `vlm_sam_refine` | VLM 出 bbox+point → 可选 SAM3 refine |

Picker / grounder 默认 `max_tokens=2048`，截断可加倍重试（至 8192）。

### 2.5 流水线（coverage）

```
annotation
  → plan_tasks(coverage)            # crop_tasks.json = 仅 GPU acquire
  → propose_and_pick | vlm_sam      # 蒙版 + mask 门禁 + picker
  → attach_crop_attributes          # 接受的 acquire 挂属性包
  → run_identity_consistency        # 每实体身份一致性门禁 → identity_audit.json
  → prune_library_by_attributes     # crop_library.json
  → expand_library_to_slots         # ≤t bind → 槽位齐全 crop_proposals.json
  → S6 人工 / 自动接受 → S7 build_gold
```

**身份一致性门禁（`identity_consistency.py`，WHO 而非 WHERE）：**

单靠 DINOv3 阈值在不同分布上会崩（Blender 干净、可分；LSDMC 真实电影里相似
外观/光照/景别轻松越过低 re-ID 阈值 → 库内混入别的实体）。因此该门禁只把
DINOv3 当**便宜的 triage**，真正的“是否同一实体”判定交给 VLM：

1. 对每个实体（`character`/`prop`；`location` 不判）嵌入其全部接受库 crop。
2. 若所有 crop 紧贴 medoid（`skip_vlm_medoid_floor` / `skip_vlm_min_pairwise`）
   → 判为单一身份，**不调 VLM、不进人工**（Blender 式易例，人工量下降主要来源）。
3. 否则把该实体**全部 crop 一次性**发给 VLM，让它对每张 crop 判两件事：
   - **`identity_visible`**：这张 crop 是否真的露出可辨识身份的视图（角色要看到
     正脸/清晰侧脸，道具要看到关键特征）。后脑勺/背影、严重模糊、过暗全黑、遮挡到
     看不出是谁 → `identity_visible=false`。这类 crop 无法作为身份锚点，**直接丢出库**
     （`accepted=false`，`identity_not_visible`），也不计入“非本实体”拒绝数。
     这解决了 8B 审核模型面对后脑勺/黑影时“凭发色轮廓硬判 same”的欠拒问题。
   - **`same_entity`**：先选出占多数的 dominant 身份，再逐张判断是否属于它；被判为
     “非本实体”且置信度达标的 crop → `accepted=false`（`identity_gate_reject`），
     不会再被 slot 绑定扩散。
4. 无 VLM auditor 时**只标记 `needs_human`、绝不凭 DINOv3 单独拒绝**；VLM 拒掉/判不可见
   了该实体**全部** crop（无可锚定身份的可见 crop 存活）时退回人工，防止误清空实体；
   VLM 拒掉 DINOv3-medoid 仅记录在审计里、不阻断实际拒绝。

**确定性暗/低信息门禁（`crop_qa.py`，VLM 之前）：** 采集时对每张 crop 在**实体像素
区域**（蒙版 `alpha>0`，而非白底合成图）算亮度均值/方差；近黑且近平（
`mean<26 且 std<16`，`location` 不判）→ `dark_low_information` → `accepted=false`。
阈值刻意保守：暗但有结构（人脸/边缘对比 → std 高）的 crop 不会被误杀。这把黑影 blob
在**进 VLM/人工之前**就拦掉，省 VLM 调用也省人工。

`acquire` 阶段的 `exclusive_assign_candidates` 亦加了 runner-up `assign_margin`：
候选对某实体的相似度必须显著高于次高实体才自动认领，否则留给 VLM，避免相近身份
在低阈值附近互相串号。`identity_audit.json` 记录每实体判定与人工/拒绝计数，量化人工节省。

| 产物 | 作用 |
|------|------|
| `crop_tasks.json` | GPU acquire 任务 |
| `coverage_plan.json` | 计划统计 / caps / 槽位计数 |
| `identity_audit.json` | 每实体身份一致性判定（cohesive/vlm/needs_human 统计） |
| `crop_library.json` | 属性去重后的接受库 |
| `crop_proposals.json` | **槽位齐全**（`acquire` + `slot_bind`） |
| `route.json` | 路由 + `task_mode` + caps |

**槽位绑定规则：**

1. 精确 `(chunk, entity)` acquire 优先  
2. 否则 ≤t **同状态**最近库 crop（再退化为任意 ≤t）  
3. 再没有 → `accepted=false` / `no_library_crop_for_slot`  

绑定复用库的 `crop_path` / 属性，设 `task_kind=slot_bind`、`bind_source_chunk_id`。

### 2.6 属性包接线（Bench）

- 模块：`vmem_bench.common.crop_attributes`、`attribute_dedup`
- 挂载：`s5_*/attach_attributes.py`（默认 classifier=`null`）
- 启用 VLM：`MEMSTRATA_BENCH_CROP_ATTR_CLASSIFIER=vlm` + base URL / model
- Gold：`build_gold` 写入 `state`（来自 `state_angle`）、`crop_attributes`、bind 元数据

### 2.7 Bench 代码图

| 关注点 | 路径（均在 `src/vmem_bench/` 下） |
|--------|----------------------------------------|
| 规划 / caps | `.../s5_*/task_planner.py` |
| 去重 + ≤t 展开 | `.../s5_*/coverage_expand.py` |
| Mask 门禁 | `.../s5_*/mask_quality.py` |
| 挂属性 | `.../s5_*/attach_attributes.py` |
| Picker（含 max_tokens） | `.../s5_*/crop_picker.py` |
| 入口 | `.../s5_*/run.py` |
| 属性包 / 去重 | `common/crop_attributes.py`、`common/attribute_dedup.py` |
| Gold | `.../s7_freeze_publish/build_gold.py`、`common/gold_helpers.py` |

### 2.8 运维硬规则（Bench 重跑）

- 重 S5 在有 GPU 的训练节点上跑，不要占用开发机。
- **禁止**大权重进 DT `/dev/shm`；训练节点 shm 可以。
- Picker 用专用小 VLM（如 8B `:8113`），勿抢过载共享 32B。
- skip-S3 标注优先级：S4 人工 &gt; S3 自动 &gt; S2 normalized；目录被清则先从 `vlm_output.json` 重建 S2。

### 2.9 实现状态与 BBB 审计（2026-07-18）

已落地并在 BBB 产物中确认：

- coverage 任务规划与 character/prop/location caps；
- SAM3 mask 门禁与 RGBA crop；
- crop 属性包挂载（当前默认 classifier=`null`）；
- 属性库剪枝与因果 `≤t` slot bind；
- `propose_and_pick` / `vlm_sam_refine` 双路由骨架。

尚未完整落地：

- entity-specific `grounding_phrase` / `static_attributes` 从 S2/S4 接入 S5；
- 生产级 closed-set VLM picker；
- staged S5 的 semantic identity QA 与跨实体冲突检测；
- exemplar/prototype identity audit；
- S6 的 identity 冲突 `must` / 普通质量 `spot_check` 分层。

BBB 当前 S5 是一个明确的失败基线，而不是可冻结质量：

- 177 个槽位 proposal，92 个 acquire 计划，77 个进入 crop library，100 个由 slot bind 复用；
- accepted character/prop proposal 中约 **62.8%** 直接混杂或继承了混杂 acquire；
- SAM3 对 character/prop 只使用 `animal` / `object` 类级 concept；
- 当前默认 `grounder=full-frame` 会选择 `FirstCandidatePicker`，多候选任务实际 **100% 选择 index 0**；
- 12 个 acquire chunk 出现不同 entity_id 使用完全相同的 frame+bbox/crop；
- S5 QA 只检查面积、近全帧和清晰度，无法发现“框很完整但身份错误”；错误库项随后被 slot bind 放大。

代表性证据：BBB `c00010` 中 `char_004`（红松鼠）、`char_005`（灰飞鼠）和
`char_006`（灰老鼠）的 accepted crop 实际都是同一只大兔子。

### 2.10 与 track-first 的复用边界

staged S5/S6 可以参考 `annotation/pipeline_track_first`，但不能直接复制整条
discovery → tracklet → re-ID 管线。Staged S1–S4 已经固定 entity_id，S5 的职责是给这些已知
实体采有限视觉库，而不是重新发现或合并身份。

适合迁移到 staged/common 的局部机制：

1. **同 chunk 跨实体 bbox/crop 冲突门禁**：不同 entity_id 的高 IoU 或同 crop 必须拒绝/换次选；
2. **closed-set picker**：候选只能分配给当前 roster 中的目标实体，不能恒取最高分实例；
3. **短 grounding phrase + static attributes**：避免用泛化 `animal/object` 或长 description 定位；
4. **exemplar/prototype identity audit**：低 margin 或更接近其他实体 prototype 时拒识；
5. **S6 分层队列**：identity 冲突为 `must`，普通属性/质量问题为 `spot_check`。

纯函数应下沉到 `vmem_bench/common/`，禁止 staged pipeline 直接依赖 track-first 的
Registry、tracklet 或运行时对象。

后续修复建议按成本分三档：

- Tier 1：同 chunk 冲突硬门禁、坏库禁止 slot bind、S6 must 卡片、生产禁用 FirstCandidatePicker；
- Tier 2：`grounding_phrase` + GDINO/fusion + 有效 Qwen picker + 轻量 prototype 审计；
- Tier 3：S4 人工 exemplar + 单帧 closed-set 联合多实例分配；只有跨 shot identity 仍失败时再考虑完整 track-first。

本轮只记录审计与设计边界，不修改 S5 选择算法、不运行 GPU、不重跑 BBB。

---

## 3. 共享属性包契约（两边镜像）

| 包 | 属性模块 | 去重模块 |
|----|----------|----------|
| Bench | `vmem_bench.common.crop_attributes` | `...attribute_dedup.select_attribute_diverse` |
| SUT | `memstrata.mllm.crop_attributes` | `memstrata.lib.dedup.select_attribute_diverse` |

| 字段 | 谁填 | 闭集 |
|------|------|------|
| `chunk_id` / `frame_index` / `seconds` | 调用方 | 数值 |
| `spatial_angle` | VLM | front / side / back / top / unknown |
| `state_angle` | VLM | default / changed / damaged / unknown |
| `shot_size` | VLM | wide / medium / close_up / extreme_close_up / insert / unknown |
| `lighting` | VLM | day / night / indoor / outdoor_overcast / artificial / backlight / unknown |
| `occlusion` | VLM | none / partial / heavy / unknown（**不进**去重桶） |

- `diversity_bucket()` = 上表四元组（无 occlusion）。
- `select_angle_diverse(spatial, state)` 是二维包装，内部仍走 `select_attribute_diverse`。
- 任一字段失败降级为 `unknown`（`source=vlm_error|null`），**永不阻塞流水线**。
- 改枚举 / schema / prompt 必须 **两边一起改**。

---

## 4. memstrata（SUT）：生产闭环里的 crop

### 4.1 目标

不负责从长视频里「挖」库；负责对**已有 crop** 打标、写入资产记忆 \(\mathcal{R}_j\)，并在容量下保持**属性多样、非冗余**。

### 4.2 视觉地层

每个 `AssetRepresentation` \(r\in\mathcal{R}_j\)：

| 字段 | 用途 |
|------|------|
| `spatial_angle` | 跨视角 |
| `state_angle` | 跨状态 |
| `temporal_tag` + `origin_chunk_id` | 时间 / 超长回忆 |
| `annotations["crop_attributes"]` | 全量属性包（含 shot/lighting/occlusion） |
| `annotations["embedding"]` | 近重复门控 |

Compose 选择：显式 `representation_id` → 过滤 → **preferred spatial/state** → 最新 chunk。

### 4.3 分类器

```
crop → CropAttributeClassifier（全量包）
     → AngleClassifier 投影出 spatial/state（兼容层）
     → Observation / AssetRepresentation
```

| 开关 | 行为 |
|------|------|
| `MEMSTRATA_CROP_ATTR_CLASSIFIER` / `MEMSTRATA_ANGLE_CLASSIFIER=null`（默认） | 不调模型；unknown |
| `=heuristic` | 文件名 stem（测试） |
| `=vlm` | OpenAI 兼容多模态 API |

- **生产路径**：`RoleAwareDecomposer` / `AssetCurator.ingest_observation` 在角度 unknown 时分类；**显式角度不被覆盖**。
- **Track A**：`ingest_packet` **不调** classifier；packet 字段权威。
- `angle_classifier.py` 是对属性包的**薄投影**，新能力加在 `crop_attributes.py`。

环境变量：`MEMSTRATA_CROP_ATTR_*` / `MEMSTRATA_ANGLE_CLASSIFIER_*`（可回退 context judger URL）。

### 4.4 Curate 门控（与 Bench 去重同构）

`AssetCurator`（`steps/curate.py`）：

1. 近重复 embedding：同桶 / 未知桶可丢；**新的已知属性桶**可保留。
2. 已知属性桶已存在 → 丢弃重复。
3. 超 `max_reps_per_asset`（默认 5）→ `select_attribute_diverse` 按桶保留。

这与 Bench S5 的 `prune_library_by_attributes` **同一套桶逻辑**，只是场景不同：
Bench 裁采集库；SUT 裁在线记忆 \(R_j\)。

### 4.5 SUT 代码图

| 关注点 | 路径（`src/memstrata/`） |
|--------|--------------------------|
| 全量属性包 | `llm/crop_attributes.py` |
| 角度兼容层 | `llm/angle_classifier.py` |
| 属性多样去重 | `lib/dedup.py`（`select_attribute_diverse` / `select_angle_diverse`） |
| 入库 / 门控 | `steps/curate.py` |
| 分解时分类 | `steps/decompose.py` |
| RGBA 喂模型 | `lib/media.py`（`load_crop_rgb_for_model`） |
| 设计总览 | `docs/method/design.md` |

---

## 5. 两端如何对齐（别踩坑）

| 话题 | Bench | SUT | 对齐点 |
|------|-------|-----|--------|
| 谁采 crop | S5 GPU 采 | 不采（吃 observation） | 契约是「有图 + 属性」，不是同一段代码 |
| 库大小 | coverage caps（8/5/12）再属性剪 | `max_reps_per_asset≈5` | 都按属性桶留多样性 |
| 槽位 / 时刻外观 | ≤t `slot_bind` → gold | compose 用 preferred angle + 最新 | 都遵循 `current_appearance` 精神 |
| 属性 schema | `common/crop_attributes` | `llm/crop_attributes` | **镜像，禁交叉 import** |
| 默认不打 VLM | classifier=`null` | classifier=`null` | CI / 离线安全 |
| 蒙版 | S5 RGBA + mask 门禁 | 读图时白底合成 | 存储蒙版、推理再合成 |

---

## 6. 变更清单（改 crop 时先看）

改属性枚举 / JSON schema / 去重桶 → **同时改** Bench + SUT 镜像 + 本文件 §3。  
改 S5 调度 / 绑定 → 更新本文件 §2。  
改 SUT 入库门控 → 更新 §4 与 `docs/method/design.md`。  
改评分外观定义 → `docs/benchmark/scoring.md`，并核对 Bench ≤t bind 是否仍一致。
