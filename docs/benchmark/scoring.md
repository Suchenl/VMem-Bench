# MemStrata-Bench 评分规范（视觉为主 · 已批准）

> ⚠️ **已被取代（DEPRECATED）**：本文件描述的是 v1 评分（ID 交集 headline + embedder 最近邻 VisualFidelity，
> 依赖 gold crop 与状态标注）。当前权威规范是 [`scoring_v2.md`](scoring_v2.md)（Visual-Coverage，VLM 判定、
> 无 crop、无状态）。本文件仅保留用于复现旧数据，**新的 benchmark claim 请勿再基于本文件**。

> 当前公开评分权威源是 [`scoring_v2.md`](scoring_v2.md) 与
> [`running_eval.md`](running_eval.md)。本文件只保留旧协议的历史说明，
> 不应作为新实验、论文表格或用户方法评测的入口。

> ## 落地状态与实测记录（截至 2026-07-21）
>
> **VisualFidelity 已实现并接入评分**（`scoring/visual.py`、`scoring/embedder.py`）：
> - 钉死打分 embedder（通用轴）= **DINOv3 ViT-B/16**（`facebook/dinov3-vitb16-pretrain-lvd1689m`），
>   由 `vmem_bench` 自持（不 import `memstrata`，AGENTS.md 规则 2），权重从
>   `$PUBLIC_MODELS_ROOT` 解析，全系统共用同一 embedder（公平不变量）。免阈值最近邻按 §3.1。
>   **D2 定稿的按 kind 路由（§3.1/§4.1）**：在 DINOv3 之外，通用轴新增 **SigLIP2**（视觉-语言语义，另报一列 VF）；
>   LSMDC 真人角色 slot 加 **ArcFace/InsightFace** 人脸轴；location slot 用 **MegaLoc(VPR)** 计分。
>   新增 embedder 复用同一「自持 + 按路径加载 + 全系统 byte-for-byte 一致」的钉死模式，DINOv3 通用轴已跑通。
> - **对不产出 instruction 的纯检索 baseline 公平**：VisualFidelity 读「SUT 选中的 crop」而非
>   instruction，故 helios/memflow/longlive_rag 得到真实非零分（实测 helios VF=0.96），
>   而被降级为诊断的 ID-Fidelity 对它们恒 0。
> - **headline 不是 VisualFidelity 单指标**（重要修正）：实测 VisualFidelity 单独作 headline 在
>   BBB 上退化——数值近饱和（0.956–0.981）且排名反转（longlive_rag 0.981 > selection_oracle
>   0.956），因为它只考「选中槽的外观」、不含召回，而召回本质是「选了哪个实体」的 ID 问题、
>   无法由外观得到。故 **headline = 视觉复合分 `MemStrataVisualScore`**：保留
>   Sufficiency/Parsimony/MemRecall/Compactness/Avoidance，仅把外观维 **ID-Fidelity → VisualFidelity**
>   替换（直接落地「Fidelity 走视觉」的意图，同时保持排名有效）。复合后排名恢复正常：
>   oracle 0.975 > memstrata 0.905 > memflow 0.846 > helios 0.771 > sliding_window 0.690 >
>   longlive_rag 0.568（`data/_runs/bbb_track_a_visual_20260721/`）。
> - 诊断随报：`visual_fidelity`（单指标）、`id_diagnostics`（含 ID-Fidelity 的纯 ID 复合分）。
> - **BBB 对视觉轴偏弱**：实体视觉可分，VisualFidelity 近饱和。要让视觉轴真正区分系统，
>   仍需含 appearance-changing 状态事件 + 相似实体的样本。
> - 入口：`scripts/memstrata/score_memstrata.py` 与 `run_gold_replay` 均新增
>   `--scoring-embedder/--scoring-embedder-weights/--no-visual`，默认 dinov3、GPU/权重缺失时
>   自动降级为 ID headline（不崩）。测试：`tests/test_visual_fidelity.py`（确定性、无 GPU）。
> - **Avoidance 已激活（确定性物化，无模型、无重跑）**：S7 冻结层 backfill
>   （`annotation/pipeline/stages/s7_freeze_publish/backfill.py`，`--only state-events`）把 VLM
>   `vlm_output.json` 里每实体的 `state_changes`（结构化 `state_change_kind` + 秒）确定性物化为
>   `state_events`：秒→chunk 用 `chunk_index.seconds_span`；有限本体/可逆闸门复用
>   `drafting.filter_state_events`（VLM 显式 kind 为权威，英文正则只用于「否决可逆」或「佐证」，
>   非英文 prose 无法分类时信任显式 kind——见该函数的放宽修正）；`deprecates` 落到**真实冻结 crop id**
>   （`chunk_id ≤ event_chunk` 的全部 rep）。写入 `entity_registry.json`（评分权威源）并同步
>   `chunk_annotations.forbidden` 与 `observations.jsonl`，**不动 `layout_hash`/crops/实体 id/seconds_span**。
>   BBB 实测：4 事件（苹果 consumed@c9+destroyed@c39、蝴蝶 destroyed@c19、兔子 appearance_changed@c26），
>   Avoidance 从 N/A→激活且可区分（memstrata 1.00 / oracle 0.97 / helios 0.82 / memflow 0.80，n=42）。
>   测试：`tests/test_state_events_backfill.py`（确定性、无 GPU）。
> - **仍 OPEN（见文末「遗留 OPEN 项」）**：§3.3 VLM 语义裁判 + §4.2 裁判 prompt 尚未落地；
>   §4.1 新增 embedder（SigLIP2 / ArcFace / MegaLoc）的 revision/sha 与接线待按 DINOv3 同款钉死模式补齐。
>   current_appearance 的 state 感知目前用「≤t 最近一张 crop」近似（自动反映最新状态），未接 §1.3
>   独立状态事件表（现走 registry 内嵌 `state_events`，等价语义）。

## 0. 一句话

**基准最终判的是视觉记忆一致性**（选出的参考图长得像不像该实体此刻该有的样子），语言/ID 只是解释性诊断。
**发布的 gold 只有两样：逐 chunk 的 prompt + 带标签的 crop 图**；其余评分量全部在装载/评分时确定性派生。
**评分要用的模型、提示词、脚本全部钉死写进本规范**，靠"钉死"而非"不用模型"保证可复现（原则 7）。

---

## 1. 发布的 gold = prompts + crops（+ 极小状态事件表）

发布包只含以下**源真相**（source of truth），其余皆为派生产物、不发布、可重建：

### 1.1 `gold/prompts.json` — 逐 chunk 的生成提示词

```jsonc
{
  "schema_version": "3.0.0",
  "movie_id": "big_buck_bunny",
  "human_reviewed": true,                 // 冻结标志（原则 6）
  "layout_hash": "…",                     // 与 crop_index 共同锁定 chunk 布局（原则 6）
  "chunks": [
    {
      "chunk_id": 7,
      "seconds_span": [125.88, 130.83],   // chunk 时间窗（人审/派生用）
      "shot_span": [12, 14],
      "prompt": "大兔子在老橡树下捡起一个红苹果。"
      // 自然剧本式散文；自然地提到本 chunk 每一个在场实体（完整性契约，见 §2.1）；
      // 严禁结构化花名册转储 / 来源提示 / 答案 ID（原则 3）。
    }
  ]
}
```

### 1.2 `assets/**` + `gold/crop_index.json` — 带标签的 crop 图

crop 图落在 `assets/{characters|props|locations}/<entity_id>/`（本地派生、可从源视频+bbox 重建）。
`crop_index.json` 是 crop 的**标签清单**（不含向量，原则 9）：

```jsonc
{
  "schema_version": "3.0.0",
  "movie_id": "big_buck_bunny",
  "human_reviewed": true,
  "layout_hash": "…",
  "entities": [
    { "entity_id": "char_big_buck_bunny", "kind": "character", "name": "大兔子",
      "identity_scope": "individual",
      "crops": [
        { "crop_path": "assets/characters/char_big_buck_bunny/c000.jpg",
          "chunk_id": 0, "state": "default",
          "bbox": [ymin,xmin,ymax,xmax], "frame_index": 3042, "quality": 0.9 },
        { "crop_path": "assets/characters/char_big_buck_bunny/c005.jpg",
          "chunk_id": 5, "state": "muddy",          // 状态变更后的新外观
          "bbox": [...], "frame_index": 3110, "quality": 0.8 }
      ]
    }
  ]
}
```

- **人工唯一要做的事 = 保证 `crop → entity_id` 归对**（crop 融合/re-ID 合并）。其余全派生。
- crop 的 `state` 标签就是状态语义：同一实体状态从 `default` 变 `muddy`，即一次外观状态变更。

### 1.3 `gold/state_events.json` — 极小状态事件表（不可逆生命周期）

state change 无法只从 crop 推出"发生在哪一 chunk、是否不可逆"，所以保留一张最小表（VLM 提议 + 人审确认）：

```jsonc
{ "schema_version": "3.0.0", "events": [
  { "event_id": "evt_apple_eaten", "entity_id": "prop_apple_01",
    "chunk_id": 6, "kind": "consumed", "description": "苹果被吃掉",
    "prior_state": "default" }        // 该 chunk 之后，prior_state 的所有 crop 变为过期引用
]}
```

`kind ∈ {destroyed, consumed, broken, acquired, attached, detached, appearance_changed}`。

> 发布包 = `manifest.json`（含源视频下载方式+sha256、切分逻辑）+ 上述三个 gold 文件 + `assets/`。
> **不发布**：embedding、`entity_registry.json`/`chunk_annotations.json`（改为派生）、`tmp/`。

---

## 2. 一切评分量从 gold 确定性派生（装载时，不写进发布包）

harness 装载时，从 §1 三文件确定性派生出评分所需的全部结构（原则 5、7）：

| 派生量 | 规则 |
|---|---|
| `present(t)` | crop_index 中标了 `chunk_id==t` 的全部 entity |
| `first_appearance(entity)` | 该 entity 出现的最小 chunk_id |
| `representations(entity)` | 该 entity 的全部 crop（按 chunk 排序） |
| `gold_instruction(t, e)` | `introduce` 若 `t==first_appearance(e)`，否则 `continuity` |
| `forbidden(t)` | 对每个 state_event（`chunk_id<t`），其 entity 在 `prior_state` 下、且 `chunk_id≤event.chunk_id` 的全部 crop |
| `current_appearance(t, e)` | e 在 chunk t 的 crop（若无则取 ≤t 内最近一张同状态 crop）；作为"此刻该有的样子"的目标图 |
| `scenario_tags(t)` | 由在场历史 + 状态事件确定性推（re-appearance / multi-instance / scene-return / state-change / none） |

### 2.1 prompt 完整性（保证召回公平，防作弊，原则 3）

- 契约：prompt 必须用自然语言提到 chunk t 的每个 present 实体（用其 name/alias，不用 canonical ID/花名册）。
- 后处理**确定性检查**：每个 present 实体的 name 或 alias 是否出现在 prompt 文本里；缺失 → flag。
- 本地模型（Qwen3-VL-8B）在自动审核时把缺失的实体**自然地补写进 prose**；确定性匹配脚本终审通过才可冻结。
- 因此 `present(t)` = "prompt 所召唤的实体"，召回可公平计分，且仍非平凡（SUT 要把自然指代映射到记忆资产）。
- **首现外观规则（introduce vs continuity）**：外观在 prompt 里只出现一次，就在该实体的 `first_appearance` chunk（`gold_instruction=introduce`）——此时把实体首现 `description` 里可辨识的外观自然写进 prompt（记忆里还没有它，外观必须在世界里落地一次，这不是泄漏）；此后所有 `continuity` chunk 只点名、不再复述外观/服装/颜色等身份属性（那是记忆测试所在，SUT 必须自己召回）。标注侧由 S3 `first_presence` 阶段在首现段把 `description` 外观注入 `action` 实现。

---

## 3. 评分：一个视觉 headline + 两组诊断

对 chunk t：**character / prop / location 全部计分**（location 不再从视觉评分剔除——D2 改为用
MegaLoc 单列「地点一致性」轴）。

### 3.1 视觉一致性（HEADLINE，钉死 embedding + 免阈值最近邻，按 kind 路由）

被测系统对每个选中实体 e 给出参考图（Track A 里必是 gold crop，其向量在评分时现算）。设：
- `q` = SUT 为 e 选用的 crop 的 embedding（钉死模型现算）；
- 候选集 = 本 chunk 全部实体的 `current_appearance(t, ·)` 的 embedding。

**免阈值判定**：`q` 的最近邻是否 = e 本人的 current_appearance，且该 crop 未过期（不在 `forbidden(t)`）。命中记 1，否则 0。逐 slot 取均值 = **VisualFidelity(t)**。

**按 entity kind 路由 embedder（D2，§4.1 钉死表）**：VisualFidelity 是**同一套免阈值最近邻判定**、
按 kind 选 embedder 的复合视觉指标——

| entity kind | 用于该 slot 的 embedder | 说明 |
|---|---|---|
| 全类通用 | **DINOv3**（主）+ **SigLIP2** | 通用视觉 + 视觉-语言语义两条正交轴，各报一列 VF |
| character（**仅 LSMDC 真人、且 target 与 selected 均检到合格人脸**）| **ArcFace / InsightFace** | 人脸细粒度身份轴；检不到脸的真人角色 slot 退回通用轴（DINOv3/SigLIP2），不进人脸轴 |
| location | **MegaLoc**（VPR）| 地点一致性轴；location 由此**被计分**，非剔除 |

- **报列**：DINOv3-VF、SigLIP2-VF 逐 kind 通用两列；LSMDC 角色另报 ArcFace 人脸轴、location 另报 MegaLoc 地点轴。
  各路由 embedder 用**同一** causal slot、免阈值最近邻与图片数量效率规则，彼此不混合 raw similarity。
  headline 复合分（§3.4）取 VisualFidelity（默认以 DINOv3 通用轴为主轴，kind 路由轴对其对应 slot 生效）。
- 为什么免阈值：不设"cosine>0.x 算对"这种拍脑袋阈值；用排名判定，抗 encoder 校准漂移。
- 状态变更的考点自然落进来：兔子第 5 chunk 沾泥，第 8 chunk 的 current_appearance 是沾泥版；SUT 若选干净版 → 最近邻可能落到"干净兔子=另一时刻"或直接因过期被判 0 → 扣分。
- `current_appearance` 缺失（该实体在 t 无 crop 且此前无同状态 crop）→ 该 slot N/A（原则 8）。

### 3.2 ID 级诊断（确定性、无模型，用来解释视觉分为什么低）

沿用 v2 §4 的集合运算，但作为**诊断**而非 headline：
- **Sufficiency / Parsimony**：选没选对实体（`|S∩P_ret|/|P_ret|`、`/|S|`）。
- **Avoidance**：`1 − |R∩F_active|/|F_active|`，有没有用过期 crop（与 3.1 的过期判定共用 forbidden）。
- **Compactness**：`mean exp(-λ(|R_s|-1))`，用了几张。
- **MemRecall**：记忆距离加权召回（长程再现是否记得）。
- **Fidelity(introduce/continuity)**：新旧分类是否正确（考记忆索引完好性）。
- 边界/反躺平规则（N/A、空集记 0）全部沿用 v2 §4 表。

### 3.3 VLM 语义裁判（二级 / 非 headline，含 Track B）· **OPEN（未落地）**

> 本节与 §4.2 裁判 prompt 是本规范当前唯一的 OPEN 项（见文末「遗留 OPEN 项」）：设计已定，尚未落地实现。

- 用途：hard bucket（state-change / multi-instance）里 embedding 最近邻含糊时，补一刀语义判断"是不是同一实体、状态对不对"；以及 Track B 生成帧的一致性（gold 里没有该图，无法用冻结 crop）。
- **绝不做 headline**（原则 7、评审启发式）。作为鲁棒性/补充指标单列，并**披露可复现级别**。
- 护栏：模型/权重/解码参数钉死、temperature=0、裁判 prompt 公开写进本规范（§4.2）、报**与人工判的一致率**。

### 3.4 汇总：唯一的 headline 复合分与权重

**headline = `MemStrataVisualScore`（视觉记忆复合分）**——即 [`schemas_and_contracts.md`](schemas_and_contracts.md)
§4 的复合分里，把外观维 **ID-Fidelity 替换为 VisualFidelity**（§3.1，按 kind 路由的多 embedder），
其余轴与权重不变。各指标先做逐 chunk 均值（跳过 N/A），再按权重归一加权：

| 指标 | 权重 |
|---|---|
| Sufficiency | 0.25 |
| Parsimony | 0.15 |
| **VisualFidelity**（替换 ID-Fidelity） | 0.20 |
| Avoidance | 0.20 |
| Compactness | 0.10 |
| MemRecall | 0.10 |

- 某指标全片 N/A 时其权重按比例重分配（同 v2 规则）。权重为约定初值，发布前附敏感性分析，不属 schema 冻结范围。
- **为什么不是 VisualFidelity 单指标**：实测单独作 headline 会退化（近饱和 + 排名反转，见「落地状态」），
  因为它只考「选中槽的外观」、不含召回；召回本质是 ID 问题，须由 Sufficiency/MemRecall 等轴承担。
- **诊断随报**（不进 headline）：ID 诊断组（含保留为诊断的 **ID-Fidelity**，§3.2）与 VLM 二级组（§3.3），
  以及 §3.1 的各路由 embedder 分列（DINOv3-VF / SigLIP2-VF / ArcFace 人脸轴 / MegaLoc 地点轴）；
  各自明确标注可复现级别与 provenance（原则 10）。

---

## 4. 钉死项（评分可复现的锚，全部写进规范）

### 4.1 评分用模型（pinned，按 kind 路由 · D2）

pinned scoring embedder 由 `vmem_bench` 自持、对所有系统 byte-for-byte 一致，权重统一按仓库
`models/model_weights/` 路径加载（不跨包 import，AGENTS.md 规则 2）：

| 用途（VisualFidelity 路由轴） | 模型 | 版本/权重 | 预处理 | 状态 |
|---|---|---|---|---|
| 通用视觉 embedding（全类，主轴）| **DINOv3 ViT-B/16** | `facebook/dinov3-vitb16-pretrain-lvd1689m`（revision 待钉） | `<size/mean/std>` | **已落地** |
| 通用视觉-语言语义（全类，第二列）| **SigLIP2** | `<hf id + revision>` | `<size/mean/std>` | 接线中（同款钉死模式）|
| 人脸 embedding（**仅 LSMDC 真人 character**，检到脸才用）| InsightFace / **ArcFace** buffalo_l | `<pack sha>` | 人脸对齐参数 | 接线中 |
| 地点 embedding（**location，计分轴**）| **MegaLoc**（VPR）| `models/model_weights/.../MegaLoc/model.safetensors`（`<weights sha>`） | — | 接线中（参考 `src/memstrata/encoders/place/vpr.py`，在 `vmem_bench` 侧独立镜像）|
| VLM 裁判 / 指令解读 | Qwen3-VL-8B（32B 备用） | `<endpoint/model id>` | temp=0 | **OPEN**（二级 & SUT planner，见 §3.3）|

> **不新增** DINOv2 / CLIP（与 DINOv3 / SigLIP2 冗余，D2）。`<…>` 为待钉的确切 revision/sha；
> encoder 稳健性检查：同一批 crop 用不同通用 embedding 各跑一遍，证明排名稳定（原则 10）。

### 4.2 VLM 裁判提示词（pinned，逐字公开）

> 占位：`prompts/judge_identity_state.md`（同一实体？状态一致？只输出离散标签 same/different + state_ok/state_changed），批准后定稿并冻结进本规范。

### 4.3 评分脚本入口（pinned）

- `scripts/memstrata/score_memstrata.py`（SUT 驱动）
- `python3 -m vmem_bench.scoring`（harness）
- 派生器 / crop embed / 免阈值最近邻 / VLM 裁判：VLM 裁判为 OPEN（§3.3）；其余落地后在此登记确切入口与默认参数。

### 4.4 LSMDC 专项视觉分与主分配置（pinned）

主分默认对所有数据集、所有 kind 用**同一通用轴 embedder**（§4.1 的 DINOv3，必要时并列 SigLIP2），
保证 BlenderOpenMovies 动画与 LSMDC 真人影片结果可直接比较。**LSMDC 真人**在通用轴之外，按 D2
路由额外接入两条**专项视觉轴**（同 causal slot、同免阈值最近邻、同图片数量效率规则，彼此不混合
raw similarity）：

- `FaceConsistency`：仅在 current target 与 selected history crop **都有合格人脸检测**的 character slot 上，用 **ArcFace** 计算；无法做人脸检测的真人角色 slot 保留在通用轴主分，不进入 ArcFace 专项分。
- `PlaceConsistency`：仅在 **location slot** 上，用 **MegaLoc/VPR** 计算（location 因此被计分，非剔除）。

> 与旧稿（源 `benchmark_run/workflow.md`）的差异：旧稿把 Face/Place 专项分定为「额外报告、**不改变 headline**」；
> 按 **D2（2026-07-22）**，这两条轴是 VisualFidelity 的 **kind 路由分支**，对其对应 slot **参与 headline**——
> 见「遗留 OPEN 项」的 superseded 记录。

配置（pinned）：

```yaml
main_score:
  # 通用轴 embedder 以 §4.1 为准（DINOv3 ViT-B/16；SigLIP2 第二列）
  model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
specialized_metrics:
  lsmdc:
    face:
      enabled: true
      encoder: arcface:buffalo_l
      eligibility: target_and_selected_face_detected
    location:
      enabled: true
      encoder: vpr:megaloc
```

---

## 5. Track A / Track B 边界（原则 1）

- **Track A（主结果、定量）**：oracle 回放；SUT 从 gold crop 里选参考。生成被抽象掉。VisualFidelity 用现算 embedding + 免阈值最近邻，**可复现**。
- **Track B（生成器在环、次要）**：SUT 驱动真实生成器产出新帧；用钉死 embedding + VLM 裁判评生成帧的身份/状态一致性——**这是把 Track B 从纯定性升级为半定量的机会**，但明确标注可复现级别，不与 Track A 主表混排。
- 禁止让下游生成器的"美学质量"进入指标（评审启发式：那样测的是生成器不是记忆）。

---

## 6. 冻结 / 发布 / provenance

- `human_reviewed:false` 的 gold 禁止正式评测；harness 启动即拒（原则 6）。
- chunk 布局属冻结范围；`layout_hash` 锁定 prompts.json + crop_index.json 一致（原则 6、2：防未来泄漏——SUT 组合 chunk t 时拿不到 ≥t 的 observation）。
- 每份结果必须携带：gold 版本 + 指标版本 + 数据版本 + **评分模型身份与运行配置** + 成本/时延（原则 10）。缺一视为不可引用。

---

## 7. 评测回放流程（确定性回放，零 VLM）

> 标注阶段（如何从原始长视频产出冻结 gold）见 [`annotation_pipeline.md`](annotation_pipeline.md)。
> 本节只管**回放/评分**：给定冻结 gold，逐 SUT 跑一遍。

对每个 chunk t（t = 0, 1, …，**严格因果**，原则 1）按 `prompt → record → observation → 指标` 回放：

| 步骤 | 信息 | 流向 | 时机 |
|---|---|---|---|
| A | `PromptPacket`（prompt 文本，无来源提示） | bench → SUT | 评分前 |
| B | `ComposedContextRecord`（选中实体 / representation + 指令 + 排除项） | SUT → bench | 评分前 |
| C | 指标计算（§3；VisualFidelity 现算 embedding + 免阈值最近邻，ID 诊断组为纯集合运算；**不调 VLM**） | bench 内部 | — |
| D | `ObservationPacket`（GT chunk 视频 + per-entity crop + 权威命名/ID + 首现描述 + 状态事件） | bench → SUT ingester | 评分**后**（generator-oracle 反馈） |

- gold 的出现清单、forbidden 清单、scenario 标签、embeddings **永不可见于 SUT**（原则 5）。
- 步骤 D 的权威 ID 传播（原则 4）使 SUT 记忆键与 gold 天然对齐，ID 诊断组退化为确定性集合匹配——这是回放阶段能做到零 VLM 的前提；VisualFidelity 用现算 embedding，也无 VLM。
- SUT 被测的能力：给定剧本式 prompt，从自己的记忆库里**选出正确的资产子集与 representation**（含拒绝干扰项、拒绝过期状态、控制上下文大小）。
- 每 chunk 评分明细与最终报告落盘；报告 `versions` 必须记录 gold 版本、指标版本、标注模型指纹（§6）。
- 契约字段结构（`PromptPacket` / `ComposedContextRecord` / `ObservationPacket`）以
  [`schemas_and_contracts.md`](schemas_and_contracts.md) §3 为权威源。

---

## 8. 数据目录布局（发布形态 + 回放产物）

发布只版本化「切分逻辑 + 标注结果 + 下载方式」；视频、切片、抽帧、crop 一律本地按需从源视频重建（gitignore）。

```
data/<dataset>/<movie>/
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

- 回放产物（每 SUT 一份）落在 run 目录（如 `data/_runs/<run_id>/`），携带逐 chunk 明细 + `versions`；
  发布包与 run 产物物理分离，run 产物不进发布包。
- gold 发布包 = `manifest.json` + `gold/`（源视频下载方式与 sha256、切分逻辑、标注结果、embedding sidecar）+ `assets/`。**不发布**：`entity_registry`/`chunk_annotations` 之外的派生结构、`tmp/`。

---

## 9. 遗留 OPEN 项与已决记录

**仍 OPEN（未落地 / 待定）**：

1. **§3.3 / §4.2 VLM 语义裁判**：设计已定（二级、非 headline、Track B 半定量），提示词
   `prompts/judge_identity_state.md` 与实现尚未落地——本规范当前唯一的实质 OPEN 项。
2. **§4.1 新增 embedder 的钉死细节**：SigLIP2 / ArcFace / MegaLoc 的确切 revision/sha 与
   `vmem_bench` 侧接线待按 DINOv3 同款「自持 + 按路径加载」模式补齐（DINOv3 通用轴已落地）。
3. **发布包是否公开答案键 / held-out**：`present/forbidden` 等由 crop_index/state_events 派生，
   等于答案键随 gold 公开。当前假设是**开放 gold**（可复现优先）；若要防刷榜需另设隐藏测试集，属未来工作。

**已决（不再讨论，历史）**：

- headline 用 **embedding 公式而非 VLM 打分**（原则 7）；headline = §3.4 的 `MemStrataVisualScore`
  复合分，外观维用按 kind 路由的 **VisualFidelity** 替换 ID-Fidelity（fairness_decisions.md D2）。
- location **计分**（MegaLoc 地点轴），不再一律从视觉评分剔除（D2）。
- **状态事件最小表**（§1.3）保留：`present/forbidden` 无法只从 crop 推出「何时不可逆变更」，故留一张最小事件表（VLM 提议 + 人审确认）。
- **current_appearance 定义**（§2）：t 无 crop 时回退到 **≤t 最近同状态 crop**（无同状态则任意状态 ≤t；仍无则 N/A）；S5 `coverage` 模式用同一规则做 slot bind（`coverage_expand.resolve_current_appearance` / `gold_helpers.resolve_current_appearance_rep`）。
- **slim gold**：发布 gold 只含 prompts + 带标签 crop（+ 极小状态事件表），其余评分量装载时确定性派生（§1、§2）。
