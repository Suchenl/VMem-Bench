# MemStrata-Bench 数据契约与指标定义（v2）

> 状态：**gold 已冻结**（`human_reviewed: true` 的实例不可原地修改，改动=bump 版本重发布，见 §5）。
> 本文件是 **schema 的唯一权威源**（schema-first）：实现代码（dataclass / JSON Schema）必须逐字段对齐本文件，不得私自增删。
> 评分与回放流程的权威源是 [`scoring.md`](scoring.md)；标注阶段见 [`annotation_pipeline.md`](annotation_pipeline.md)。

通用约定：

- 所有 JSON 顶层带 `schema_version`（semver）。gold 文件冻结后只允许 bump 版本重新发布，不允许原地修改。
- ID 命名：`{kind_prefix}_{slug}`，如 `char_big_buck_bunny`、`prop_apple_01`、`loc_meadow`；
  representation ID：`{entity_id}@c{chunk_id:03d}[.{n}]`。多实例（multi-instance）必须编号区分。
- 路径一律为相对数据实例根目录的相对路径；新运行的 crop 落在
  `assets/{characters|props|locations}/<entity_id>/`（本地派生、gitignore、可从源视频 +
  bbox 重建）。旧的平铺 `assets/<entity_id>/` 与 legacy `derived/assets/<entity_id>/`
  保持可读，gold JSON 只引用相对路径。
- **embedding 绝不内联进 JSON**（设计原则 #10）：一律存 `gold/embeddings.safetensors`，
  JSON 只存字符串 `embedding_key`。任何契约中出现 float 数组 embedding 字段均视为违规。

---

## 0.5 production roster_seed.json（人工确认的标注输入）

生产 gold **必须**由人工确认的 canonical roster 约束；自动 discovery 只允许产出 proposal/debug
草稿，不能冻结。seed 与 gold 分离，它是标注输入和 provenance，不进入 SUT 契约。

```jsonc
{
  "version": 1,
  "movie_id": "big_buck_bunny",
  "human_confirmed": true,
  "entities": [{
    "entity_id": "char_big_buck_bunny",      // 稳定 canonical id；不得由 VLM 重命名/reslug
    "name": "Big Buck Bunny",                // kind 内唯一 canonical name
    "kind": "character",                     // character | prop | location
    "identity_scope": "individual",          // individual | category | scene
    "description": "Large white rabbit …",
    "grounding_phrases": ["large white rabbit"], // 第一项是 detector 主 phrase
    "aliases": ["bunny", "white rabbit"],   // 语义/审核辅助，不自动创建实体
    "exemplar_crops": ["seed/exemplars/bunny_front.jpg",
                       "seed/exemplars/bunny_side.jpg"], // 相对 seed 文件解析
    "static_attributes": {"species": "rabbit", "primary_color": "white"},
    "allowed_state_events": ["appearance_changed"]
  }]
}
```

- `individual`：角色或有连续生命周期的关键 prop；必须至少一个 exemplar，可声明有限状态事件。
- `category`：benchmark 需要但不追踪单个实例生命周期的同类 prop；禁止状态事件。
- `scene`：仅允许 `location`，供 prompt 场景上下文；headline metrics 已剔除 location，不进入身份
  alias 高优先级审核。
- `allowed_state_events` 只允许
  `destroyed | consumed | broken | acquired | attached | detached | appearance_changed`。
- 显著 track 若不能映射到 seed，必须进入 `unknown/reject` finding；不得强分类或凭空创建 gold entity。

---

## 1. gold/entity_registry.json（冻结实体注册表）

```jsonc
{
  "schema_version": "2.0.0",
  "movie_id": "big_buck_bunny",
  "human_reviewed": true,                  // 冻结标志；false 时禁止用于正式评测
  "annotation_provenance": {               // 可复现性指纹
    "vlm": "qwen3-vl-32b", "embedder": "dinov3-vit-s16",
    "perception_backend": "gdino_track",   // track-first：gdino_track（路线A）| sam3_track（路线B，消融）
    "tracker": "bytetrack_local",          // iou（旧贪心）| bytetrack_local（默认：两段关联+运动预测+DINO 外观融合，确定性）| boxmot_botsort（消融：工业 BoT-SORT，非确定性，绝不进主 gold）
    "reid": "dinov3-vits16",
    "text_embedder": "qwen3-embedding-4b", // 文本↔文本 embedding（roster 语义去重 / prompt 完整性检查）；未启用则为 null
    "crop_classifier": "siglip2-base-patch16-512", "face_encoder": "insightface-buffalo_l",
    "identity_resolution_mode": "seeded", // 生产默认：canonical seed 直接拥有身份；cluster_vlm/greedy 仅 proposal/消融
    "roster_mode": "seeded",              // seeded（可冻结）| proposal（strict freeze 拒绝）
    "production_mode": true,
    "precluster_linkage": "complete",      // cluster_vlm 模式的预聚类 linkage：complete（默认，抗链式误合并）| average
    "service_placement": "fastest",        // 常驻服务放置模式：fastest（一服务一卡）| packed（按峰值显存打包共卡）| none（进程内单例回退）
    "pipeline_version": "…", "review_patch_sha256": "…"
  },
  "entities": [
    {
      "entity_id": "char_big_buck_bunny",
      "kind": "character",                 // character | location | prop
      "name": "Big Buck Bunny",            // 权威命名（命名神谕）
      "description": "A giant, chubby grey rabbit …",   // 首现时进 prompt 的外观描述
      "first_chunk": 0,
      // --- 时间元数据（§services_and_time.md Q3；由 tracklet∩chunk 的 presence 帧区间确定性算出；
      //     人审+分析+时间指标用，SUT 不消费，不改既有 chunk 粒度打分）。秒 = 帧号 / layout.fps。---
      "presence_spans": [[72, 410], [1180, 1355]],   // 该实体全部在场帧区间（闭区间，按帧号升序，不重叠）
      "first_frame": 72,  "first_seconds": 3.0,       // 精确首现（= 最早 span 起点）；first_chunk 保留
      "last_frame": 1355, "last_seconds": 56.46,      // 末现（= 最晚 span 终点）
      "screen_time_seconds": 21.4,                    // 在场总时长（spans 并集帧数 / fps）= 重要性/显著度信号
      "max_absence_frames": 770, "max_absence_seconds": 32.08,  // 相邻 span 间最长缺席 = 被测试的记忆跨度（再现距离）；单段实体为 0
      "static_attributes": {"species": "rabbit", "primary_color": "grey", "size_class": "large"},  // 稳定身份键（species/subcategory、primary_color、size_class、object_type…），consolidation identity funnel 用：同名同 kind 但 static_attributes 冲突（如 species=rabbit vs bird）判为不同实体。自由字符串字典，绝不承载动作/姿态/情绪。人审可见；SUT 不消费（不进 PromptPacket/ComposedContextRecord）。向后兼容：旧 gold 无此字段反序列化为 {}
      "representations": [
        {
          "representation_id": "char_big_buck_bunny@c000",
          "chunk_id": 0,
          "crop_path": "assets/characters/char_big_buck_bunny/c000.jpg",
                                                        // 本地派生、gitignore、可重建（旧平铺/derived 均兼容）
          "bbox": [ymin, xmin, ymax, xmax],            // 0-1000 归一化；用于 crop 复现与人审定位
          "bbox_source": "grounding_dino",             // grounding_dino | tracker（track-first 跟踪插值框，参与 identity 匹配）| sam3（路线B 分割框，参与匹配）| vlm_fallback（自动 flag）| full_frame（仅 location 类无明确 landmark 时允许，不 flag）
          "frame_index": 3042,                          // crop 取自的绝对帧号（按内容分挑帧，跳过黑场/均匀帧）
          "embedding_key": "char_big_buck_bunny@c000", // gold/embeddings.safetensors 中的键
          "mask_path": null,                            // 可选，路线B(sam3) 的实例掩码 relative path；路线A 为 null
          "state": "default",                           // 状态标签，与 state_events 呼应
          "qa": {"verified": true, "rounds": 1, "flagged": false, "grounding_score": 0.42, "track_id": 3, "reid_score": 0.71, "face_score": 0.83, "cluster_group_index": 5}  // 自检闭环留痕（原则 #11）；grounding_score=该 rep 的 GroundingDINO 置信（0.0 表示 full_frame/vlm_fallback 整帧兜底）。track-first 追加可选元数据：track_id=镜头内 tracklet 本地 id；reid_score=跨镜头 re-ID 复用的融合余弦（greedy 模式，新建实体不带）；face_score=人脸 cue 命中时的 ArcFace 余弦（§3.7，仅 character 且检到脸时才有）；cluster_group_index=identity_resolution（cluster_vlm 模式）产出的最终身份簇序号，供 tmp/identity_resolution.json 与人审 UI 反查该 rep 属于哪次聚类/VLM决策，同实体内所有 rep 共享同一值。均为标注元数据，不进 SUT 契约；cover 选择与 crop_audit 跳过用 grounding_score
        }
      ],
      "state_events": [                    // Avoidance / state-change 的 GT（决策 D4）；挂在受影响实体名下
        {
          "event_id": "evt_apple_eaten",   // 示例：此事件实际应挂在 prop_apple_01 名下
          "chunk_id": 6,                   // 事件发生的 chunk（该 chunk 的 prompt 必须叙述之，原则 #9）
          "frame_index": 3120,             // 可选：状态变化的精确帧（尽力而为；不可定位时为 null，chunk_id 必存）
          "seconds": 130.0,                // 可选：= frame_index / layout.fps
          "description": "the apple is eaten",
          "deprecates": ["prop_apple_01@c003", "…"]   // 此后即为过期引用的 representation
        }
      ]
    }
  ]
}
```

## 2. gold/chunk_annotations.json（冻结逐 chunk 标注）

```jsonc
{
  "schema_version": "2.0.0",
  "movie_id": "big_buck_bunny",
  "human_reviewed": true,
  "chunks": [
    {
      "chunk_id": 7,
      "shot_span": [12, 14],               // 聚合的镜头区间（闭区间，含端点）
      "frame_span": [3021, 3140],          // 帧区间（闭区间 [first,last]，含端点；相邻 chunk 不共享帧号）
      "seconds_span": [125.88, 130.83],    // = frame_span / layout.fps（人审/分析用；不进打分）
      "prompt": "Big Buck Bunny picks up a red apple under the old oak tree.",
      // ↑ 剧本式；首现实体内联 description；禁止来源/检索提示（决策 D3）
      "present": ["char_big_buck_bunny", "prop_apple_01", "loc_meadow"],   // Sufficiency/Parsimony 的 GT
      "first_appearances": ["prop_apple_01"],
      "gold_instructions": [               // Fidelity 的 GT（仅评分用，不进 prompt）
        {"entity_id": "char_big_buck_bunny", "requirement": "continuity",  // continuity | introduce
         "note": "same appearance as previous chunks"},
        {"entity_id": "prop_apple_01", "requirement": "introduce"}
      ],
      "forbidden": [                       // Avoidance 的 GT；由 state_events 推导物化
        {"representation_id": "prop_apple_01@c003", "reason": "evt_apple_eaten"}
      ],
      "prompt_completeness": {             // 可选标注元数据（Q2 缺陷B）：每个在场实体 name/description 与 prompt 的文本 embedding 余弦；低于阈值 flagged=true → 进人审。不进 SUT、不进打分。
        "scores": {"char_big_buck_bunny": 0.71, "prop_apple_01": 0.44},
        "flagged": ["prop_apple_01"], "threshold": 0.5
      },
      "scenario_tags": ["re-appearance"]   // re-appearance | multi-instance | scene-return | state-change | none
    }
  ]
}
```

## 3. SUT 交互契约（bench ↔ SUT，纯 JSON，零代码导入）

### 3.1 PromptPacket（bench → SUT，评分前）

```jsonc
{ "schema_version": "2.0.0", "chunk_id": 7, "prompt": "…" }
```

### 3.2 ComposedContextRecord（SUT → bench，评分前）

沿用 `common/schemas.py` 并收紧：`asset_id`/`exclusions` 必须使用 ingestion 反馈中获得的权威 ID；
删除 v1 `AssetRef.embedding: list[float]` 字段（违反原则 #10，且评分已不依赖 SUT 上报的 embedding）。

```jsonc
{
  "schema_version": "2.0.0",
  "chunk_id": "7",
  "selected": [
    {
      "asset_id": "char_big_buck_bunny",
      "representation_ids": ["char_big_buck_bunny@c000"],  // 选用的具体 representation（Compactness/Avoidance 依据）
      "function": "subject",
      "strength": "required"
    }
  ],
  "instruction": {
    "per_asset": [ {"asset_ref": "char_big_buck_bunny", "requirement": "continuity"} ],
    "exclusions": ["prop_apple_01@c003"]   // SUT 主动排除的过期 representation
  },
  "memory_keys": ["…"],                     // 诊断：当前记忆库全部键
  "timing_ms": 0.0, "model_calls": 0
}
```

### 3.3 ObservationPacket（bench → SUT ingester，评分后 · generator-oracle 反馈）

> **协议更正（2026-07-24，权威）**：在"真实 segment 评测"协议下，bench 交给 SUT 的**只有** `chunk_id` +
> `chunk_video`（该 chunk 的**原始 segment 视频**，用来**替代 SUT 生成器的产出**、消除生成噪声）。
> bench **不再下发** `observations[]`（`crop_path` / 权威 `entity_id` / `description`）——那是**已废弃的
> gold-replay / 命名神谕**形态。感知（detect→crop→embed）与实体命名/去重现在**全在方法侧（SUT）**完成：
> SUT 自己看 segment、自己抠图、自己给工作 id。真实评测的 ObservationPacket 实际形如
> `{"schema_version":"2.0.0","chunk_id":7,"chunk_video":"chunks/chunk_007.mp4"}`；下面保留旧字段仅为兼容
> gold-replay 离线自测。（连带影响：§4.1 第 3 条"去重由命名神谕代劳"在真实协议下**不再成立**，去重是方法侧
> 能力；是否把去重纳入正式评测轴，待定。）

```jsonc
{
  "schema_version": "2.0.0",
  "chunk_id": 7,
  "chunk_video": "chunks/chunk_007.mp4",
  "observations": [
    {
      "entity_id": "prop_apple_01",        // 权威 ID/命名（命名神谕）
      "kind": "prop",
      "name": "Red Apple",
      "description": "…",                   // 仅首现时提供
      "crop_path": "assets/props/prop_apple_01/c007.jpg",
                                                    // per-entity crop（不是整帧！；旧平铺/derived 均兼容）
      "representation_id": "prop_apple_01@c007"
      // 可选（方法侧视觉地层；冻结协议不要求 gold 标注）：
      // "spatial_angle": "front|side|back|top|unknown",
      // "state_angle": "default|changed|damaged|unknown",
      // "temporal_tag": "chunk_7"
    }
  ],
  "state_events": [ {"event_id": "evt_apple_eaten", "deprecates": ["…"]} ]
  // 状态事件对 SUT（System Under Test，被测系统）可见：按"提示词即完整生成源"
  // 原则（#9），状态变化已在发生 chunk 的 prompt 中叙述，observation 反馈再给出
  // 结构化事件；SUT 需要知道世界变化才能做生命周期管理。
  // 但"哪些 chunk 禁止引用哪些 representation"的 forbidden 物化表（评分 GT）永不可见。
}
```

> **Freeze note (2026-07-16)**：`spatial_angle` / `state_angle` / `temporal_tag` 为 SUT 可选消费字段；
> 冻结 Track A gold / 六指标 / hard-case buckets **不要求**这些字段，缺省按 `unknown` 兼容。

### 3.4 review_patch.json（人工审查 → 管线回填）

```jsonc
{
  "schema_version": "2.0.0",
  "merges": [["prop_apple_01", "prop_apple_02"]],        // 误分裂实体归并
  "splits": ["char_rodent_01"],                           // 误归并实体拆分（拆分后需重跑归并再审）
  "renames": {"char_rodent_01": "Frank the Squirrel"},
  "drops": ["prop_background_blob"],                      // 非实体噪声
  "field_edits": [ {"path": "chunks[7].prompt", "value": "…"},
                   {"path": "entities[char_big_buck_bunny].description", "value": "…"} ]  // path 形式：chunks[N].field 或 entities[<entity_id>].field（review.py apply_patch 支持）
}
```

---

## 4. 指标定义（确定性集合运算，逐 chunk）

> **HEADLINE 指标已换血（见 [`scoring.md`](scoring.md) 与 [`../baselines/fairness_decisions.md`](../baselines/fairness_decisions.md) D2）**：
> 复合 headline 的外观维现为 **VisualFidelity**（按 entity kind 路由的多 embedder：全类 DINOv3 + SigLIP2；
> LSMDC 真人角色 ArcFace；location MegaLoc 且**计分**），**取代**下表中作 headline 的 **ID-Fidelity**。
> 下面的 **Fidelity（introduce/continuity 分类）保留为诊断**（`id_diagnostics`），不再是 headline 外观维。
> 复合分名与权重以 `scoring.md` §3.4 为准（Fid 槽 = VisualFidelity）。**location 在 headline 中不再被剔除**——
> 下文「地点是场景上下文」一段仅适用于本节的 ID 诊断集合运算，视觉 headline 按 D2 对 location 计分。

对 chunk t，记：

- `P(t)` = gold `present` 中 `character`/`prop` 实体；`N(t)` = 对应的 gold `first_appearances`；
  **可检索集** `P_ret(t) = P(t) − N(t)`（首现实体在 t 之前从未被观察过，SUT 记忆中
  不可能存在，纳入分母会无辜罚分；首现实体的正确行为是在指令中声明 `introduce`，由 Fidelity 计分）。
- `S(t)` = SUT 选中的 `character`/`prop` 实体集；`R(t)` = 对应 representation 集；`R_s` = 实体 s 被选中的 representation。
- `G(t)` = gold 指令集；`I(t)` = SUT 指令集。
- `F_active(t)` = 所有满足 `event.chunk_id < t` 的状态事件所废弃的 representation 集合
  （事件发生的 chunk 本身不禁：事件正是在该 chunk 中被演出来的）。

| 指标 | 公式 | 边界规则（反躺平） |
|---|---|---|
| **Sufficiency** | `\|S ∩ P_ret\| / \|P_ret\|` | `P_ret=∅` → N/A（该 chunk 不计入均值）；`S=∅, P_ret≠∅` → 0 |
| **Parsimony** | `\|S ∩ P_ret\| / \|S\|` | `S=∅, P_ret≠∅` → **0**（漏选不等于精简）；`S=∅, P_ret=∅` → N/A |
| **Compactness** | `mean_{s∈S} exp(-λ·(\|R_s\|-1))`，λ=0.2 | `S=∅` → N/A |
| **Fidelity** | 对每条 gold 指令 `(entity, requirement)`，requirement ∈ {continuity, introduce}：SUT 指令集中存在同实体同 requirement 的条目记 1，否则 0；取均值 | `G=∅` → N/A；`I=∅, G≠∅` → 0 |
| **Avoidance** | `1 − \|R ∩ F_active\| / \|F_active\|` | 该 chunk `F_active=∅` → N/A；**整部影片无状态事件 → 影片级 N/A，绝不记 1.0** |
| **MemRecall**（记忆距离加权召回，新增） | 设 `Ret(t)` = 在 t 再现且此前有过缺席的可检索实体（`s∈P_ret(t)` 且 s 在某 `t'<t` 在场、在 `(t', t)` 间缺席）；每个 s 的记忆距离 `d_s` = 距上次在场的 chunk 数；`MemRecall(t)=Σ_{s∈Ret} w(d_s)·[s∈S] / Σ_{s∈Ret} w(d_s)`，`w(d)=log(1+d)` | `Ret(t)=∅` → N/A；**整片无再现实体 → 影片级 N/A**。纯 ID+距离运算，无阈值/无 VLM/无随机 |
| **MemStrata Score** | 各指标先做逐 chunk 均值（跳过 N/A），再按权重归一加权；权重默认 `{Suf .25, Par .15, Com .1, Fid .2, Avo .2, MemRecall .1}`，某指标全片 N/A 时其权重按比例重分配 | — |

**长程记忆图（分析附表，冻结协议内）**：评分报告额外输出
- `memory_length_events`：每个再现 (chunk, entity) 一条 `{chunk_id, entity_id, memory_length=d_s, selected}`——同 segment 多实体分别算长度；
- `memdist_curve`：按 `memory_length` 聚合，`recall` = 该距离上被选中的实体比例；并按 chunk 的 **max** `d` 附带六指标均值（横轴距离、纵轴指标）。
- `memdist_auc`：对曲线做归一化梯形积分 `auc_norm = ∫y dd / (d_max-d_min)`（完美平坦 y≡1 → 1.0）；报告值为 `float|null`，对 `recall` 及曲线上出现的 chunk 指标各算一份。不报告原始面积。
- `horizon_curve[*].memdist_by_entity`：每 chunk 的 per-entity 距离字典。
不改 gold 标注轴、不改六指标定义；`metric_version` ≥ `2.3.1`。

**效率报告（不计入 headline quality score）**：每个 `ComposedContextRecord` 都携带
`timing_ms` 与 `model_calls`；评分器聚合为 `efficiency`：
`total/mean/median/p95/max_composition_ms` 与 `total_model_calls`。它只测
`handle_prompt` 的在线上下文组合/检索，排除模型冷启动、gold 加载和
post-score observation ingest。这样可比较检索成本，但不会让系统以牺牲六项质量
指标来换取更高 headline。

补充：

- 全部为 ID 精确匹配，无阈值、无 VLM、无随机性；同一输入必然同一得分（设计原则 #6/II）。
- **地点是场景上下文，不是被评分资产**：`location` 仍在 prompt 与 ObservationPacket 中供 SUT 建立场景条件，但评分器在 Sufficiency、Parsimony、Compactness、Fidelity、Avoidance 和 MemRecall 前同时剔除 location 的 gold、SUT selection/instruction 与 location state-event 表征。因此选择或不选择 location 都不影响 headline metrics。
- **Fidelity 的真实含义**：`introduce` vs `continuity` 的正确分类要求 SUT 查询自身记忆判断
  "该实体是否已见过"——它实测的是 SUT 记忆索引的完好性（记忆若把同一实体错误分裂，
  回归实体会被误报为 introduce）。
- 评测报告由 `scoring/metrics.py::aggregate_scores` 产出（入口 `python -m vmem_bench.scoring`），
  schema：
  `{score_name: "MemStrata Score", "MemStrata Score": float, weights: {suf,par,com,fid,avo,memrecall},
    metrics_active: [metric], effective_weights: {metric: float},
    n_applicable: {metric: int}, per_metric_mean: {metric: float|null},
    per_scenario_tag_mean: {tag: {metric: float}}, per_memdist_bucket_mean: {near|mid|far: {metric: float}}, num_chunks: int,
    versions: {gold_schema, chunks_schema, layout_schema, metric_version:"2.3.1", movie_id, layout_hash},
    horizon_curve: [{chunk_id, metrics:{metric: float|null}, scenario_tags,
                     memdist_by_entity:{entity_id: d}, memdist, timing_ms, model_calls}],
    memory_length_events: [{chunk_id, entity_id, memory_length, selected}],
    memdist_curve: [{memory_length, n_entities, recall, n_chunks?, <metric>?}],
    efficiency: {scope, n_chunks, total_composition_ms, mean_composition_ms,
                 median_composition_ms, p95_composition_ms, max_composition_ms,
                 total_model_calls}}`。
  `n_applicable` = 每指标计入的 chunk 数（N/A 跳过）；`per_metric_mean` 中 `null` = 该指标全片 N/A；
  `versions` 携带 gold/指标/管线版本指纹（§5 缺一视为不可引用结果）。
- 权重为约定俗成的初始值，发布前需附敏感性分析；不属于 schema 冻结范围。

### 4.1 协议性质声明（必须诚实写进论文/README）

1. **实体级选择在本协议下接近可解析**：D3（权威命名）+ 原则 #9（prompt 完整性）意味着
   `P(t)` 大体可从 prompt 文本解析获得。一个"解析 prompt 实体名 → 查记忆键"的朴素
   baseline（prompt-parser）预期在实体级 Sufficiency/Parsimony 上近满分。**本协议的区分度
   在于**：representation 级选择（选哪个状态的哪张参考图）、生命周期（Avoidance）、
   上下文成本（Compactness）、新旧分类（Fidelity），以及 retrieval 系 baseline
   （embedding 相似检索 / 滑窗 / 全历史）在实体级会犯的过选/漏选错误。prompt-parser
   应作为必报的参考 baseline，而非被回避。
2. 难度消融开关 `prompt_naming: authoritative | generic`：generic 档 prompt 对非主角
   实体用不定指描述（"an apple"），SUT 需自行完成实体链接，留作 hard 变体，不进主表。
3. **去重由命名神谕代劳**：ObservationPacket 直接下发权威 ID（原则 #4），等于 bench 替
   所有 SUT 完成实体去重。这是有意的边界——本 benchmark 隔离评测**组合/选择/生命周期**
   能力，不评测去重；SUT 自身的去重质量属系统实验，另行评测。

## 5. 冻结与发布规则

1. `human_reviewed: false` 的 gold 禁止用于正式评测（harness 启动即校验并拒绝）。
2. 冻结后修改 = bump `schema_version` / 数据版本并重新走审查，旧版本保留。
3. **chunk 布局属于冻结范围**：gold 标注与 `gold/chunk_index.json` 强耦合（全部标注按 chunk 编号），
   `gold/chunk_index.json` 随 gold 一起冻结并计算 layout hash（legacy: `layout/chunk_index.json` 仍可读）；
   harness 启动时校验运行侧 chunk 布局 hash 与 gold 一致，不一致即拒跑。改 `max_frames` 等切分配置 = 新数据版本。
   **片头/片尾剔除**：非叙事片段（片头字幕/片尾字幕/厂标）在切分前剔除，记录于
   `chunk_index.json` 与 `manifest.json → layout` 的 `excluded_segments`
   （`[{shot_span, frame_span, seconds_span, reason: opening_credits|end_credits}]`，闭区间帧号）。
   被剔除的帧不属于任何 chunk 的 `frame_span`，评测与 packet 导出**绝不**把这些时间段交给 SUT；
   剔除结果参与布局并因此被 layout hash 覆盖（改剔除配置 = 新数据版本）。
4. 发布包 = `data/<dataset>/<movie>/` 中的 `manifest.json` + `gold/`（源视频下载方式与 sha256、切分逻辑 `chunk_index.json`/`shot_boundaries.csv`、标注结果、embedding sidecar）+ `assets/`；视频/切片/抽帧/crop 等 `tmp/`（legacy `derived/` 与 `build/`）均本地重建，不进包。
5. 评测报告必须携带 gold 版本 + 指标版本 + 管线指纹，缺一视为不可引用结果。

---

## 6. 常驻服务层内部契约（build 期，非 SUT-facing）

> 设计见 [`services_and_time.md`](services_and_time.md)。这些是**标注管线↔模型服务**的内部契约，
> 不进 gold、不进 SUT。图像**传路径不传像素**（帧/crop 均在共享 FS）。

### 6.1 感知服务 `pathhttp`（GDINO / DINOv3 / InsightFace / 可选 SigLIP）

```jsonc
// POST /detect  (GroundingDINO)
{ "frame_path": "tmp/frames/f0003042.jpg", "phrases": ["grey rabbit", "red apple"],
  "score_threshold": 0.35, "max_area": 0.65 }
// -> { "detections": [ {"bbox":[ymin,xmin,ymax,xmax], "score":0.42, "phrase":"grey rabbit"}, ... ] }

// POST /embed  (DINOv3；batch)
{ "paths": ["tmp/frames/f0003042.jpg", "..."] }
// -> { "embeddings": [[...], ...], "dim": 384 }        // 落 safetensors 由 client 负责

// POST /face   (InsightFace；detect-is-the-gate)
{ "crop_path": "tmp/candidates/xxx.jpg" }
// -> { "embedding": [...512], "det_score": 0.9 }  或  { "embedding": null }   // 无脸=门关

// POST /siglip (仅 crop_classify_method=siglip 时启动)
{ "crop_path": "...", "labels": ["a rabbit","a bird"] }
// -> { "ranked": [["a rabbit",0.91],["a bird",0.03]] }
```

### 6.2 文本 embedding 服务（Qwen3-Embedding-4B, OpenAI 兼容）

走 vLLM `/v1/embeddings`（`{model, input:[texts]} -> {data:[{embedding:[...]}]}`）。client 用于
roster 语义去重（name+description 余弦）与 prompt 完整性检查。

### 6.3 `tmp/services.json`（legacy: `build/services.json`）（放置结果，client 据此连）

```jsonc
{ "schema_version": "2.1.0", "placement": "fastest",
  "services": [
    {"key":"vlm","transport":"openai","base_url":"http://127.0.0.1:8002/v1","model":"Qwen3-VL-32B-Instruct","gpu":null,"external":true},
    {"key":"text_embed","transport":"openai","base_url":"http://127.0.0.1:8003/v1","model":"Qwen3-Embedding-4B","gpu":5,"pid":12345},
    {"key":"gdino","transport":"pathhttp","base_url":"http://127.0.0.1:8010","gpu":6,"pid":12346},
    {"key":"dino","transport":"pathhttp","base_url":"http://127.0.0.1:8011","gpu":7,"pid":12347}
  ] }
```

放置模式 `fastest`（一服务一卡）/ `packed`（按 `resident+peak` 首次适配递减装箱）/ `none`（进程内单例回退，
`--no-services`）。容量预检遵循 `gpu-service-capacity.mdc`：按 `min_free_mib` 过滤、保 headroom、不杀保活进程、
空卡不足即停并报告。
