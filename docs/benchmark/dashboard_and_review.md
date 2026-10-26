# Track-First 看板 + 人工审核**前端** · 实现规格（后端已就绪）

> **本文现在是纯前端任务书。** 后端（事件心跳 + 所有读写 API）已经实现并跑通，前端**只需按下面钉死的
> 契约去渲染 SSE 事件、调用 HTTP 接口**，不要改后端、不要自行臆造字段。
>
> 目的：为 **track-first 标注流水线**（`annotation/pipeline_track_first.py`）配一个既能**实时监控**、
> 又能**人工审核 + 修改 gold** 的单页前端。
>
> **后端已完成（无需你做）**：SSE 事件已含逐 shot / 逐 entity 心跳（§2）；`server.py` 已实现
> `GET /status`、`GET /gold`、`GET /review/patch`、`POST /review/patch|apply|freeze`（§5，真实契约见该节）。
> 你的活是 §7 的前端 UI + §9 的验收项；§6 是**后端可选扩展，不是你的任务**。
>
> 关联：`annotation_tracking_internals.md`（流水线设计）、[`schemas_and_contracts.md`](schemas_and_contracts.md)（数据契约权威源）、
> [`services_and_time.md`](services_and_time.md)（服务层/时间字段）、[`annotation_pipeline.md`](annotation_pipeline.md)（标注阶段）、
> 现有 `web/server.py`（后端，勿改）+ `web/static/index.html`（旧 VLM-first 看板，你要取代它）。
> 人机协同**评审策略**（三层门 / 决策卡 / 模型边界）见文末 §10——本文件同时是「审核 UI 规格」与「评审策略」的权威源。
>
> **术语**：`chunk`=评测最小单元（连续若干 shot）；`shot`=一个镜头（切点之间）；`entity`=资产
> （character/prop/location）；`representation`=某 entity 在某 chunk 的一张裁剪证据。

---

## 0. 为什么要重写（现状问题）

现前端 `web/static/index.html` 是给**旧 VLM-first** 流水线写的，监听 `chunk_start / role_start /
role_end / registry / chunk_done`。track-first 发的是**另一套事件**，且在最耗时的 per-shot 跟踪阶段
长时间只发一个 `identity`，所以前端会「停在 cast_roster 一动不动」。已在后端补了逐 shot/逐 entity 心跳
（见 §2），本规格据此重做前端；审核部分则把现有「离线 `review.html` → 导出 patch → CLI apply/freeze」
（`annotation/review.py`）升级为**前端内直接编辑 + 保存**。

## 1. 架构

- **后端**：`vmem_bench/web/server.py`，Python **stdlib-only**（`http.server`），**已实现全部接口，勿改**。
  提供 SSE（`/events`）、图片（`/img`）、状态/读 gold（`/status`、`/gold`）、审核读写（`/review/*`）。
- **前端**：单页应用，静态文件放 `web/static/`。允许用现代框架（React/Vue 等）**打包成静态产物**放进
  `web/static/`，或纯原生 JS——由实现方定，但**产物必须是 server 能直接吐的静态文件**，运行时不依赖 node。
- **两种模式**（同一页面，按运行状态自动切换或 Tab 切换）：
  1. **Live 监控**：`GET /events` 的 SSE 流（run 进行中）。
  2. **Review 审核**：读 `GET /gold` 等（run 结束、`run_done` 后），可编辑并 `POST` 保存补丁 / 应用 / 冻结。
- **数据都在磁盘**：一个 run 的产物目录 `<out>/`（server 以 `--out` 指向它）。前端不直接读文件系统，
  一律走 server HTTP。图片走 `/img?p=<相对路径>`（已实现，仅暴露 `<out>/assets/`、`<out>/tmp/candidates/` 与 legacy `<out>/derived/` 子树）。

## 2. Live 事件契约（SSE，权威）

SSE 端点 `GET /events`：每条 `data:` 是 `tmp/events.jsonl`（legacy: `build/events.jsonl`）的一行 JSON。特殊控制帧
`event: reset`（新 run / 日志被截断时前端应清空重建）。每条事件都有 `ts`（float 秒）和 `kind`。
下表是 **track-first 会发出的全部 kind**（按时间顺序），实现前端时**只需处理这些**：

| kind | 时机 | 关键字段 | UI 建议 |
|---|---|---|---|
| `run_start` | 开跑 | `movie_id`, `video`, `backend`, `stage="chunking"` | 顶栏显示片名 + “切镜头中” |
| `layout` | SBD/切分完成 | `n_chunks`, `n_shots`, `fps`, `layout_hash`, `stage="roster"`, `chunks:[{chunk_id,frame_span,shot_span}]` | 建 chunk 网格；存每 chunk 元数据 |
| `roster_start` | 进入演员表发现 | `n_keyframes_budget`, `vlm_batch`, `stage="roster"` | 阶段徽标切到 roster（roster 阶段开始心跳） |
| `roster_progress` | **每个 VLM roster 批次后** | `done`, `total`, `n_known`, `stage="roster"` | roster 进度条推进 + 累积候选数（避免停在 layout 一动不动） |
| `roster_resumed` | 复用已 checkpoint 的演员表 | `n_entities`, `stage="roster"` | 跳过发现，直接进入去重/跟踪 |
| `roster_semantic_dedup` | 演员表语义去重后 | `before`, `after`, `merged`, `stage="roster"` | 演员表面板提示合并数 |
| `cast_roster` | 全片演员表定稿 | `n_keyframes`, `n_entities`, `entities:[{name,kind}]`, `stage="roster"` | 展示候选演员表（还没裁剪证据） |
| `tracking_start` | 进入逐镜头跟踪 | `n_shots`, `stage="tracking"` | 显示进度条 0/n_shots |
| `track_progress` | **每个 shot 结束** | `shot`, `n_shots`, `n_entities`, `frame`, `elapsed`, `eta_seconds`, `stage="tracking"` | 进度条推进 + 实体累计数 + ETA（核心心跳） |
| `identity` | 跨镜头 re-ID 完成 | `n_entities`, `n_tracklet_spans` | 跟踪阶段完成标记 |
| `crop_qa` | 裁剪质检剔除后（可选） | `n_flagged`, `method` | 提示剔除数 |
| `naming_start` | 进入逐实体命名 | `n_entities`, `stage="naming"` | 命名进度条 0/n |
| `naming_progress` | **每个 entity 命名后** | `done`, `n_entities`, `name`, `stage="naming"` | 命名进度推进 |
| `name_error` | 某实体命名失败（不致命） | `entity_id`, `error` | 告警一行，不阻塞 |
| `naming_done` | 命名结束 | `n_entities` | 命名完成 |
| `chunk_done` | **每个 chunk 的 prompt 起草后** | `chunk_id`, `n_present`, `n_first`, `seconds` | 点亮该 chunk 网格格子（这是逐 chunk 实时进展） |
| `run_done` | 全部完成 | `n_chunks`, `n_entities`, `n_flagged_chunks`（`gold_dir` 已剔除） | 切到「可审核」状态，加载 gold |

**阶段模型（给 UI 用）**：`chunking → roster → tracking → identity → naming → drafting(chunk_done×N) → done`。
大部分墙钟时间在 `tracking`（128 shot），其次 `naming`（每 entity 一次 VLM）。`drafting` 阶段才逐 chunk 快速点亮。
前端应有一个**总进度/阶段指示器**：roster 用 `roster_progress.done/total`，tracking 用 `track_progress.shot/n_shots`，naming 用 `naming_progress.done/n_entities`，
drafting 用累计 `chunk_done / layout.n_chunks`。

> 注意：track-first **不发** `chunk_start / role_start / role_end / registry`（那是旧流水线的）。
> 前端不要依赖它们。`registry`/资产库的实时增量在 track-first 里没有逐 chunk 推送——资产要等 run_done
> 后从 `GET /gold` 全量读；Live 阶段只用 `cast_roster` 的名字列表 + `track_progress` 的累计计数占位。

## 3. Gold 数据契约（审核阶段读，权威源见 `schemas_and_contracts.md` / `common/schemas.py`）

run_done 后，`<out>/gold/` 下有：
- `entity_registry.json` → `EntityRegistry`
- `chunk_annotations.json` → `ChunkAnnotations`
- `embeddings.safetensors`（向量，前端不用）
其它：`<out>/gold/chunk_index.json`（切分；legacy `layout/chunk_index.json`）、`<out>/tmp/annotation_qa.json`（每 chunk QA/flagged；legacy `build/`）、
`<out>/assets/{characters,props,locations}/<entity_id>/*.jpg`（裁剪 + `cover.jpg`；
旧平铺 `assets/<entity_id>/` 和 legacy `derived/assets/` 仍接受）。

**Entity**（可审核对象，字段来自 `common/schemas.py::Entity`）：
```jsonc
{
  "entity_id": "str", "kind": "character|prop|location", "name": "str", "description": "str",
  "first_chunk": 0,
  "static_attributes": {"species":"...", "primary_color":"...", ...},   // 稳定身份属性，可编辑
  "representations": [{
     "representation_id":"str", "chunk_id":0, "crop_path":"assets/characters/<eid>/xxx.jpg",
     "bbox":[ymin,xmin,ymax,xmax(0-1000)], "bbox_source":"grounding_dino|vlm_fallback|full_frame|tracker",
     "frame_index":123, "embedding_key":"str", "state":"default",
     "qa":{"verified":bool,"rounds":int,"flagged":bool,"grounding_score":float}
  }],
  "state_events": [{"event_id":"str","chunk_id":0,"description":"str","deprecates":["rep_id"],
                    "frame_index":null,"seconds":null}],   // frame_index/seconds 为 advisory
  // Q3 时间元数据（确定性算出，展示用；编辑 entity 后可选重算，不影响 SUT）：
  "presence_spans":[[start,end],...], "first_frame":int|null, "first_seconds":float|null,
  "last_frame":int|null, "last_seconds":float|null, "screen_time_seconds":float|null,
  "max_absence_frames":int|null, "max_absence_seconds":float|null
}
```

**ChunkAnnotation**（可审核对象，来自 `common/schemas.py::ChunkAnnotation`）：
```jsonc
{
  "chunk_id":0, "shot_span":[s0,s1], "frame_span":[f0,f1], "seconds_span":[t0,t1],
  "prompt":"str",                              // 可编辑（核心）
  "present":["entity_id",...],                 // 可编辑（增删 present）
  "first_appearances":["entity_id",...],       // 由 present + first_chunk 派生，改 present 后重算
  "gold_instructions":[{"entity_id":"","requirement":"continuity|introduce","note":""}],
  "forbidden":[{"representation_id":"","reason":"event_id"}],
  "scenario_tags":["state-change|multi-instance|re-appearance|..."],   // 编辑后需重算
  "prompt_completeness":{...}                   // Q2 报告，展示用
}
```

## 4. 审核能编辑什么（沿用 `review.py` 的补丁语义，扩展到前端）

现有离线补丁 `review_patch.json`（`schema_version:"2.0.0"`）支持这些操作，**前端应直接产出同款补丁**
（这样后端 `apply_patch` 复用，零重复逻辑）：

```jsonc
{
  "schema_version":"2.0.0",
  "merges":[["target_eid","source_eid"], ...],   // 把 source 并入 target（rep/state_event 合并）
  "drops":["entity_id", ...],                      // 删除该 entity（并从所有 chunk.present 移除）
  "renames":{"entity_id":"新名字", ...},
  "field_edits":[
     {"path":"chunks[<cid>].prompt","value":"新 prompt"},
     {"path":"entities[<eid>].description","value":"新描述"}
  ],
  "splits":[...]   // 现未自动化：后端只告警，需手工改 JSON（见下「缺口」）
}
```
`apply_patch` 会：应用 merges/drops/renames/field_edits → 重算每 chunk 的 `present/first_appearances/
gold_instructions/forbidden/scenario_tags` → 写回 gold。`freeze` 会 lint（`common/gold_lint.py`
`strict_review=True`），通过才置 `human_reviewed=true`（未冻结的 gold 评测会拒收）。

**前端把可编辑面收敛到 apply_patch 已支持的集合**（rename / drop / merge / prompt / description）即可。
其它更强编辑（增删 `present`、`static_attributes`、`state_events`、删单张 `representation`、split）
属后端可选扩展（§6），**本次前端不做**。

## 5. 后端已实现的 HTTP 接口（**直接调用，勿改后端**）

全部已在 `server.py` 实现并跑通。基址 = 前端页面同源（server 以 `--out <run目录>` 启动）。响应均为 JSON
（除 `/`、`/events`、`/img`）。

| 方法 路径 | 作用 | 真实返回 / 入参 |
|---|---|---|
| `GET /` | 页面 | `static/index.html` |
| `GET /roster` | canonical roster 前置确认页 | `static/roster.html`；run 完成后可从当前 registry/crops bootstrap，也可继续编辑 draft |
| `GET /roster-seed` | 读取 seed/editor 数据 | `{seed,candidate_crops,source,confirmed}`；优先 confirmed→draft→gold proposal |
| `POST /roster-seed` | 保存/确认 seed | body=seed；`_confirm=false` 保存 draft；`_confirm=true` 过 `roster_seed.py` 严格校验后原子写 `roster_seed.json` |
| `GET /events` | SSE 事件流（§2） | `text/event-stream`；控制帧 `event: reset`；数据帧 `data: <一行事件 JSON>` |
| `GET /img?p=<相对路径>` | 图片 | 仅暴露 `<out>/assets/`、`<out>/tmp/candidates/` 与 legacy `<out>/derived/` 子树（jpg/png），越权/穿越→404 |
| `GET /status` | Live/Review 判定 | `{"kind":str|null,"stage":str|null,"movie_id":…,"n_chunks":…,"n_entities":…,"done":bool}`（`done` = gold 已存在） |
| `GET /gold` | 读 gold | 完成时 `{"registry":{…},"chunks":{…},"qa":[…],"layout":{…},"human_reviewed":bool,"done":true}`；**未完成→HTTP 409** `{"ok":false,"done":false,"error":"…"}` |
| `GET /review/patch` | 取回草稿 | 上次 `POST` 的草稿 JSON，或 `{}` |
| `POST /review/patch` | 存草稿（不落 gold） | body = review_patch(§4)；→ `{"ok":true}` |
| `POST /review/apply` | 应用补丁→回写 gold | body = review_patch，或 `{}` 表示用已存草稿；成功→**应用后的完整 `/gold` 载荷**；失败→HTTP 400 `{"ok":false,"error":"<原因>"}` |
| `POST /review/freeze` | 冻结 gold（过 lint 才成） | 成功→`{"ok":true,"human_reviewed":true}`；lint 失败→HTTP 400 `{"ok":false,"error":"<lint 文本，多行>"}` |

> 行为要点（前端据此处理）：所有 `POST /review/*` 在 **run 未完成（`done=false`）时返回 HTTP 409**，
> 前端应据此禁用审核操作、停留在 Live 视图。`apply` 成功直接回吐新 gold，前端拿它刷新即可，无需再 `GET /gold`。
> 写入是原子的、单锁串行，前端正常单人审核即可。
> `POST /roster-seed` 是**运行前**的本体确认，不受 `done` 限制；确认时会拒绝重复 ID/name、kind/scope
> 不一致、individual 无 exemplar、非法 event policy 等错误。

## 6.（后端可选扩展 · **非前端任务**，需要时由后端加）

当前 `apply_patch` 支持的编辑集合：**renames / drops / merges / field_edits(`chunks[<cid>].prompt`、
`entities[<eid>].description`)**。前端**先把可编辑面收敛到这个集合**即可（§4）。
若日后要支持「增删 present / 改 static_attributes / 改 state_events / 删单张 representation / split」，
需要先在后端 `apply_patch` 扩展对应分支——**这不是本次前端交付范围**，前端遇到暂不做即可。

## 7. 前端 UI 建议（不强制，但要覆盖以下信息）

- **顶栏**：片名、当前阶段徽标（chunking/roster/tracking/naming/drafting/done）、总进度条、
  Live/Review 模式切换、`layout_hash` 短哈希。
- **Roster 视图**（`/roster`）：候选实体保留/删除、canonical ID/name/kind/scope、phrases/aliases、
  3–5 exemplar 选择、有限 event policy；先存 draft，人工确认后才生成 production seed。
- **Live 视图**：
  - 阶段进度：tracking 段用 `track_progress`（shot x/n + ETA），naming 段用 `naming_progress`。
  - chunk 网格：`layout.chunks` 建格，`chunk_done` 点亮（present 数、用时）。
  - 演员表：`cast_roster` 名字 chips；run_done 后替换为带 cover 的真实资产卡。
  - 右侧事件流：把 §2 每类事件渲染成一行人话（现前端已有 `fmt()` 可参考）。
- **Review 视图**（run_done 后）：
  - **资产库**：每个 entity 一张卡（cover + 所有 rep 缩略图 + name 可改 + description 可改 +
    static_attributes 可改 + drop 勾选 + merge into 选择 + 坏 rep 单删）。flagged（`qa.flagged` /
    低 grounding_score / `bbox_source=vlm_fallback`）高亮。
  - **chunk 列表**：prompt 可编辑（textarea）、present chips 可增删、state_events 可编辑、
    scenario_tags/seconds_span 展示、prompt_completeness 展示；`annotation_qa.json` flagged 的 chunk 高亮，
    风险高的（state-change/multi-instance/re-appearance）置顶（沿用 `review.py::_chunk_review_risk`）。
  - **时间轴（Q3 加分项）**：用 `entity.presence_spans`（绝对帧）+ fps 画每个 entity 的出没条，
    直观看「首次出现 / 消失 / 回归（max_absence）」，这正是 MemRecall 想考的记忆距离。
  - **操作条**：Save draft（`POST /review/patch`）、Apply（`POST /review/apply`）、
    Freeze（`POST /review/freeze`，显示 lint 结果）。编辑态与已保存态要有明确视觉区分。

## 8. 约束与非目标（前端侧）

- **不改后端**：所有数据经 §5 的 HTTP 接口，前端不直接读文件系统。
- 前端可用框架，但**最终产物必须是 `web/static/` 下 server 能直接吐的静态文件**（运行时不依赖 node）。
- **编辑只经审核补丁改 gold，不碰评测口径**：Q3 时间字段、prompt_completeness 是 metadata；评分在 `scoring/`，前端不碰。
- `POST /review/*` 在 `done=false` 时返回 409：前端据此在 Live 阶段禁用审核操作。
- 图片只经 `/img`。冻结失败（lint 不过）时前端要展示后端返回的错误文本。

## 9. 前端最小验收清单

1. Live：新 run 从 run_start 到 run_done，阶段徽标 / 进度条 / chunk 网格 / 右侧事件流全程有动静，
   tracking 阶段用 `track_progress` 逐 shot 推进（含 ETA），naming 用 `naming_progress`（不再停在 cast_roster）。
2. Review：`done=true` 后 `GET /gold` 加载，改 name/description/prompt、勾 drop、填 merge into →
   `POST /review/patch` 存草稿 → `POST /review/apply` 后用返回的新 gold 刷新看到变更 →
   `POST /review/freeze` 成功后 `human_reviewed=true`；freeze 失败时展示 `error` 文本。
3. Roster：从 gold proposal 加载候选，完成删改/选择 exemplar 后保存 draft；非法 seed 确认失败并显示
   原因，合法 seed 确认后 `human_confirmed=true`。
4. 产物为静态文件放入 `web/static/`，运行时不依赖 Node。

---

## 10. 评审策略（人机协同：三层门 + 决策卡 + 模型边界）

> §1–§9 是**审核 UI 的实现规格**（前端渲染什么、调什么接口）；本节是它背后的**评审策略**——
> 机器与人各该做什么、审核项如何分层与排序、模型可用到什么边界。两者不冲突：策略决定 UI 收敛到哪个可编辑面（§4）与决策卡呈现（§7）。

### 10.1 结论

标注主链路要追求的不是「模型代替人做所有判断」，而是把人从浏览 JSON、重复查找证据和机械修复中移开：
**机器负责发现、聚合、排序和验证可形式化的错误；人只裁决模型无法可靠区分的视觉身份与叙事语义。**
Track-First 已具备该方向的基础：确定性 `present`、严格 lint、`auto_review`、人工 disposition 冻结门、
多 crop 起草证据、review-only identity candidates，以及 location 不参与 headline **asset-selection** 评分
（注：视觉 headline 侧 location 已按 D2 用 MegaLoc 计分，见 [`scoring.md`](scoring.md)；此处指 ID 诊断的选择评分）。
它尚未通过真实 BBB 重跑证明「最低人工成本」；该结论必须由候选数、人工分钟数、建议接受率、strict-lint
和抽样误差共同验证。

### 10.2 思考路径

1. **先区分错误性质。** 路径、引用、派生字段、空值和 schema 问题有唯一正确答案，应由确定性代码处理；「两个 crop 是否同一角色」「某变化是否不可逆」没有仅靠规则可得的答案，不能伪装成自动化。
2. **再区分机器动作。** 自动拒绝、自动生成可逆 patch、给人建议、交由人裁决的安全等级不同。高置信不是单一模型分数，而是多证据一致且没有反证。
3. **最后优化人的操作。** 人工审核的单位应是一个可以回答的决策，而非文件或 chunk。先给结论、再给最少但足够的证据、最后提供少量可撤销动作。

### 10.3 三层审核策略

**A. 确定性自动 gate**（无需模型或人工）：schema/ID/路径完整性、`present` 与 `first_appearances` 的派生一致性、
非法 representation 引用、空/占位 prompt、无效 state-event、冻结所需 disposition 缺失。任何失败均阻止 freeze，不静默降级。

**B. 高置信机器建议与有限自动修复**：机器可生成 patch 或「接受建议」，但每次都保留前后差异、证据、阈值/模型版本，并可撤销。
身份合并**只有同时满足**以下条件才有资格进入该层：

- 同 kind，静态属性无冲突；
- body、face、class、text 至少两路强一致，或规范名一致且视觉证据强一致；
- 不存在同帧共现等时间线反证；
- patch 后局部重算和 strict lint 通过。

即使达到条件，也应抽样人工审计（建议每片至少 5%，且每类至少若干项）并据接受率/撤销率校准阈值。
prompt 文本相似度和单次 VLM 判断只能产生 finding，不能单独自动改写 gold。

**C. 人工裁决**（必须升级人工）：多实例角色 / 相似道具的身份判断、模型或证据通道相互冲突、不可逆 state-event、
影响大量未来 chunk 的 merge/split、以及自动化抽样审计。人工最终决定写入 review patch + disposition；
freeze 只接受 strict lint 为零且必审项均已处置的实例。

### 10.4 Human-friendly 审核体验（决策卡）

统一队列中的每一项只提一个问题，并按下式排序：

`预计评分影响 × 受影响 chunk 数 × 证据分歧 × 审核优先级`

- **身份卡** — 问题：「这两个实体是否相同？」展示双方 2–3 张高质量、多样代表 crop、首末出现时间线、同屏反证、
  body/face/class/text 信号、受影响 chunk/prompt 摘要与推荐动作。动作限定：合并 / 保持独立 / 需要更多证据。
- **状态卡** — 问题：「该变化是否不可逆并应废弃旧 representation？」展示事件前后 crop、发生 chunk 的 prompt、
  候选 `deprecates`、未来受影响 chunk。动作限定：确认不可逆 / 非不可逆 / 证据不足。
- **Prompt 卡** — 问题：「prompt 是否覆盖已确定的 present roster？」展示 sampled frames、给 VLM 的实体证据与当前 prompt。
  动作限定：覆盖充分 / 漏实体 / 叙述冲突；只有后两项需要最小文本编辑。

别名候选应按**连通簇**一次呈现，避免反复判断 A-B、B-C、A-C。每次动作后即时展示 patch、局部重算结果与剩余 blocker 数，并支持撤销。

**当前已落地（对应 §5/§7 的 Review 收件箱）**：dashboard 的 Review 模式已提供「审核收件箱」——
`tmp/review_queue.json` 中的 identity / state / prompt / lint 项按确定性优先级呈现；identity 候选按连通簇合并；
双实体卡显示代表 crop、时间跨度与多路相似度，审阅者必须填写理由才能暂存「合并为左侧实体」或「保持独立」；
状态卡展示不可逆事件、废弃 references 与影响范围，但不伪造未实现的 state-event 编辑功能；prompt/lint 卡可跳转到对应实体/chunk 的完整编辑上下文。
所有动作只更新浏览器中的 review patch/disposition 草稿，仍须经 §5 的保存 / 应用 / lint / freeze 流程。
状态事件用 `state_event_reviews`（确认 / 驳回 / 受限编辑都需理由；原始事件与人工结论追加存 `tmp/state_event_review_pairs.jsonl`，
当前处置存 `tmp/state_event_dispositions.json`）；应用后重算 `forbidden` 与 state-change tags；freeze 拒绝任何尚未有人工决定的剩余事件。
前端「预览 Lint」在临时副本应用当前 patch 并返回 strict-lint blocker，不写真实 gold。

### 10.5 模型使用边界

- 初版**不需要新增模型**：现有 DINO/face/class/text 信号与现有 VLM 足够生成队列与证据。
- 后续可加一个**受约束的证据裁判**：仅对已筛选的 identity/state 卡、基于固定 crop 与时间线输出「同一 / 不同 / 证据不足」及依据；
  **不得发现新实体或直接改 gold**。
- 若裁判结果用于**自动 merge**，它必须与起草/命名模型独立（不同模型家族或至少不同 endpoint），
  并先在人工已裁决样本上测量 precision、误合并率与节省分钟数；未达预设 precision 门槛时，裁判**只能排序，不能自动执行**。

### 10.6 近期实施顺序与验收

1. 先建立 `tmp/review_queue.json`：合并 auto-review、identity candidates、annotation QA 与 lint 的证据为单一、只读、可排序队列。
2. 在现有 review 页面展示身份 / 状态 / prompt 决策卡与剩余 blocker 摘要；不改变 patch 或 freeze 语义。
3. patch 后只重算受影响实体/chunk，显示局部 lint 和派生字段差异。
4. 在真实 BBB 运行记录自动通过率、人工处置数/分钟、建议接受率、抽样误差与 freeze 成功率；用这些指标校准阈值。
5. 仅在用户批准模型与 precision 门槛后，引入独立证据裁判或自动 merge。
