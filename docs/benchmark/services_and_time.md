# 设计规格：常驻服务层 + GPU 放置双模式 + 时间标注/指标 + tracker 升级

> 状态：**设计草案，待用户批准后进入实现**（2026-07-09）。遵循 schema-first / 禁止提前编码：
> 本文冻结边界与契约后才写代码。覆盖用户三问（Q1 tracker、Q2 文本/图文 embedding、Q3 时间信息）
> 与新硬要求：**所有被用到的模型在生产前常驻为后端服务；用不到的模型绝不启动；提供两种 GPU 放置模式。**
> 关联：[`annotation_tracking_internals.md`](annotation_tracking_internals.md)、[`schemas_and_contracts.md`](schemas_and_contracts.md)、
> [`design_principles.md`](design_principles.md)、[`references.md`](references.md)。

---

## 0. 用户已拍板的决策

| 项 | 决策 |
|---|---|
| 常驻边界 | **被用到的模型全部常驻服务**；用不到的（如默认关闭的 SigLIP）**不启动** |
| 放置模式 | 两种：**fastest**（一服务一卡、分散、降单卡功率提速）/ **packed**（按峰值显存打包共卡） |
| Embedding 服务栈 | **vLLM**（与现有 :8002 同栈，OpenAI 兼容 `/v1/embeddings`） |
| 时间信息落点 | **扩展现有** `entity_registry.json` + `chunk_annotations.json`（不新建 timeline 文件） |
| 时间→指标 | **新增一个时间/记忆距离相关的正式指标**（不止元数据） |
| boxmot | **引入**作消融后端（默认仍用我们的确定性 tracker） |
| 进程内单例澄清 | 已确认：进程内单例=一部片加载一次全程复用，**非**每次推理重载 |

---

## 1. 常驻服务层（persistent service layer）

### 1.1 原则

- **只启动被用到的模型**：由 run 的 config 推导"启用模型集合"，只对该集合起服务。
- **每个被用到的模型 = 一个常驻服务**；标注进程只做 client。VLM/Embedding 走 OpenAI 兼容 HTTP；
  感知模型（GDINO/DINOv3/InsightFace/可选 SigLIP）走**基于文件路径**的轻服务。
- **关键点：感知 IPC 传"路径"不传"像素"**。帧/crop 都已落在共享 FS（`derived/frames`、
  `derived/candidates`），服务读路径→跑模型→回 JSON（boxes/embeddings/faces）。这样规避了
  "图像 IPC 昂贵"这一顾虑（这也是规则里感知模型默认不建独立服务的主要代价，被"传路径"消解）。

### 1.2 服务清单（service manifest，每个模型一条）

每个可服务模型登记一条静态元数据（`services/registry.py`）：

```python
ServiceSpec(
  key="gdino",                       # 逻辑名
  served_model_name="grounding_dino",
  weights="models/model_weights/...",# 或 hub id / 公共根
  transport="pathhttp",              # "openai"(vLLM) | "pathhttp"(感知路径服务)
  resident_vram_mib=2048,            # 常驻显存(实测优先, 保守估计兜底)
  one_inference_peak_mib=1024,       # 单次推理峰值增量
  min_free_mib=4096,                 # 选卡阈值(=resident+peak+headroom)
  default_port=8010,
  enabled_when=lambda cfg: cfg.perception_backend == "gdino_track",
)
```

初版保守估计（H800 141GB/卡，均很小，除 LLM）：

| 服务 | 何时启用 | resident | peak | 备注 |
|---|---|---|---|---|
| VLM Qwen3-VL-32B | 总是（roster/命名/draft） | ~70GB | +数 GB | **已在 :8002**，复用不重起 |
| Embedding Qwen3-Embedding-4B | 启用 roster 语义去重 / prompt 完整性检查 | ~10GB | +2GB | vLLM `--task embed`，:8003 |
| GroundingDINO | `perception_backend=gdino_track` | ~2GB | +1GB | pathhttp |
| DINOv3 | 总是（re-ID/关键帧/crop-QA） | ~1GB | +0.5GB | pathhttp（batch 接口） |
| InsightFace buffalo_l | `use_face=true` | ~2GB | +0.5GB | pathhttp |
| SigLIP2 | `crop_classify_method=siglip` | ~2GB | +0.5GB | pathhttp；默认关→不启动 |
| SAM3/3.1 | `perception_backend=sam3_track` | 待定 | 待定 | 路线B消融，另接 |

> 估计值先落文档，首启后用实测 `nvidia-smi` 回填、更新本表（experiment_memory 同步）。

### 1.3 GPU 放置：两种模式（`services/placement.py`）

预检遵循 `gpu-service-capacity.mdc`：读每卡 `memory.free/used`，按 `min_free_mib` 过滤，保留
headroom（≥10GiB 或峰值 20% 取大者），**不杀** gpu.py/occupy 等保活进程，空卡不足则停止并报告。

- **fastest 模式**：每个服务独占一张空卡（VLM 已独占其卡）。分散→单卡不争用、可跑更高时钟，
  推理最快。需要的空卡数 = 服务数。空卡不够则回退提示或降级到 packed。
- **packed 模式**：先算每卡预算 = `free - headroom`，用 first-fit-decreasing 按
  `resident+peak` 把服务装箱到尽量少的卡（大服务优先）。省卡，但同卡服务分时争用、峰值需并存校验
  （同卡服务 `sum(resident) + max(peak)` ≤ 预算）。

放置结果落 `build/services.json`（哪个服务在哪张卡、端口、pid、状态），client 据此连。

### 1.4 启动器 + 客户端（自包含在 benchmark 内）

- `scripts/start_services.py`（或 `python -m vmem_bench.services.launch`）：读 config→推导启用集合→
  预检+放置→在训练节点用 **tmux/nohup** 逐个起服务（不前台挂 SSH，遵守规则）→写 `services.json`→
  健康探活（HTTP `/health` 或 `/v1/models`）。幂等：已在跑且健康则复用。
- pathhttp 服务实现：`vmem_bench/services/perception_server.py`，FastAPI/stdlib http，端点
  `POST /detect|/embed|/face`，body 传绝对/相对路径列表，返回 JSON。服务内模型是**该服务进程的单例**。
- client：`vmem_bench/services/clients.py` 提供 `GdinoClient/DinoClient/FaceClient/EmbedClient/
  SiglipClient`，与现有 `VlmJudger` 并列。**pipeline_track_first / gdino_track / reid / face /
  crop_classify 改成调用 client**，而非进程内 `from_pretrained`。
- 回退：`--no-services`（调试）时用旧的进程内单例路径，保证可离线单测。

> 注意与规则的张力：规则默认"轻感知模型走进程内单例、不建独立服务"。用户明确要求全部常驻并给两模式；
> 本设计用"传路径"把独立服务的代价降到最低，并保留 `--no-services` 进程内回退，兼顾两者。

---

## 2. Q1 — tracker 升级 + boxmot 消融

- **默认 tracker（确定性、自包含、可单测）升级**：吸收 ByteTrack/BoT-SORT 的两点思想——
  ① 两段关联（先高分框、再用低分框补）；② 轻量运动预测（常速线性/卡尔曼 + `scipy` Hungarian 替代贪心），
  外观线索继续用 DINOv3 门控（BoT-SORT 的 appearance-fusion，我们已实现一半）。修 provenance：
  `tracker` 从错误的 `"bytetrack"` 改为真实值 `"iou_dino"`（升级后 `"bytetrack_local"`）。
- **消融后端**：`tracker = iou | bytetrack_local | boxmot_botsort`。`boxmot_botsort` **pip 安装
  boxmot**（不 vendored，稳定依赖，符合更新后的原则7），仅消融用；它自带 ReID/整帧依赖=非确定性，
  故**绝不进默认 gold**，只出现在"我们的确定性 tracker vs 工业 BoT-SORT"对照表。
- schema：`annotation_provenance.tracker` 值域扩展；不变更 rep 字段。

## 3. Q2 — 文本 embedding（Qwen3-Embedding-4B）+ 图文（SigLIP）

- **文本↔文本用 Qwen3-Embedding-4B**（CLIP 文本编码器弱、短、只为图文配对训练，不适合文本语义匹配）。
  常驻 vLLM :8003。补两个**现存缺陷**：
  - **缺陷A roster 语义去重**：`merge_roster` 现按名字小写精确匹配→"grey rabbit/the bunny/Big Buck
    Bunny"分裂。改为：先精确键合并，再对不同键的 (name+description) 文本 embedding 余弦≥阈值的做二次合并
    （kind 相同才合并；static_attributes 冲突则不合并，沿用 identity funnel）。确定性（阈值固定）。
  - **缺陷B prompt 完整性/命名一致**：track-first 已删 VLM verifier，原则#9 无人守。加确定性检查：
    每个在场实体的 name/description 与 chunk prompt 的文本 embedding 相似度低于阈值→flag 该 chunk 进人审
    （不自动改 prompt）。比子串匹配鲁棒、比 VLM 便宜可复现。
- **图文↔SigLIP2**：仅 `crop_classify_method=siglip` 时用（类别标签检查）。默认关→按用户"用不到不启动"
  →**不起服务**；启用时才起 pathhttp 服务。默认 crop-QA 仍是 DINO 原型法。
- schema：`annotation_provenance` 增 `text_embedder`（如 `qwen3-embedding-4b`）；chunk 增可选
  `prompt_completeness`（{entity_id: score, flagged: bool}，标注元数据，不进 SUT）。

## 4. Q3 — 时间信息（扩展现有契约）+ 新增时间指标

> 打分主体仍 chunk 粒度、确定性不变；新增时间字段为**元数据**，新增指标**单列**、可 N/A、不改旧指标。

### 4.1 schema 增量

`entity_registry.json` 每 entity 增（**大多已由 rep_spans 算出，零额外成本**）：
```jsonc
"presence_spans": [[first_frame, last_frame], ...],   // 该实体所有在场帧区间(闭区间, 已按 tracklet∩ 得到)
"first_frame": 3042, "first_seconds": 126.75,          // 精确首现(=最早 span 起点)；first_chunk 保留
"last_frame": 8210,  "last_seconds": 342.08,           // 末现
"screen_time_seconds": 88.5,                           // 在场总时长(spans 并集/fps) = 重要性/显著度信号
"max_absence_frames": 4120, "max_absence_seconds": 171.7  // 最长缺席=被测试的记忆跨度(再现距离)
```
`chunk_annotations.json` 每 chunk 增 `seconds_span: [t0,t1]`（=frame_span/fps，便于人审/分析）。
`state_events` 每事件增可选 `frame_index` / `seconds`（尽力而为，至少保留 `chunk_id`）。
`layout/chunk_index.json` 已有 `fps`；秒均由 fps 换算，不冗存。

### 4.2 新增指标：Temporal Memory-Distance（工作名，定确定性公式）

动机：MemStrata 本质测长程记忆；再现距离越大越难。用 4.1 的 `max_absence` / 再现间隔把每个
"re-appearance"事件标上**记忆距离**，据此：
- **分层报告**：把 chunk/事件按记忆距离分桶（近/中/远），对现有 Suf/Par/Fid/Avo 分桶求均值，得到
  **随记忆距离衰减曲线**（比现有 `horizon_curve` 按 chunk 序号更本质）。
- **单值指标 `MemDist-weighted Sufficiency`（候选）**：对可检索集里"曾消失又再现"的实体，按其再现距离
  加权 Sufficiency（距离越大权重越高），衡量"越久远越容易漏检索"。纯 ID+距离运算、无阈值/无 VLM。
- 归入 `scoring/metrics.py`，`versions.metric_version` bump；旧指标与权重不变，新指标先作**附表**报告，
  经敏感性分析后再决定是否并入主 MemStrata Score（沿用"权重需附敏感性分析"约定）。

> 待你确认：新指标是先作**附表**（推荐，安全），还是直接并入主 Score 权重？

---

## 5. 实现顺序（批准后）

1. 冻结本文 §1.2 服务契约 + §4.1 schema 增量到 `schemas_and_contracts.md`（schema-first）。
2. `services/`：registry + placement(双模式) + perception_server + clients + launch；`--no-services` 回退。
3. Q1 tracker 升级 + boxmot 消融后端 + provenance 修正 + 单测。
4. Q2 roster 语义去重 + prompt 完整性检查（EmbedClient）+ 单测。
5. Q3 时间字段落盘（pipeline 保留 rep_spans 全量）+ 新指标 + 单测。
6. 起服务（fastest 模式，空卡 5/6/7）→ H800 全链路重跑 BBB → 前端(:7865) 看结果 → 对照旧 baseline。

## 6. 待你最终确认的点

1. 放置模式默认用哪个（建议 fastest，卡够）？两模式都实现。
2. §4.2 新指标先作**附表**还是直接进主 Score？（建议附表）
3. §1.2 显存估计仅初值，首启后实测回填——认可这个流程？

---

## 7. 决策记录：为什么不做「帧级时序定位」，何时才做（2026-07-09 与用户敲定）

### 7.1 出现时刻——已完整记录，无需再加

每个实体的 `presence_spans = [[s1,e1],[s2,e2], …]` 已经是它**所有出现区间**（帧级、相邻合并），来自
tracklet frame span；`first_frame/last_frame` 只是派生便捷字段。「每次出现的进/出时刻」本就在里面，
**不是只有首末**。「某实体在某 chunk 内的 entry/exit 帧」= `presence_spans ∩ chunk.frame_span`，
用时现算即可，不落冗余字段。→ **出现侧到此为止，够了。**

### 7.2 状态转变时刻——gold 默认 chunk 粒度，只加「advisory 帧」，不上时序重模型

**结论：状态事件以 `chunk_id` 为权威且唯一被评分的时间戳；额外让 drafter 顺带吐一个 best-effort
`event_frame` → 映射为 `StateEvent.frame_index/seconds`，标注为 advisory、永不参与评分。**
**现阶段不引入 Marlin-2B 等时序事件定位重模型。**

为什么不做帧级（确定性论证，供后续 agent/设计原则参考）：
1. **无消费者**：当前所有指标（MemStrata Score、MemRecall）都是 chunk 粒度；SUT 是逐 chunk 生成/评分，
   状态"发生在 chunk k"与评分单位天然对齐。sub-chunk 精度目前没有任何指标/任务在用。
2. **chunk 已不粗**：chunk ≈ 120–360 帧 @24fps ≈ 5–15s，"哪个 chunk"本身已相当局部。
3. **无确定性检测器**：任意语义状态（"桃子被吃掉""树被射中"）没有确定性检测器能定位精确帧，
   帧级只能靠模型估计 → 有噪声、不可复现，违背 gold 的确定性 + 可人审底线。
4. **成本收益**：为一个没人消费的字段常驻一个时序重模型（显存 + 运行时 + 审核负担），投入产出不成正比。

何时才升级到真正的帧级时序定位（触发条件 + 阶梯）：
- **先决条件**：先设计出一个**奖励时序精度的指标或任务**（例如"SUT 是否在正确时刻切换状态"）。没有它就不做。
- **升级阶梯（先免费后重模型）**：① 先用**已有 drafter 的 advisory `event_frame`**（零新增模型、不评分）——
  已实现（`vlm_roles.draft_chunk` 传 `frame_indices`，`drafting.state_events_from_draft` 映射并 clamp 到
  chunk span）；② 只有当 advisory 帧被证明太糙 **且** 确有指标依赖它时，才引入 **Marlin-2B** 这类时序
  事件定位模型（走常驻服务）。**先有需求 → 再要精度 → 先 VLM advisory → 实在不够再上重模型**，不倒着上。

> 通用原则（可推广到其它"要不要更精细标注"的决策）：**没有下游消费者的精度不要提前做（YAGNI）；
> 要做也先用手头已有的免费信号，重模型是最后一档。** 见 `.cursor/rules/cheapest-reliable-tool.mdc`、
> `ponytail.mdc`、`specification-driven-development.mdc`。
