# MemStrata-Bench 追踪/re-ID 内部机制：Track-First（检测+跟踪打骨架，VLM 只命名）

> **本文是 track-first 追踪/re-ID 的内部机制文档（被 `pipeline_track_first` 代码引用）；阶段级流程见
> [`annotation_pipeline.md`](annotation_pipeline.md)。**
>
> 状态：**已实现并接入生产**。track-first 的检测→跟踪→re-ID 骨架、确定性 presence、per-crop 质检
> 均已在 `annotation/pipeline_track_first.py` 落地并离线自测；身份判定的**生产默认 = `seeded`（人工
> roster 约束的封闭集分配，§10）**，`cluster_vlm`（§9）/`greedy`（§3.1）仅作显式 proposal/消融开关。
> §7 记录 2026-07-09 的定向决策（历史留档）。
> 关联：诊断依据见 [`pitfalls.md`](pitfalls.md)；阶段级 S1–S7 流程见 [`annotation_pipeline.md`](annotation_pipeline.md)；
> 数据契约以 [`schemas_and_contracts.md`](schemas_and_contracts.md) 为准；参考工作与 2026 调研见
> [`references.md`](references.md)；常驻服务层/GPU 双模式/时间指标见
> [`services_and_time.md`](services_and_time.md)；track-first 实时看板 + 人工审核前端规格见
> [`dashboard_and_review.md`](dashboard_and_review.md)。本文件聚焦"追踪/re-ID/身份判定的机制与契约增量"。

---

## 0. 一句话

把"跨 chunk 身份一致性"从 **VLM 每 chunk 从零重发现 + 事后对齐** 改成 **确定性检测器出框 →
跟踪器连成 track（持久 ID）→ 外观 re-ID 跨镜头认人**；VLM 从"每 chunk discover/verify/per-crop
audit"降级为"给已稳定的实体命名 + 起草 prompt"。目标：身份碎片化从根上消除、VLM 调用量降一个数量级。

---

## 1. 为什么现行设计又慢又差（证据）

真实产物 `data/.../big_buck_bunny/_archive_20260709_045601_pre_rerun/build/checkpoint.json`（跑到 chunk 22，
实际 `vlm_model=Qwen3-VL-32B-Instruct`）：

1. **同一实体被拆成多个 entity_id**（致命）：`char_white_rabbit` vs `char_white_rabbit_character`、
   `char_pink_butterfly` vs `..._character`、`char_brown_squirrel` vs `..._character`、
   `prop_red_apple` vs `prop_red_apple_prop`，以及一批 `_prop`/`_location` 双胞胎。本 benchmark 全部价值
   在于跨 chunk 追踪实体再现/延续，身份一裂开，continuity/Fidelity 的 GT 直接失真。
2. **语义身份错误且被 prompt 加重**：主角 `char_big_buck_bunny`（洞里"深灰生物"）与 `char_white_rabbit`
   （"白色大胖兔"）其实是同一角色被拆开；discovery prompt 明令 `Do NOT use prior knowledge` 反而阻止
   了"维持同一反复出现主角"的判断。
3. **17% chunk 是空壳兜底**：c11/16/19/22 prompt 全是 `"The scene continues in this location (chunk N)."`，
   present 只有 `loc_chunk_0NN_setting`——`_fallback_chunk_annotation` 在 discovery 返回空时注入的占位符。
4. **跑了四轮修复，连一部 10 分钟片子的 `gold/` 都没冻出**：只有 `_archive` 里的 checkpoint。

**根因（非症状）**：现行单 chunk 环路
`VLM discover → ground → embed → VLM draft → VLM verify → per-crop VLM audit`，把最贵最不稳的 VLM 放在
"身份一致性"这个本应由 tracking/re-ID 确定性解决的位置上。四轮修复（`grounding_phrase`、
`static_attributes` 门禁、`normalize_entity_name` 剥后缀、crop pruning、fail-fast、checkpoint…）全是补丁，
没有一项引入"稳定的 track 级身份"这个正确抽象。`pitfalls.md` 自己的耗时公式
`total_time ≈ chunks × qa_rounds × branches × (discover + draft + verify + per_crop_vlm)` 已经说明——
几乎每一项都是 VLM 调用。这违反本仓库 `AGENTS.md` #5/#10（确定性优先、能用专用感知就别上 LLM）。

---

## 2. 成熟开源打标仓库的共识设计（借鉴来源）

| 仓库 | 关键设计 | 我们借鉴 |
|---|---|---|
| labelVid (GroundingDINO+Florence-2+AutoDistill) | **detector-first**：检测器出框，VLM 不做发现 | 身份骨架交给检测器，不让 VLM 逐 chunk discover |
| VLM-AutoYOLO | 关键帧抽取(scene/motion/interval)+**SSIM 去重**；SAM 精修；human-in-the-loop | 抽帧加运动/去重感知（可选） |
| **humos**（机器人 GT） | **ByteTrack/BoT-SORT 持久 ID 跨帧跟踪**；**SigLIP 零样本分类比 VLM 快 ~200×**；VLM 只判状态迁移 | ← 本重构两大主借鉴 |
| SwiftAnnotate | annotator–validator 两段式 | 保留"独立校验"思想，但校验降级为确定性/SigLIP，不再每 chunk 重 VLM |

共识：**确定性感知打骨架（检测→跟踪→re-ID），VLM 只做语义命名/描述/剧本化。**

---

## 3. 新架构（Track-First）

### 3.0 统一路线，对一切影像鲁棒（不是只为动画）

本系统必须能标注**任何影像**——动画、真人电影、纪录片、UGC……**默认走同一条路线**，不为风格分叉。
这条 track-first 路线天生模态无关：检测器（开放词表）、跟踪（IoU/外观）、re-ID（DINOv3 外观 + 静态属性）
对像素一视同仁，不预设"卡通脸 / 真人脸"。真正需要按对象**类别**（kind，不是风格）分路的只有两处，且都封装在统一接口内：

- **按 kind 分支（而非按 style）**：`character` 走"外观 + 可选人脸 cue"，`prop` 走纯外观，`location`
  走场景级归属（§3.4）。这套分支对动画和真人是同一份代码——真人多一路人脸 cue 命中、动画少一路而已，
  见 §3.7 的**自门控**设计（有脸就用、没脸就退化，绝不硬判"这是不是脸"）。
- **风格差异当作参数，不当作代码分叉**：`reid_threshold`、检测器置信度、`track_fps` 等按数据分布标定
  （真人镜头运动快可提 `track_fps`、卡通外观稳可降），但**流程与代码路径唯一**。只有当统一路线在某模态
  上实测不达标时，才允许加分支——先统一，验证后再按需分化。

> 一句话：**统一像素路线 + 按 kind（非按 style）的两处封装分支 + 自门控的人脸增强**，保证"卡通兔子"和
> "真人演员"用同一套确定性骨架被追踪、被认人。

### 3.1 身份的两级建立（确定性）

- **镜头内跟踪（intra-shot tracking）→ tracklet**：每个 shot 是连续帧。按 `track_fps`（默认 3–4 fps）
  在 shot 内抽帧，检测器（GroundingDINO，开放词表）出框，喂给跟踪器（ByteTrack / BoT-SORT）得到
  **tracklet**（一个 tracklet = 该 shot 内一个物体实例，带本地持久 track_id）。同框多实例天然被 track_id
  区分——直接解决"松鼠/兔子/蝴蝶 crop 混进同一实体"。
- **跨镜头再识别（cross-shot re-ID）→ 全局 entity_id**：每个 tracklet 取其成员 grounded crop 的 DINOv3
  embedding 均值作为外观签名（可叠加 SigLIP 语义类别作硬约束），与已有全局实体做 re-ID 匹配。匹配上则
  复用 entity_id，否则新建。**身份由外观相似度决定，不由 VLM 起的名字决定**——从根上消灭 `_character`
  后缀式碎裂。默认仍"宁分勿合"，误分由人审 merge 修。

### 3.1a 演员表的关键帧选择（避免"过多/过少"——检索结论）

演员表的目标是"用尽量少的帧覆盖全片所有不同实体/外观"。两头都危险：帧**太多** → VLM token 超限 +
幻觉（KFS-Bench 明确指出"帧更多 ≠ QA 更好"，MLLM 幻觉会给采样引入随机性）；帧**太少** → 漏掉后
出场/稀有实体。2026 年主流的训练无关关键帧选择（KTV / KFS-Bench / AdaRD-Key / EVFR / Katna /
Mixpeek）做法高度一致：**cheap 稠密候选 → cheap 去重 → cheap 编码 → 多样性选子集 → 质量过滤 →
只对选中的少量帧上 VLM**。落到我们（已有 shot 结构）：

1. **候选（scene-adaptive，不用固定高 fps 全片铺）**：利用已有 shot 边界，每个 shot 低 fps（默认 ~2fps）
   抽候选帧（Mixpeek scene-adaptive budget：`min/max_frames_per_scene`）。
2. **去重**：pHash / LUV 色差 / SSIM 杀近重复帧（Katna 用 LUV 绝对差；Mixpeek 用 pHash+汉明距离）。
3. **编码 + 每 shot 取代表**：DINOv3 编码候选，**每 shot 做 K-medoids 取最接近质心的真实帧**作代表
   （KTV：cluster→取质心最近帧，减少时间冗余），每 shot 留 1–2 张。
4. **压到全片预算**：跨 shot 再去重（BBB 很多 shot 是同一片草地），对全片代表帧做 **Farthest Point
   Sampling / K-medoids** 压到固定预算 K（EVFR/Mixpeek：FPS 在 embedding 空间选最远点，保证多样、无
   冗余；AdaRD-Key/KFS-Bench：diversity+relevance 联合目标）。
5. **质量过滤**：Laplacian 方差去模糊 + 亮度/熵过滤（Katna），别把黑场/糊帧/字幕帧喂给 VLM。
6. **上 VLM（分批）**：K 张分批（每批 8–12 张）喂 VLM 发现实体，跨批合并演员表（按 name+static+外观
   去重）。**分批是关键**——一次塞太多图正是 token 超限 + 幻觉的根源（KFS-Bench）。

预算示例（BBB，68 shots ≈ 14k 帧）：每 shot 1 medoid → 68 代表 → 去重+FPS → **K≈32**（约 4 批 ×8）。
既覆盖全部不同场景/角色，又不撑爆单次 token。演员表仍允许**增量**：跟踪阶段若出现演员表里没有的
持续 track（连续 ≥`track_min_len` 帧的显著新物体），补一次 VLM 命名加进表（§3.1），兜住"初始采样漏掉
的后出场角色"。

> 全部候选筛选（抽帧/pHash/DINO 编码/K-medoids/FPS/模糊过滤）都是**确定性 + cheap**，符合
> `cheapest-reliable-tool.mdc`；VLM 只落在最后一步的少量选中帧上。参考：KTV(k-medoids 质心帧)、
> EVFR(FPS，DINOv2+FAISS)、Katna(LUV 差分+直方图 K-means+Laplacian 模糊过滤)、
> KFS-Bench(ASCS 自适应相似+聚类；帧多≠好)、AdaRD-Key(relevance+diversity max-volume)、
> Mixpeek(scene-adaptive + pHash dedup + k-medoids/FPS)。

### 3.1b 感知后端抽象：双路线可插拔（消融用）

身份骨架（§3.1）里"帧 → 每帧检测 → tracklet"这一段封装成**一个感知后端接口**，下游（re-ID、
per-crop 质检、presence、命名、prompt）完全后端无关。这样能**同时实现两条路线、跑完再比效果**（论文消融）：

```python
# annotation/perception/base.py（自包含，仅依赖 tracking.py 的 Detection/Tracklet dataclass）
class PerceptionBackend(Protocol):
    name: str  # 进 annotation_provenance
    def track_shot(self, frames: list[Frame], roster: list[RosterEntry]
                   ) -> list[Tracklet]: ...  # 一个 shot 的帧 → 带持久 track_id 的 tracklet
```

| 路线 | 后端实现 | 检测 | 跟踪 | 掩码 | 说明 |
|---|---|---|---|---|---|
| **A（默认）** | `gdino_track` | GroundingDINO 开放词表出框（roster 的 `grounding_phrase`） | 自包含 class-aware IoU 跟踪（`tracking.py`，可选 BoT-SORT 升级） | 无（bbox） | 已实现骨架；模态无关，成熟稳 |
| **B（消融）** | `sam3_track` | SAM3 / SAM3.1 概念分割（文本 concept prompt = roster 名） | SAM3 原生视频传播（memory）得到跨帧同 ID 掩码/框 | 有（mask） | 一步出"检测+跟踪+掩码"，crop 更干净、遮挡更稳；作对照后端 |

两后端**吐同一种 `Tracklet`**（B 额外带 mask，可选写入 `Representation.mask_path`），re-ID 及之后完全共用。
消融只切 `perception_backend = gdino_track | sam3_track`，其余流程与参数不变，直接对比重复实体率 / crop 纯度 /
presence 准确率 / 墙钟时间。SAM3 权重就绪前，A 是主路，B 的接口与 stub 先建、权重到位即插。

> 自包含性（原则 7）：两后端都实现在 `benchmarks/VMem-Bench/src/vmem_bench/annotation/perception/` 内，
> 只 import 第三方库 + 本 benchmark 内模块，不引用 MemStrata 外任何代码；模型权重从公共根按路径加载。

### 3.2 VLM 的新职责（大幅收缩）

| VLM 任务 | 频次（旧 → 新） | 说明 |
|---|---|---|
| 实体发现 | 每 chunk × 轮 × 分支 → **全片一次性构 cast roster** | 从跨全片的少量代表帧发现"演员表"（name/kind/`grounding_phrase`/`static_attributes`），供检测器全程复用 |
| 实体命名/描述 | 每 chunk 重命名 → **每全局实体一次** | tracking+re-ID 定完身份后，VLM 看该实体最佳 crop（可拼图）给权威 name+description 一次 |
| prompt 起草 | 每 chunk（保留） | present 列表已由 tracking 确定，VLM 只做剧本化叙述 + 首现实体内联描述 |
| 状态事件检测 | 每 chunk verify（保留但收缩） | 仅对 **embedding 漂移 / 消失-再现** flag 的实体-chunk 调用，判定不可逆变化 → `deprecates` |
| per-crop 同一性审计 | 每 crop VLM → **删除，改 SigLIP/DINO 分类** | 见 3.3 |
| presence 查全/查准 | 每 chunk VLM verifier → **删除，改确定性** | present = 与该 chunk frame_span 相交的 tracklet 的全局实体集，见 3.4 |

粗略调用量：`旧 ~ chunks×rounds×branches×(discover+draft+verify+per_crop)`
→ `新 ~ 1(roster) + N_entities(命名) + chunks(draft) + flagged(状态事件)`。约降一个数量级。

### 3.3 SigLIP 替代 per-crop VLM 审计

per-crop"这张 crop 真的是实体 X 吗"用 `google/siglip2-base-patch16-512`（已在公共库）或 DINOv3 原型
余弦做零样本分类：crop 向量 vs 实体原型向量（及 roster 的其它候选名）取最近。零样本、确定性、可复现、
比 VLM 快两个数量级——符合 `cheapest-reliable-tool.mdc`。VLM 只在 SigLIP 落入灰区时兜底（可选）。

### 3.4 presence 变确定性（详解）

**要解决的字段**：每个 chunk 的 `present`（该 chunk 里出现了哪些实体，是 Sufficiency/Parsimony 的 GT）
和 `first_appearances`（其中哪些是首次出现，是 Fidelity 的 GT）。

**旧做法（VLM verifier，慢且会错）**：让一个 VLM 看该 chunk 抽的 ~12 帧，逐项判断
`presence_recall`（该出现的实体都在吗？——漏判就少标）和 `presence_precision`（有没有多标不在的？）。
这是**语义判断**：VLM 可能把一闪而过的实体漏掉、把背景物体幻觉成"在场"、或对同一实体因换了名字而
重复计数。慢（每 chunk 一次 VLM）且不可复现。

**新做法（纯集合运算，零 VLM）**：presence 本来就是"检测 + 跟踪的结果"，直接读出来即可。

```
present(chunk t) = { tracklet 所属的全局 entity_id
                     | tracklet.frame_span 与 chunk_t.frame_span 有交集 }
first_appearances(chunk t) = { e ∈ present(t) | registry[e].first_chunk == t }
```

**worked example（BBB chunk 7，frame_span=[1666,1812]，聚合 shots 8–10）**：
跟踪在这几个 shot 内产出若干 tracklet，各自已被 re-ID 归到全局实体——

| tracklet(本地 track_id) | phrase | 帧区间 | re-ID → 全局实体 |
|---|---|---|---|
| t12 | grey rabbit | 1668–1805 | `char_big_buck_bunny` |
| t13 | white flower | 1690–1770 | `prop_white_flowers` |
| t14 | tree branch | 1720–1812 | `prop_tree_branch` |

→ `present(7) = {char_big_buck_bunny, prop_white_flowers, prop_tree_branch, ...}`；其中
`prop_white_flowers.first_chunk==7` → 进 `first_appearances`；`char_big_buck_bunny` 之前 chunk 已注册
→ 不进 first_appearances（它在本 chunk 是 continuity）。**全程无 VLM，精确可复现。**

**边界规则**：
- 只出现 <`track_min_len` 帧的物体（单帧误检）不成 tracklet → 不计入 present（"一闪而过不算在场"，
  与旧 `grounding_min_frames` 精神一致，且天然挡住背景噪声误标 presence）。
- **locations 特殊处理**：地点是场景级、通常整帧、不适合框选跟踪。location 的 present 由"该 shot 的
  场景归属"决定（full-frame 代表帧 + 场景 embedding 的 name/近邻匹配，见 reid.py 的 signature=None
  名字路径），不走 tracklet∩frame_span。character/prop 走 tracklet，location 走场景归属。
- 这样 `presence_recall/precision` 不再需要 VLM verifier；verifier 仅保留可选的"prompt 是否忠实/完整"
  语义检查（Fidelity 的 gold_instructions 与 prompt 起草由 VLM 做，但 present 本身不再问 VLM）。

**为什么这直接治 baseline 的病**：旧做法里 presence 依赖 VLM 每 chunk 重判 + 实体名，名字一漂移
（`white_rabbit` vs `white_rabbit_character`）就会把同一实体在 present 里算成两个、Fidelity 把回归实体
误判为 introduce。新做法 present 来自"确定性检测+跟踪+re-ID 后的全局 ID"，名字漂移在 re-ID 阶段已被
外观合并消化，presence 自然干净。

### 3.7 人脸：自门控的多线索 re-ID cue（不先判"是不是脸"）

**用户的顾虑（正确）**：若先用一个模型判"这张 crop 是不是人脸"、判是了再上 ArcFace，会 (a) 多一次模型调用
低效，(b) 那个"是不是脸"分类器本身会误判、把非脸送进 ArcFace 或把真脸漏掉。**解决办法：取消"先判后调"，
让人脸检测器的检测结果本身就是门控信号，且人脸只是外观 re-ID 的一条附加线索、不是替代。**

设计要点：

1. **只在 `character` 上尝试，且检测即门控**：re-ID 已知每个 tracklet 的 kind（来自演员表）。仅对
   `kind==character` 的代表 crop 跑一次**人脸检测+对齐+编码一体**的模型（InsightFace `buffalo_l`：
   RetinaFace 检测 + ArcFace 512d 一次前向）。**"有没有脸"直接由检测器返回 0/N 个框决定——不存在单独的
   "是不是脸"分类步骤，也就没有那一步的误判和额外调用**。检测到脸 → 取最大/最清晰脸的 ArcFace 向量作
   `face_sig`；没检到（动画脸、背影、遮挡、非人角色）→ `face_sig=None`，本条 cue 缺席，无任何副作用。
2. **人脸是加权 cue，不是替代**：re-ID 相似度是多线索**融合**，而非"有脸就只看脸"：

   ```
   sim(track, entity) = w_body · cos(dino_body)                      # 永远在，主线索
                      + w_face · cos(face_sig, entity.face_sig)      # 仅两侧都有脸时计入
                      + w_class · cos(siglip_class)                  # 语义类别（可选，硬约束/加权）
       （门禁：kind 相同 且 static_attributes 兼容，见 consolidation._static_compatible）
   ```

   人脸缺席时权重**自动重归一化到 body+class**——所以动画、真人统一一条公式，动画自然退化成"外观为主"，
   真人在有正脸时获得强判别的 face cue。误检一张脸也不致命：它只贡献一个加权项，被 body/class 稀释，
   不像"先判后调"里误判会直接改变是否调用 ArcFace 的**控制流**。
3. **阈值与保守性**：`face_sig` 命中且 `cos(face) >= face_strong` 时可作为**强证据**允许在 body 略低于
   `reid_threshold` 时仍判同人（同一角色不同姿态 body 会漂，正脸更稳）；但**绝不**允许仅凭 face 低分就
   合并（避免双胞胎/相似脸误合）。整体仍"宁分勿合"，误分人审 merge 兜底。
4. **落盘**：命中人脸的 representation 在 `qa` 里记 `face_score`；实体的 `face_sig` 与 body 签名一样存
   safetensors sidecar（原则 10），不进 JSON。人脸编码器指纹进 `annotation_provenance.face_encoder`。

> 为什么统一且鲁棒：同一段 re-ID 代码对动画/真人都跑，差别只是 `face_sig` 是否为 None——**由数据自证，
> 不由我们写 if style=="live_action"**。符合 §3.0 的"按 kind 不按 style、自门控"原则与
> `cheapest-reliable-tool.mdc`（InsightFace 检测即门控，比"VLM 判是不是脸"确定、快、可复现）。

### 3.5 机制如何落进阶段级流程（阶段级权威见 annotation_pipeline.md）

本文不再单列一份端到端步骤表——**阶段级 S1–S7 流程以 [`annotation_pipeline.md`](annotation_pipeline.md)
为权威**。本文的机制在该流程中的落点：§3.1（逐 shot `track_shot` → tracklet）、§3.1/§3.7（跨镜头
re-ID 多线索融合）、§3.3（per-crop 质检）、§3.4（确定性 presence / first_appearances）是 **S5
视觉库/身份判定**所依赖的内部机制；roster 发现/命名/prompt 起草（VLM）落在 S1/S3。

> 实现状态：上述机制已在 `annotation/pipeline_track_first.py` 落地并离线自测（presence/
> first-appearance/roster 选帧/re-ID/融合/crop-QA 共 35 项）。per-crop 质检默认
> `crop_classify_method="prototype"`——复用 re-ID 的 DINOv3 crop embedding，把"离本实体原型更远、离另一同
> kind 实体原型更近（超过 margin）"的 rep 判为混类并连同其 presence span 一起剔除（`audit_registry_crops`
> + `_prune_reps`），**零新模型、确定性、离线可测**；`crop_classify.py` 另提供 `SiglipCropClassifier`
> 作图文零样本替代/消融（GPU）。location 采用"逐 shot 场景 embedding → re-ID 聚为地点实体"的确定性归属。

---

## 4. 契约增量（尽量小；scoring/harness/publish 尽量零改）

**gold / SUT / 指标契约 shape 不变**：`present`、`first_appearances`、`gold_instructions`、`forbidden`、
`state_events`、`Entity`、`Representation` 字段全部保留，scoring/harness/publish 无需改。仅新增**标注元数据**
（不进 SUT 契约、不参与评分）：

1. `Representation.bbox_source` 增加取值 `tracker`（框来自跟踪插值/关联，非单帧检测）与 `sam3`（B 后端的
   分割框）；保留 `grounding_dino | vlm_fallback | full_frame`。identity 匹配认 grounded 框（`grounding_dino`/`tracker`/`sam3`）。
2. `Representation.qa` 增加可选 `track_id`、`reid_score`、`face_score`（人脸 cue 命中时的 ArcFace 余弦）。纯元数据。
3. `Representation.mask_path`（可选，relative path）：B 后端(SAM3)产出的实例掩码；A 后端为空。
4. `annotation_provenance` 增加 `perception_backend`（`gdino_track`/`sam3_track`）、`tracker`、`reid`
   （如 `dinov3-vits16`）、`crop_classifier`（`siglip2-base-patch16-512`）、`face_encoder`（`insightface-buffalo_l`）指纹。
5. `AnnotationConfig` 增字段（默认值见 §6）：`perception_backend`、`track_fps`、`tracker`、`track_min_len`、
   `track_iou_threshold`、`reid_threshold`、`reid_w_body/w_face/w_class`、`face_encoder`、`face_strong`、
   `crop_classifier_model`、`crop_classify_threshold`、`cast_roster_mode`（`global|per_shot`）。

> 契约第 4 行"不得私自增删"——上述增量批准后，同一次改动内写进 `schemas_and_contracts.md` §1/§4，
> 避免 SDD 漂移（`pitfalls.md` F2 教训）。

---

## 5. 代码影响（组件级，实现阶段才动）

| 模块 | 变化 |
|---|---|
| `annotation/chunking.py` | 不变 |
| `annotation/grounding.py` | 复用；`ground_batch` 用于 shot 内逐帧检测 |
| `annotation/embedding.py` | 复用（tracklet 签名 + re-ID） |
| `annotation/tracking.py` | **已建**：自包含 class-aware IoU 跟踪，检测→tracklet（BoT-SORT 留升级路径） |
| `annotation/perception/base.py` | **新增**：`PerceptionBackend` 协议 + `Tracklet` 契约（后端无关下游） |
| `annotation/perception/gdino_track.py` | **新增（路线 A，默认）**：GroundingDINO 出框 + `tracking.py` 关联 |
| `annotation/perception/sam3_track.py` | **新增（路线 B，消融）**：SAM3/3.1 概念分割 + 原生视频传播；权重到位即插 |
| `annotation/reid.py` | **已建**：跨镜头 re-ID；扩为多线索融合（body + 自门控 face + class） |
| `annotation/face.py` | **新增**：InsightFace `buffalo_l` 检测即门控 → `face_sig`（仅 character，§3.7） |
| `annotation/crop_classify.py` | **已建**：默认 DINOv3 原型余弦 per-crop 质检（`audit_registry_crops`，零新模型）+ 可选 `SiglipCropClassifier` |
| `annotation/consolidation.py` | Registry 加 `face_embeddings`/`class_embeddings`（多线索存储）；旧 consolidate 保留供旧路参考 |
| `annotation/roster.py` | **已建**：演员表关键帧选择（候选采样→去重→medoid→FPS 到预算）+ 跨批 roster 合并（确定性） |
| `annotation/vlm_roles.py` | **已加** `discover_roster`（全片一次）+ `name_entity`（每实体一次）；`draft_chunk` 复用；旧 discover/verifier 保留 |
| `annotation/pipeline_track_first.py` | **新增（主环路）**：§3.5 端到端编排；确定性 presence/first-appearance；复用 chunking/drafting/events/`_persist` |
| `annotation/pipeline.py` | 旧 VLM-first 主环路，保留 `_persist`/`_write_checkpoint` 等复用助手；`run.py` 已切到 track-first |
| `annotation/run.py` | **已切**：CLI 默认走 `annotate_movie_track_first`，`--perception-backend/--track-fps/--reid-threshold/--no-face` |
| `scoring/` `runner`/`publish` | **不变**（契约 shape 未变） |
| `annotation/review.py` | 基本不变（merges/splits/renames 依旧适用） |

---

## 6. 建议默认参数

```
# --- 演员表关键帧选择（§3.1a）---
roster_candidate_fps = 2.0        # 每 shot 候选抽帧率（cheap 稠密候选）
roster_dedup_method = "phash"     # phash | ssim | luv：近重复帧去重
roster_dedup_phash_hamming = 6    # pHash 汉明距离阈值（<= 视为重复）
roster_per_shot_k = 1             # 每 shot K-medoids 代表帧数（1–2）
roster_global_budget = 32         # 全片 FPS/K-medoids 压到的关键帧总预算 K
roster_vlm_batch = 8              # 每次 roster 发现 VLM 调用喂入的帧数（分批防 token 超限/幻觉）
roster_blur_laplacian_min = 60.0  # Laplacian 方差下限（低于=糊帧，丢弃）
cast_roster_mode = "global"       # 全片一次性演员表 + 跟踪期增量扩充；per_shot 作对照

# --- 感知后端（§3.1b，模态无关，消融切换）---
perception_backend = "gdino_track"  # gdino_track(A,默认) | sam3_track(B,消融)

# --- 跟踪 / re-ID / 身份判定（§3.1, §3.7, §9, §10）---
track_fps = 3.0                   # shot 内跟踪抽帧率（真人运动快可提，卡通稳可降；参数不分叉代码）
tracker = "bytetrack_local"       # 默认：两段关联+运动预测+DINO 外观融合，确定性（值与 schemas_and_contracts.md 一致）；iou=旧贪心，boxmot_botsort=工业 BoT-SORT 消融（非确定性，不进主 gold）
identity_resolution_mode = "seeded"  # 生产默认：人工 roster seed 约束的封闭集分配（§10）；cluster_vlm(§9)/greedy(§3.1) 仅 proposal/消融
track_min_len = 2                 # tracklet 至少覆盖 2 个采样帧（滤单帧误检，沿用 grounding_min_frames 精神）
track_iou_threshold = 0.3
reid_threshold = 0.55             # 融合相似度跨镜头合并阈值（偏保守/宁分勿合，人审 merge 兜底）
reid_w_body = 1.0                 # DINOv3 body 外观权重（主线索；缺席其它 cue 时自动占满）
reid_w_face = 0.6                 # 人脸 cue 权重（仅两侧都检到脸时计入，自门控，§3.7）
reid_w_class = 0.3               # SigLIP 语义类别权重（可选）
face_encoder = "insightface-buffalo_l"  # RetinaFace 检测即门控 + ArcFace 512d 一次前向；无脸→cue 缺席
face_strong = 0.5                # 人脸强证据阈值：达标可在 body 略低于 reid_threshold 时仍判同人；绝不单凭脸合并

# --- per-crop 质检 / VLM（§3.3, §3.2）---
crop_classifier_model = "google/siglip2-base-patch16-512"
crop_classify_threshold = 0.25    # SigLIP 类别 margin，低于则送 VLM 灰区兜底
vlm_model = "Qwen/Qwen3-VL-8B-Instruct"   # 命名/起草用 8B，不用 32B
```

**运行环境（2026-07-09 定）**：实跑放 **H800**（显卡更快，用户指定）；感知模型（GroundingDINO/DINOv3/
SigLIP2）进程内单例，需要 transformers ≥4.57 的 Python。VLM 走**常驻
vLLM OpenAI 服务**（H800 上另起 8B；本机 `:8000` 已有一个 `Qwen3-VL-8B-Instruct` 服务可作备用/本机调试）。

---

## 7. 已拍板决策（2026-07-09，用户确认，进入实现）

- **A. 迁移方式 = 替换**：track-first 直接替换旧 discover+consolidation 主路（ponytail 删繁）。
- **B. 跟踪/感知后端 = 双路线消融**：路线 A 自包含 class-aware 跟踪（`tracking.py` 已建，BoT-SORT 留升级）；
  路线 B SAM3/3.1 分割+视频传播后端，**两条都完整实现，跑完再比效果作论文消融**（§3.1b）。
- **C. 演员表粒度 = global**：全片一次性发现 cast + 跟踪期增量扩充；`per_shot` 仅作对照。
- **D. prompt 约束**：命名阶段**放开跨 chunk 身份识别**（不再 `Do NOT use prior knowledge` 阻止认出反复
  出现的主角），**描述仍只依据像素**。身份主要由确定性 re-ID 定，VLM 命名不再制造碎裂。
- **E. 统一路线 + 鲁棒（新）**：默认一条模态无关路线覆盖动画/真人/一切影像，只按 kind（非 style）在两处
  封装分支，风格差异当参数不当代码分叉（§3.0）。
- **F. 人脸 = 自门控多线索 cue（新）**：仅 character，InsightFace 检测即门控，人脸作加权 re-ID cue 融合、
  缺席自动退化，不设"是不是脸"分类步、不因误判改控制流（§3.7）。
- **G. 自包含定义（原则 7 已改写）**：自包含 = `memstrata`↔`vmem_bench` 不互相 import + 不 import
  `benchmarks/VMem-Bench/` 外任何**代码**；第三方库与外部**模型权重**（按路径加载）不算耦合，允许使用。

---

## 8. 预期收益与验证口径（步骤 2 baseline 对照）

- **质量**：重复实体对（同一物体多 ID）数 → 目标 0；空壳兜底 chunk 占比 17% → 目标 0；
  `derived/assets/<id>/` 混类 crop → SigLIP 质检后显著下降。
- **速度**：VLM 调用总数、总 token（`summary.total_tokens` 已可观测）降一个数量级；单 chunk 墙钟时间。
- 步骤 2 将写只读脚本从旧 checkpoint 量化上述基线（重复实体率 / fallback 率 / VLM 调用数），
  作为改造前后对照，落到 `experiments/probe/memstrata_annotation_baseline/`。

---

## 9. 身份判定 v2：VLM 主导批量聚类（2026-07-13，用户拍板，已实现）

### 9.0 为什么第一版 track-first 的身份判定仍不够（根因，不是症状）

§3.1 把跨镜头身份判定做成**在线贪心最近邻聚类**：`reid_assign` 按镜头顺序逐个把 tracklet 跟
registry 里已有实体比较融合余弦，`>= reid_threshold` 就并入否则新建。这套设计在 `memstrata_annotation_baseline`
（重复 id 23.4%）和 `sam3_exemplar_bbb` 探针（exemplar 余弦 margin 常只有 ~0.1，argmax 选错物种）
之后仍反复出问题，人工审核负担比直接标注还高。根因三点：

1. **任务被建模错了**：roster 已经先验知道全片大致有哪些实体，"tracklet 是谁"其实是小候选集（同
   kind/identity_group 通常 1–5 个候选）上的**封闭集分类**，却被当成**开放集聚类**用一个全局绝对阈值
   做 yes/no——阈值无法同时满足"不误合并"和"不误分裂"。
2. **判别信号本身够不上任务粒度**：DINOv3/SigLIP 是语义级通用表征，不是为 instance-level 细粒度
   re-id 训练的，同类视觉相似对象（红狐狸/红松鼠/苹果）判别力不足。
3. **在线贪心不可逆**：局部一步错只能靠事后人工 merge/`identity_candidates`/`identity_adjudication`
   补；这些"事后擦"本身就是审核负担的主要来源，量级常超过真实身份判断数量。

### 9.1 新设计：确定性预聚类 + VLM 权威判定（不再是灰区兜底）

翻转"谁是主力"：VLM 从"chunk 级 discover/verify"降到"tracklet 聚类粒度的判定"（调用量仍降一个
数量级，但比"零 VLM"的第一版多了一层，是刻意的、认为值得的权衡——见 §9.4），embedding 只做候选缩小
/预聚类；聚类从在线贪心改为离线批量，允许纠错传递。实现在
[`annotation/identity_clustering.py`](../annotation/identity_clustering.py)（纯算法，零 VLM/GPU）+
[`annotation/identity_resolution.py`](../annotation/identity_resolution.py)（编排）：

1. **按 (kind, identity_group) 分桶**：与旧 `allowed_entity_ids` 同样的限制（检测短语是廉价稳定的
   身份先验）。
2. **确定性预聚类**：`identity_clustering.cluster_by_linkage`，**complete-link 或 average-link**
   （默认 complete），刻意不用简单连通分量——连通分量等价于 single-link 聚类，在噪声 embedding 下
   一条弱"桥"（如角度不好的兔子 crop 恰好跟松鼠 crop 有 0.5 相似度）就会把不同个体链式误合并；
   complete-link 要求跨簇**最差**那对相似度也过线，正式化了 `reid.py` 里原有的
   `cluster_min_similarity` 补丁思路，把它从"贪心 assign 之后的补救检查"提升为聚类本身的主准则。
3. **VLM 簇内校验（权威，非灰区）**：`vlm_roles.AnnotatorRole.verify_cluster`——每个多成员候选簇一次
   调用，看代表 crop 网格判断是否单一个体，若不是则拆分（可能进一步拆给未参与该次调用的成员，按
   body 余弦最近邻确定性归并回拆分结果）。
4. **VLM 跨簇合并（权威）**：复用 `vlm_roles.AnnotatorRole.group_same_individuals`（原只用于 review-only
   的 `identity_adjudication.py`），对同 kind、不同 identity_group 的簇做一次跨簇合并判定——这是
   治"white_rabbit vs big_buck_bunny"这类短语碎裂的正式机制，不再只是 review 阶段的建议。
   static 属性冲突对 VLM 的合并判定有**硬否决**（不管 VLM 怎么判，冲突的静态属性直接否决合并）。
5. **Roster 完整性检查**：最终身份簇若有效证据量（tracklet observation 数）达到阈值、但其成员从未
   命中任何 roster 短语，产生 finding（"可能是 roster 漏发现的实体"），而不是静默强分类给最相似的
   错误候选。

人审呈现（后续 review UI 落地）从"pairwise 候选卡片"改为"按最终身份簇一次性审阅"——一个实体一张卡，
展示全片多样化代表 crop 网格，工作量与最终实体数（十几个量级）成正比，而不是与候选对数/tracklet 数
成正比。

### 9.2 与旧路径的关系：不新建第二条 pipeline，config 开关切换

`config.identity_resolution_mode`：**生产默认 `"seeded"`**（§10，人工 roster seed 约束的封闭集分配）；
本节的 `"cluster_vlm"` 与旧 §3.1 在线贪心 `reid_assign` 的 `"greedy"` 均**降级为 proposal/消融开关**
（原样保留，用户可控的显式选择，非自动探测降级）。三条路径共享同一个 `pipeline_track_first.py` 主环路，
只是阶段 4 内部按开关分支；`reid.py` 提取了 `commit_tracklet_observation`（entity 创建 + representation
记账的共用尾段），各路径都调用它，避免重复实现"落盘"这一步。

契约 shape 不变：`Entity`/`Representation` schema 零改动，新增的聚类/身份判定溯源全部落在已有的
自由字段——`Representation.qa.cluster_group_index`（该 rep 属于哪个最终身份簇，供人审 UI 反查）、
`EntityRegistry.annotation_provenance.identity_resolution_mode`/`precluster_linkage`（可复现性指纹）。
不发布、不进 SUT 契约的过程留痕（预聚类候选、VLM 校验/合并 finding）写 `tmp/identity_resolution.json`
（镜像 `identity_candidates.json`/`identity_adjudication.json` 的定位）。

### 9.3 关键身份决策用 32B（模型档位，用户拍板 2026-07-13）

`verify_cluster`/`group_same_individuals`（identity resolution 热路径）现在是**权威决策**而非灰区
兜底，出错代价高——用 `--judge-base-url`/`--judge-model`（现有 hybrid serving 机制，默认 32B）而不是
`--vlm-base-url` 的快模型。这不是新增插件：`run.py` 已有的 `judge_role` 概念（原本覆盖 roster 发现/
命名/identity adjudication/auto-review 投票）天然覆盖这两个新调用点，零新增 CLI 插线。roster 发现/
命名/prompt 起草等生成类任务仍用 8B（`pitfalls.md` 的"不默认让 32B 承担所有子任务"经验不变——identity
决策是例外，不是推翻）。

### 9.4 预期效果与已知权衡（诚实评估，非承诺；量化以 BBB 重跑为准）

**质量/低人工量/审核友好**是本次重构直接瞄准的目标：审核项从"candidate pair 数量"（会远超真实实体数）
降到"最终实体簇数量"（同真实实体数阶）；判定信息量从"单一余弦分数"升到"多张代表 crop + VLM 判断"。
**效率**有一个刻意的、认为值得的小幅让步：本次往阶段 4（原本零 VLM）加回按"候选簇/最终实体"计数的
VLM 调用（数十次量级），比旧 chunk 级设计（每 chunk×round×branch，成百上千次）小得多，但比"零 VLM"
的第一版是净增加——换可靠性和可审核性的质变。独立的簇校验/合并调用**必须**走线程池+多 endpoint 并发
（用户确认显卡/endpoint 数量充裕），墙钟时间才能压到接近单次调用延迟而不是线性叠加；`tracking.py`/
`track_parallel.py` 的镜头级并行不受本次改动影响，仍是全片墙钟时间的主要瓶颈。这些数字的精确值以
`bbb_rerun_eval`（对照 §8 的 `memstrata_annotation_baseline` 口径）实测为准，本节不作先验保证。

---

## 10. 身份判定 v3：人工 seed 约束的封闭集分配（2026-07-13 路线转向）

BBB `h800_parallel_v1` 暴露了 v2 的边界：批量聚类能改善已有 tracklet 的 merge/split，却不能修复
错误 roster 本体、错误物种/static attributes、漏 tracklet 和开放式状态事件；64 个 chunk 虽然
`0 flagged`，仍有 125 条 prompt omission 和大规模事后审核队列。因此 v2 降级为 proposal/消融，
生产 gold 改为：

1. 人工一次确认 benchmark-relevant canonical roster、稳定 ID、多视角 exemplar 与 identity scope；
2. individual tracklet 在同 kind seed exemplar 间做封闭集分类；绝对分数或 top-2 margin 不足即
   `unknown/reject`，禁止强分类；
3. category prop 按 seed phrase 聚合，不虚构实例生命周期；scene/location 只作 prompt context；
4. seed ID/name/description 锁定，VLM 不再重命名或 reslug；
5. 状态事件限制到有限 ontology，并受每个 individual seed 的 allowed policy 约束；
6. prompt canonical-name/event 覆盖、unknown、seed 漏证据、alias split、无效事件均为 freeze 阻断项；
7. review unit 是一个 canonical entity 的全片 crop 网格或一个 entity state timeline，不再是 pairwise
   聚类候选图。

实现入口：`annotation/roster_seed.py`、`run.py --roster-seed`、`identity_resolution_mode=seeded`。
自动 discovery 必须显式 `--proposal-only`，其产物 provenance 为 `roster_mode=proposal`，strict freeze
拒绝。
