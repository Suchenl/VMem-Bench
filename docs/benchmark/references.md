# MemStrata-Bench Annotation — References & 2026 Landscape

本文件回答两件事：

1. **我们参考了哪些工作、具体借鉴了什么设计**（自动打标框架 / MOT / ReID / 感知与对齐模型）。
2. **2026 年是否已有"长视频角色一致性 gold 自动构建"的成套方案**（一次最新调研的结论）。

> 说明：这里列的是*设计/方法*层面的借鉴。运行时的实际实现按 `vmem_bench_design_principles.md`
> 原则 7：可加载外部**权重**、可 import**稳定不变**的第三方库，但会改动的算法一律**在
> `` 内自实现/vendored**。所以下面很多"借鉴"是*照搬思路、自己写代码*，而非直接依赖它们的仓库。

---

## Part A — 参考的工作与借鉴点

### A.1 自动打标框架（auto-labeling frameworks）

| 工作 | 是做什么的 | 我们借鉴了什么 |
|---|---|---|
| **Grounded-SAM 2** (IDEA-Research) | 文本→框（GroundingDINO）→ 掩码（SAM2）→ 视频里传播/跟踪。开放词表自动打标的事实标准组合。 | 我们的 `gdino_track` 后端就是它的前半段："roster 短语 → GroundingDINO 逐 shot 检测 → 跟踪"；`sam3_track` 后端对应它的"→ 掩码/传播"后半段。整个 detect→(segment)→track 组合直接照搬其思路。 |
| **Autodistill** (Roboflow) | 用基础大模型（foundation）自动打标，再蒸馏出一个小的专用模型。核心哲学："大模型只用来 bootstrap 标签"。 | 借鉴"大模型省着用"的哲学：VLM 只做 roster 发现 + 每实体命名，**身份由确定性检测/跟踪/re-ID 决定**，把 VLM 调用从 `chunks×rounds×branches×(discover+draft+verify)` 砍到 `1(roster)+N(命名)+chunks(draft)`。我们不蒸馏 student，但"少用大模型"的动机一致。 |
| **Roboflow (supervision)** | 开源 `supervision` 库：检测框集合、ByteTrack 封装、标注可视化；配套的自动标注 SaaS。 | 借鉴其"置信度阈值化的检测框集合 + ByteTrack 关联"范式：我们的 `grounding.detect_all`（返回阈值以上全部框，而非只取最优）+ `tracking.py` 的 IoU 关联。 |
| **FiftyOne** (Voxel51) | 数据集策展/可视化：embedding 去重、离群点(mistakenness)发现、样本挑选。 | 借鉴 embedding 驱动的**去重 + 错标发现**：`roster.py` 的关键帧 embedding 去重/最远点采样；`crop_classify.py` 的原型审计（把"更像另一个实体"的 crop 标为可疑）就是 mistakenness 思路。 |
| **CVAT / Label Studio** | 人机协同标注 UI：逐帧/逐 track 标注、审核、插值。 | 借鉴"人审是最终闸门"的范式：`review.html` 只展示**被 flag 的项 + 随机抽检样本**，人力随残余误差而非语料规模增长；`human_reviewed` 冻结门。web dashboard 的 track/实体审核 UX 也受其启发。 |

### A.2 多目标跟踪（MOT）与重识别（ReID）研究

| 工作 | 是做什么的（简答 item 2） | 我们借鉴了什么 |
|---|---|---|
| **ByteTrack** (Zhang et al., 2022) | **单镜头内**的多目标跟踪：每帧检测器给出一堆框，ByteTrack 用卡尔曼预测 + IoU 把**跨帧**的框连成一条条轨迹(tracklet)，给每个目标一个稳定的**局部 ID**。它的关键点是"连低分框也拿来二次匹配"，减少漏跟。 | `tracking.py` 的 shot 内 tracklet 关联：IoU 匹配 + 轨迹延续（`max_miss` 容忍短暂遮挡）。这一步产出"同一 shot 里这是同一个体"的确定性局部身份。 |
| **BoT-SORT** | ByteTrack 的增强版：在 IoU/运动之外，加入**相机运动补偿**和**外观 embedding(ReID)** 一起做关联，遮挡/相机晃动下更稳。 | 借鉴"运动 + 外观融合关联"：我们在 IoU 之上加了 `track_appearance_gate`（DINOv3 crop embedding 相似度门控），避免把长得完全不同的两个体错连成一条轨迹。 |
| **Person ReID: LaST (2105.15076) / MARS** | **跨镜头/跨相机**重识别：不同镜头拍到的人，用外观 embedding + 聚类/度量学习判断是不是同一个人。LaST 从 2k+ 部电影用半自动工具 PLabel（检测框→人工分配 ID）构建。 | `reid.py` 的**跨 shot re-ID** 就是这件事：把每条 tracklet 的平均 embedding 去和全局 registry 的实体原型比余弦，超阈值就合并成同一 `entity_id`；用 static-attribute 冲突化解 over-merge。"电影里做 ReID + 半自动+人工修"直接对标 LaST。 |

> 一句话总结 item 2：**ByteTrack/BoT-SORT 是"单镜头内把逐帧的框连成轨迹、给稳定局部 ID"的跟踪算法**；
> 跨镜头把这些局部 ID 认成同一角色是 **ReID** 干的活。我们把这两步拆开：shot 内用 tracking，shot 间用 re-ID。

### A.3 感知 / 对齐 / 人脸模型（作为权重加载，非代码耦合）

| 模型 | 类型 | 用途 |
|---|---|---|
| **GroundingDINO** | 开放词表检测（图像+文本→框） | 逐 shot 用 roster 短语定位角色/道具。 |
| **SAM2 / SAM3(3.1) / Sa2VA** | 可提示分割 + 视频掩码传播 | `sam3_track` 备选后端：掩码→bbox、跨帧传播身份（消融 route B）。 |
| **DINOv3** (Meta) | 自监督**图像**embedding | crop/帧 embedding：re-ID 原型、关键帧去重、crop-QA。**纯图像-图像**。 |
| **SigLIP2 / CLIP / OpenCLIP / jina-clip-v2** | **图文对齐** | `crop_classify` 的可选 SigLIP2 零样本"crop 语义是否匹配文本短语"检查。**图像-文本**，与 DINO 不可混用（见 `crop_classify.py` 头注）。 |
| **InsightFace (buffalo_l = RetinaFace + ArcFace)** | 人脸检测 + 512d 嵌入 | `face.py` 自门控人脸线索：检测器"检到脸"本身就是门，检到才把 ArcFace 向量并入 re-ID 融合，无需单独的"是不是脸"分类器。 |

---

## Part B — 2026 最新调研：有没有现成的"长视频角色一致性 gold 自动构建"方案？

**结论（TL;DR）：没有可直接拿来即用的成套方案，但我们采用的 track-first 配方，正是 2026 年学术界/工业界
独立收敛出的共识。** 现有工作分两类，都不是"给因果流式记忆 benchmark 造 gold"：

### B.1 为"训练"身份保持生成模型而造数据的 pipeline

| 工作 (2026) | 它的自动构建 pipeline | 和我们的关系 |
|---|---|---|
| **Memento** (arXiv 2606.14667) | Qwen3-VL 生成全局故事 caption + **无代词固定 subject 清单**；**ByteTrack** 跟踪主体；挑"目标占画面最大"的 2 帧做重建目标。 | 与我们的 roster(固定 ID)+track+best-crop **几乎一模一样**。差别：它造的是训练用重建目标，不是 benchmark gold。 |
| **Gloria** (arXiv 2603.29931) | 角色中心 anchor 帧抽取：global 随机采样、viewpoint 用 GVHMR 估姿态、expression 用 EmotiEffLib+**Gemini judge** 过滤。 | 借鉴：多视角 anchor 选择 + MLLM judge 过滤。仍是生成训练数据。 |
| **AnimeShooter** (arXiv 2506.03126) | 动画多镜头数据集：Gemini 生成 story-level+shot-level 层级标注；**Sa2VA 按角色 ID 分割**；InternVL 质量过滤。 | 与我们的层级(roster→shot)+按 ID 分割高度一致。聚焦动画、面向生成。 |
| **OmniHuman** (arXiv 2604.18326) | 人物中心：**身份连贯跟踪** + 每身份 bbox 元数据 + DWPose 全身姿态。 | 借鉴：identity-coherent tracking + per-identity 元数据结构。面向人物生成/驱动。 |
| **ConsisID** (PKU-YuanGroup, data_preprocess) | YOLO 取 face/head/person 框 → SAM-2 掩码 → Qwen-VL caption，**支持多 ID 标注**。 | 佐证：detect→segment→caption 的多 ID 流水线是 2026 标准件；我们的组件选型主流。 |

### B.2 面向"视频理解/评测"的实体跟踪 benchmark

| 工作 | 它是什么 | 和我们的关系 |
|---|---|---|
| **NarrativeTrack** (apple/ml-NarrativeTrack) | 首个"细粒度实体跟踪"视频理解 benchmark，**全自动标注**（detect+track → Gemini 标注/过滤 → QA），自动 QA ~70% 正确，人工验证子集。 | **精神上最接近**（实体中心 + 全自动标注 + 人审）。但它评的是 VideoLLM 的 QA 理解能力，不是"记忆/上下文管理"，没有 oracle passthrough、没有 lifecycle/state。 |
| **LaST** (arXiv 2105.15076) | 大规模电影 person-ReID，半自动 PLabel（检测框→分配 ID→人工修）。 | 佐证"电影 ReID + 半自动+人工修"的做法。纯 ReID 数据集，无生成/记忆语义。 |

### B.3 为什么没有"正好我们要的那一套"

上述工作都缺我们 benchmark 的**独有目标**（见 `vmem_bench_design_principles.md`）：

- **因果流式 + Generator-Oracle passthrough**：评的是 SUT 的**主动记忆/上下文组合**，GT chunk 直接回灌，
  没人把 gold 构建成这种流式契约。
- **生命周期/状态事件 + presence 确定性**：每 chunk 谁在场、谁首现、什么状态变了，用 tracklet∩chunk-span
  **确定性**算出来（`pipeline_track_first.presence_for_chunks`），而非 MLLM 判断。
- **面向 gold 而非训练数据**：别人产的是 caption/anchor/重建对去*训练*生成器；我们产的是可被多个 baseline
  流式消费、可打分的**评测 gold**。

**因此我们的定位**：**复用**他们验证过的组件（ByteTrack/BoT-SORT、SAM2/3/Sa2VA、GroundingDINO/YOLO、
DINOv3、SigLIP、ArcFace、DreamSim/CLIP 度量），但"如何拼装成一套流式记忆 benchmark 的 gold 构建器"
是我们的贡献。track-first 重构与 2026 主流不谋而合，反过来验证了这条路线是对的。

---

## 引用（arXiv / repo）

- Memento — Reconstruct to Remember for Consistent Long Video Generation, arXiv 2606.14667
- Gloria — Consistent Character Video Generation via Content Anchors, arXiv 2603.29931
- AnimeShooter — Multi-Shot Animation Dataset for Reference-Guided Video Generation, arXiv 2506.03126
- OmniHuman — Large-scale Dataset and Benchmark for Human-Centric Video Generation, arXiv 2604.18326
- NarrativeTrack — apple/ml-NarrativeTrack (GitHub)
- ConsisID — PKU-YuanGroup/ConsisID (GitHub, data_preprocess)
- LaST — Large-Scale Spatio-Temporal Person Re-identification, arXiv 2105.15076
- ByteTrack — Zhang et al., ECCV 2022 (arXiv 2110.06864)
- BoT-SORT — Aharon et al., arXiv 2206.14651
- Grounded-SAM 2 / GroundingDINO / SAM 2 — IDEA-Research (GitHub)
- Autodistill / supervision — Roboflow (GitHub)
- FiftyOne — Voxel51 (GitHub)
- CVAT (GitHub) / Label Studio — HumanSignal (GitHub)
- DINOv3 — Meta AI; SigLIP2 — Google; InsightFace (buffalo_l) — deepinsight (GitHub)
