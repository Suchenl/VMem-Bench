# MemStrata-Bench 运行手册（端到端评测流程）

> 状态：**当前权威运行手册**。评分公式/数据形态见 [`scoring_v2.md`](scoring_v2.md)。
> 本文讲**怎么把一部影片从 S4 金标准跑到分数**，以及**这套评测到底在测什么、什么绝对不能做**。
> 任何要碰本 benchmark 的人或 agent，**先读 §0 三条铁律 + §1 SUT 契约**，再看后面的操作。

---

## 0. 三条铁律（硬约束，违反即评测无效，不要再问第二遍）

1. **金标准 = S4 人核标注，且只有文本。**
   gold 一律由 [`build_gold_from_s4_review.py`](../../scripts/vmem_bench/maintenance/build_gold_from_s4_review.py)
   从 `tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json` 转出。
   gold **只含文本**（roster + 逐 chunk present / first_appearances / prompt / seconds_span），
   **不含任何 crop 像素**（representation 的 `crop_path` 一律为空，这是刻意的）。
   **禁止**使用任何非 S4 来源的旧 gold。

2. **参考图（context）永远是 SUT 自己产的，bench 不发图。**
   SUT 通过**观察真实 chunk 视频**、自己做感知/分解，把**自己的 crop** 存进**自己的记忆库**。
   打分打的就是这些 SUT 自产图。**禁止**把 gold crop（或任何 bench 提供的像素）当作 SUT 的记忆/context——
   那是已废弃的错误协议（见 §5 的警告）。

3. **SUT 只能看到 prompt + 视频，永远看不到"答案"。**
   每个 chunk，bench 只给 SUT：① 该 chunk 的 **prompt 文本**；② 该 chunk 的**真实视频片段**。
   **prompt = S4 人核过的剧本 `action`（+ 音效）原文，逐字照搬**，实体只按剧本自然叙述被提到（谁在场靠 prose 自然点名，
   这正是 name-anchor 的来源）。**绝不注入** `Canonical entities in this chunk: ...` 这类把 present 全名单显式列出的后缀
   （已于 2026-07-24 连同 `ensure_prompt_entity_coverage` 一并删除；prompt 完整性只"度量"不"改写"，见 §4）。
   **绝对禁止**把 gold 的 `present` / `first_appearances` / roster 的 entity_id 列表喂给 SUT——
   那等于把"这段该出现谁"的答案直接告诉被测系统。"该有谁"只在 **bench 打分侧**使用，SUT 侧不可见。

> 这三条是本 benchmark 的立身之本。gold=S4 文本、图 SUT 自产、SUT 不看答案——三者缺一，得到的分数没有意义。

**输入档（公平性轴，两档都要跑、并列报告）**——`bench_adapters/causal/runner.py --input-mode`：
- `name_anchored`（默认，主设定）：prompt = S4 剧本 prose 原文，实体按叙述自然点名。
- `description_provided`：在 name-anchored 之上，用确定性规则在 prompt 尾部追加"该实体长什么样"的
  外观描述，**只描述该 prompt 里已被点名、且属于身份实体（`kind==character`）的对象**（不泄漏 present/roster）。
  **道具（prop）/场景（location）不描述**——给"开阔草地""苹果"加外观对实体检索是纯噪声，还会稀释靠文本
  saliency 的 baseline（MemFlow）的角色信号、把 prompt 顶过 umT5 的 512-token 上限，这正是早期"啥都描述"
  导致 descprov 反而比 name_anchored 更差的原因（2026-07-25 修正为仅描述 character）。给靠外观/描述匹配
  记忆的系统一个公平的文本抓手，避免只测 name_anchored 偏袒靠名字检索的系统。输出名 `<adapter>__descprov`，
  两档并存分别打分。细节与报告方式见 [`../experiments/fairness_experiment_plan.md`](../experiments/fairness_experiment_plan.md)。

---

## 1. SUT 契约（一张表看清边界）

**SUT = 任何消费本 benchmark 的系统**（`memstrata` 是其中一个 SUT，与所有 baseline 在同一套 S4 gold 上平等受测；
"SUT" 不特指 memstrata）。逐 chunk 按时间序驱动：

> **双评测口径（重要，别再误解）**：`memstrata` 与所有 baseline（longlive_rag / memflow /
> memflow_sma / iamflow / slotmem …）**地位完全对等**，都只是 SUT。它们会在**两套评测**下被衡量：
> ① **本记忆 benchmark**（当前文档）——把真实 segment 替代生成器产出、只考"记忆机制"，产出
> precision/recall/F1/redundancy/selection_efficiency + 时间效率指标；
> ② **真实生产管道**（`methods/MemStrata/production/`，`montage produce`）——把同一个 SUT 接进
> 端到端"剧本→生成→回看建记忆→再生成"的闭环，考它在有生成器噪声、真实长片下的**产出质量与稳定性**。
> 因此 memstrata **不是**"裁判/bench 侧"，baseline 也**不是**只在 bench 里出现；两套评测里它们都是被测对象，
> 接口一致（prompt+视频进、记忆/context 出）。写代码/文档时不要把 memstrata 特殊化成 bench 的一部分。

| 环节 | bench → SUT（输入） | SUT → bench（输出） | SUT 绝不可见 |
|---|---|---|---|
| 组合 context | 该 chunk 的 **prompt 文本** | 一组它**自己存的参考图**（context） | `present` / `first_appearances` / roster ids |
| 观察记忆 | 该 chunk 的**真实视频片段** | （无，仅更新自己的记忆库） | gold crop / 任何 bench 像素 |

- 时序纪律：先 `handle_prompt`（用**当前已建**的记忆 compose 出 context），**再** `handle_observation`
  （把本 chunk 视频交给 SUT 更新记忆）。SUT 在为 chunk t 组合 context 时，**不能**先看到 chunk t 的视频观察。
- prompt 里带实体名字是**允许**的（这正是"点名唤起记忆"的机制）；带结构化的 present/first_appearances **不允许**。
- **所有 SUT（含全部 baseline：helios / retrieval / causal 等）一视同仁**：都只用 prompt + 真实视频自建记忆、自产图，
  **没有任何系统能走 gold-crop 回放**。
- **bench 只交两样东西给 SUT，且绝不发图**：① 该 chunk 的 **prompt 文本**；② 该 chunk 的**原始 segment 视频**。
  这段 segment 是**用来替代 SUT 生成器的产出**——真实系统里 SUT 会先按 context 生成一段视频再回看建记忆，
  这里直接把真实 segment 塞给它，**消除生成器噪声**，让评测只聚焦"记忆机制"。
  bench **不做任何感知、不产任何 crop、不下发任何图像或 ID 答案**。
- **感知 + 记忆一律在"方法侧（SUT）"，不在 bench**：SUT 收到 segment 后，自己决定怎么把它变成记忆——
  检测 / 抠图 / 编码 / 存储 / 召回全由该 SUT（baseline）的 adapter 完成。多个 baseline 可**共用一套方法侧感知前端**
  （同样的 detect→crop→embed），这属于"方法侧工具"，与"bench 发图"是两码事；它们真正的差别只在**记忆 / 选择策略**。
- **adapter 不得替 SUT 维护或补写记忆/检索结果**：每个 SUT 的记忆读写必须来自它自己的原生机制
  （例如 MemStrata 的 `AssetBank`/`MemoryUpdater`/`IntentInterpreter+compose`，SlotMem 的
  `RoleWiseSlotMemoryBank`，MemFlow 的 KV bank/SMA routing，IAMFlow 的 `agent_memory_bank`，
  LongLive-RAG 的 latent descriptor pool）。bench adapter 只允许做两类胶水：
  ① 按本协议把 `prompt -> compose`、`real segment -> observe/write` 串起来；
  ② 把 SUT 内部已经选择/保留的 memory 表示**投影成带 `source_seconds` 的 temporal refs**，供 bench
  从真实视频物化参考帧。**禁止**在 adapter 里新增"如果 SUT 没召回就补最近帧/补 gold/补 roster"这类
  fallback；真实 SUT 返回空就记录为空，分数自然反映该 read path 的能力。
- 打分（Stage 2）只看 context 这组 SUT 自产图：是否覆盖了 continuity 实体、有没有乱召回、冗不冗余；
  "该有谁"来自 S4 gold 文本，只在 bench 侧用。

---

## 2. 术语

- **frozen gold（= S4 人核）**：`<movie>/gold/`，`human_reviewed=true`，只含文本 GT，评测只读。
- **context / visual selection**：SUT 为每个 chunk 从**自己建的记忆库**里组合出的一组**自产**参考图。
- **continuity 实体**：`present \ first_appearances`，即"之前见过、这次该靠记忆调回"的实体；recall 只在它们上算。

---

## 3. 三阶段总览

| 阶段 | 干什么 | 需要 GPU/VLM？ | 产物 |
|---|---|---|---|
| **Stage 0** | 从 S4 人核标注转出 `gold/`（文本 GT） | 否 | `<movie>/gold/{chunk_index,entity_registry,chunk_annotations}.json` |
| **Stage 1** | 每个 SUT 观察真实 chunk 视频、**自建记忆库**、逐 chunk 产出**自产** context | **是**（感知/分解：GroundingDINO/SAM3/DINOv3 等） | `outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/<system>.json`（指向 SUT 自存 crop/temporal refs） |
| **Stage 2** | VLM 视觉覆盖打分 | **是**（`qwen3-vl-32b` judge；DINOv3 仅用于 `redundancy_sim` 诊断列） | `outputs/evaluation/trackA/<system>/<dataset>/<movie>/_visual_score/<system>/score.json` |

---

## 3.1 分辨率约定（统一预处理到 480p / 832×480）

**规矩：所有喂进模型/judge 的视频像素统一预处理到 832 宽 × 480 高**（Wan2.1-T2V-1.3B 原生尺寸，16:9），
不跑源分辨率（源片常是 720p/1080p，又慢又贵，且判分不需要）。三处强制统一，均已落到代码：

| 环节 | 位置 | 处理 |
|---|---|---|
| Stage 1 段落切割 | `baselines/bench_adapters/causal/runner.py:_cut_segment` | ffmpeg `scale=832:480` 落盘 |
| Stage 1 SUT 观测解码 | `baselines/bench_adapters/causal/_video_io.py`（`WAN_W=832,WAN_H=480`） | 解码即缩放到 832×480 |
| Stage 1 参考帧抽取 | `baselines/bench_adapters/causal/frame_materializer.py:_cut_frame` | ffmpeg `scale=832:480` |
| Stage 2 judge 视频 clip | `src/vmem_bench/scoring/visual_coverage.py:_cut_clip`（`JUDGE_CLIP_W/H`） | ffmpeg `scale=832:480` |
| Stage 2 参考图内嵌 | 同上 `_img`（`JUDGE_IMG_MAX_SIDE=384`） | 发送前再降到最长边 384px |

**为什么是 480p 且公平**：SUT 全程只在 480p 上感知，judge 在同一分辨率上判分既公平又快；judge 参考图进一步压到
384px 以控 token（大足迹系统一 chunk 可达 40+ 图）。发布版固定这些常量，不要临时调大。

> 已存在的旧产物（`_segments/*.mp4`、`_ref_frames/*`、`_clips/*.mp4`）因有 `if out.is_file()` 缓存**不会自动重切**；
> 想让旧片享受 480p 提速/一致性，删掉对应缓存目录后重跑即可（解码端 `read_segment_pixels` 本就再缩放到 832×480，
> 张量不变，仅影响速度/落盘）。

---

## 4. Stage 0 — 从 S4 人核标注转出 gold（文本 GT）

```bash
cd benchmarks/MemStrata
PYTHONPATH=src $PY scripts/vmem_bench/maintenance/build_gold_from_s4_review.py \
    --movie-dir data/BlenderOpenMovies/big_buck_bunny
```

读 `tmp/pipeline/s4_segment_sampling_human_review/human_revised_annotation.json`，
转成标准 3 文件 gold（`chunk_annotations.json` 带 present/first_appearances/prompt/seconds_span，
所有 representation 的 `crop_path` 为空）。**S4 gold 就是最终要发布的金标准。**

- **三文件各司其职、不重复**：`chunk_index.json` = 薄布局（chunk_id + shot/frame/seconds spans + `layout_hash`）；
  `chunk_annotations.json` = 逐 chunk 富 GT（present/first_appearances/prompt/...，**scorer 从这里读**）；
  `entity_registry.json` = 实体 roster。三者分别被 scorer / harness+freeze+publish / roster 读取，缺一不可。
  （2026-07-24 起 `chunk_index` 不再冗余承载富标注。）
- **prompt = S4 剧本 `action`（+ 音效）原文**，转出时**不再注入任何 canonical-entity 后缀**
  （`ensure_prompt_entity_coverage` 已删除，见铁律 3）。gold 里字段名统一为 `prompt`，不保留 `action` 字样。
- **gold 是最小包含**：每部只有上述三个 JSON，无 crop 像素、无 embedding、无 tmp。

## 5. Stage 1 — 每个 SUT 自建记忆库、产出自产 context（需要感知服务）

**本 benchmark 的核心，也是最容易做错的一步。** 正确协议（严格遵守 §0 / §1）：bench 逐 chunk 按时间序做两件事，
其余全在方法侧：

1. **把 prompt 交给 SUT** → SUT 用**当前已建记忆** compose 出 context 并存下（`handle_prompt`）。这份 context 就是打分对象。
2. **把该 chunk 的原始 segment 视频交给 SUT**（替代生成器产出，消除生成噪声）→ SUT 的 adapter 自己对这段真实视频
   做感知 + 记忆（detect→crop→embed→存储，`handle_observation`），更新自己的记忆库，供后续 chunk 召回。

`visual_selections/<system>.json` 里的 `crop_abspath` **必须来自 SUT 自己的 memory 表示经 temporal ref 物化得到的真实帧/图**，
绝不能是 `gold/crops/...`，也不会有任何 bench 下发的图；若某 SUT 内部 memory 不是显式 crop（如 KV、latent、slot），
adapter 只能把它已经保留/召回的 source time 投影成 `source_seconds`，再由 bench 统一切真实参考帧。

- **bench 侧**只负责：解析源视频、按 `seconds_span` 切出每 chunk 的**原始 segment**、驱动上面两步循环、收集并保存
  各 SUT 的 context、最后打分。**bench 不感知、不抠图、不发图。**
- **方法侧（每个 baseline 的 adapter）**负责：感知（detect/crop/embed）+ 记忆机制（存储/去重/召回/生命周期）+ compose。
  多个 baseline 可共用同一套**方法侧感知前端**（保证公平、只烧一次 GPU），它们的差别只在记忆/选择策略——这仍是方法侧工具，
  不是"bench 发图"。

> ⚠️ **不要用 `scripts/vmem_bench/compare/run_movie_benchmark.py` + `BenchReplayAdapter` 当 Stage 1。**
> 那是**已废弃的 gold-crop 回放**：把 gold 观察包里的 crop 直接灌进 SUT 记忆 = bench 发图，违反铁律 2；
> 且它会把 gold 的 present 通过观察包泄露给 SUT，违反铁律 3。在 S4 gold 下观察包 `crop_path` 全空，
> 这条路本就跑不出任何像素。**仅可用于纯 ID 契约的离线自测，不能产出用于打分的 context。**
>
> 正确的"SUT 观察真实 segment → 自建记忆"的 Stage 1 驱动器（切 chunk 视频 + 起 crop-acquisition 服务 +
> 逐 chunk 并行驱动所有 SUT）**尚在接线中**；接好后在此补最终命令、`<system>` 命名映射与产物校验。

## 6. Stage 2 — VLM 视觉覆盖打分

前置依赖（缺一不可）：

1. **VLM judge 常驻**：`qwen3-vl-32b`，OpenAI 兼容 `POST /v1/chat/completions`，默认 `http://127.0.0.1:8110`
   （`--api` 可改）。`temperature=0`、`fps=2.0` 已钉死。**没起服务就跑不了 Stage 2。**
2. **源视频**：按 `data/dataset_dirs.txt` 解析（见 §7）。scorer 用 ffmpeg 现切 chunk 片段。
3. **ffmpeg**：默认 `ffmpeg`（`--ffmpeg` 可改）。
4. **DINOv3**（可选）：给 `redundancy_sim` 列用；torch/权重不可用时该列为 `null`，不阻塞 headline。

```bash
cd benchmarks/MemStrata
# 冒烟：只打前 5 个 chunk 验证链路；全量去掉 --limit
PYTHONPATH=src $PY -m vmem_bench.scoring.visual_coverage \
    --movie  data/BlenderOpenMovies/big_buck_bunny \
    --system <system> \
    --video  ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4 \
    --limit  5
```

`<system>` = `outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/` 下的文件名去掉 `.json`
（每个要对比的 SUT 各跑一次）。产物：
`outputs/evaluation/trackA/<system>/<dataset>/<movie>/_visual_score/<system>/{score.json,details.json}`；
chunk 片段缓存在同一 run 目录的 `_clips/`。

## 7. 源视频路径怎么解析

`data/dataset_dirs.txt` 是数据集根目录清单（登记在 benchmark 内，路径可指向大盘）：

```
BlenderOpenMovies: ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos
LSMDC: ${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched
```

某部影片的视频 = `<该数据集根>/<movie_id>/<video_file>`，例：
`big_buck_bunny` → `.../BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4`。

## 8. 影片级 → 语料级

单片 per-chunk → 影片级均值（`score.json.summary`）；跨语料再对影片取宏平均（见 [`scoring_v2.md`](scoring_v2.md) §4.8）。
发布随附 noise floor（`scoring_v2.md` §5）。

## 9. 常见坑

- **用了非 S4 的旧 gold** → 违反铁律 1。gold 必须由 Stage 0 从 S4 转出。
- **context 里出现 `gold/crops/...`** → 违反铁律 2，图必须 SUT 自产，评测无效。
- **把 present/roster 喂给了 SUT** → 违反铁律 3，等于泄题，评测无效。
- **Stage 2 连不上 8110 / 超时** → `qwen3-vl-32b` judge 没起或不健康。
- **`redundancy_sim` 全 `null`** → 打分机无 torch/DINOv3 权重；不影响 headline。

---

## 附录 A：解释器

本机没有裸 `python`，仓库根 `.venv` 软链当前是坏的。用任一装好依赖的 conda env（如 `vace`，与 scorer 默认 ffmpeg 同 env）：

```bash
PY=python3
# 下文所有 python 都用 $PY；orchestrator 内部用 sys.executable 起子进程，继承同一解释器。
```
