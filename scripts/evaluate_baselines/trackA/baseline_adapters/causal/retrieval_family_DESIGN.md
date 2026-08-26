# 自建检索 baseline 家族（新协议重写）· 设计规范

> **治理定位（红线）**：本文件下的 `seg_uniform` / `seg_dinokey` /
> `seg_framererank` / `frame_text` 是自研检索 **controls/ablations**，实现名必须带
> `_ablation` 后缀；`recency` / `bm25_desc` / `random` 是 `_ctrl`
> 对照。它们只能用于消融和诊断，**不能进入正式 baseline 表**，避免把代理检索当作方法行。
> 正文 baseline 表仍只放外部/既有方法；本族用于解释粒度、预算、描述输入和粗到细检索因素。

> 术语统一：一律用 **segment**（金标准单位），不再用 chunk。
> 协议：SUT 只拿到 **segment 的 prompt + 该 segment 的真实视频**；自建记忆、自己检索、输出**带时序身份的证据**
> （`source_seconds`），由 `frame_materializer` 落成参考图给 VLM 打分。旧的 `retrieval_baselines.py`
> （吃 gold entity/crop、按实体 top-k）**作废**，按本规范在 `baseline_adapters/causal/` 重写为因果 adapter。

## 记忆空间（四个变体共用）

每观测一个 segment，存：
- **segment 文本嵌入**：对该 segment 的 prompt 文本用文本编码器编码（text-text 检索用）；
- **segment 帧嵌入**：对该 segment 均匀抽的若干帧编码（跨模态文本→帧检索用 SigLIP2；DINO 关键帧用 DINOv3）；
- 每帧记 `source_seconds`（时序身份，用于 materialize）。

## 四个变体（= 检索粒度 × 抽帧策略）

| 变体名(建议) | 逻辑 | 抽帧 |
|---|---|---|
| `seg_uniform` | text→segment 文本检索，取 top-k segment | **均匀抽帧**：直接对选中 segment 视频按 `fps∈{0.5,1,2}` 抽（消融） |
| `seg_dinokey` | text→segment 文本检索，取 top-k segment | **DINO 关键帧**：段内密采后按 DINOv3 相似度做多样性挑选（贪心最远点/聚类），取代表关键帧 |
| `seg_framererank` | **粗→细**：先 text→segment 选 top-k segment，再在其内 text→frame 跨模态精检索抽帧 | 文本-帧 rerank |
| `frame_text` | text→frame 直接跨模态检索（对**所有**已存帧），取 top-k 帧 | 直接帧检索 |

（对应你给的：`text-segment + uniform` / `text-segment + dino keyframe` / `text-segment + text-frame` / `text-frame`。）

## 抽帧参数（可配置，做消融）

- **均匀抽帧用 fps**，不是固定 K：`RETR_UNIFORM_FPS ∈ {0.5, 1, 2}`（默认 1），即直接 `ffmpeg -vf fps=` 抽，
  避免固定 K 太少丢信息、对本方案不公平。
- `seg_dinokey`：段内先 fps=2 密采 → DINOv3 编码 → 贪心最远点选 `RETR_KEY_PER_SEG`（默认 3）关键帧。
- top-k segment：`RETR_TOPK_SEG`（默认 5）；`frame_text`/`seg_framererank` 的 top-k 帧：`RETR_TOPK_FRAME`（默认 8）。
- **总预算上限** `RETR_BUDGET`（默认 16 帧/段）保证与其它 baseline 的 budget 可比（budget 影响 precision/sel_eff）。
- 编码器按相似度空间分开：text→segment 的文本-文本召回可用
  `Qwen3-Embedding-4B`；text→frame 必须使用 `SigLIP2` 的共享 text/image
  tower；关键帧多样性使用 `DINOv3 ViT-B/16`（与打分侧 redun_sim 一致）。

## 落地位置

核心 skill 在 `src/memstrata/skills/memory_retrieval/`，与 `composition/` 平级：

- `RetrievedRef.source_seconds` 直接对齐 causal bench 的 `RetrievedItem.source_seconds`；
- `Retriever.retrieve(query, *, as_of_seconds, budget)` 是更通用的帧/段级协议；
- 不把相似检索塞进 `composition`，保持 MemStrata 公理 6：Compose 只做确定性
  model-free 解引用，不做全库检索。

bench 侧只有薄适配器 `baseline_adapters/causal/retrieval_family.py`，负责把 `RetrievedRef`
转成 `RetrievedItem`，并由同一个 `runner.py` + `frame_materializer.py` 落帧。

## runner 统一策略

- `--budget B`：runner 按 adapter 返回的 `score` 排序裁剪到 top-B；支持
  `B∈{1,2,4,8,16}`，主表固定单一 B，`--budget-sweep` 用于消融。
- RRF 融合：`seg_framererank_ablation` 可返回粗召回 rank 与帧精排 rank，
  runner 用 Reciprocal Rank Fusion（`k=60`，只用 rank，不做加权和或跨模型分数标定）
  得到最终候选，再执行同一 budget 裁剪。
- `--input-mode description_only`：只对 prompt 中已经出现的注册名做确定性替换，
  输出中性指代 + 外观描述；没有 description 时跳过/回退到中性化 prompt。该档不泄露
  `present` / roster，规则对所有系统一致。它依赖写路径把 asset/rep description 补齐；
  description 为空时结果会退化，这是预期诊断信号。

## encoder provider

`src/memstrata/encoders/base.py` 新增：

- `siglip2`：共享 text/image 空间，用于 text→frame，不能和 Qwen 文本向量直接做帧相似度；
- `qwen3_embedding`：text→text（例如 text→segment 粗召回）；若设置 `MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT`，
  走 OpenAI-compatible `/embeddings` server，否则按 `PUBLIC_MODELS_ROOT` 解析本地权重；
- 默认 `hash` provider 保持无依赖冒烟能力，重模型只在构造 provider 时懒加载。
