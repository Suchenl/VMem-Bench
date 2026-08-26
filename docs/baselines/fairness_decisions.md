# Baseline 公平对比：最终决定（2026-07-22 固化）

> 本文是"如何跑 baseline 才算最公平对比"的**权威决定记录**。跑 / 改 / 评 baseline 前先读。
> 实现层面的 Track A 定义见 [`track_a.md`](track_a.md)；具体实验矩阵见
> [`../experiments/fairness_experiment_plan.md`](../experiments/fairness_experiment_plan.md)。
> 术语以 [`glossary.md`](../../../../methods/MemStrata/docs/glossary.md) 为准。

## 背景

对 `benchmarks/MemStrata` 跑 baseline 的逻辑做过一次公平性审查，发现若干问题并逐条讨论确认。
核心工程是干净的（所有系统走同一 `score_run()` harness、同一份 frozen gold、同一套
prompt/observation packet、同一个 pinned scoring embedder；frame→entity 投影统一用 frozen gold
并硬拒 future/current leakage）。以下是需要落地的**公平性决定**。

## 决定

### D1. Prompt 含实体规范名 = 刻意设计；两种给名设定（name-anchored / description-only）

- Prompt 里字面写实体规范名是**刻意设计**，且所有系统都通过 observation packet 拿到同样的
  name（首现附 description）。因此 MemStrata 靠"更会用名字"赢是**正当结果**，不是泄露。
  （检索 baseline 也累积 name+description 记忆，差异是"精确匹配 vs 语义匹配"的范式差异。）
- **name-anchored = 主设定 / 真实设定**：主表报这个。
- **description-only = 鲁棒性压力测试**（附录），对所有系统一视同仁：把 name
  从 observation **和** prompt 一并去掉/改述，只留 description。
  - 关键：MemStrata 在 description-only 下 **不走** `no_name_anchor` 退化路径（那会掉到 recency，冤枉方法），
    而是走它的**语义/描述匹配**路径——依旧是"匹配式主动组合"，只是匹配 key 从 name 变成 description。
  - 目的：证明 MemStrata 的优势不只来自名字这条易路，在描述这条难路上也成立。

### D2. VisualFidelity 用按实体类型路由的多 embedder（不用 DINOv2/CLIP）

pinned scoring embedder 由 bench 拥有、对所有系统 byte-for-byte 一致。改为**按 entity kind 路由**：

| entity kind | embedder | 说明 |
|---|---|---|
| 全类通用 | **DINOv3**（现有）+ **SigLIP2**（新增） | 通用视觉 + 视觉-语言语义，两条正交轴，各报一列 VF |
| character（**仅 LSMDC 真人**） | **ArcFace / InsightFace** | 人脸细粒度身份；动画片不算此列（可按真人片/动画片分组报） |
| location | **MegaLoc**（VPR） | 地点一致性。→ location **不再一律从视觉评分剔除**，改为单列地点一致性轴 |

- **不新增** DINOv2 / CLIP（与 DINOv3 / SigLIP2 冗余）。
- MegaLoc 复用 `models/model_weights/.../MegaLoc/model.safetensors`（按路径加载，不跨包 import；
  参考 `src/memstrata/encoders/place/vpr.py` 的实现，在 `vmem_bench` 侧独立镜像一份）。

### D3. 检索 baseline 报多档 k

`text/frame/fusion` 检索 baseline 主表报 **k ∈ {1, 3, 5, budget-matched（按每 chunk 平均 present 数）}**
多行，兑现"budget-matched"承诺；有时间再扫全 k 出曲线。现状是 env 默认写死 `top_k=5`。

### D4. 全影片 × 全 baseline（原则），靠工程提速而非牺牲忠实度

- **原则**：每部影片都跑所有 baseline。
- 因果对手（memflow / iamflow / longlive_rag）的记忆决策由其模型内部 KV/attention/entity-score
  决定，**机制 forward 不能省**（省了就是被踢出主表的 `*_budget_proxy`，不忠实）。
- 但"11 小时"是**串行错觉**，trace 生成逐片独立、天然可并行。提速杠杆（全部忠实）：
  1. **跨影片 × GPU 并行**：走 `scripts/tgpu_fs.py` 共享文件系统队列；墙钟 ≈ 最慢单片。
  2. **共享一次性 GT VAE encode**：memflow/iamflow 已共用 `memflow_latents.pt`，longlive_rag 统一到只 encode 一次/片。
  3. **iamflow 用现成 vLLM 环境**服务 Qwen3-4B(LLM)+Qwen3-VL-2B(VLM)（同权重，忠实提速）；跑不通再装。
  4. **缓存确定性中间量**（VLM per-block 视觉分、DiT per-layer KV 指纹）→ 后续扫参近乎免费。
- **禁用**：truncate 历史、蒸馏 / 更小模型、name/budget 代理。

### D5. scripted / agentic 系统移出定量主表

- 因果（无全局计划、逐 chunk 自回归）：`helios / longlive_rag / memflow / iamflow`
  与论文 setting 一致 → **可对比，进主表**（`CAUSAL_WITH_TRACE`：后三者各需每影片 GT trace）。
- **`decmem` 排除出定量主表**（非"能不能跑得快"的问题，是任务模态不匹配）：
  1. 硬件：LTM 的 block-sparse kernel 要 **sm_90a**（H100/H200/**H800**），A800(SM80) 不行——
     这一条 H800 节点能解决；
  2. **输入模态（更根本、H800 也解决不了）**：DecMem 的 Sparse Global Memory/LTM 以 **WorldMem
     控制流**（配对 MP4 + NPZ 动作/位姿轨迹）为条件；被动电影没有 action/pose track，其既定
     conditioning **无法构造**。伪造轨迹=不忠实，关 LTM 走 dense=删掉被测机制本身。
  → 因此 decmem 不进主表，`DecMemAdapter` 直接 `NotImplementedError`，仅保留 `decmem_budget_proxy`
    作诊断消融（明确非方法行）。若将来做 action-conditioned 的 benchmark 分支再单独评。
- 全局计划 / 脚本化 / agentic：`ViMax / MovieAgent / VideoMemory / StoryMem / Memento / MM-StoryAgent`
  **非因果**（先出整段计划再渲染，能看到未来）→ **移出定量主表**，仅在附录做"非因果 agentic 系统"
  定性说明。代码里保留 `external/scripted/` converter，但不进 leaderboard。

### D6. SUT 记忆像素通道：开通，但评分口径不变

- `RetrievedItem` 开通可选 `image_path` 字段，允许图像原生 SUT 把自己 compose 出的参考图交给
  bench；latent / KV / 纯时间戳系统仍走时间戳回切整帧路径。
- 护栏保持硬约束：自带图仍必须带 `source_seconds` 且严格早于当前 chunk 起点；路径必须存在、
  不含 `/gold/` 段，并位于本 run 工作区内，否则丢弃并记录 provenance 失败。
- 采用口径 a：不改 `visual_coverage` 指标语义，参考图继续作为 UNLABELED 图像判断其中可见哪些
  gold 实体；crop 和整帧进入同一集合口径。
- 当前只让 `memstrata` 适配器发出自身 rep crop 的 `image_path`。检索 / latent 基线保持不变
  (`image_path=None`)，继续由 materializer 按时间戳回切整帧；若未来某个图像原生 baseline
  要切换，必须在同一契约和护栏下显式记录。

## 由此暴露的、需要一起修的运行逻辑问题

- 两个 orchestrator baseline 名单不一致：`run_bbb_track_a.py` 不跑检索 baseline；
  `run_movie_benchmark.py` 默认 `--visual` 关闭。→ 统一 baseline 名单，多影片默认走 visual。
- 非 visual 的 ID headline 里 Fidelity 对所有 baseline 结构性为 0（占权重 0.2）→ 只有 visual
  composite（VisualFidelity 换掉 ID-Fidelity）才公平 → 全线走 visual。
