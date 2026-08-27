# TrackA Stage2 评分加速：设计评估报告

## 0. 核心结论（先行）

**慢的主因不是判官模型太大，而是服务并发被 KV cache 锁死。** 这与"换小模型"的直觉相反，也解释了为什么 8B 探针只快 1.77x。

铁证有三条：

1. `start_qwen32_vllm.sh:62` 用 `--max-num-seqs "${MAX_NUM_SEQS:-1}"`，即每个 vLLM 副本一次只跑 **1 条序列**。
2. `ReviewerEndpointPool` 的设计就是「每个 URL 一次一个租约」，并在 docstring 里明确写了它是为 `MAX_NUM_SEQS=1` 副本设计的：

```56:62:src/vmem_bench/annotation/pipeline/stages/s3_segment_auto_review_revise/reviewer_pool.py
class ReviewerEndpointPool:
    """Queue of reviewer base URLs with one-at-a-time leases per URL.

    ``MAX_NUM_SEQS=1`` vLLM replicas should expose one URL each.  Workers block
    on ``acquire`` until a free endpoint is available, so pool size equals the
    maximum number of concurrent segment reviews.
    """
```

3. **最有说服力的一条**：我对已有 run 做了回归 `score_ms ~ a + b·dur + c·n_extra_calls`，得到每一次 `_distinct_views` 子调用的边际成本是 **3.1s（sita）/ 5.2s（Pianist）/ 7.8s（Halloween）**。而这个子调用只发 2-4 张 384px 小图、`max_tokens=64`、实际输出约 10 个 token（`{"distinct": 2}`），真实算力需求不到 0.5 秒。**3-8 秒里绝大部分是排队等租约**，不是计算。

换模型救不了排队，这就是 8B 只有 1.77x 的根本原因。

VRAM 算术验证了 `max_num_seqs=1` 是被迫的，也指出了出路（Qwen3-VL-32B：64 层 / 8 KV head / head_dim 128 → 256 KB per token）：

| 拓扑 | 权重占用 | KV 预算 | 可容纳 10k-token 请求 |
|---|---|---|---|
| TP=1, util 0.85（现状） | 64 GB | 4 GB | **约 1.6 路** ← 所以只能设 1 |
| TP=2, util 0.85 | 32 GB/卡 | 36 GB/卡 | 约 15 路 |
| TP=2, util 0.90 | 32 GB/卡 | 40 GB/卡 | 约 16 路 |

---

## 1. 成本解剖（实测，非估算）

在 `retrieval_frame_text_ablation__B16` 的已有产物上测得：

| 观测项 | 数值 | 含义 |
|---|---|---|
| `_distinct_views` 子调用总数 / 主调用总数 | 380 / 358 ≈ **1.06** | 一半以上的 VLM **请求数**是这个廉价子调用 |
| 子调用边际成本 | 3.1 / 5.2 / 7.8 s | 纯排队损耗，占 sita 平均 19.1s 的约 27% |
| duration 归因份额 | 12%（Pianist）/ 27%（sita）/ 63%（Halloween） | 视频 prefill 目前**不是**主导项 |
| duration 无关常数项 | 5.9-14.1 s | decode + 固定开销主导 |

**一个被我实测排除的假设**：我原本怀疑服务端全量解帧是瓶颈（`--media-io-kwargs num_frames=-1` 会解全部帧，clip 是 24fps/240 帧却只采样 fps=2 用约 20 帧）。实测 832×480 的 191 帧 h264 全解码仅 **0.21s**，降采样到 fps=2 后 0.10s。**所以「预先把 clip 抽成 fps=2」不值得做，省不到 0.1 秒。** 记录此负结果以免浪费工程量。

真正的常数项来自 batch=1 下的自回归 decode：每 segment 输出约 350 token（16 条 `{"i":0,"present":true,"entity_id":"char_001"}` + `missing`），A800 上 32B bf16 每 token 约 31ms → **约 11 秒纯 decode**，且此时显存带宽利用率不到 3%。

---

## 2. 度量–证据依赖图（做任何替换前必须先看这张图）

这是判断「改哪里会动到论文数字」的唯一依据。读 `score_segment()`（`visual_coverage.py:351-433`）可得：

| 上报指标 | 依赖的判断 | 是否 headline |
|---|---|---|
| `precision` | 仅 per-ref `present` | 是（进 f1） |
| `recall` / `recall_all` / `f1` | 仅 `missing` 集合 | **是（headline）** |
| `redundancy_vlm`、`selection_efficiency` | `entity_id` 分组 + `_distinct_views` 计数 | 否 |
| `redundancy_sim` | DINOv3 自相似度，已是确定性 | 否 |

**关键推论**：`entity_id` 只影响 redundancy 与 selection_efficiency，**完全不影响 precision，也不影响 headline recall/f1**。而占 27% 墙钟时间的 `_distinct_views` 子调用，服务的正是这两个非 headline 指标——而其中 `redundancy_sim` 已有确定性孪生版并已在 leaderboard 中并列上报。这是风险最低、收益最直接的下手处。

---

## 3. 对用户 WeDetect / DINO / 聚类级联方案的评估

### 3.1 决定性的量化反驳：省不下 video call

现行协议把**一个 segment 的全部 16 张 ref 摊进同一次 video call**。因此除非廉价通道把这 16 张**全部**高置信裁决，否则这次 video call 照付。我在 532 个 segment / 8440 张 ref 上实测的歧义率很高：

- 33%（2329/8440）的 ref 被判 `present=false`
- 33%（2770/8440）的 ref 被判 `entity_id="none"`
- 81.6% 的 segment 有非空 `missing`

单图自动裁决率 q 对应的 segment 跳过率是 q¹⁶：

| q（单图自动裁决率） | 0.90 | 0.95 | 0.97 | 0.99 |
|---|---|---|---|---|
| P(16 张全部裁决) | 0.185 | 0.440 | 0.614 | 0.851 |

跨视角人物 re-ID 在电影素材上做到 0.97-0.99 并不现实；现实区间 0.90-0.95 对应**整体加速仅 1.2-1.8x**。相比之下第 4 节的 A 类改动是 4-6x 且几乎零科学风险。**级联方案的「加速/风险比」明显更差。**

### 3.2 一个可能致命的公平性问题：循环评估

MemStrata 自己就把 WeDetect-Ref 当**默认** crop 后端，并用 DINOv3 做身份门控与关键帧选择：

```62:64:scripts/evaluate_baselines/trackA/baseline_adapters/causal/memstrata.py
# WeDetect-Ref is the DEFAULT crop backend (describe->bbox); the isolated service normally
# listens here. Override with MEMSTRATA_WEDETECT_URL; set it to "" only to force it off.
_DEFAULT_WEDETECT_URL = "http://127.0.0.1:8710"
```

同文件 `_build_perception()`（221-302 行）注释亦写明「DINOv3 是 identity gate / bank reconciliation / keyframe selection 都要用的」。此外消融族 `retrieval_seg_dinokey_ablation__B16`（`retrieval_family.py`）本身就用 DINO 选关键帧。

**让判官使用被评方法自己的感知骨干打分 = 自我偏好 / 循环评估**，这是审稿人会第一时间抓住的问题，且与 `AGENTS.md` 的公平性契约精神冲突。若一定要用，只能用于非 headline 指标，并附上敏感性分析。

### 3.3 三个硬工程约束

1. **导入边界**：`vmem_bench` 不得 import `memstrata`（MemStrata
   repository 的 `AGENTS.md` 规则 2，`scoring/embedder.py:5-7` 亦明文重申）。
   因此不能直接复用 `wedetect_client.py`，必须在 bench 侧另写一份 HTTP 客户端。
2. **许可证**：WeDetect 是 GPL-v3，现行做法是隔离进程、绝不 import（`wedetect_client.py:1-15`）。把它放进一个**要公开发布的 benchmark 的评分链路**，有发布层面的影响，需提前确认。
3. **服务不返回 embedding**：`WeDetectRefGrounder.ground()` 只返回 `([y0,x0,y1,x1] on 0-1000 grid, score)`（`wedetect_client.py:71-109`）。用户设想的「用它自己的 embedding 聚类检测框」在当前服务上**不可用**，得改服务或另用 DINOv3 对裁剪框补一次 embedding（额外成本）。

### 3.4 一个会被静默做错的语义漏洞：场景 / 地点实体

判官提示词对场景类实体的 `present` 定义是「视频**发生在该场景**里」：

```368:369:src/vmem_bench/scoring/visual_coverage.py
        "1) present:该图代表的对象是否**出现在视频**中(true/false)。"
        "若为场景/地点,present 指视频**发生在该场景**里。\n"
```

这是**全图级**判断，检测框在原理上无法表达。我的抽样里 `loc_001` 类 ref 有 236 条。纯检测级联会静默把这类实体判错。正确做法是按 `kind` 路由到 place recognition——好消息是仓库已有 `MegaLocScoringEmbedder`（`scoring/embedder.py:209`）。

### 3.5 廉价通道未必更便宜（需先算清）

`missing` 需要对 roster 每个实体做「是否可见」的闭集 grounding。以 Halloween 为例：`n_present≈3.1` 实体 × 30 帧 = **约 93 次 grounding 调用/segment**。按每次 50-100ms 计就是 5-9 秒——与当前一次 VLM 调用同量级。**在动工前必须先实测 WeDetect 单次 grounding 延迟**，否则可能做完发现没省钱。

### 3.6 方案中确实成立的部分

- **fps=2 抽帧 + DINO 去重保留 25-50% 关键帧**：这是「证据裁剪」，方向正确，且只要对所有 SUT 用同一个 pinned sampler 就满足 AGENTS.md 规则 4 的对称预处理要求。但**要等到第 4 节 A1 做完才有大收益**（见 4.4）。
- **用 embedding 做重复度判定**：已经在做（`redundancy_sim`），扩展它是自然的。
- **按 kind 路由的身份匹配器**：`entity_registry` 里有 `kind`（`visual_coverage.py:461`），而 `embedder.py` 已备好 DINOv3 / SigLIP2 / ArcFace（LSMDC 真人脸，168 行）/ MegaLoc（地点，209 行）。这套工具箱比单一 DINO 阈值合理得多。

### 3.7 定位建议

把完整感知级联做成 **「开发期快速 scorer」**：用于方法迭代时的快速反馈，永不用于论文/leaderboard 数字，并在 README 里标注它与 32B 判官的实测一致度。这样它的价值被保留，科学风险被隔离。

---

## 4. 四个加速设计

### A1 — 服务拓扑与并发修复（**A 类：协议保持**，强烈建议先做）

- TP=2、4 个副本、`MAX_NUM_SEQS=8~12`、`GPU_MEM_UTIL=0.90`。`start_qwen32_vllm.sh` 已把这三项都参数化（`TENSOR_PARALLEL_SIZE` / `MAX_NUM_SEQS` / `GPU_MEM_UTIL`），**无需改这个脚本**。
- 唯一实质代码改动：`ReviewerEndpointPool` 增加「每 URL 多槽位」（如 `slots_per_url`），并在 `PooledJudgeCaller` 透传。**必须做成 opt-in**，因为这个 pool 与标注流水线 S3 共用。
- 预期 **4-6x**。理由：并发从 8 路提到约 32-48 路；虽副本数减半，但 TP=2 使单 token 延迟减半，且 batch>1 让显存带宽从约 3% 利用率提上来。按 prefill-bound 上限估算约 7000 seg/h。
- 风险：**低**。模型权重、提示词、采样参数、输入全不变，唯一变化是 batched kernel 的归约顺序导致 greedy 解码非位级可复现。这必须用第 6 节的噪声底实验量化，不能口头断言。

### A2 — 消除 `_distinct_views` 的调度损耗（**A/B/C 三档**）

现状是在 group 循环里**串行**发子调用，每次都抢一个独占租约（`visual_coverage.py:398-410`）。三档选择：

- **A2a（A 类，精确）**：把同一 segment 的各 group 子调用**并行化**，并让它们不与主调用竞争独占租约。输出语义完全不变。预期 **1.25-1.4x**，风险低。
- **A2b（B 类）**：把所有 group 合并成一次调用。语义「意图等价」但非精确——VLM 同时看到多组会改变计数行为，需验证。
- **A2c（C 类）**：弃用 `redundancy_vlm`，`selection_efficiency` 改用 DINO 判重。措辞代价很小（`redundancy_sim` 本来就已并列上报，且 threshold-free），但引入一个需在 held-out 上冻结的阈值，且要重跑或后处理。已有先例脚本 `recompute_redundancy_sim.py` 可参考。

顺带一个小项：`_mean_pairwise_sim` 全局持有 `_EMB_INFER_LOCK`（`visual_coverage.py:296-297`），把所有 worker 线程的 DINO 推理串行化，约占 5%。改成批量 embed 即可。

### A3 — 输出 token 压缩（**B 类**）

约 350 个 decode token 是 batch=1 下的主导成本。把 `{"i":0,"present":true,"entity_id":"char_001"}` 换成紧凑行格式（如 `0:1:3`，即 索引:present:roster槽位），输出可压到约 90 token，decode 时间约降 3.5x → 端到端 **1.6-2.2x**。风险：**中**。改输出格式会改变模型行为，必须做一致性验证。注意 A1 做完后负载转为 prefill-bound，A3 的边际收益会下降——**A3 应在 A1 之后重新评估，不要并行下注**。

### B1 — 证据裁剪：DINO 关键帧去重（**B/C 类**，即用户思路的可辩护形态）

fps=2 抽帧后用 DINOv3 去重，保留 25-50% 关键帧再交给判官。**时序很重要**：现在 duration 只占 12-27% 成本，做了收益有限；但 A1 完成后负载翻转为 prefill-bound，此时减少视频 token 会有**近线性**收益。预期（A1 之后）**1.3-1.6x**。风险：中——判官看到的证据变少，`missing` 可能因实体只出现在被丢弃帧里而漏判，需专门验证 recall 不系统性偏移。必须对所有 SUT 用同一 pinned sampler，并作为 protocol v2.3 记录 + 附校准表。

### C1 — 完整感知级联（**C 类**，不建议用于 headline）

即第 3 节评估的用户原方案。预期 1.2-1.8x，风险高（循环评估 + 阈值拟合 + 场景语义漏洞 + headline recall 被挪到检测器的工作点上）。建议按 3.7 定位为开发期 scorer。

### 组合预期

| 路线 | 预期加速 | 剩余 109k segment 耗时 |
|---|---|---|
| 现状 | 1x | 约 80-90 h |
| A1 | 4-6x | 15-22 h |
| A1 + A2a | 5-8x | **11-18 h** |
| A1 + A2a + B1 | 7-12x | 7-13 h |

**只做 A1 + A2a（都是 A 类，协议保持）就能把 80h 压到 11-18h**，且论文措辞不需要任何改动。这是我的核心建议。

---

## 5. 各路线所需的模型能力（精确对应）

| 能力 | 用途 | 现有可用件 | 缺口 |
|---|---|---|---|
| Referring grounding（describe→box） | 判 `missing`：roster 实体是否可见 | `WeDetectRefGrounder.ground()` | 只在 SUT 包内；bench 侧需另写客户端；GPL；**单次延迟未实测** |
| 开放词表检测 + 区域 embedding | 视频侧检测聚类 | — | WeDetect **不返回 embedding**，需扩服务或补 DINOv3 |
| 身份相似度 embedding（按 kind 路由） | 判 `present`、`entity_id` 分组 | DINOv3 / SigLIP2 / ArcFace / MegaLoc 均已在 `scoring/embedder.py` | 需 kind→embedder 路由与 held-out 冻结阈值 |
| 场景级地点识别 | 场景类 `present`（"发生在该场景里"） | `MegaLocScoringEmbedder`（209 行） | 检测器**无法**替代，必须单列 |
| 时序聚合 | 帧级证据 → segment 级 present | — | max-over-frames + 迟滞，防单帧误报 |
| 最终 VLM 仲裁 | 场景级 present、歧义身份 | 现行 32B | 级联下仍不可省 |

---

## 6. 建议立刻做的实验（E1：判官噪声底 + 并发等价性）

这个实验一举两得：既验证 A1，又补上论文**已承诺但尚未产出**的数字——模块 docstring 自己写着要报噪声底：

```52:54:src/vmem_bench/scoring/visual_coverage.py
The "what should be present" side (roster + present set) is FROZEN gold, so scoring
is reproducible; only the per-image visual judgement uses the (pinned) VLM. Report
this together with a measured human-agreement / noise-floor number.
```

**样本**：250 个 segment，从**已有 32B 参考分**的 run 中分层抽样（sita 358 / Pianist 235 / Halloween 128 已全部打过分，参考分免费），按 duration 三分位 × `n_present`（低/高）分层，跨 Blender 与 LSMDC 两个数据集。

**三个 arm（顺序很重要）**：
1. **A0-replay**：用**当前配置**重跑同 250 段 → 建立判官内在重放噪声底。**这一步不能省**，否则 A1 的 delta 无法解释。
2. **A1**：TP=2 / `max_num_seqs=8` / 4 副本，重跑同 250 段。
3. 对照已存档的 A0 原始分。

**比较指标**：
- 吞吐：seg/h（端到端墙钟）
- 判断级一致性：per-ref `present` 位精确一致率；per-ref `entity_id` 一致率；`missing` 集合 Jaccard
- 段级：`precision` / `recall` / `f1` 的 MAE 与 Pearson
- 电影级：`f1_mean` / `recall_mean` 的 |Δ|，以及**排名是否反转**

**通过门槛**（A1 vs A0，且需 ≤ A0-replay 噪声底的同量级）：
- 吞吐 ≥ **3x**
- per-ref `present` 一致率 ≥ **0.97**
- 段级 `recall` / `f1` MAE ≤ **0.03**
- 电影级 |Δ f1_mean| 与 |Δ recall_mean| ≤ **0.01**
- 校准子集上**无系统排名反转**

0.01 这个门槛的依据是现有 leaderboard 相邻系统间距约 0.04（0.6769 / 0.6328 / 0.5834），留了 4 倍余量；**但需注意那份 leaderboard 是 `n_chunks=6` 的小样本产物，正式定阈时应用大样本间距复核**。

**要改动的文件**（按改动量排序）：
1. `src/vmem_bench/annotation/pipeline/stages/s3_segment_auto_review_revise/reviewer_pool.py` — `ReviewerEndpointPool` 加 opt-in 的 `slots_per_url`（唯一实质改动）
2. `src/vmem_bench/scoring/judge_service.py` — `PooledJudgeCaller.__init__` 透传槽位数
3. `src/vmem_bench/scoring/visual_coverage.py` — `run()` 的 `workers` 默认值（500-502 行）解绑 pool size；A2a 则并行化 `score_segment` 的 group 子调用（398-410 行）
4. **无需改**：`start_qwen32_vllm.sh`（`TENSOR_PARALLEL_SIZE` / `MAX_NUM_SEQS` / `GPU_MEM_UTIL` 已参数化）、`stage2_service.py`（`--workers` / `STAGE2_WORKERS` 已存在）
5. 新增对比脚本：可直接改自 `experiments/results/probe/tracka_stage2_8b_probe_20260727_2305/compare_probe.py`（8B 探针已有现成的指标比对逻辑），产物按 `experiments/` 布局落盘

**建议不要做的两件事**：直接换 8B（已被证伪）；以及在 A1 之前投入 C1 级联（收益 1.2-1.8x，却要承担循环评估这个可能致命的审稿风险）。

---

两点说明：我没有修改任何文件；期间只读取了仓库产物、复制了一个 clip 到 `/tmp` 做解码计时，未向正在运行的 GPU 服务发送任何请求（GPU 1 当时 100% 占用，判断为有作业在跑）。另外本会话只挂载了 `cursor-ide-browser` 一个 MCP，没有 NotificationServer，因此约定的完成通知未能发出。