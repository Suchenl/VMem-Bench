# baseline_adapters — 新协议 SUT 适配层（总览 + 完成方式 + 忠实度反思）

本目录是 MemStrata-Bench **新协议**下所有 baseline 的 bench 适配代码。vendored 原始仓库
（`baselines/Causal/<name>/`）保持**零改动**，所有胶水都写在这里。

- 因果 baseline 的落地在 [`causal/`](causal/)，协议细节见 [`causal/README.md`](causal/README.md)。
- 权威协议（三条铁律 + SUT 契约）见 [`../../docs/benchmark/running_eval.md`](../../docs/benchmark/running_eval.md) §0/§1。

> 一句话协议：bench 每 segment 只给 SUT 两样——**prompt 文本 + 真实 segment 视频**（真实 segment
> **替代 SUT 生成器的产出**以消除生成噪声）。SUT 用**自己的**感知/记忆/检索，返回带**时序身份**的
> 记忆项；bench 侧 `frame_materializer` 按时序一致性从真实源视频切出对应帧做 VLM 视觉覆盖打分。

---

## 1. 每个模块的具体完成方式

所有因果 adapter 实现同一个 `CausalMemoryAdapter` 契约（`causal/contract.py`）：
`reset(movie)` → 每 segment `compose(prompt)`（先）→ `observe_segment(真实clip)`（后）→ `finalize()`。
三者共用 `causal/_video_io.py` 把真实 segment 解码成 Wan 像素张量（480×832、[-1,1]、16fps，VAE 时序 stride 4）——
这是**纯解码 IO，不是感知/记忆**，每个 baseline 仍用自己的 VAE + 原生记忆/检索。

| baseline | 原生记忆空间 | 原生检索 | 我们的落地方式 | 完成度 |
|---|---|---|---|---|
| **LongLive-RAG** | 每 latent 一个 **AE 描述子** 的增长池 | **AE 余弦 top-k**：用最近 latent 描述子查询，池 = `[sink, 池, recent_exclude]` 之间；`memory_size=6/recent_exclude=5/sink_size=1` | 直接引 `wan.modules.vae._video_vae` + `ae.model.LatentAE`（**绕开 DiT/flash_attn**）；`observe` 把真 segment 编码成 latent → 每帧 AE 描述子入池并打源秒标签；`compose` 复刻发布的余弦 top-k，返回命中项源秒 | **完整实现，可跑**（无需生成器前向） |
| **MemFlow (w/o SMA)** | **sink + 局部窗 + KV bank**（bank 为逐层文本显著性 top-k 压缩的历史块） | **写入**：`compress_kv_bank` 逐层文本 top-k；**读取**：整块 bank 进 attention，不再筛 | teacher-force 真 latent（`context_noise=0`，**该 segment prompt** 条件化）逐 segment 前向填 KV/bank；**指纹追溯**把存活 bank 块映射回源 latent（逐层 + 投票）；`compose` 汇总 sink∪局部窗∪bank 的源秒 | **忠实实现，已 GPU 验证**（wan2_1 env，5 segment：base+LoRA 生效、126 参考帧、future_dropped=0） |
| **MemFlow (with SMA)** | 同上（**写入与 w/o SMA 完全相同**） | **读取**：`dynamic_topk_routing_attention` 用 φ=mean-pool 紧凑描述子 `φ_q·φ_k` 从 `sink∪bank` 路由 **top-3 chunk** 再进 attention（`SMA=True`） | 复用 `memflow.py`（`sma` 开关）：reset 时把所有 self-attn 的 `SMA` 置 True，hook 路由捕获 φ 选中的历史 chunk（query=真段，仅留早于当前 chunk 的源）；`compose` 引用回填于 `observe`（runner 循环后 materialize） | **忠实实现，已 GPU 验证**（wan2_1 env，5 chunk：31 模块置 SMA、118 参考帧、future_dropped=0；`memflow_sma`） |
| **IAMFlow** | **实体感知 active-memory 帧**（每帧带 KV 切片 + 关联实体） | `select_frame_from_chunk`（DiT 注意力 entity_score + 0.3·VLM 视觉分融合；上游函数名保留）+ prompt 边界 `retrieve_initial_frames` 贪心实体覆盖 | 正式 adapter 内自包含 fp8→bf16 反量化和 HF/vLLM 后端 glue：teacher-force 真 latent 逐 segment 前向填 KV/crossattn（**该 segment prompt** 条件化）+ 归档/驱逐 + VLM 帧打分；`compose` 用 LLM **逐 segment 因果**抽实体（不预计算全 prompt）→ `retrieve_initial_frames`；frame_id→源秒登记表 | **忠实实现，已 GPU 验证**（vace env，**无需新环境**；15 segment：18 参考帧、segment 9–14 角色复现后正确召回、future_dropped=0） |

各 baseline 所需权重/环境见对应 adapter 模块头注释；运行方式见 `causal/README.md` §运行。

---

## 2. 为什么这个接入方式是合适的（核心反思）

**接入策略 = teacher-force 真实 segment 的 latent 过各 baseline 的原生记忆写入 + 原生检索，再把检索项按时序一致性物化成真帧。**

这与用户设定一致，也是**隔离记忆机制、同时消除生成噪声**的最小损伤方式，理由：

1. **为什么不让 baseline 自己生成再抽记忆？** 那会把「生成器质量」混进「记忆机制」的评测——两个不同 baseline 的差异会被各自生成器的画质/漂移污染。用真实 segment 替代生成产出，正是为了让所有系统在**同一份视觉内容**上比记忆。
2. **为什么用 teacher-forcing 而不是改检索？** 我们**不改**任何 baseline 的检索算法（余弦 top-k / 文本显著性 top-k / 实体覆盖都原样保留），只把「喂给记忆的像素」从「自己生成的」换成「真实的」。检索逻辑 100% 是 baseline 自己的。
3. **为什么按时序一致性物化成帧？** 各 baseline 的记忆原生形态不同（描述子 / KV 块 / 实体帧），无法直接互比。统一的可比标的是「它检索出的记忆对应源视频的哪一时刻」——把该时刻的真帧交给 VLM 打视觉覆盖，方法中立、不注入 gold、不给参考图贴标签。

**是否有替代方案更好？** 唯一的替代是「让 baseline 全程原生生成 + 只记录检索」——但这重新引入生成噪声（用户明确要去除），且不改变检索忠实度问题。故 teacher-forcing 是正确路线；剩下的忠实度风险纯粹是「在 teacher-forcing 下如何**忠实捕获**原生选择」的实现问题，而非路线缺陷。

---

## 3. 和原始检索方案一样吗？有损性能吗？（逐 baseline 诚实评估）

### LongLive-RAG —— 忠实，几乎无损 ✅

- **相同**：AE 模型、余弦 top-k、`sink/recent_exclude/memory_size` 参数、查询=最近 latent 描述子（`latent_descriptors[-1]`）——全部与原实现一致。描述子改由**真 latent** 计算，正是预期的「真内容、无生成噪声」替换。
- **差异（可忽略）**：(a) 我们在 chunk 起点查询一次，原实现在生成该 chunk 时逐 block 查询；对「为该 prompt 组合的记忆」而言，chunk 粒度更贴切。(b) 我们不把检索回灌进生成（因为不生成），这是评测设计而非损伤。
- **结论**：无实质性能损失。

### MemFlow —— 忠实，已 GPU 验证 ✅（两处缺口已补）

- **缺口1 已修**：`compress_kv_bank` 的 bank 选择是**文本条件**的。observe 前向现按**该 chunk 的 prompt** 编码 crossattn，文本显著性信号真实。
- **缺口2 已修**：不再用 recency 占位。`compress_kv_bank` 逐字复制 K/V 块，故用**指纹匹配**把每个存活 bank 块映射回其被提交时的源 latent（逐层 + 投票聚合），与原 Track-A 驱动器 `run_bank_trace.py` 同法。这是 MemFlow 真实的文本显著性选择，非近似。
- **长视频 RoPE**：按整片时长扩展时序 RoPE 表（精确、非近似），避免超出默认 1024 位置。
- **验证**：wan2_1 env（torch2.6+flash_attn2.6.3）5 chunk 跑通，base+LoRA 生效（trainable 19.78%），126 参考帧、`future_dropped=0`，检索标记 `memflow_sink+local+kv_bank_fingerprint_trace_on_real_latents`。
- **SMA 变体（`memflow_sma`）**：MemFlow 源码另有 `dynamic_topk_routing_attention`（φ=mean-pool 紧凑描述子的 `φ_q·φ_k` top-3 chunk 路由），由 `SMA` 开关门控、shipped 配置默认关。**写入与 w/o SMA 完全相同，只差读取**。adapter 以 `sma=True` reset 时把所有 self-attn 的 `SMA` 置 True，并 hook 路由记录 φ 选中的历史 chunk（用真段作 query，仅保留早于当前 chunk 的源，因果合法）；因路由是 query 相关的，`compose` 返回的 rec 在 `observe` 内按引用回填（runner 循环后统一 materialize）。5 chunk GPU 验证：31 模块置 SMA、118 参考帧、`future_dropped=0`。打分（judge=qwen3-vl-32b，limit 5）SMA 全面更好：prec 0.44→0.56 / rec 0.67→1.00 / f1 0.62→0.84，帧数 31.5→29.5。

### IAMFlow —— 路线忠实，但当前实现**有损**，须先补 ⚠️

- **缺口1**：观察前向用空 prompt → crossattn 文本缓存非实体条件 → `entity_score` 退化。**须把 prompt（及 LLM 实体）喂进 observe 前向。**
- **缺口2**：`select_frame_from_chunk` 当前传 `current_entity_ids=[]` 且无 `visual_scores` → 选帧近似均匀。**须传真实实体 + 真帧的 VLM 视觉分（0.3 权重）。**
- **缺口3**：frame_id→源秒 目前用 chunk 起点近似，须改成 evicted-frame 精确索引。
- **影响**：未补前，IAMFlow 的实体感知检索被抹平，结果不代表其真实能力。

---

## 4. 当前可跑性与阻塞

| baseline | 权重 | 环境 | 现在能跑吗 |
|---|---|---|---|
| **MemStrata（本系统）** | ✅ SAM3 (`facebook/sam3`) + DINOv3 (`facebook/dinov3-vitb16-...`)（`PUBLIC_MODELS_ROOT`）；GroundingDINO 缺 → SAM3-concept-only | ✅ 复用 `helios`（torch2.10；SAM3 走 vendored `sam3_transformers59`(tf5.9) 前置 PYTHONPATH） | **能**（本轮已 GPU 验证：big_buck_bunny name_anchored limit=4） |
| LongLive-RAG | ✅ 已就位（`wan_models/Wan2.1-T2V-1.3B` + `ae_latent_mem.pt`/`longlive_base.pt`/`longlive_lora.pt`） | 复用 `vace`（torch2.5+einops，仅需 VAE+AE） | **能**（本轮已跑样本） |
| MemFlow | ✅ `KlingTeam/MemFlow/{base.pt,lora.pt}`（symlink 进 repo） | ✅ 复用 `wan2_1`（+omegaconf/peft） | **能**（本轮已 GPU 验证） |
| IAMFlow | ✅ `Causal_Video_Generation/IAMFlow/{iamflow_fp8.safetensors,tinyvae.pth}` | ✅ 复用 `vace`（fp8→bf16 反量化）+ 可选 `vllm` 服务 | 待 §3 实体/VLM 接线修复 + 验证（**无需新 env**） |
| SlotMem | ✅ LoRA/encoder `Causal_Video_Generation/SlotMem/ckpt/{stage1,stage2}/{stage*_high,stage*_low}.pt` + native base `Wan-AI/Wan2.2-I2V-A14B` | `vace`（flash_attn 2.8.3 + torch 2.5.1）；SlotMem 用自带 vendored diffsynth，非 lightx2v | ✅ 已接入：VAE 编码真 segment，选 SlotMem 单 bank timestep 加噪，单次 native DiT 前向注意力探针抽角色 slot，再用 stage2 encoder/writer 写 `RoleWiseSlotMemoryBank`；**正式 TrackA 禁用 distilled/lightx2v Wan2.2**（可加载但烟测画质不可用） |

> 权重与环境登记见 [`WEIGHTS.md`](WEIGHTS.md)。**LongLive-RAG、MemFlow 现已端到端 GPU 验证**；IAMFlow 用
> `vace` 反量化 bf16 跑 DiT（正式 adapter 内自包含），LLM/VLM 走
> vace HF 或 vllm 服务，无需专用环境；待实体/VLM 接线修复后验证。
