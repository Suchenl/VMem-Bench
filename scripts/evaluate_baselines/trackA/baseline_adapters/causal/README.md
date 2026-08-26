# baseline_adapters/causal — 新协议因果 baseline 适配层

本目录放**每个因果 baseline 的 bench 适配代码**。vendored 原始仓库（`baselines/Causal/<name>/`）
保持**零改动**：所有 bench 胶水都写在这里，绝不塞进第三方代码，便于审查与重拉。

> 权威协议见 [`docs/benchmark/running_eval.md`](../../../docs/benchmark/running_eval.md) §0 三条铁律 + §1 SUT 契约。
> 本层是那套协议在因果 baseline 上的落地。

## 为什么会有这一层（旧路线已删除）

旧的 `src/vmem_bench/baseline_adapters/` 走的是 **gold-replay / online-gold / gold-id 映射**：
把 gold 的 crop 像素或 gold 实体 id 灌给 SUT，再把检索结果映射回 gold 实体做 ID 级打分。
这违反铁律 2/3（bench 不发图、不泄答案），**已按用户决定整体删除**。

新协议：bench 每 segment 只给 SUT 两样——**prompt 文本 + 真实 segment 视频**（真实 segment 用来
**替代 SUT 生成器的产出**，消除生成噪声）。SUT 用自己的感知/记忆机制观察真实 segment、建自己的记忆、
自己检索。检索到的记忆项带**时序身份**（源视频绝对秒数 / 源 chunk id），由 bench 侧
[`frame_materializer.py`](frame_materializer.py) **从真实源视频切出对应帧**作为参考图，交
`vmem_bench.scoring.visual_coverage` 做 VLM 视觉覆盖打分。这就是「把检索到的记忆用时序一致性
映射成 frame」——方法中立，不注入 gold、不给参考图贴标签。

## 数据流

```
runner.py  ──每 segment 时间序──▶  adapter.compose(prompt)      → RetrievedMemory(带时序身份)
                                  adapter.observe_segment(真实clip) → 更新原生记忆(先 compose 后 observe)
      │
      └▶ frame_materializer  ──时序身份→切真帧──▶ outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/<name>.json
                                                     │
                                                     └▶ scoring.visual_coverage (VLM 打分)
```

- **因果护栏**：物化时丢弃「源时间 ≥ 当前 chunk 起点」的项（因果 SUT 只能取过去），计数记进 manifest。
- **bench 侧只做**：切真实 segment、驱动 compose/observe、切参考帧、写 manifest。不感知、不抠图、不发图、不给 gold。

## 每个 baseline 要实现什么（`CausalMemoryAdapter`，见 `contract.py`）

因为这些因果 baseline（LongLive-RAG / MemFlow / IAMFlow 等）本质都是**视频生成器**，
其原生循环是「生成 chunk → 从生成帧抽记忆 → 注入下一 chunk」。新协议要用真实 segment **替代生成产出**，
所以每个 adapter 的核心 hook 是：

1. `reset(movie)`：加载该 baseline 的模型/VAE/记忆管理器（用它**自己的 conda 环境**）。
2. `observe_segment(obs)`：把**真实 segment** 过该 baseline 自己的编码器/VAE，走它**原生的记忆写入**
   （latent / KV bank / role-wise slot / frame buffer），给每条记忆打上源秒数/源 chunk 的时序标签。
   **不做生成**——只借它的记忆写入路径。
3. `compose(req)`：用它**原生的检索算法**（AE 余弦 / 注意力相关性 / 角色 slot 命中 …）从当前记忆里
   选出若干项，返回 `RetrievedItem(source_seconds/source_chunk_id, evidence_kind, ...)`。
4. `finalize()`：可选，返回配置/记忆规模/检索模式等 run 级元数据。

每个 adapter 模块须暴露 `build_adapter()` 工厂，供 `runner.py --adapter <module>` 调用。

## 运行

```bash
# 每个 baseline 用装好对应依赖的 Python（见各 adapter 模块头注释）
PY=python3
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal
$PY runner.py --adapter longlive_rag \
  --movie-dir ../../../../../assets/trackA/BlenderOpenMovies/big_buck_bunny --limit 5
# 产物：outputs/evaluation/trackA/<system>/<dataset>/<movie>/visual_selections/<adapter>.json（+ _ref_frames/ 切出的真帧）
# 之后照常跑 scoring.visual_coverage --system <adapter>
```

## runner 的批量保护（所有因果 baseline 共用）

一个 `--movie-list` 是一批**互相独立**的作业，长跑（memflow_sma 单段 11–165 s、整片 8 h+）
让下面几件事必须成立，否则会成小时级的算力浪费：

- **单片失败不连坐**：`main()` 里每部电影单独 try/except，失败只记一条
  `failed` summary 并继续下一部，最后仍以 exit 31 汇总。此前一部电影抛异常会把排在它
  后面的整条 list 全部丢掉。
- **锁能自愈，也不会误抢**：`.stage1.lock` 现在写 `pid=... host=... start=...`，并在**每段
  之后 heartbeat**（`_touch_job_lock`）。抢锁前判活：同机就探 pid，跨机就看 heartbeat 是否
  超过 `VMEM_STAGE1_LOCK_STALE_MINUTES`（默认 45 min，远大于最慢单段）。
  没有 `host=` 字段的**旧格式锁一律视为存活**，绝不抢——线上确实存在合法持锁 8 h+ 的
  memflow_sma 进程，抢它就会出现两个 runner 干同一部片。这也让手工「清理 stale lock」
  的批次不再必要，那正是之前造成重复 runner 的原因。
- **进度行自带 ETA**：`[stage1] ... segment=i/N ... eta_min=...`。单段耗时随同节点占用在
  11–165 s 之间摆动，肉眼无法判断一个长跑值不值得留，所以直接把答案打出来。
- **adapter 可拒单**：adapter 若实现 `preflight(movie) -> str | None`，runner 会在
  `reset()` 前调用；返回字符串即跳过该片并记录原因。IAMFlow 用它做宿主内存预算检查，
  见 [`docs/baselines/tracka_iamflow_host_memory.md`](../../../../../docs/baselines/tracka_iamflow_host_memory.md)。

回归测试：`benchmarks/VMem-Bench/tests/test_trackA_stage1_job_lock.py`、
`test_iamflow_host_memory_guard.py`（无 GPU、无权重，<1 s）。

## 现状

| baseline | env | adapter | 原生记忆 / 检索 | 状态 |
|---|---|---|---|---|
| **MemStrata（本系统）** | torch + transformers（SAM3 vendored bundle 前置 PYTHONPATH） | `memstrata.py` | 分层 AssetBank（SAM3-concept+DINOv3 感知写入）/ IntentInterpreter 名锚 + model-free compose | **TrackA minismoke PASS**：BlenderOpenMovies:`big_buck_bunny` + LSMDC:`0001_American_Beauty`，各 limit=6，均生成 `visual_selections`。 |
| SlotMem | torch 2.5 + flash-attn 2.8（仅诊断） | `slotmem.py` | 角色 slot（RoleWiseSlotMemoryBank） | **不进入无 oracle 主表**：其 released interface 需要外部/scripted `role_names` 才能稳定定位 slot；这属于 oracle-role / Scripted 诊断条件，不符合 TrackA/B prompt-only 因果生产评测。runner 通过 `scripts/evaluate_baselines/trackA/.disable_slotmem_mainline` 在主线中快速跳过。 |
| LongLive-RAG | torch 2.5 + einops | `longlive_rag.py` | 自编码 latent 描述子 + AE 余弦 top-k | **TrackA minismoke PASS**：两数据集各 1 样本 limit=6 均通过；纯描述子运算，无需生成器前向。 |
| MemFlow (w/o SMA) | torch 2.6 + flash-attn 2.6 | `memflow.py` | sink + 局部窗 + KV bank（文本显著性 top-k） | **TrackA minismoke PASS**：两数据集各 1 样本 limit=6 均通过。 |
| MemFlow (with SMA) | 同上 | `memflow.py`(`sma=True`/`memflow_sma`) | 同上，读取用 `dynamic_topk_routing_attention` φ top-3 chunk 路由 | **TrackA minismoke PASS**：两数据集各 1 样本 limit=6 均通过。 |
| IAMFlow | torch 2.5 + flash-attn（fp8→bf16 反量化） | `iamflow.py` | 实体感知 active-memory 帧（entity+VLM 融合选帧） | **TrackA minismoke PASS**：两数据集各 1 样本 limit=6 均通过。 |

> 三者共用 `_video_io.py` 解码真 segment 为 Wan 像素张量（480×832、[-1,1]、16fps），这是纯 IO，不是感知/记忆——每个 baseline 仍用**自己的 VAE + 原生记忆/检索**。
>
> IAMFlow 全量 TrackA 建议把 LLM/VLM 放到长期 OpenAI-compatible vLLM 服务，避免每个 worker
> 自己加载 Qwen3-4B / Qwen3-VL-2B。设置 `IAMFLOW_LLM_ENDPOINT`、`IAMFLOW_VLM_ENDPOINT`
> 后 `iamflow.py` 会走 HTTP；未设置时保留原 HuggingFace in-process fallback。当前
> Stage-1 的启动说明和 helper 脚本见
> `experiments/results/e2e/tracka_full_20260726/IAMFLOW_SERVICE.md`。
>
> 完成度说明：
> - **LongLive-RAG** 检索是纯描述子运算（无需跑生成器前向），已按发布的 `latentmem` 规则（`memory_size=6/recent_exclude=5/sink_size=1`）完整实现，**已 smoke 通过**（torch 2.6，12 参考帧）。
> - **MemFlow / IAMFlow** 的记忆写入发生在生成器前向内部，用 teacher-force 真 latent 单次前向填原生记忆；两数据集 minismoke 已确认这些写入/读取路径均可产生可物化的 `visual_selections`。SlotMem 仅保留为 oracle-role 诊断，不参与主线批量。
> - **检索族（真实编码器）**：文本 Qwen3-Embedding + 帧 `seg_uniform`（均匀采样）变体此前已 smoke 通过（27 参考帧、future_dropped=0）。默认 `seg_framererank`（SigLIP2 帧-文本重排）在 部分 transformers 5.x 环境下会因 `get_text_features(...)` 返回 `BaseModelOutputWithPooling` 而非张量（`.float()` 崩），属版本不匹配、非数据/权重问题；CPU 探测下 **transformers 4.57.x + torch 2.5–2.10** 均可，SigLIP2 text+image tower 均正常返回 Tensor（`text_feat_shape=(1,768)`，`SIGLIP2_OK`）。故 `seg_framererank` 变体请用 transformers 4.57 的 Python 运行（`RETR_VARIANT`/`MEMSTRATA_RETRIEVAL_VARIANT` 选变体；SigLIP2 权重 `${PUBLIC_MODELS_ROOT}/google/siglip2-base-patch16-512`）。
> - **Track A 批量前置状态**：**PASS**。BlenderOpenMovies:`big_buck_bunny` 与 LSMDC:`0001_American_Beauty` 各 1 个样本已覆盖当前无 oracle 主表行（MemStrata、LongLive-RAG、MemFlow、MemFlow-SMA、IAMFlow、frame_text、seg_uniform、seg_dinokey、seg_framererank）。SlotMem 已移至 Scripted/oracle-role 诊断类。
> - 尚未做完整语料打分：Stage 2 VLM scoring + `aggregate_trackA_outputs.py` 汇总仍待启动。
>
> 各 baseline 所需权重见对应 adapter 模块头注释（均需 Wan2.1-T2V-1.3B backbone + VAE + 各自 ckpt，放在 vendored 仓库根下）。
