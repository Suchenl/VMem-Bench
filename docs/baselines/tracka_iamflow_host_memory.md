# IAMFlow 在 Track A 上的宿主内存伸缩性

本文是 IAMFlow Track-A 基线 `exit 137` 问题的权威说明：根因、已加的保护、以及
论文里必须怎么报。调度背景属于历史运行记录，不是公开运行入口。

## 根因

`exit 137` 从来不是抢占、坏节点或过度调度，而是 IAMFlow 自身归档结构造成的
**cgroup OOM**——它的宿主内存占用随时间线长度线性增长。

每归档一帧，`MemoryBank._extract_frame_kv_all_blocks` 会把该帧从 DiT KV cache 里
切出来、`.cpu()` 搬到宿主内存，然后**永久**存进两个地方：
`frame_archive[fid].kv_cache` 和 `_frame_kv_store[fid]`。两者都没有任何淘汰逻辑。
`max_memory_frames`（=3）只约束 *active* 集合，不约束归档；而 Track-A adapter 传的是
`save_frames_to_disk=False`，所以 `_save_frame_kv` 直接 return，也不会落盘。

对 Wan2.1-T2V-1.3B，单帧归档的代价是：

```
30 blocks x 2 (k,v) x 1560 tokens x 1536 dim x 2 B (bf16) = 274.2 MiB
```

adapter 每个 bench segment 提交 `segment_latents // num_frame_per_block` 个 block，
每个 block 归档一帧，即 `(duration / 0.25) // 3` 帧。一个 15 秒的 LSMDC segment
归档 20 帧 = **每段 5.4 GiB 宿主内存**。

用这个公式对照线上实测：

| movie | segments | 平均段长 | 归档 KV 帧 | 预测总 GiB | 结果 |
|---|---:|---:|---:|---:|---|
| `0053_Rendezvous_mit_Joe_Black` | 221 | 7.3 s | 2071 | 555 | completed |
| `sita_sings_the_blues_part1` | 164 | 14.9 s | 3259 | 873 | OOM |
| `0013_Halloween` | 128 | 14.9 s | 2442 | 654 | completed |
| `1005_Signs` | 296 | 9.8 s | 3763 | 1008 | OOM |
| `0017_Pianist` | 235 | 13.8 s | 4287 | 1148 | OOM |

关键指标是 `RssAnon`（`RssFile` 始终不到 100 MB，即不可回收的匿名内存），内核日志
也直接印证了机理：

```
Memory cgroup out of memory: Killed process ... anon-rss:545592208kB   # 545 GB
/sys/fs/cgroup/memory/memory.limit_in_bytes = 858993459200             # 800 GiB
```

H800 pod 也只放宽到 900 GiB，帮助有限。

## 为什么这是「总时长 / 归档帧数上限」而不是「吞吐问题」

Track A 上的成败与片长强相关，与运气无关：

* 已完成 35 部：段数中位数 **55**，最长 300。
* 未完成 57 部：段数中位数 **198**，最短的失败者 100。

但更准确地说，上限不是段数本身，而是**归档 KV 帧数**：`sum((duration / 0.25) // 3)`。
短段的 300 段可能还能跑完，15 秒长段的 164 段也可能超过预算。按约 274 MiB/归档帧、
800 GiB pod 计算，单个 IAMFlow runner 的上限约为 3000 个归档 KV 帧，而且必须是 pod
内唯一的重进程。由此得到两条硬结论：

* **每个 pod 永远只能跑 1 个 IAMFlow runner。** 两个装不下，而且第二个会把已经
  跑了几小时的第一个一起拖进 OOM。
* **高归档帧数电影不可能完整完成**：这既可能是很多短段组成的长片，也可能是
  `sita_sings_the_blues_part1` 这种只有 164 段但几乎每段 15 秒的样本。加 GPU 无效，因为
  真正的约束是每个 pod 的宿主内存。

段数只能做粗略预警；调度和论文报告应使用归档 KV 帧数或总时长解释失败。

## 修了什么，以及故意没修什么

所有改动都在 adapter
（`scripts/evaluate_baselines/trackA/baseline_adapters/causal/iamflow.py`）里，
`baselines/Causal/IAMFlow/` 下的 vendored 代码保持原样，只在实例层面打补丁。

### 数值完全等价的回收（默认开启）

1. **丢掉 `FrameInfo.pixel_frame`。** 它在整个 vendored 仓库里只被赋值、从未被读取
   （`to_dict()` 也不含它），而且它是解码像素块的一个 *view*，留着就把整块像素钉死
   到本片结束。
2. **给 `pipe._sync_pixel_store` 加上界**（保留最近 `8` 个 chunk key）。它只会被
   `_get_chunk_pixels` 按 eviction lag（3 个 chunk）和当前 chunk 读回；vendored 代码
   往 `_sync_pixel_order` 里 append 却从不 pop。

两者都不影响检索、打分或生成。合计约省下 10% 的增长——值得做，但远不足以解决问题。

### 没修：KV 归档本身

归档是算法必需的。`retrieve_initial_frames` 可以按实体覆盖召回**任意**历史帧，
`get_memory_kv` 随后需要该帧的 KV 切片，所以给它加上界就改变了 IAMFlow 的记忆能力。
两个被否掉的替代方案：

* **落盘 spill** 在数值上是等价的（`_load_frame_kv` 本来就支持从 `{frame_id}.pt`
  回读），但 `1007_Spider-Man1` 要写约 1.6 TB，一部 200 段的电影约 590 GB，而共享盘
  已经 97% 满。不可行。
* **降精度存切片**会改变数值。

所以不存在「既省内存又忠实」的修法。O(时间线) 的宿主内存是该方法本身的性质，这就是
诚实的结论。

## 保护与开关

| env var | 默认 | 作用 |
|---|---|---|
| `VMEM_IAMFLOW_RSS_LOG` | `1` | 每段输出 `[iamflow][rss]`：`anon_gb`、实测 `slope_gb_per_seg`、`proj_final_gb`、`budget_gb` |
| `VMEM_IAMFLOW_RSS_WATCHDOG` | `1` | 主动抛 `HostMemoryBudgetExceeded`，而不是等着被 SIGKILL |
| `VMEM_IAMFLOW_MAX_RSS_GB` | *(由 cgroup 推导)* | `RssAnon` 硬上限（GB） |
| `VMEM_IAMFLOW_RSS_BUDGET_FRACTION` | `0.92` | 取 cgroup 上限的多少作为预算 |
| `VMEM_IAMFLOW_MAX_RSS_SLOPE_GB_PER_CHUNK` | `0`（关） | 实测斜率超过该值就中止 |
| `VMEM_IAMFLOW_RSS_WARMUP_SEGMENTS` | `3` | 拟合斜率前跳过的段数（前几段被模型加载主导） |
| `VMEM_IAMFLOW_PROJECTION_FLOOR_FRACTION` | `0.5` | 只有 `RssAnon` 达到预算的该比例后才允许按预测中止，保证 `--limit` 冒烟不会自杀 |
| `VMEM_IAMFLOW_PROJECTION_MARGIN` | `1.15` | 预测峰值超过 `预算 x 该系数` 才中止，吸收预测误差 |
| `VMEM_IAMFLOW_PIXEL_STORE_KEEP` | `8` | 保留的解码像素块数（必须大于 eviction lag 3） |
| `VMEM_IAMFLOW_PREFLIGHT_ENFORCE` | `0` | 开启后在开跑前直接拒绝超预算的电影，默认只告警 |
| `VMEM_IAMFLOW_PREFLIGHT_CALIBRATION` | `0.75` | 缩放 preflight 日志里的静态上界 |
| `VMEM_IAMFLOW_MAX_ARCHIVE_KV_FRAMES` | `0`（关） | **可选变体**，见下节 |

watchdog 依据的是**实测**增长，这是它可信的原因。preflight 只是上界（没有抽取到实体
的段会跳过归档），`0001_American_Beauty` 预测 919 GiB 却真的跑完了——所以拒绝必须是
opt-in，默认只告警。

主动中止远好于被 SIGKILL：增量 `visual_selections` checkpoint 已经落盘、进程带着
`iamflow_host_memory_budget_exceeded` 和具体 movie/segment 退出、`finalize.json` 里
还有一段 `host_memory` 记录峰值、斜率和预算。

### 阈值是按线上实测校准的

阈值不是随手取的。用实测斜率回放六部真实电影（判定「注定失败」的口径是
`峰值 >= cgroup - 20 GiB`，因为同 pod 还住着两个 vLLM server，约 10 GB 宿主 RSS），
默认参数必须做到「该拦的全拦、不该拦的一个不碰」：

| movie | 段数 | 实测 GiB/段 | 预计峰值 | watchdog 中止于 | 内核会 OOM 于 |
|---|---:|---:|---:|---:|---:|
| `0053_Rendezvous_mit_Joe_Black` | 221 | 2.4 | 542 GiB | 不中止 ✓ | — |
| `0001_American_Beauty` | 300 | 2.3 | 702 GiB | 不中止 ✓ | —（线上确实跑完了） |
| `0013_Halloween` | 128 | 5.5 | 716 GiB | 不中止 ✓ | — |
| `0017_Pianist` | 235 | 3.6 | 858 GiB | **第 99 段** | 第 219 段（省 2.7 GPU·h） |
| `1005_Signs` | 296 | 4.7 | 1403 GiB | **第 86 段** | 第 189 段（省 1.5 GPU·h） |
| `1048_Gran_Torino` | 732 | 1.07 | 795 GiB | **第 677 段** | 擦线，不可存活 |

第一版用 `0.85` 的预算系数会误杀 `0013_Halloween` 和 `0001_American_Beauty`——后者线上
实际跑完了。所以预算系数提到 `0.92`，并加了 `VMEM_IAMFLOW_PROJECTION_MARGIN`。这组
用例已固化成回归测试（`test_watchdog_calibration_against_measured_runs`），改阈值必须
先让它继续通过。

## `VMEM_IAMFLOW_MAX_ARCHIVE_KV_FRAMES` 是变体，不是修复

设为 > 0 会限制多少归档帧保留 KV 切片：优先丢最旧的，且永不丢当前 active 的帧。
元数据保留，所以实体覆盖召回仍会把老帧排进来，但 `get_memory_kv` 会静默跳过切片已
丢失的帧（它容忍 `None`），于是被召回的记忆集合变小了。

这改变了基线行为。按 `AGENTS.md` 的公平契约，这样得到的数字必须
报成 **`IAMFlow-boundedKV(K)`** 并写明 `K`，**绝不能**报成 `IAMFlow`。只在「否则完全
跑不出来」的电影上使用，并在表里标注。

## 论文报告口径

`tab:main-results` 目前整张还是 `\tbd`，没有任何已发布数字受影响，口径可以自由决定。
两个符合契约的选项：

1. **推荐。** 主表对能覆盖 92 部的系统保持全量；IAMFlow 单独一行并写明 `n`，加脚注
   说明截断源于实现的内存伸缩上限（每归档 KV 帧 ~274 MiB，约 3000 个归档帧触顶），
   而非方法质量。
2. 另给一张「共同子集」表，把所有系统都限制在 IAMFlow 能跑完的电影上。

这条增长曲线本身就是一个值得进附录的 benchmark 结果：VMem-Bench 讨论的正是长视频记忆
的成本，「该基线在 800 GiB 预算下无法处理小时级时间线」是可复现、可量化、有内核日志
背书的发现，不只是一次失败。

## 验证

```bash
python3 -m pytest \
  tests/test_iamflow_host_memory_guard.py \
  tests/test_trackA_stage1_job_lock.py -q
```

不需要 GPU、不需要模型权重、不 import vendored 包，总耗时远小于 1 秒。
