<user_visible_high_level_summary>
审计完成，结论先行：**memflow_sma 没卡死**（77–165s/段是算法+同节点争抢，H800 上只要 11.5s/段），剩 2345 段、关键路径约 2.4h，8h 内能完；**IAMFlow 不是调度问题而是宿主内存泄漏**——每段泄 2.3–4.5 GB anon RSS，pod cgroup 上限只有 800 GB(A800)/900 GB(H800)，dmesg 已抓到 `Memory cgroup out of memory ... anon-rss:545 GB` 的 exit137 铁证；已完成的 35 部中位 55 段、未完成的 57 部中位 198 段且最短失败者 100 段，**13 部超 320 段的电影在不改代码前提下永远跑不完**。所以「1–2 个 A800 节点 / 8h 跑完 TrackA 所有剩余基线」不可行（IAMFlow 全量约 211 GPU-h 且并发被内存锁死在每 pod 1 个）。建议 Plan A：只补完 memflow/memflow_sma（约 3h，仅需在 h800 n2 加 3 个空闲卡、不需要起任何 vLLM 服务），立刻停手不再新起 IAMFlow，把 IAMFlow 作为「按时长截断的子集基线 + 线性内存增长成本发现」写进论文。另外发现真正的关键路径不在 stage1 而在 stage2 评分：736 个 (movie×system) 只完成 148 个、当前约 8 个/小时 ⇒ 还需约 73h，光靠本机那一个 32B judge 8h 内绝无可能填满主表。已给出精确的 kill/keep/launch 清单（含 3 个确认在做重复劳动的进程）与判活规则（nvidia-smi 的 utilization.gpu 在这些节点上恒为 100%、完全不可信）。注：本会话没有可用的 NotificationServer MCP，通知需由父代理发送。
</user_visible_high_level_summary>
<response>
# TrackA 基线独立审计报告

## 结论先行

1. **memflow_sma 没有卡死，也不该杀。** 慢是「算法固有 + 同节点争抢」的叠加，不是 hang。剩余 2345 段 / 10 部电影，关键路径约 **2.4h**，8h 内可以完成。
2. **IAMFlow 不是调度问题，是宿主内存泄漏。** 每段泄漏 2.3–4.5 GB anon RSS，而 pod 的 cgroup 上限只有 **800 GB（A800）/ 900 GB（H800）**。exit137 全部是 cgroup OOM，不是抢占。**13 部超过 320 段的电影在不改代码的前提下永远跑不完**，与给多少卡无关。
3. **「1–2 个 A800 节点 / 8h 跑完 TrackA 所有剩余基线」不可行**，差两个数量级。最接近目标的做法是 Plan A（下文）。
4. **真正的关键路径不在 stage1，而在 stage2 评分**：主表需要 736 个 (movie × system) 评分，目前只完成 148 个，速率约 8 个/小时 ⇒ 还需 **约 73 小时**。stage1 就算今晚全完，主表也填不满。

---

## 一、memflow_sma 审计

### 为什么看起来「卡死」

`memflow_sma` 适配器在整部电影跑完前**只写一个增量 checkpoint**，`_adapter_work/` 里只有最终的 `finalize.json`，`_ref_frames/` 的 9000+ 张图是结尾一次性落盘。所以 `LSMDC/1031_Quantum_of_Solace/` 目录 3 小时零文件是**正常现象**，不是死。

真正的存活证据在 stderr 日志（`logs/kml-a800_n0_g3__memflow_sma.log`，mtime 一直是当前时间），里面每段都有一行：

```
[stage1] memflow_sma__B16/LSMDC/1031_Quantum_of_Solace chunk=122/174 id=121 done compose_ms=0.03 observe_ms=97… n_retrieved=70
```

### 慢的真实量级与归因

`compose_ms ≈ 0.03`（检索本身免费），`observe_ms` 是全部成本 —— 即 Wan causal forward + SMA routing capture + VAE encode。实测：

| 电影 | 节点 | observe/段 | 同 pod 并发 |
|---|---|---|---|
| 1019_Confessions | **h800 n0 g4** | **11.5 s** | 2 runner |
| 1007_Spider-Man1 | a800 n1 g5 | 31 s | 4 runner |
| 0019_Pulp_Fiction | a800 n3 g2 | 31 s | 3 runner |
| 1033_Sherlock | a800 n5 g1 | 39 s | 2 runner |
| 1046_Australia | a800 n2 g1 | 53 s | 2 runner |
| 0013_Halloween | a800 n3 g7 | 83 s | 3 runner |
| 1031_Quantum | a800 n0 g3 | 97 s | 1 runner |
| 0026_The_Big_Fish | a800 n1 g0 | 112 s | 4 runner |
| 0017_Pianist（早期） | a800 n0 g3 | 160 s | 高并发期 |

结论：**11.5 s 到 160 s，跨度 14 倍。** 算法本身在 H800 上只要 11.5 s/段；A800 上 31 s；同节点堆 3–4 个 runner 时退化到 80–160 s。所以「memflow_sma 天生 100s/段」是错的 —— 一半是硬件，一半是你自己造成的同节点争抢。每个 runner 只吃 8 GB 内存和 0.5–0.9 个核，CPU 不是瓶颈，退化来自显存带宽 / PCIe / 共享 ceph I/O。

### 老 RUN 该不该杀 —— 判定规则

**先纠正一个致命的误判依据：这些节点上 `nvidia-smi` 的 `utilization.gpu` 恒为 100%，连只占 455 MiB 的空卡也报 100%。它完全不可用于判活。** 请用下面的规则：

| 判据 | 做法 |
|---|---|
| 存活 | `logs/*.log` 的 mtime < 5 分钟；且间隔 3–5 分钟两次采样 `chunk=X/Y` 里的 X 有推进 |
| 显存 | 用 `memory.used`（memflow_sma 稳定 26–30 GB），不要用 util |
| **不可杀** | 一部电影**没有中途续跑**：`runner.py::_selection_complete` 只在 manifest **完整**时跳过，杀掉 = 丢掉全部已跑 GPU 时间 |

**允许杀的四种情况（其余一律保留）：**
1. 该电影 manifest 已完整（`len(chunks)==expected` 且每个 chunk 都有 `retrieval_timing`）→ 当前进程在做重复劳动；
2. 两个 PID 共用同一个 movie-list / 同一部电影；
3. 日志 >15 分钟无推进；
4. IAMFlow 专属：`已用 RssAnon / 已跑段数 × 总段数` 已超出 cgroup 余量 → 与其白烧几小时，不如立刻杀。

---

## 二、IAMFlow 审计：exit137 的确切机理

### 铁证

```
A800 node2, pid 46205 (iamflow, 0013_Halloween):
  RssAnon: 492 GB     RssFile: 98 MB      VmSwap: 0
  cgroup memory.limit_in_bytes = 858993459200  (= 800 GiB)

dmesg:
  Memory cgroup out of memory: Killed process 105376 (python)
    total-vm:609520272kB  anon-rss:545592208kB   ← 545 GB 匿名内存
  Memory cgroup out of memory: Killed process 122736 (python)
    anon-rss:312537212kB
```

`RssFile` 只有 98 MB，说明这 492 GB 是**真实匿名内存**，不是 mmap 的页缓存，不可回收。泄漏速率（三个节点独立采样）：

| 进程 | 段进度 | RssAnon | GB/段 |
|---|---|---|---|
| a800 n2 / 0013_Halloween | ~88/128 | 486 GB | 5.5 |
| a800 n1 / 0017_Pianist | ~119/235 | 433 GB | 3.6 |
| a800 n5 / 0053_Rendezvous | ~181/221 | 432 GB | 2.4 |
| h800 n0 / 1005_Signs | 18/296 | 85 GB | 4.7 |

### 与成败分布完全吻合

- **已完成 35 部**：段数中位数 **55**，最长 300（0001_American_Beauty）。
- **未完成 57 部**：段数中位数 **198**，**最短的失败者是 100 段**。

即：短片全过，长片全挂，分界线在 ~300 段 —— 正好是 `800 GB ÷ ~2.5 GB/段`。这不是随机调度抖动，是**时长决定的确定性 OOM**。

### 推论（很硬）

- **每个 pod 最多 1 个 IAMFlow runner。** 两个必挂，而且会连带把跑了几小时的那个一起 OOM 掉。
- **13 部 >320 段的电影（1048=732, 1049=663, 1007=544, 1024=498, 0028=491, 0041=397, 0019=396, 1033=380, 1006=373, sita_p2=358, 1026=345, 1046=343, 1019=321）在不改代码前不可能完成。**
- 当前 4 个在飞的 IAMFlow 里，按泄漏速率外推：n5/0053 会成功；n2/0013 勉强（预计 ~708 GB，擦边）；**n1/0017_Pianist 预计 855 GB，会在第 ~215/235 段挂**；**h800 n0/1005_Signs 预计 1400 GB，会在第 ~185/296 段挂**。后两个是注定白烧。
- H800 也救不了：cgroup 只放宽到 900 GB，而 IAMFlow 的 `observe_ms` 在 H800 上是 20–51 s vs A800 的 40–80 s，**只快不到 2 倍**（对比 memflow_sma 的 2.7 倍）。

### 为什么没有 25–30 个稳定独占 runner

不是调度水平问题，是三重硬约束叠死：
1. **宿主内存**：每 runner 峰值 300–550 GB，pod 只有 800 GB ⇒ 1 runner/pod ⇒ 全公司 9 个 pod 上限 9 个并发。
2. **服务占卡**：每个跑 IAMFlow 的 A800 节点要占 2 张卡给 vLLM（已验证 node2：`:8100` Qwen3-4B on GPU6 21.7 GB，`:8101` Qwen3-VL-2B on GPU7 29 GB，两个都健康）。
3. **单卡显存**：runner 自己吃 44–53 GB，A800 80 GB 卡放不下第二个。

### 还有没有可优化路径

有，但都不在 8h 内：泄漏几乎肯定在 vendored IAMFlow 的 `pipe.agent_memory_bank.frame_archive`（随 chunk 单调增长、疑似存解码后的像素张量/crop）、`vlm_agent` 按 `p{pid}_c{cid}` 键的永不淘汰缓存、以及 `_run_sync_vae_vlm` 里解码出的 block 像素。`pipe.max_memory_frames` 这个字段存在但需核实是否真的在执行淘汰。修法：archive 落盘存路径而非张量 + 显式 `del`/`gc` + 有界 LRU。验证成本很低：1 张卡 + 1 部 128 段短片，30 分钟内用 per-segment `RssAnon` 曲线就能确认斜率是否压到 <0.1 GB/段。

---

## 三、可行性判定

TrackA stage1 剩余量（按 gold `chunk_index.json` 精确统计，全库 92 部 / 16548 段）：

| 系统 | 完成 | 剩余段 | 剩余电影 |
|---|---|---|---|
| longlive_rag | 16548 / 16548 | **0** | ✅ 全完 |
| 4 个 retrieval_* | 92/92 各自 PASS | 0 | ✅ 全完 |
| memflow | 16420 | **128** | 1（0013_Halloween，在飞，第 32/128） |
| memflow_sma | 14203 | **2345** | 10 |
| iamflow | 2752 | **13796** | 57 |

**成本核算：**
- memflow + memflow_sma = 2473 段 ≈ **20 GPU·h**，但按电影串行、关键路径 = 1007_Spider-Man1 剩 284 段 × 31 s ≈ **2.4h**。
- iamflow = 13796 段 × ~55 s ≈ **211 GPU·h**。8h 完成需要 **26 个稳定 runner**；内存把并发锁死在 **9 个**（全部 9 个 pod 各 1 个）⇒ 理想下限 **23h**；而且其中 5841 段属于那 13 部永远跑不完的电影。**1–2 个 A800 节点 ⇒ 2 个 runner ⇒ 105h。**

**判定：8h 内跑完「TrackA 所有剩余基线」不可行；1–2 节点约束下更不可行。** 但——**8h 内跑完 memflow + memflow_sma 完全可行，只需 ~3h 和 3 张空闲卡。**

---

## 四、Plan A / B / C

### Plan A（推荐，最省、最稳）
1. **保留全部在飞的 memflow / memflow_sma**（它们不可续跑，杀了就白烧）。
2. **只补 3 个空闲槽**（h800 n2 有 6 张 646 MiB 的空卡，11.5 s/段），跑三部无人认领的：`0032_The_Princess_Bride`(161)、`0033_Amadeus`(90)、`daily_dweebs`(6)。
   - **这三个不需要起任何 vLLM 服务** —— 已核对 worker 脚本，memflow_sma 的启动行不带任何 `IAMFLOW_*` endpoint，纯本地权重。直接回应「别盲目起服务占显存」。
3. **IAMFlow 立刻停手**：不再新起任何 runner，让已经在飞的 4 个自然收敛（预计只有 1–2 个能活）。
4. **IAMFlow 在论文里降级为「按时长截断的子集基线」**，并把内存曲线本身写成一个成本发现（见第六节）。
5. 释放出的算力全部转去 **stage2 评分**（真正的瓶颈）。

预计：**T+1.3h memflow 全完；T+3h memflow_sma 全完**；总增量成本 ≈ 3 张 H800 卡 × 1h。

### Plan B（若审稿口径必须要 IAMFlow 全 92）
先花 30 分钟 + 1 张卡定位并修掉泄漏，再重排。但即使修好（假设降到 <20 GB 常驻，此时瓶颈变成单卡 44–53 GB 显存 ⇒ 每节点 6–7 个 runner），211 GPU·h ÷ 40 runner ≈ 5.8h —— **需要 5–6 个节点全开**。**「8h + 全量 IAMFlow + 1–2 节点」在任何情况下都不成立。** 这条路只在你愿意放弃「1–2 节点」约束时才存在。

### Plan C（最省，兜底）
完全不再动 IAMFlow，连在飞的 4 个都不等（杀掉注定 OOM 的 n1/0017_Pianist 和 h800n0/1005_Signs，省下 2 张卡），IAMFlow 就以现有 **35 部完整结果** 定稿。适用于「今晚就要交表、不接受任何不确定性」。

---

## 五、精确的下一步动作清单

### 立即 kill（3 个，都在做确定无价值的工作）

| 节点 | PID | 已跑 | 理由 |
|---|---|---|---|
| a800 n2 g1 | **39780** | 315 min | `memflow_sma/1046_Australia` 的 manifest **已完整**（343/343 chunk 全有 timing，另一个进程先完成了）。`materialize_record_checkpoint` 是**合并写**、不预填，所以完整 = 真完成。纯重复劳动。 |
| a800 n4 g3 | **57817** | 160 min | 同理，`memflow/1008_Spider-Man2` manifest 已完整（225/225）。纯重复劳动。 |
| a800 n1 g2 | **24750** | **7 min** | 7 分钟前刚起的 IAMFlow。同 pod 已有 pid 110987 占 433 GB，cgroup 只有 800 GB ⇒ **两个必然一起 OOM**。杀新的（7 min），保老的（201 min）。 |

> 顺带：`1046_Australia` / `1008_Spider-Man2` 存在「一部电影被两个进程跑过」的历史，落盘 manifest 的 provenance 建议在写表前核一遍是哪次运行产出的。

### 可选 kill（Plan C 才做）
- a800 n1 g7 pid 110987（0017_Pianist，235 段，433 GB@119 段 ⇒ 外推 855 GB）：**注定在 ~215/235 段 OOM**，再烧 2h 也拿不到结果。
- h800 n0 g3 pid 1159674（1005_Signs，296 段，85 GB@18 段 ⇒ 外推 1400 GB）：**注定在 ~185/296 段 OOM**。

### 一律保留
- 全部 6 个 memflow_sma（n0/126310、n1/19638、n1/108104、n3/74067、n3/76031、n5/110643、h800n0/1159336）—— 无中途续跑，杀 = 白烧。
- a800 n3 g4 pid 80398（memflow 0013_Halloween，partial-block 修复验证）—— 这是 memflow 的最后一部，第 32/128，observe 47–53 s，**ETA ≈ 1.3h**，补丁看起来生效了（已经跑过 32 段没再报 64-vs-42）。

### 新起 3 个（前置检查后再动手）
先按 `gpu-service-capacity` 规则确认 h800 n2 是我方配额节点（`tgpu -c kml-h800 -node 2 bash -lc 'tmux ls; nvidia-smi'`；该节点现在 6 张卡是 646 MiB 空闲，g0/g7 被别人占着）。确认后在 g1/g2/g3 各起一个 memflow_sma，**不起任何服务**：

- `0032_The_Princess_Bride`（161 段，当前状态 CLEARED_RETRY，**无 runner**）
- `0033_Amadeus`（90 段，被串在 n0_g3 的 1031 后面排队，抢出来单独跑）
- `daily_dweebs`（6 段，几分钟的事，可与 0033 合并成一个 list）

同时确认 `0020_Raising_Arizona`（145 段，状态是**假 PASS**、manifest 缺失）已被 n1/108104 接手 —— 它刚跑完 0026，list 里的第二部正是 0020；若 10 分钟后日志没出现 0020，就补一个槽。

### 论文表格处理（`_tables/experiments/tracka_main_results.tex`）
主表现在整张是 `\tbd`，`leaderboard.md` 只有 `n=2`，所以**还没有任何数字被 IAMFlow 污染** —— 现在决定口径的成本是零。

VMem-Bench 的 `AGENTS.md` 公平契约要求「每个 SUT 收到逐字节相同的输入」。把 IAMFlow 放进 92 部主表会变成子集对比，违反可比性。两个合规选项：
- **(推荐)** 主表 8 个系统保持全 92 部；IAMFlow 单独一行并显式标注 `n=35`，脚注写明截断原因是**实现的宿主内存随时间线线性增长（≈2.5–4.5 GB/段，800 GB 预算下 ~300 段封顶）**，而非方法质量；
- 或者额外给一张「共同子集」表，所有系统都只报 IAMFlow 能跑完的那 35 部。

**这件事其实是加分项而非污点**：VMem-Bench 的卖点之一就是长视频记忆的成本轴，「IAMFlow 无法在 800 GB 预算下处理小时级时间线」是一个可复现、可量化、有 dmesg 证据的实证结论。建议在附录放一张 `RssAnon vs 段数` 的线性拟合图。

---

## 六、必须先解决的更大问题：stage2 评分才是关键路径

审计过程中发现的最重要一件事，**优先级高于 IAMFlow**：

- 主表需要 **92 电影 × 8 系统 = 736** 个 `_visual_score`，目前完成 **148**，最近 1 小时只完成 **8** 个。
- 按段数算：**4645 / 132384**（3.5%）。
- 剩余 588 个 × ~8/h ⇒ **约 73 小时**。
- 全部压在**本机**一个 `qwen3-vl-32b`（`:8110`）上，4 个 `visual_coverage` worker，单部电影跑 2.5 小时（我看到 `0019_Pulp_Fiction` 已跑 2h27m）。judge probe 显示 32B 判 3 个 chunk 要 83 s（≈28 s/chunk）。
- 8B judge 那条降本路线**已被证否**：`judge_compare_8b_vs_32b` 报告写明 40 个 8B endpoint 全部 502，结论「不可用于论文评分」。

**建议把接下来的 GPU 预算投到这里，而不是 IAMFlow**：h800 n2 有 6 张 141 GB 空卡，足够跑多个 Qwen3-VL-32B 副本并把 scorer 并发从 4 提到几十。同时**本机 worker 数不要超过 4**（现在正好 4 个 + 两个常驻 vLLM，本机 load 已经不低）—— 扩容全部走 H800，这正是你担心的「CPU 打满影响保活」的正确规避方式。

---

## 风险与注意

- **不要为了「集中到 1–2 个节点」去杀在飞的 memflow_sma**。没有中途续跑，那等于把已经烧掉的 20+ GPU·h 全部作废，还会把 2.4h 的关键路径重置成 4.7h。已经分散在多节点的在飞任务是**沉没收益**，让它跑完是最省的选择。
- **不要在任何一个 pod 上放第 2 个 IAMFlow runner**，包括 H800。这不是「可能 137」，是 800 GB 上限下的算术必然，而且会把先跑的那个一起拖死。
- **别再用 `utilization.gpu` 判活**，这些节点上它恒为 100%。用「日志 chunk 推进 + memory.used」。
- 起新槽前先 `tmux ls` 确认是我方配额节点，非配额节点会在模型加载后 60–90 s 回收进程。
- 共享存储 `/data` 已用 97%（2.0P / 剩 71T），且 `ls` 大目录出现过 >30 s 的延迟。落盘密集的 `_ref_frames`（单部电影 9425 个文件）还在持续写入，写表前建议确认剩余配额。
- 我这次只做了只读审计，没有杀进程、没有起服务、没有改任何代码。
- 本会话没有可用的 NotificationServer MCP（该项目只挂载了 `cursor-ide-browser`），完成通知需要由父代理发送。
</response>
