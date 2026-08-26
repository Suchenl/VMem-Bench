# 公平性实验计划（对应 baseline 公平性决定）

> 决定的"为什么"见 [`../baselines/fairness_decisions.md`](../baselines/fairness_decisions.md)。
> 本文只讲"要跑哪些实验 / 代码改什么 / 验收标准"。

> **术语与新管线更新（2026-07-24）**
> 1. **不再用 "Regime A/B" 这种叫法**，改为语义明确的两个输入档：
>    - **name-anchored**：prompt = S4 剧本 prose 原文，复现实体按自然叙述被点名。
>    - **description-provided**：在 **name-anchored 之上**，用确定性规则在每段 prompt 尾部
>      追加"该实体长什么样"的外观描述（`[实体外观参考] 名字：描述；…`），**只描述该 prompt 里
>      已被点名的实体**（信息不超出 prompt 已有的名字 → 不泄漏 present/roster）。
>    - 这**取代**了旧的 "description-only"（把名字从 prompt 里抹掉）思路：新档是**加描述信息**，
>      不是去名字。旧的 `apply_input_regime`（name-stripping，仅存在于已废弃的 gold-replay 路径）作废。
> 2. **落点在新协议管线**：`baselines/bench_adapters/causal/runner.py --input-mode
>    {name_anchored,description_provided}`。description-provided 的输出/打分用独立系统名
>    `<adapter>__descprov`，两档并存。下表 D 行的旧 orchestrator 实现仅作历史记录。
> 3. **报告方式**：两档结果**都进主表**（不一定按列排布，可分区/分行呈现），并列报告，
>    不用其中之一单独下结论。**paper 暂不改**，先在本文与相关中文实验文档登记。

## 语言轨道与 gold 数据特性（2026-07-24）

**语言设计决定**：主表**统一中文**（name-anchored / description-provided 两轨），**不做全量 2×2 四轨**。
- 依据：所有当前 baseline（MemFlow / LongLive-RAG / IAMFlow / SlotMem）均用 **umT5-xxl** 文本编码器
  （Wan2.1 中英双语），中文并不会"卡死"任何 baseline，故语言轴不是主表公平性的必要条件。
- 语言鲁棒性放**附录子集研究**：用 **12 部原生英文/混排片**（见下）做中/英对照，实证语言 Δ 很小 →
  正面回应"benchmark 是否偏袒某语言"。这 12 部英文是**原生**的（非机翻），en 侧零翻译噪声。

**gold `description` 已改写为"首次出现时的纯外观"**（2026-07-24，人工 pilot + grok 子代理批处理）：
- 覆盖全部 92 部 / 2803 实体，仅改 `description`，其余字段与结构不变，JSON 零损坏。
- 规则：只留体貌（体型/颜色/材质/部位/发型/服饰/形状），删用途/性格/职业/身份/动作/能力/
  时序/叙事/跨实体关联；多造型只取首现；**保持每条原文语言，不翻译**。
- 用途：description-provided 档在 prompt 尾部按此追加"实体外观参考"。

**数据特性记录（暂不改，后续再议）**：
1. **混合语言 gold → 已统一中文**（2026-07-24 完成）：原 92 部中约 81 部 prompt 以中文为主、约 11 部
   以英文为主，且个别片（CASABLANCA、Amadeus、sita_part1 等）**片内中英混排**。已对 **12 部**做中文
   归一化（翻译 canonical `prompt` + 中文纯外观 `description`，专有名词按 `entity_registry.name`
   保留英文以保住 name-anchoring）。**原英文快照**（英文 prompt + 英文纯外观 description）保存在各片
   `gold/variants/en/`，作附录**语言鲁棒性子集的 en 变体**（原生英文、零机翻噪声）。
   - 归一化清单（12 部）：`caminandes_2_gran_dillama`、`pepper_carrot_ep3`、`sita_sings_the_blues_part1`、
     `CASABLANCA`、`Charade`、`Clerks`、`O_Brother_Where_Art_Thou`、`The_Big_Lebowski`、`Amadeus`、
     `Harry_Potter_and_the_Half-Blood_Prince`、`The_Great_Gatsby`、`The_Ugly_Truth`。
   - 校验：12 部 JSON 全通过、英文残段≤6%（多为下条截断残句）、name-anchoring 覆盖不变、en 快照齐全。
3. **部分 prompt 原文截断**：少数 S4 产出的 prompt 本身在 ~220 字符处被截断（如 O_Brother 若干段、
   Amadeus chunk3），非翻译引入。当前按残句忠实处理，**先记录、后议**。
2. **13 条 appearance-less 实体**：原文只有身份/职业/关系、本无外观（如"委员会男1""麦克风""手机"
   "笔记本电脑"），按"不编造"原则改写后 `description` 为空。当前**保持空**，description-provided
   对这些实体不追加后缀。清单：
   `Ist_das_Leben_nicht_schoen`(char_011/014/016)、`Raising_Arizona`(char_006/007/008/009/012/015)、
   `The_Queen`(prop_009)、`This_is_40`(char_012/prop_006/prop_007)。

## 代码改动矩阵

| # | 改动 | 涉及文件（预估） | 状态 |
|---|---|---|---|
| A | 统一两 orchestrator 的 baseline 名单；多影片默认 `--visual` on | `scripts/vmem_bench/compare/baseline_sets.py`（新，唯一真源）、`run_bbb_track_a.py`、`run_movie_benchmark.py` | ✅ 代码完成 |
| B | scoring 加 SigLIP2 + ArcFace + MegaLoc embedder，按 entity kind 路由；VisualFidelity 报多列 | `src/vmem_bench/scoring/embedder.py`、`visual.py`（`build_visual_suite`）、`runner.py` | ✅ 代码完成（全跑需 numpy/权重环境） |
| C | 检索 baseline 报多档 k（{1,3,5,budget-matched}） | `baseline_adapters/run_gold_replay.py`（`--retrieval-top-k`）、两 orchestrator（`@k{n}` 循环） | ✅ 代码完成 |
| D | name-anchored / description-provided 两档输入（见顶部术语更新） | **新管线**：`baselines/bench_adapters/causal/runner.py`（`--input-mode` + `_apply_description_provided`，只描述 prompt 已点名实体，输出名 `<adapter>__descprov`）。（历史）旧 gold-replay：`vmem_bench/scoring/runner.py::apply_input_regime`（description-only，已废弃） | ✅ 新管线代码完成 + 变换离线验证（BBB 全 52 段变换正确、no-name 段不追加、不泄漏 present）；⏳ GPU 并列重跑待确认后执行 |
| E | 因果 trace 全影片生成 + 并行 + iamflow vLLM + 共享 encode + 缓存 | `scripts/vmem_bench/compare/generate_causal_traces.py`（`--movies all`/`--skip-complete`/`--submit`/iamflow endpoint 透传）、`scripts/baselines/iamflow/run_agent_trace.py`（vLLM HTTP LLM/VLM + 分数缓存）、`scripts/baselines/iamflow/servers/serve_iamflow_vllm.sh`（新） | ✅ 代码完成（E1 并行/E2 vLLM/E3 共享 encode 核对/E4 缓存）；⏳ E5 = GPU 上实跑，需节点 |
| F | scripted 系统移出定量主表（仅附录定性） | `baseline_sets.py`（`assert` 主表∩附录=∅）、`run_gold_replay.py`（external 报错指向 D5） | ✅ 代码完成 |

## 实验清单

### E1. 主表（name-anchored，全影片 × 全 baseline，visual headline）
- 系统：MemStrata(-fast) + 因果(helios/longlive_rag/memflow/iamflow) + 检索(text/frame/fusion@多 k) + 诊断(full/recency/sliding/oracle)。
- 影片：全部 12 部 frozen Blender + LSMDC（就绪后）。
- headline：VisualFidelity composite，多 embedder 列（DINOv3 / SigLIP2 / [ArcFace 仅真人] / [MegaLoc location]）。

### E2. description-provided 公平性对照（进主表，不必分列）
- 同系统、同影片，输入换成 **description-provided**（name-anchored 之上，对 prompt 已点名实体
  追加确定性外观描述后缀；见顶部术语更新）。
- 目的：给"靠外观/描述匹配记忆"的系统（尤其纯视觉、不吃名字的 baseline）一个公平的文本抓手，
  避免只报 name-anchored 而系统性偏袒靠名字检索的系统（如 MemStrata 的 name-anchoring）。
- 报两档并列 headline；量化每个系统在"名字→描述"输入变化下的稳健性/增益。

### E3. 检索 baseline k 敏感性
- k ∈ {1,3,5,budget-matched}；主表报多行，附录扫全 k 曲线。

### E4. embedder 敏感性
- 同一批 selection，不同 pinned embedder（DINOv3 vs SigLIP2 vs ArcFace vs MegaLoc）下 VisualFidelity 对比，验证结论不依赖单一编码器。

## E 运行手册（GPU，E5 待在节点上执行）

> 跑全部影片 × 全系统的**完整交接指令 + 产物保留清单**见
> [`run_all_movies_handoff.md`](../../../../methods/MemStrata/docs/experiments/run_all_movies_handoff.md)（可直接交给跑实验的 agent）。

因果 trace 生成由 `scripts/vmem_bench/compare/generate_causal_traces.py` 统一驱动，逐 stage
幂等可续跑。当前 trace 库存（`data/BlenderOpenMovies/<movie>/gold/`）：

- 4 齐全：`big_buck_bunny`、`caminandes_2_gran_dillama`、`caminandes_3_llamigos`、`charge`
- 缺 iamflow：`cosmos_laundromat_first_cycle`
- 未开始：其余 17 部 frozen 影片（`sita_..._part1` 无 frozen gold，不算）

**步骤（在 KML 节点上）**

1. 起 vLLM 服务（从 `vllm` env，DiT 仍留在 `vace` env）：
   ```bash
   bash scripts/baselines/iamflow/servers/serve_iamflow_vllm.sh 0        # LLM :8100 + VLM :8101 同卡
   curl -s http://127.0.0.1:8100/v1/models             # ready 后再跑
   ```
2. 单卡串行（`vace` python）跑全部未完成影片：
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
     python3 \
     scripts/vmem_bench/compare/generate_causal_traces.py --movies all --skip-complete \
       --iamflow-llm-endpoint http://127.0.0.1:8100/v1 \
       --iamflow-vlm-endpoint http://127.0.0.1:8101/v1 --iamflow-vlm-cache
   ```
3. 多卡/多节点并行（D4.1，墙钟≈最慢单片）：每个 target 先起一个 worker
   `python scripts/tgpu_fs.py worker --cluster kml-a100 --node 1`，再 `--submit`：
   ```bash
   ... generate_causal_traces.py --movies all --skip-complete \
       --iamflow-llm-endpoint ... --iamflow-vlm-endpoint ... --iamflow-vlm-cache \
       --submit kml-a100:1:0 kml-a100:2:0
   ```

忠实性保证：DiT/VAE 始终在本地 `vace` 进程跑（与已冻结的 BBB trace KV 数值一致），
vLLM 只承接 IAMFlow 本就用 vLLM 的 LLM/VLM（同权重、贪心解码）；不 truncate、不换小
模型、不用 budget 代理。

## 验收标准

- 一条命令即可产出"全 baseline + visual headline"的完整对比（A 完成）。
- 每部影片的因果 trace 存在或明确 `skipped_no_trace`（绝不伪造行）；全影片覆盖后无 skip（E 完成）。
- 主表 headline 对所有系统公平（VisualFidelity 换掉 ID-Fidelity，检索 baseline 不再因无 instruction 被判 0）。
- **name-anchored 与 description-provided 两档都进主表、并列报告**，不用其中之一单独下结论
  （避免只测 name-anchored 偏袒 MemStrata、不利 baseline）。
