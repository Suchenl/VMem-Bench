# MemStrata · Baselines

> **⚠️ 已被取代（2026-07-22）：** baseline 选择 / 公平性的**当前权威**是
> [`fairness_decisions.md`](fairness_decisions.md)。本文件保留为**历史/策略上下文**（对位思路、
> 外部系统调研、ViMax 边界分析等仍有参考价值）。凡与 `fairness_decisions.md` 冲突处，以后者为准——
> 特别是：**主定量表现在只放因果系统**（`helios / longlive_rag / memflow / iamflow / decmem`），
> 脚本化 / agentic 系统（ViMax / MovieAgent / VideoMemory / StoryMem / Memento / MM-StoryAgent）
> 按 **D5** 移出主表、仅进附录定性说明。

> 本目录是 **baseline 策略与对照记录**（文档）。代码侧实现见
> `../src/vmem_bench/baseline_adapters/`（`MemoryEvidenceRecord` 映射、诊断 baseline、
> 外部框架薄转换器）。`benchmark_run/adapters/` 仅保留向后兼容 re-export。
> 原则（2026-07-17 修订）：Track A = **online gold-replay**（`baseline_adapters` 的 online adapter）；**不**包装各家 `inference.py`。真实出片用官方脚本，属 Track B。详见 `track_a.md`。

最新外部 baseline 核验表见 [`external_baseline_audit.md`](external_baseline_audit.md)。该文档的 2026-06 分类仍有参考价值；但当前论文 claim 已收敛为"因果长视频生成的分层记忆管理"，所以主表优先级改按"可观测 memory decision 是否能映射到帧/实体证据"排序，而不是按 generator 是否耦合直接降级。

---

## 复现：如何拿到外部 baseline 源码

这些第三方框架的版本由各自 adapter 的 release metadata 管理；本冻结分支不包含内部 submodule 清单。

```bash
# 在 Montage 树内（拆仓前）
python baselines/sync_baselines.py

# 或只拉一类
python baselines/sync_baselines.py Scripted
python baselines/sync_baselines.py Causal
```

`Scripted/<name>` / `Causal/<name>` 是本地 checkout（已被 `baselines/.gitignore` 忽略）。清单与 sync 脚本随 MemStrata 拆仓一起带走即可。

---

## 0. 对位思路：为什么 baseline 难找

MemStrata-Bench 的 Track A 是 **generator-passthrough / gold-replay**——评的是"**生成前/生成中被选中的历史视觉证据是否正确**"。MemStrata 原生产物是 `ComposedContextRecord`（具名资产 ⊕ role ⊕ lifecycle ⊕ instruction ⊕ forbidden）；其他 baseline 可以先导出 `MemoryEvidenceRecord`（retrieved frame / reference image / latent block / KV block），再映射回历史帧与实体后统一评分。Track B 则在同一批样本上把 context 放进真实生成环路，评最终视频 rollout。

因此对位 baseline 必须**显式暴露或可插桩回溯"为每个 chunk 选了哪些历史视觉证据"**，不要求它原生就是实体级资产管理器。几类工作要区分：

- **显式实体/资产记忆**（如 MemStrata / IAMFlow / VideoMemory）：原生维护 entity registry、asset bank 或 visual memory bank，是最贴近"分层实体记忆管理"claim 的主表候选。
- **帧级因果记忆**（如 MemFlow / StoryMem / Memento）：原生选历史帧、keyframe 或 subject memory；若能导出每步 retrieved frame/keyframe，就进入 Track A 主表。
- **latent / KV 级检索记忆**（如 LongLive-RAG / DecMem / SlotMemory）：即使原生耦合 generator，也可把 latent/KV block 回溯到 source chunk/frame 后纳入 Track A；缺失的 role/lifecycle/instruction 字段如实留空。
- **agentic 制片系统**（ViMax / MovieAgent / …）：显式维护 character/scene/asset bank 并为每个 shot 选素材，且 generator-free（可换生成器）→ **可对位**，是本 bench 的重要外部系统候选。
- **窗口 / 全历史 / 压缩上下文**：不是本文主冲突对象；只保留为诊断或 appendix，不进入 Table 1 主线。

---

## 1. 外部可复现对照（开源、可适配）

| baseline | 代码 | 对位点 / 适配方式 |
|---|:--:|---|
| **MemFlow** | ✅ cloned `Causal/MemFlow/` | 因果长视频生成最直接对手；导出每个 chunk 的 prompt-conditioned historical frame retrieval，映射回实体/状态后进 Track A 主表 |
| **IAMFlow** | ✅ cloned `Causal/IAMFlow/` | identity-aware memory / global entity IDs；最贴近显式实体记忆路线，检查其 entity registry、frame archive、active memory buffer 输出 |
| **VideoMemory** | ✅ cloned `Scripted/VideoMemory/` | 脚本化视频生成的 Visual Memory Bank，管理 character / scene / prop assets；检查 storyboard→memory→visualization 阶段产物能否导出逐 shot asset selection |
| **StoryMem** | ✅ cloned `Scripted/StoryMem/` | 脚本化视频生成的 keyframe memory bank；导出每 shot 自动抽取/选择的 memory keyframes，映射到历史帧与实体 |
| **Memento** | ✅ cloned `Scripted/Memento/`（sparse） | 脚本化视频生成的 dual-query memory：identity-relevant long memory + local context；需插桩导出 story-selected / shot-selected memory items |
| **LongLive-RAG** | ✅ cloned `Causal/LongLive-RAG/` | latent retrieval memory；导出 top-K latent entries `M_t`，回溯 source chunk/frame 后纳入 Track A |
| **DecMem** | ✅ cloned `Causal/DecMem/` | Sparse Global Memory + Anchored Local Memory；适合作为 latent/block 级因果/外推长程记忆 baseline，算力风险高 |
| **Helios** | ✅ cloned `Causal/Helios/` | 实时/因果长视频生成基础模型；不是实体记忆方法，但可作为 Causal 侧强生成器/基础设施 baseline，检查其 hierarchical token compression / context 输出是否能映射历史视觉证据 |
| **ViMax** (HKUDS, 2026-06) | ✅ cloned `Scripted/ViMax/` | 脚本化/agentic 视频生成系统，Visual Asset Planning + Asset Indexing + Character/Env Tracking；取 per-shot reference-selection 产物对位 |
| **MovieAgent** (showlab, 2025) | ✅ cloned `Scripted/MovieAgent/`（sparse） | 脚本化视频生成的 character bank + 分层 CoT，为每 shot 选 character/reference；需检查结构化落盘产物 |
| **Context-as-Memory** | ❌ coming soon | 记忆/上下文式候选；项目页显示 GitHub coming soon，放码前不进入主表 |
| **SlotMemory** | 🟡 watchlist | object-centric KV memory；项目页称有 code/checkpoints，需核验 GitHub 是否非空、可运行 |
| **OmniMem** | ❌ coming soon | explicit sparse KV retrieval；官方 repo 当前显示 code release coming soon，不能进 selected set |
| **FreeMem** | ❌ no official GitHub | AAAI 2026 tuning-free hierarchical memory；无可用官方实现，相关工作/排除项 |
| **MM-StoryAgent** (X-PLUG, 2024-08) | ✅ `X-PLUG/MM_StoryAgent`（Apache-2.0, 306★） | role-consistent 图像选择（轻量对照，脚本化 storybook） |
| **StoryAgent**(CSVG) (2411.04925) | ❌ 代码未放（"upon acceptance"，仅 demo 结果仓库） | storyboard 选 reference subject（LoRA-BE）；放码后再纳入。**注意 ≠ 上面 MM-StoryAgent**（同名不同工作） |
| **DreamFactory** (2408.11788) | ❌ 无官方 repo（拟 acceptance 后放） | image vector DB + keyframe 迭代；自带 CSFD / Cross-Scene Style 指标可借鉴；放码后再纳入 |

**适配**：以上系统优先用 **gold-replay** 跑——把其"为 chunk i 选了哪些历史视觉证据"的中间产物转成 `MemoryEvidenceRecord` / `ComposedContextRecord`，与 GT 逐 chunk 对位评 Sufficiency / Parsimony / Currency / Role / Instruction。若某系统没有显式 lifecycle/currency/负例规避字段，应在 adapter 中如实留空或映射为其最接近的公开产物，再由统一指标报告覆盖不足，而不是手工补答案。

**Table 1 范围（当前权威 = [`fairness_decisions.md`](fairness_decisions.md) D5，取代下方 2026-07-16 旧修订）**：定量主表只放**因果**系统（`helios / longlive_rag / memflow / iamflow / decmem`，与论文逐 chunk 自回归 setting 一致）。**脚本化 / agentic 系统（ViMax / MovieAgent / VideoMemory / StoryMem / Memento / MM-StoryAgent）为非因果（先出整段计划再渲染，能看到未来），移出定量主表、仅在附录做定性说明**；代码里保留 `external/scripted/` converter，但不进 leaderboard。`CoAgent` / `Context-as-Memory` / `OmniMem` / `FreeMem` 在可用代码不足前不进入 selected set。

> 下方 2026-07-16 旧修订仅作历史记录：曾把主表优先级排为 **MemFlow + IAMFlow + VideoMemory + StoryMem/Memento + LongLive-RAG + ViMax/MovieAgent**，把 agentic 系统也放进 generator-free 主表——这已被 D5 推翻。

**Track B 范围**：真实生成闭环对比仍单列为 generator-in-the-loop table。外部框架候选是 StoryMem、MemFlow、Helios、LongLive-RAG、LongLive、Memento、IAMFlow、DecMem 等完整长视频生成系统；Montage 侧候选是接入同一 producer/context 机制的 VACE、LongCat-Video、LTX-2.3、MultiShotMaster 后端。Track B 可借 StoryMem 的 ST-Bench / IAMFlow 的 NarraStream-Bench 作为初始脚本集合，但需要转换到 MemStrata-Bench 的样本/提交格式，并补充 lifecycle、forbidden、state-change、scene-return 等 hard cases。

### 1.1 ViMax 导出可行性核验（2026-06-26，已扒源码 `HKUDS/ViMax`）

**结论：可以干净导出，适配可行——但只能填满我们 schema 的一个子集；缺的轴恰是本 bench 要暴露的。**

ViMax 的 working-dir 全是**结构化 JSON、逐阶段落盘**（读产物无需重跑 LLM）：

```
working_dir/
├── characters.json                       # CharacterInScene[]: identifier_in_scene / static_features / dynamic_features(衣着)
├── character_portraits_registry.json     # {identifier:{front/side/back:{path,description}}}  ← 角色资产库(多视图)
├── camera_tree.json                      # Camera[]: active_shot_idxs / parent_shot_idx / missing_info  ← 依赖图(复用/续写关系)
├── shots/{idx}/
│   ├── shot_description.json             # ff_desc / ff_vis_char_idxs / lf_* / motion_desc / cam_idx / variation_type
│   ├── first_frame_selector_output.json  # ★ {reference_image_path_and_text_pairs:[(path,usage_text)], text_prompt}
│   ├── last_frame_selector_output.json   # ★ 同上(shot 内)
│   ├── first_frame.png / new_camera_*.png / transition_video_*.mp4 / video.mp4
└── final_video.mp4
```

**核心产物** = `shots/{idx}/{frame_type}_selector_output.json`：`reference_image_path_and_text_pairs` 就是"该镜头选了哪些素材（路径）+ 每个怎么用（text）"，`text_prompt` 是生成指令。**1 shot ≈ 1 chunk**，该 JSON 即该 chunk 的 Composed-Context 代理。资产身份由路径反查：`character_portraits/{idx}_{id}/{view}.png`→identity 资产；`first_frame.png`→continuation/previous-state；`new_camera_*.png`→camera/scene 参考。

**映射到我们的 Composed Context**：

| 我们的字段 | ViMax 可导出? | 来源 / 说明 |
|---|:--:|---|
| selected assets（具名） | ✅ | `reference_image_path_and_text_pairs` 路径反查身份 |
| role/type | 🟡 部分 | identity(portrait) / continuation(first_frame) / camera(new_camera) 可推；**scene/style/prop/state 无独立 typed 资产**（烘焙进帧） |
| instruction（如何用） | 🟡 | per-image text + `text_prompt`，自由形式生成 prompt，非结构化 position/action/transform |
| forbidden / negative | ❌ | ViMax 无负例/禁用集 |
| lifecycle / currency | ❌ | 角色库恒在（registry persist），无 superseded/过期状态 → costume/state-change、scene-return-stale 无表示 |
| cost / efficiency | ✅ | selector 是 LLM 调用，可计 model_calls(候选数)/时延 |

→ 缺 forbidden/lifecycle/scene-style-state **不是适配缺陷，而是 ViMax 结构上没有**。这些字段是否导致实际差距必须由同一评测协议验证，不能在文档里预设结论；这也是 MemStrata-Bench 需要精心构造 hard cases 的原因。

**推荐适配（generator-free 注入）**：ViMax 的 generators 是依赖注入（`RenderBackend`，`image_generator`/`video_generator` 可换）。注入 **stub generator** 把候选帧替换为样本的真实/oracle 帧，**只跑它的 planning + reference-selection（LLM）链路**，取其 selection 决策 → 与本 bench 的 gold-replay 协议一致，最省钱、最可控、可复现（不烧 ViMax 的视频生成 API）。
- 备选（纯产物读取）：让 ViMax 端到端正常跑（需其 image/video API key），跑完读 `selector_output.json`——更贵，且后续镜头 selection 依赖已生成帧。
- ✅ **adapter 思路已落地**（2026-06-29，generator-free，零花费）：scripted/agentic converter 归入
  `src/vmem_bench/baseline_adapters/` 的 `external/scripted/`（**注意：ViMax 等已按 D5 为附录定性，不进 leaderboard**）。
  - `vimax_to_records(workdir, *, id_map, chunk_ids)`：读 `shots/{idx}/{first,last}_frame_selector_output.json` 的 `reference_image_path_and_text_pairs` → `ComposedContextRecord`（路径启发式反查 identity/kind/function；usage_text→instruction.placement_action；forbidden/lifecycle 留空；`model_calls`=候选数）。`id_map`（ViMax 标识→GT key）做显式跨空间对齐，未映射的原样透传（不蒙）。
  - `load_normalized(path)`：通用入口，读任何已规范化成本 schema 的 `.json/.jsonl`（MovieAgent / MM-StoryAgent vendor 后由其 adapter 导出 `composed_context.jsonl`，直接打分，不在此臆造它们的内部 schema）。
  - 历史备注：早期一键运行脚本挂在已**退役**的 `Cine250K` 适配 kit 下（该 kit 已废弃，不要复用其路径）。

### 1.2 实跑 ViMax 拿 working-dir（2026-06-29，run kit 就绪）

**与本 bench 一致：generator-free / gold replay，0 花费、不调任何生成 API。** ViMax 内部把每个 shot 先出帧/肖像（image），再动画成视频；我们只读它每个 shot 的**参考选择**（组合决策 `*_frame_selector_output.json`），不读视频。gold replay 下把它要"生成"的帧**替换成金标样本的真实帧**，和 Montage 的 oracle backend 切源片同理。

**关键澄清（回答"DeepSeek 够不够"）：不够。** `reference_image_selector` **末步必走多模态**——把候选图的真实像素 base64 塞给 `chat_model`（`agents/reference_image_selector.py` 的 `image_url`）。DeepSeek-chat 纯文本、且无视觉模型 → 选择器失效。所以做选择的 VLM 要能看图。

**配方（kit 已备，全程 $0）**：
- `chat_model` = **本地 InternVL3.5**（lmdeploy 起 OpenAI-compatible，视觉可用）→ $0，与 Montage 感知后端复用同一服务。
- `image_generator` = `OracleFrameGenerator` → 喂金标样本的**真实资产帧**作候选（不调生成 API）→ $0。
- `video_generator` = `StubVideoGenerator`（从不打分视频）→ $0。

> **历史备注（run-kit 已退役）：** 上述 run-kit 原挂在已废弃的 `Cine250K` 适配目录下
> （`vimax_stub_backend.py` 的 `OracleFrameGenerator` / `StubVideoGenerator`、
> `vimax_montage_eval.yaml`、自带"婚礼"剧本的 `run_vimax.py` 等），并在 ViMax 自己的环境
> （pydantic/moviepy/langchain）里配一个本地视觉 VLM（如 lmdeploy 起 InternVL3.5）+ oracle 帧注入跑通。
> **该 kit 与其路径均已退役，不要复用**；ViMax 已按 D5 降为附录定性说明。上面的"配方"仅保留其
> generator-free / oracle 注入的**方法思路**作为历史参考。

---

## 2. 边界：为什么 ViMax 是"对照候选"而非唯一竞品（反 salami）

MemStrata 标题 = *"From Passive Retrieval to Active Composition"*。ViMax 这类系统包含参考选择、资产索引、依赖图和条件生成工程，因此可以作为外部 agentic 对照，但它不是唯一比较对象；Table 1 的主对比范围已收敛为生成器解耦的外部上下文组织器。逐条对比（左列均来自 ViMax 论文）：

| 维度 | ViMax (2026-06) | MemStrata |
|---|---|---|
| 本质机制 | **RAG**（检索 screenplay 全局上下文）+ **graph-based dependency**（检测共享视觉元素→依赖图→用前置 shot 生成内容做 reference conditioning） | **Active Composition**：从 Production Asset Space 主动选资产并分配使用角色与约束 |
| 资产本体 | 前置 shot 生成内容作 reference，依赖**单调累积**，无生命周期 | role-typed assets + **lifecycle**（引入/复用/修改/废弃）+ 资产关系 + 失败样本 + 未来约束 |
| 主动废弃/负向 | **无**——检测到依赖就 condition；"越相似越复用"在换装/旧机位/过期状态上正好犯错 | **forbidden carryover + 负例规避 + costume/state change + scene-return 排除过期状态** |
| 记忆经济学 | 无去冗余/无损丢弃/成本目标 | **Low-Redundant Restore + Low-Error Discard + Composition 成本** |
| 使用指令 | 选了 ref 直接 condition，不产"如何用" | 每个资产配**显式 instruction**（如何用它生成该 chunk） |
| context 地位 | 隐式工程中间态 | **显式、可解释、可评测的一等产物** |
| 评测 | 最终视频一致性（ViMax-Bench，**generator-dependent**） | context 本身（**generator-free / gold-replay**） |
| 系统目标 | 端到端出片（idea→video），asset 只是工程模块 | 不出片、不训生成器，单点研究记忆组合机制+表示 |

**为什么仍值得纳入**：ViMax 能导出 per-shot reference selection，适合在 generator-free setting 下转成 Composed Context 评分。它是否在 Currency / 负例规避 / state-change / Instruction Fidelity 上落后，应由数据和同一协议给出，而不是预设。该对照的作用是补强外部系统比较，而不是替代检索式主 baseline。

**两条红线（必须守住）**：
1. "为镜头选素材"这一动作 ViMax/MovieAgent 都有，**不能当 MemStrata 卖点**——新颖性须收敛到 lifecycle + 主动废弃/forbidden + 记忆经济学（Low-Redundant Restore / Low-Error Discard）+ 显式 instruction + generator-free 评测。
2. ViMax 为 **2026-06 并发工作**，匿名评审按 concurrent 处理、自包含引用。

---

## 3. 内部 fallback 与诊断对照（非 SOTA，自实现）

已落代码 `../src/vmem_bench/baseline_adapters/`（诊断 baseline，原 `passive.py` 路线）。这些方法不应作为 Table 1 的主要攻击对象；它们只承担两类作用：外部方法不可运行时的最低 fallback，以及系统诊断/appendix。

| baseline（代码 name） | 机制 | 预期短板 |
|---|---|---|
| `text_retrieval` / `Text-Retrieval` | q=下一段提示词/当前 plan；k=历史 chunk 摘要 | 内部强检索 fallback |
| `keyframe_retrieval` / `Visual Retrieval` | q=当前帧/计划视觉 embedding；k=历史 chunk embedding | 内部强视觉检索 fallback |
| `Text-Visual Retrieval` | 融合文本与视觉相似度 | 内部最强检索 fallback，优先补齐 |
| `full_history` | 每 chunk 灌入全部累积记忆 | 仅诊断/appendix；不是主攻击对象 |
| `sliding_window_1` | 只注入最近 window 内用过的资产 | 仅诊断/appendix；不是主攻击对象 |
| `oracle_selection` | selection-only 诊断上界 | 仅诊断/appendix；不进入主表主线 |
| **Montage 主动组合**（本工作） | active composition | 与检索式长上下文方法和外部系统比较 |

---

## 4. 与代码的关系

- 本文档：baseline **策略/对照记录**（外部可复现短名单、ViMax 边界、内部 fallback/诊断定位）。
- `../src/vmem_bench/`（baseline adapters 另立）：内部 fallback/诊断 baseline 的**可运行实现**（`passive.py` / `base.py` / `demo.py`）。
- scripted/agentic 对照适配器归入 `../src/vmem_bench/baseline_adapters/external/scripted/`（ViMax 工件→`ComposedContextRecord`，MovieAgent/MM-StoryAgent 走 `load_normalized`）；按 D5 这些系统仅进附录，不进 leaderboard。
- 真要跑某第三方 baseline：把其开源代码 **vendor 进本目录** `baselines/<model>/`（连同 LICENSE），适配器只读其落盘工件转 `ComposedContextRecord`——**本 bench self-contained，不引用 bench 之外的任何文件**。
