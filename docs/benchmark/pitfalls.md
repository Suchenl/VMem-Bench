# MemStrata-Bench Annotation Pitfall Notes

本页记录 MemStrata-Bench 离线标注流水线中已经踩过的坑、症状、根因和修复方向。目的不是替代
`workflow.md` / `schemas_and_contracts.md`，而是让后续 agent 不再重复把同一类错误写回代码。

## 2026-07-08: Big Buck Bunny 标注质量差、flagged 多、速度慢

### 症状

- `derived/assets/` 中出现明显资产混杂：
  - `char_big_buck_bunny` 混入蝴蝶 crop。
  - `char_red_fox` 混入松鼠/兔子 crop。
  - `prop_tree_branch` 的 cover 变成紫鸟。
  - `prop_tree_trunk` 混入松鼠或蝴蝶。
- 多个 chunk 被标为 `flagged`，尤其多实体同框的 chunk。
- 单个 flagged chunk 常耗时 200-370 秒；Big Buck Bunny 运行到 chunk 15 时已经产生 70+ 次
  discover/draft/verify 级别的 VLM 调用，且 verify 内部还会为每个 crop 额外调用 VLM。
- 旧运行中 `gold/` 迟迟未写出，说明标注管线还停留在 build/candidate 阶段，不能当正式 gold 使用。

### 根因

1. **Grounding query 错误**  
   旧实现把完整 `description` 交给 GroundingDINO。描述里包含动作、相邻物体、情绪、时间顺序等文本，
   多实体同框时 detector 很容易被上下文词带偏。例如要定位 Big Buck Bunny，却抓到 Red Fox。

2. **exact-name 复用绕过身份门禁**  
   `consolidate_observation()` 先按 `(kind, name)` 直接复用已有实体。只要 VLM 复用了同名，
   错 crop 就会被追加到正确实体的 representations 中，污染资产库。

3. **没有落实 decomposition 设计中的静态身份漏斗**  
   设计文档要求“静态身份与动态状态解耦”和三级匹配漏斗：大类过滤、静态属性软匹配、
   视觉/VLM 仲裁。旧实现只有 kind/name/embedding/VLM judge，缺少静态属性冲突拦截。

4. **QA 重试不能修 crop 错误**  
   QA feedback 会让 annotator 修 prompt 文本，但不会让 GroundingDINO 更会定位。若失败只剩
   `crop_match`，继续跑第 2/3 轮通常只是重复浪费 VLM 调用。

5. **过度依赖大 VLM**  
   默认 `qwen3-vl-32b` 被同时用于 entity discovery、draft、verify、crop_match 和 same-entity
   arbitration。32B 模型本身慢，再叠加多 branch、多 round 和 per-crop 审计，成本会指数放大。

6. **模型名与 vLLM served-model-name 错配**  
   若服务实际以 `Qwen3-VL-8B-Instruct` 启动，而 client 发送 `qwen3-vl-32b`，OpenAI-compatible
   endpoint 会返回 404。旧 client 会把 404 当可重试错误，导致每个失败分支额外等待多次 backoff。

7. **后端池大小被误当成单 chunk 分支数**  
   多个 annotator/verifier endpoint 的目的首先是提高数据集级吞吐：不同 chunk/attempt 可以轮询不同后端。
   旧实现把 `max(len(annotators), len(verifiers))` 直接当成每个 chunk 的 QA branch 数，导致开 8/16 个服务时，
   单个 chunk 也会发起 8/16 路 discover/draft/verify，速度和 flagged 风险一起膨胀。

### 已采用的修复

- Discovery schema 增加 `grounding_phrase`：用于定位的短名词短语，不允许动作、情绪、时间句或周边物体。
- Discovery schema 增加 `static_attributes`：仅保存稳定身份属性，如 species/subcategory、primary_color、
  size_class、object_type、location_type 等。
- GroundingDINO 改用 `grounding_phrase`，不再默认使用长 description。
- `Registry` 增加运行期 `static_attributes`，exact-name 复用和 embedding 复用都必须通过静态属性兼容检查。
- `crop_match` 失败的 representation 会在提交前从 entity representations 和 embedding store 中剪掉，
  防止坏 crop 污染 asset bank。
- 若某轮 QA 只剩 `crop_match` 失败，提前停止重试，避免继续消耗 VLM。
- location / full-frame crop 不再做逐 crop VLM 审计；presence checks 已覆盖场景是否存在，
  full-frame location crop 再做 same-entity VLM 判断收益低、成本高。
- VLM client 对 HTTP 4xx（尤其 model not found 的 404）改为 fail-fast，并在错误中提示检查
  `base_url` 和 `model` / `--served-model-name` 是否一致。
- `branches_per_chunk` 与 endpoint pool 解耦：默认每个 chunk 只跑 1 个 annotator/verifier 分支，
  但会按 chunk/attempt 轮询 `--annotator-urls` / `--verifier-urls` 后端池；只有显式设置
  `--branches-per-chunk > 1` 才做单 chunk 多样本采样。

### 落地状态（2026-07-08 实施完成，含测试）

上述"已采用的修复"已全部落进代码，并在原有修复之上补齐了深层设计项。代码入口：

- `common/schemas.py` — `Entity` 新增向后兼容的 `static_attributes: dict[str,str]`（进 gold，人审可见）。
- `annotation/config.py` — 新增 `branches_per_chunk` / `verifier_model` / `verifier_video_for_retry` /
  `grounding_min_frames` / `static_overlap_threshold` / `crop_audit_score_threshold` / `review_seed`；
  `grounding_score_threshold` 调到 0.35（≥ GroundingDino.box_threshold，使其真正起精选作用）。
- `judger/vlm.py` — `_call_api` 对 HTTP 4xx fail-fast（错误信息含 model 名）+ 5xx/网络退避重试 +
  JSON parse 轻量无 backoff 重试；`judge_same_entity` 文本路径 prompt 改为严格、default 不合并。
- `annotation/vlm_roles.py` — discovery schema 加 `grounding_phrase`+`static_attributes`；
  draft schema 加 `deprecates_representations`，roster 带每个实体历史 rep_id；
  `VerifierRole` 加 `crop_audit_score_threshold`，crop 审计跳过 location/full_frame/高分 crop，新增 `verify_chunk_video`。
- `annotation/consolidation.py` — 三级身份漏斗：kind → 静态属性兼容门禁（`_static_compatible`，阈值
  `static_overlap_threshold`）→ embedding 双阈值 + VLM 灰区仲裁；默认偏"宁分勿合"；description 改首版非空优先；
  `grounding_score` 写入 `rep.qa`；`present_payload` 带 `grounding_score`/`bbox_source` 供 verifier 决策。
- `annotation/pipeline.py` — `_ground_and_crop` 用 `grounding_phrase` + 时序一致性（`min_frames` 帧检出）+
  返回 `grounding_score`；`_branch_role_pairs` 按 chunk/attempt 轮询 endpoint 池；`_only_crop_failures` +
  crop-only early-stop；`_union_feedback` 取所有分支失败并集；`_prune_failed_crops` 提交前剪坏 rep/空实体；
  `_best_cover_rep` 选最高 grounding score 的 rep 作 cover；`_fallback_chunk_annotation` discovery 全空时
  注入兜底 location（不再 RuntimeError）；attempt≥2 且开关开且 verifier 支持视频时物化 chunk clip 走视频核对
  （clip 落 `derived/clips/`，失败回退帧）。
- `annotation/drafting.py` — `state_events_from_draft` 支持 VLM 指定 `deprecates_representations`（精确废
  弃，留空才全量）；multi-instance 检测用实体所有 rep embedding 的均值代表向量（不再是 `reps[0]`）。
- `annotation/review.py` — spot-check 改风险加权采样（state-change/multi-instance/re-appearance/低分/
  vlm_fallback 优先）+ `seed=None` 默认随机（可复现时传 int）；`apply_patch` 在 merge/drop/rename 后用
  gold 的 embeddings sidecar 重算受影响 chunk 的 `scenario_tags`；`field_edits` 支持 `entities[<id>].description`
  （补偿 description 首版优先后人审修描述的能力）。
- `annotation/run.py` — CLI 加 `--branches-per-chunk` / `--verifier-model` / `--grounding-min-frames` /
  `--grounding-score-threshold` / `--static-overlap-threshold` / `--crop-audit-score-threshold` /
  `--verifier-video-for-retry` / `--no-verifier-video` / `--review-spot-check` / `--review-seed`；
  verifier 用独立 model（per-role），`VerifierRole` 传 threshold。

测试（确定性，无 VLM/GPU；`cd benchmarks/MemStrata && PYTHONPATH=src <venv-python> tests/...`）：

- `tests/test_annotation_pipeline.py`：9 个 unit checks + stub E2E（含 QA 重试、crop pruning、patch/freeze/publish）全过。
- `tests/test_schemas_v2.py`：v2 契约 round-trip 全过。
- `tests/test_annotation_fixes.py`：新增 7 项自检 —— deprecates 精确化、union feedback、best cover、
  fallback location 幂等、grounding_phrase 生效、时序一致性、apply_patch 重算 scenario_tags —— 全过。

### 第二轮优化落地（2026-07-08，identity 污染 / 成本可观测 / 长视频韧性）

第一轮修完 Pitfall_Notes 已列的根因后，复审又发现一批 identity / 成本 / 韧性缺陷，已全部落地并加测试：

- **identity 污染（A1）**：`consolidation.best_match` 只用 `bbox_source=="grounding_dino"` 的 rep embedding 做跨
  chunk 比对；`vlm_fallback`/`full_frame` 的整帧 embedding（含背景+其他实体）不再参与 identity matching，且
  当前观察若为非 grounded crop 则直接跳过 embedding 灰区路径（只靠 name+static 合并，宁分勿合）。堵住了
  静态属性门禁之外仍存在的「整帧 embedding 误合并」路径。
- **灰区仲裁可靠性（A2）**：`consolidate_observation` 灰区调 `judge_same_entity` 时改传候选实体的 best
  grounded crop（图-图，`_best_grounded_crop`），仅在候选无 grounded crop 时回退 description（图-文）。图-图
  比首现文本描述可靠。
- **静态属性大小写门禁（A5）**：`_static_compatible` 对 key+value 做 lower 归一化，避免 VLM 跨 chunk 产出
  `Species`/`species` 时 key 集不相交 → 门禁静默失效。
- **剪枝精确性（A3）**：`vlm_roles` 的 `crop_match` check 现在带 `representation_id` + `name` 字段，
  `_prune_failed_crops` 优先按 rep_id 精确剪，不再靠解析 detail 字符串。
- **per-crop 审计成本（A4）**：`_should_audit_crop` 一并跳过 `vlm_fallback`（整帧审计必然过 = 纯 VLM 浪费），
  靠 presence_precision + 人审兜底。
- **grounding_phrase 边界（A6）**：`_ground_and_crop` 逐个 strip 取第一个非空，避免空白 grounding_phrase
  短路成空 phrase 喂给 detector。
- **token 可观测（B7）**：`vlm._call_api` 解析 `response.usage` 累加到 per-judger 计数器，pipeline 在 run_done
  汇总 annotator+verifier 的 prompt/completion tokens，`summary["total_tokens"]` + `evlog emit("usage")`。
  Pitfall 强调成本优化，现在有了反馈信号。
- **日志体积（B8）**：chunk 级 `registry` 事件改为只 dump 本 chunk touched 的 entity（delta），全量 registry
  在 `run_done` 时 `registry_final` emit 一次。原 O(chunks*entities) 全量 dump（68-chunk stub 已 962KB）不再
  膨胀。
- **discovery prompt 预算（C9）**：`known` 列表带 `static_attributes` 且按 `known_entity_limit`（默认 60）
  截到最近 N 个实体、description 截断 120 字符，防长视频 prompt 无界膨胀。
- **deepcopy COW（D11）**：`_run_branch` 的 registry 改为「entities deepcopy + embeddings 浅拷贝 dict」，
  分支追加新 embedding / 改 entity 不污染原 registry，省掉每分支 deepcopy 全部 embedding 向量的开销。
- **checkpoint/resume（D10）**：每 `chunk_done` 写 `build/checkpoint.json` + `checkpoint_registry.json` +
  `checkpoint_embeddings.safetensors`；`annotate_movie(resume=True)` + CLI `--resume` 从断点恢复
  （校验 layout_hash 一致才恢复），跳过已完成 chunk。长视频后期崩了只重做当前 chunk，不重来。

测试新增 9 项（`test_annotation_fixes.py` 现 16 项全过）：best_match 跳过非 grounded embedding、灰区图-图
仲裁、fallback crop 不走 embedding 路径、crop_match check 带 rep_id、`_should_audit_crop` 跳 vlm_fallback、
`_static_compatible` 大小写归一化、空白 grounding_phrase 回退、checkpoint write/load round-trip、COW registry
隔离。`test_annotation_pipeline.py`（9 unit + e2e，含 checkpoint 写入）/`test_schemas_v2.py` 仍全过。

### 第三轮优化落地（2026-07-08，B8 前端回归 / 契约漂移 / correlated errors / batch 推理 / checkpoint 写盘成本 / 发布校验）

复审深入前两轮未触及的模块（chunking/grounding/embedding/web/publish）与跨文件契约，发现两项前两轮
自己引入的债务 + 四项新优化点，全部落地：

- **F1 — B8 前端回归（自引入债务，必修）**：第二轮 B8 把 chunk 级 `registry` 事件改 delta 后，`web/static/index.html`
  的 `renderAssets` 仍是替换式（`innerHTML=""` 后只画 delta 几个）→ 每个 chunk 完成后资产库只剩本 chunk touched
  实体，之前的全消失。修：前端 `state.assets` map，`registry` 事件 delta upsert by `entity_id`，`registry_final`
  （run_done 全量）整体替换一次；`resetAll` 清 `state.assets`；`fmt`/`addLog` 加 `registry_final` 分类。
- **F2 — 契约漂移（自引入债务，SDD 硬规则）**：前两轮加进 gold 的字段没同步 `docs/schemas_and_contracts.md`
  （契约第 4 行明令"不得私自增删"）。补：`Entity.static_attributes`（consolidation identity funnel 用，人审可见，SUT
  不消费）、`Representation.qa.grounding_score`（0=full_frame/vlm_fallback，标注元数据，不进 SUT 契约）、
  `review_patch.field_edits` 的 `entities[<id>].field` path 形式。
- **F3 — correlated errors（principle #11）**：`vlm._call_api` 固定 `temperature=0.0`，`branches_per_chunk>1` 时多
  分支同 prompt 同 frames 同 temperature → 产出高度相关，违反独立性。修：`_call_api` 加 `temperature` 参数；
  `AnnotatorRole`/`VerifierRole` 的 discover/draft/verify 方法加 `temperature` 透传；`config.diversity_temperature`
  （默认 0.3）；pipeline 在 `attempt>=2` 或 `branch>=1` 时传 `diversity_temperature`，branch 0 / attempt 1 保持
  0.0（canonical run 可复现）。评测确定性不受影响（指标是 gold 上的集合运算，不碰 VLM 输出）。
- **F5 — embedder 真批量（性能）**：`DinoV3Embedder` 原 `embed_image` per-crop 调用，lock 序列化 → pipeline 的
  `ThreadPoolExecutor` 并行变串行。加 `embed_batch(images)` 一次 forward 多图（CLS token 批归一化），pipeline 优先
  `hasattr(embedder,"embed_batch")` 走批量、stub 无则 fallback `pool.map(embed_image)`。grounder 不批量：GroundingDINO
  跨图异 phrase 的 batch 语义复杂易错，per-call lock 保留（`ponytail:` ceiling 已标，升级路径是支持 batched
  open-vocab detection 的后端）。
- **F6 — checkpoint 写盘成本（长视频韧性）**：D10 每 chunk 重写全量 `checkpoint_embeddings.safetensors`，长 video
  后期 O(chunks×embeddings) 写盘是 checkpoint 主成本。修：`config.checkpoint_embedding_interval`（默认 5），
  `checkpoint.json`+`checkpoint_registry.json` 每 chunk 写（轻），safetensors 每 N chunk 写 + run_done 全量写一次；
  `_load_checkpoint` 在 sidecar 滞后于 `last_chunk_id` 时截断 resume 点到 sidecar 覆盖 chunk，并删 to-be-rerun chunk
  的 rep（防 consolidate 重跑重复）。
- **F7 — publish 发布校验**：`publish._check_frozen` 原只查文件存在 + `human_reviewed`。补：`schema_version` 跨
  layout/gold 文件一致（契约 §0）+ `layout_hash` 存在（契约 §5.3，harness 启动校验依赖），不一致/缺失即拒发布。

未做（记录为后续，非标注逻辑修复）：

- **bench scoring/harness 缺失** → ✅ **第四轮已落地**（见下"第四轮落地"段）：`vmem_bench/scoring/`
  消费 gold + SUT `ComposedContextRecord` 算 5 指标 + harness 校验 + CLI（in-process / records-dir）。
  `../method/design.md` 原 YAGNI 属分阶段，现 bench 包已可真正跑评测。
- **grounder 并行变串行**：`self._lock` 序列化 GPU 推理是线程安全必需；F5 已用 batch 化 embedder 缓解，
  grounder batch 见 F5 ceiling。

测试新增 6 项（`test_annotation_fixes.py` 现 22 项全过）：temperature 透传、`embed_batch` 空短路、
`write_embeddings=False` 跳 sidecar、checkpoint 滞后截断、publish schema_version 不一致、publish 缺 layout_hash。
`test_annotation_pipeline.py` 9 unit 全过（e2e 需 ffmpeg/ffprobe，环境依赖）；`test_schemas_v2.py` 全过。

### 后续不要再做

- 不要把长 description 当 grounding query。定位短语必须是短名词短语。
- 不要让 exact-name 直接合并实体；同名也必须过静态属性门禁。
- 不要把 verifier 发现的坏 crop 复制进 `derived/assets/<entity_id>/`。
- 不要依赖 QA 多轮来修 grounding 错误；grounding 错误要在定位/裁图阶段解决。
- 不要默认让 32B VLM 承担所有子任务。
- 不要把 annotator/verifier 端口数当成单 chunk branch 数；端口池用于吞吐，`branches_per_chunk`
  才是单样本冗余采样开关。

### 第四轮落地（2026-07-08，bench scoring/harness —— 第三轮"未做"项的实现）

第三轮把 bench scoring/harness 列为"后续缺口"，本轮把它补齐：`vmem_bench` 从只有
annotation+publish+web 升级为**可真正跑评测**（消费 gold + SUT `ComposedContextRecord` 算 5 指标）。
契约 §3/§4 已冻结，本轮是 impl 阶段（SDD 时序合规）。

新增模块 `src/vmem_bench/scoring/`：

- **`metrics.py`** — 5 指标逐 chunk 确定性集合运算（契约 §4 字面实现，无阈值/VLM/随机）：
  `Sufficiency=|S∩P_ret|/|P_ret|`、`Parsimony=|S∩P_ret|/|S|`、`Compactness=mean exp(-λ(|R_s|-1))`
  （λ=0.2）、`Fidelity=|G∩I|/|G|`、`Avoidance=1-|R∩F_active|/|F_active|`。边界规则（反躺平）严格按
  契约表：`P_ret=∅→Suf N/A`、`S=∅,P_ret≠∅→Suf/Par 0`、`S=∅,P_ret=∅→Par N/A`、`S=∅→Com N/A`、
  `G=∅→Fid N/A`、`I=∅,G≠∅→Fid 0`、`F_active=∅→Avo N/A`。`F_active(t)=∪{event.deprecates: event.chunk_id<t}`
  ——event 自身 chunk 不计入（该 chunk 内旧状态仍是合法参考，event 之后才禁用）。
  `aggregate_scores`：per-metric 均值跳过 N/A + `n_applicable` 计数 + per-`scenario_tag` 分桶均值 +
  权重重分配（某指标全片 N/A——如整片无 state_event 的 Avoidance——其权重按比例分给其余指标，绝不静默
  记 1.0）+ `MemStrata Score` 加权 headline。
- **`runner.py`** — `load_gold` 加载 + harness 启动校验（`human_reviewed` 两文件皆真、`schema_version`
  跨 layout/gold 一致、`layout_hash` 存在且与 gold `annotation_provenace.layout_hash` 一致、文件齐全；
  不满足抛 `HarnessError`，CLI 退出 2）。`build_observation_packet` 从 gold 构造 `ObservationPacket`
  （首现实体带 description、重现不带、`state_events` 按 chunk 过滤）。`run_replay` 逐 chunk：
  **prompt→收 record→（外层算指标）→observation**——observation 在评分后才发，SUT 在 chunk t 看不到
  chunk t 的 oracle 反馈（契约 §3.3 post-scoring 设计）。`score_run` 串起回放+评分+聚合+versioned 报告
  （含 `horizon_curve` per-chunk 诊断 + timing/model_calls）。
- **`__main__.py`** CLI：`--movie-dir`、`--sut-mode {in-process,records-dir}`、`--weights-json`、`--report`。
  in-process 用 bundled `BenchReplayAdapter`+`HashEmbedding`（确定性参考 SUT + harness 自检，无 GPU/权重）；
  records-dir 评离线 SUT 预产的 `chunk_<NNN>.json`（SUT 已自行回放，observation 不重投）。

**发现的固有特性（非 bug，contract 字面行为，记录备查）**：回放顺序 prompt→record→observation 决定
**chunk 0 的 `handle_prompt` 时 SUT 记忆为空** → 首现 chunk `S=∅` → Suff/Par/Com N/A，而首现 chunk 的
gold instruction 是 `introduce`（G≠∅）→ `I=∅` → **Fidelity=0**。即"首现 chunk 的 introduce 指令 SUT 记忆
空无法满足"是 memory-replay 设计的必然结果（实体经 post-scoring `ObservationPacket` 才进记忆）。这把
Fidelity 的有效评测集中到 continuity chunk（SUT 有记忆、能选能下 continuity 指令）。若后续要让首现 chunk
Fidelity 可评，是 contract/spec 层决策（如让 gold_instructions 只对 P_ret 实体设），不在 impl 层改。

测试 `tests/test_scoring.py` 19 项全过：5 指标单元 + 6 边界用例（N/A/反躺平/过选惩罚/多 rep 惩罚/
deprecated 保留→Avo 0、避让→Avo 1）+ F_active 边界 + 聚合（影片级 Avoidance N/A 权重重分配、scenario_tag
分桶）+ harness 5 项拒绝（unfrozen/schema 不一致/缺文件/layout_hash 不一致）+ observation packet
（首现 vs 重现、state_events 按 chunk）+ 端到端 `score_run` 过 `BenchReplayAdapter`。CLI smoke 两模式
（in-process / records-dir）报告一致（0.9333，确定性）。全回归 6 套件无回归（replay_roundtrip /
production_dedup / scoring / annotation_fixes / schemas_v2 / annotation_pipeline）。

剩余 ceiling（不在本轮）：records-dir 模式不驱动 stateful 外部 SUT 的 observation（in-process 已覆盖
stateful 回放；live file-IPC 模式留给后续）；真实 embedder（DinoV3）接入 in-process CLI 走 `--embedder`
配置（当前 HashEmbedding 足够确定性自检 + 单元评测）。

**收尾（dead code 清理 + 契约同步，SDD 漂移修复）**：

- 删 `common/report.py`（`aggregate_report`/`ChunkScore` 全仓无 caller，被 `scoring/metrics.py::aggregate_scores`
  取代；且旧版不支持 N/A 跳过 / 权重重分配，无法复用）。删后全回归 6 套件无破坏、无 import 残留。
- 契约 §4 第 195 行原"报告结构沿用 `common/report.py`"是 SDD 漂移（实现已用 `scoring/metrics.py`）→ 改为
  指向 `scoring/metrics.py::aggregate_scores` 并补完整报告 schema（`n_applicable` / `per_metric_mean`
  含 null / `per_scenario_tag_mean` / `versions` / `horizon_curve`）。同时修契约 §5 重复的 `## 5.` 标题。
- README 目录树去 `report.py`，加 `scoring/` 模块说明 + 入口。
- `test_scoring.py` 加 2 项 CLI 测试（in-process 与 records-dir 报告一致性、unfrozen gold → exit 2），
  现 21 项全过。

## 2026-07-13: 身份识别"聚类+阈值"失效，人工审核比直接标注还难 → VLM 主导批量聚类重构

### 症状

- Track-first 第一版（§annotation_tracking_internals.md §3.1）跨镜头身份判定是在线贪心最近邻聚类
  （`reid_assign`：融合余弦 `>= reid_threshold` 就并入，否则新建），仍然反复出问题：
  `memstrata_annotation_baseline` 探针量出 23.4% 重复实体 id；BBB v3 仍有 3 组身份别名要人工合并、
  12 个 must_review；`sam3_exemplar_bbb` 探针（2026-07-12）里 argmax exemplar 余弦经常选错物种，
  且第一二名 margin 常只有 ~0.1（如 f9263：animal_red 仅 0.31）。
- 用户反馈：人工审核候选对（pairwise identity card）的工作量比直接看一遍视频标注还大——自动化没有
  减少判断量，只是把"人看一遍视频标一次"换成了"机器做很多次不可靠的局部判断 + 人事后审核每一个可疑
  判断"，而可疑判断的数量（candidate pair 数）远超真实身份判断数量（全片实体数）。

### 根因（非症状，三点）

1. **任务被建模错了**：roster 已先验知道全片大致有哪些实体，"tracklet 是谁"其实是小候选集上的
   **封闭集分类**，被当成**开放集聚类 + 全局绝对阈值**——阈值无法同时满足"不误合并"和"不误分裂"。
2. **判别信号够不上任务粒度**：DINOv3/SigLIP 是语义级通用表征，不是为 instance-level 细粒度 re-id
   训练的；同类视觉相似对象（红狐狸/红松鼠/苹果）在其空间里判别力不足，这是能力天花板，不是再调一个
   参数能解决的。
3. **在线贪心不可逆**：局部一步错只能靠事后人工 merge/`identity_candidates`/`identity_adjudication`
   补——这些"事后擦"本身就是审核负担的主要来源。

### 已采用的修复（2026-07-13 实施完成，见 annotation_tracking_internals.md §9 详述）

- 新增 `annotation/identity_clustering.py`：**complete-link/average-link 层次聚类**替代"简单连通
  分量"式的隐式 single-link——避免噪声 embedding 下一条弱"桥"链式误合并不同个体。纯算法，CPU-only，
  零 VLM/GPU，`tests/test_identity_clustering.py` 专门有链式误合并的对照测试。
- 新增 `annotation/identity_resolution.py`：确定性预聚类 → VLM 簇内校验（`vlm_roles.verify_cluster`，
  权威判定，非灰区兜底）→ VLM 跨簇合并（复用 `group_same_individuals`，权威，不再只是 review-only
  建议）→ roster 完整性检查（大 track 未匹配 roster → finding，不强分类）。独立的簇校验/跨簇合并调用
  走线程池并发（用户确认显卡/endpoint 充裕）。
- `reid.py` 提取 `commit_tracklet_observation`（entity 创建 + representation 记账的共用尾段），
  新旧两条路径（`config.identity_resolution_mode = cluster_vlm | greedy`）共用，不重复实现落盘逻辑。
- 身份关键决策（`verify_cluster`/`group_same_individuals`）走 `judge_role`（`--judge-base-url`，
  默认 32B）——这两步现在是权威决策，出错代价高，不再套用"默认不用 32B"的旧经验；roster 发现/命名/
  prompt 起草仍用 8B，`run.py` 已有 hybrid serving 机制零新增插线。
- 契约零改动：新增溯源全部落在已有自由字段——`Representation.qa.cluster_group_index`、
  `EntityRegistry.annotation_provenance.identity_resolution_mode`/`precluster_linkage`；过程留痕写
  `tmp/identity_resolution.json`（镜像 `identity_candidates.json` 的定位，不进发布包）。
- 测试：`tests/test_identity_clustering.py`（14 项，含链式误合并对照）+
  `tests/test_identity_resolution.py`（20 项，VLM mock）+ `tests/test_pipeline_track_first.py` 新增
  2 项（config 开关的源码级回归守卫 + artifact round-trip）。全套 `pytest tests -q` 206 passed（另 7
  个环境相关的既有失败与本次改动无关，clean tree 上同样失败，已核实非回归）。

### 后续不要再做

- 不要把跨镜头身份判定做成"单一全局阈值 + 贪心在线 assign"——即使调阈值调得再细，开放集聚类的框架本身
  就无法同时满足"不误合并"与"不误分裂"；封闭集分类（有 roster）应该走批量聚类 + 权威裁判，不是在线贪心。
- 不要用简单连通分量（graph + 阈值）做预聚类——等价于 single-link，噪声 embedding 下极易链式误合并；
  要用 complete-link/average-link。
- 不要把"聚类候选"当成最终决策——聚类只用来把 VLM 要看的图片数量压下来，真正的身份判定权威来源是
  VLM 权威判定这一层，不是聚类阈值本身。

## 2026-07-13: cluster_vlm 仍产出虚假安全 → 人工 seed 约束的生产路线

`big_buck_bunny_h800_parallel_v1` 证明 v2 聚类只修到了下游 merge/split，未修 roster 本体：
64 chunk 的 summary 为 `0 flagged`，但 58/64 有 findings、125 次 prompt 漏实体、角色物种系统性错标、
review queue 仍有 50+ 卡。根因不是再调一个聚类阈值：

1. 自动 roster 的错误 name/kind/static attributes 会成为所有下游 hard gate 的错误前提；
2. track-first 的 `qa_report.flagged` 曾只等价于 `present` 是否为空，findings 不阻断；
3. 自动 discovery/cluster/naming 同时在决定 ontology、identity 和文案，错误相关且无法独立校验；
4. location 不参与 headline metrics，却制造最高优先级 alias 审核；开放式 state event 把 visible/in-focus
   误当 irreversible change。

已转向 v3：

- 生产 CLI 必须 `--roster-seed <human_confirmed.json>`；自动 discovery 只能显式
  `--proposal-only`，strict freeze 拒绝 proposal provenance。
- canonical seed 固定 entity ID/name/description、identity scope、多视角 exemplar 和有限 event policy。
- individual tracklet 做同 kind 多 exemplar 封闭集分类；低 score/margin → `unknown/reject`，不强分类。
- category prop 不做实例生命周期；location 仅由帧作为 prompt context，不再成为 identity review 资产。
- 状态事件只允许有限 ontology；prompt canonical-name/event 覆盖确定性补齐。
- unknown、seed 漏证据、prompt omission、invalid event、alias split 全部 flag 并阻断 freeze。
- 审核改为一个 canonical entity 一张 crop-grid 卡、一个 entity 一条 state timeline；不再回到 pairwise 候选爆炸。

### 后续不要再做

- 不要把 `0 flagged` 当质量信号，除非 flagged 已覆盖所有 blocking findings。
- 不要让 VLM/聚类创建生产 entity_id 或改写人工 seed 的 canonical identity。
- 不要用更多 judge 副本掩盖错误 ontology；上游本体错误时并行只会更快地产生错误。
- 不要为不参与指标的 location 或一次性背景物体制造身份审核工作。

## Speed Notes

当前慢主要来自三个乘法项：

```text
total_time ~= chunks * qa_rounds * branches * (discover + draft + verify + per_crop_vlm)
```

其中：

- 正常 chunk：1 round * 2 branches，已经需要 2 次 discover、2 次 draft、2 次 verify。
- flagged chunk：3 rounds * 2 branches，膨胀到 6 次 discover、6 次 draft、6 次 verify。
- 每次 verify 还会对多个 entity crop 调 `judge_same_entity()`；如果 chunk 有 6 个实体，就是额外最多
  12 个 per-crop VLM 请求。
- 使用 32B VLM 时，每次结构化图文请求的延迟都明显高于 8B/2B 级模型。
- 404/model-name mismatch 不是吞吐问题；必须先修模型名，否则重试只会放大等待时间。
- endpoint 数量不应该进入上式的 `branches`，除非显式提高 `branches_per_chunk`；否则它只决定多个
  chunk/attempt 能分摊到多少后端。

推荐分工：

| 子任务 | 建议模型/策略 | 原因 |
|---|---|---|
| entity discovery | Qwen3-VL-8B 或同级 8B VLM | 需要语义，但不需要 32B。 |
| draft prompt | Qwen3-VL-8B | 只需生成结构化 prompt，32B 收益有限。 |
| presence verifier | Qwen3-VL-8B；必要时抽样复核用 32B | 二元检查为主，8B 足够作为第一道门。 |
| crop_match | 默认只对非 location 且非 full_frame 的 crop 做；优先用 detector score + static attrs + embedding | per-crop VLM 最容易爆调用数。 |
| gray-zone same-entity arbitration | 只在静态属性兼容且 embedding 落入灰区时调用 VLM | 符合 decomposition 漏斗，避免全量 VLM。 |
| final human review | 只看 flagged / low-confidence chunk | gold 冻结前必须人审，但不需要所有 chunk 都走重 VLM。 |

推荐运行策略：

1. 先用 8B VLM 生成 draft gold，并开启静态属性门禁和 crop pruning。
2. 只把 `flagged`、低 grounding score、`vlm_fallback`、多实例同框 chunk 送 32B 复审。
3. 默认 `branches_per_chunk=1`；多个 endpoint 只作为后端池轮询。只有确实需要单 chunk 多样本投票时，
   再显式提高 `BRANCHES_PER_CHUNK` / `--branches-per-chunk`。
4. 对 Big Buck Bunny 这类动画样本，优先用短 phrase + DINO/GroundingDINO/静态属性过滤；
   不要让大 VLM 反复裁判本可由专用感知模型解决的问题。
5. 启动前用 `/v1/models` 或服务日志确认实际 served model name；CLI `--vlm-model` 必须完全匹配。

## Checklist Before Freezing Gold

- `gold/entity_registry.json` 和 `gold/chunk_annotations.json` 已存在。
- `human_reviewed` 在冻结前保持 false，人工审查后再 `freeze`。
- `derived/assets/*/cover.jpg` 不含明显错类 crop。
- `build/annotation_qa.json` 中 flagged chunk 已人工看过。
- 任一 chunk 若存在 `vlm_fallback` 的 character/prop crop，应人工复核。
- `qwen3-vl-32b` 只用于必要复审，不作为所有阶段默认模型。
- annotation run 日志里不应出现 404；出现 404 先查 model name，不要继续跑。

## 2026-07-20: BDY-A800 本地健康但 console S3 视频派发超时

### 结论

BDY-A800 的 32 个 Qwen3-VL-8B rank 可以在节点本地完成真实图像 + JSON-schema
任务，但**当前不得用于 console 发起的 BBB S3/S5 视频生产派发**。在此链路修复并以
真实视频样本复验前，console 只派发 H800 与 gpu-a800 endpoint。

### 症状与证据

- 以 `10.252.*` 控制 IP 注册时，BBB S3 出现大量 `URLError: [Errno 110] Connection timed out`；
  这些错误是 `RETRYABLE_ERROR`，不能当作 gold 证据。
- 改用 nodes.tsv 里可路由的主机名后，开发机可对四个节点的
  `/v1/chat/completions` 成功得到 HTTP 200 / `OK`。
- 32 rank 的本地多模态健康检查均通过，但这只证明服务在节点本地可用，不证明 BBB 的
  `video_url` 生产请求可端到端完成。

### 可能原因

1. `10.252.*` 是 BDY 节点控制/任务队列地址，不是 console 的稳定应用 HTTP 路由。
2. 本地 data-URL 图像请求与 S3 的 `file://` 视频请求经过的媒体读取、网络路由和超时路径不同；
   前者成功不能推出后者成功。
3. console 的旧任务会把 endpoint URL 快照写入 job 命令；服务重启后旧 job 不会自动替换错误 IP。

### 数据本地性假设（未证实，禁止据此改管线）

BDY 将 Qwen3-VL 权重 stage 到节点本地 `/tmp/memstrata_public_models` 后可以稳定加载，
这表明 Ceph 远程读取确实可能是模型加载的瓶颈。S3 的 `video_url=file://...` 则要求
服务节点再读取源视频；若该视频不在节点本地，远程视频读取、解码和 HTTP 请求可能形成另一条
高延迟路径。

但现有 BBB 错误首先是 `urlopen` 到 `10.252.*` 的连接超时，不能证明视频本地性就是根因。
在任何“复制视频到计算节点本地”的实验前，必须先在节点本地做两项只读验证：

1. `stat` / 小范围解码确认该具体源视频路径可读；
2. 同一服务分别接收小 data-URL 图像和同路径 `file://` 视频，记录 HTTP、耗时和服务端错误。

在上述证据齐全前，保持管线逻辑不变，并继续禁止 BDY 承担 console 的 BBB 视频生产派发。

### 强制规避

- 计算节点操作走你自己的作业队列，不要从开发机 SSH 进训练节点改服务。
- BDY 的 `FLEET_ADVERTISE_HOST` 只允许 nodes.tsv 的 `host` 字段；禁止 `10.252.*`。
- 发现 BDY 视频派发 timeout 时，立即给 BDY endpoint 写 `break`，停止 console 派发；
  重新提交 job 时只使用 H800 + gpu-a800，直到 BDY 真实视频复验通过。

### 2026-07-20 追加：shared-FS executor 原型仍未跑通

为避免开发机到计算节点的 HTTP 路由，console 已能写共享 job manifest 并把 runner
提交到节点本地。runner 在节点本地检查源视频、发现本地
`127.0.0.1` VLM 后调用原有 batch。

该原型在 node0 上长时间占用单 worker，且 node-local VLM supervisor 未形成可访问
`/v1/models` 的 listener；没有产生可验证的 S3 result。故截至本次记录：

- **BDY shared-FS 完整 S3 仍为未验证/不可用**；
- 不应将其暴露为 production console 选项；
- 不应继续用完整 44 分钟电影作为 smoke test；
- 若以后恢复此方向，先以单个 10 秒 clip、单个本地 rank、流式 runner log 验证，
  再考虑完整 movie 或 hybrid 分片。
