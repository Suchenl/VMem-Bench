# S1 · `s1_vlm_annotation`（起草 + 起草期审核）

本目录是 S1 的**可复现权威落点**：官方起草提示词、起草期为何审、失败→动作映射、以及发给起草 VLM 的**稳定修复模板**。

日期化作业台（如 `data/_vlm_rerun_kit_YYYYMMDD/`）只承载当次 per-movie 填充；合并清场后 kit 槽位会删，**模板必须留在这里**。

## 三步（作者侧）

1. **起草**：Qwen3.7 Plus + 本目录 `prompt_qwen3_7_plus_quick_v5.md`（生产基座；`v6` 为实验/对照，非默认）。
2. **格式/覆盖审计 + 目标化重标**：确定性扫描 `vlm_output.json` 的覆盖、重叠、大洞、超长段、实体引用等；对需语义重写的洞，用人机流程把 `prompts/*.md` 填成 `SEND_TO_VLM.md` → 上传原片 → 回填 JSON → agent 合并。角色分工见下。
3. **人工审实体**：确认 canonical roster / exemplar seed（设计见 `docs/track_first_redesign.md` §10）。**尚未**完整挂进 `pipeline/orchestrator`；`pipeline_track_first/roster_seed.py` 是相关代码线索。

产出：格式可过 gate、实体经确认的 `movie_dir/vlm_output.json`，再交确定性 S2。

## 角色（勿写错）

| 角色 | 做什么 | 不做什么 |
|---|---|---|
| 起草 VLM（Qwen3.7 Plus） | 按 v5 / 补丁产出或修补 JSON | 不负责合并进正式文件 |
| 审计 agent（实跑多为 Cursor/Grok） | 扫描失败、分类动作、生成 kit、合并、清场 | 一般不直接“改语义”替代 VLM |
| 人 | 复制 `SEND_TO_VLM`、上传原片、粘贴结果；确认实体本体 | 不靠人肉手改大段 screenplay |

「Grok 审格式」在论文/底稿中应理解为：**起草期半人工审计 + 目标化重标编排**，不是「Grok 单独改写正式 JSON」。

## 为何必须审（动机）

起草模型常见失败：头/尾截断、scene/segment 重叠、内容大洞、`duration_seconds > 15`、实体时间戳越界、dangling ID、输出截断/脏粘贴。  
**S2 只做 schema/归一/lint，不自动补洞、不按语义拆超长段**；这些洞若拖到 S3 视频复核，成本更高且难归因。因此覆盖与格式问题应在 S1 作者侧清掉。

详细验收项与动作映射见 [`audit_checklist.md`](audit_checklist.md)。  
修复提示词模板见 [`prompts/`](prompts/)。

## 与下游边界

- **原始 S1 文件语义上是草稿**；S2 起写派生产物，不改写正式 `vlm_output.json`（编排约定）。
- 起草期修复**可以**（且应当）改写正式 `vlm_output.json`——那是作者侧定稿，不是 S3 自动 revise。
- 合并进正式文件后必须清场：见 `benchmarks/MemStrata/AGENTS.md` **规则 3**。

## 文件索引

| 文件 | 用途 |
|---|---|
| `prompt_qwen3_7_plus_quick_v5.md` | 生产起草基座 |
| `prompt_qwen3_7_plus_quick_v6.md` | 对照/实验稿 |
| `audit_checklist.md` | 审什么、何时算过、失败→动作 |
| `prompts/*.md` | 各类修复补丁的稳定模板 |
| `prompts/assemble.md` | 如何拼成可粘贴的 `SEND_TO_VLM.md` |
