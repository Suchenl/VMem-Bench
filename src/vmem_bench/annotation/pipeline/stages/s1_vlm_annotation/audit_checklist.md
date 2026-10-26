# S1 起草期审计清单（可复现）

用途：对 `movie_dir/vlm_output.json` 做**确定性**扫描后，决定是否需要人机 VLM 重标，以及用哪类补丁。  
本清单记录 2026-07 实跑（LSMDC + BlenderOpenMovies）中确认有效的分类；实现脚本可换，**判定语义应保持稳定**。

## 1. 硬验收（合并进正式文件前必须过）

| 项 | 通过条件 |
|---|---|
| 可解析 | 单文件合法 JSON；无 thinking/markdown 围栏残留 |
| 时长字段 | `video_duration_seconds` = 片长 T（秒） |
| 覆盖 | scene/segment 时间并集覆盖 **[0, T]**（允许极小缝，见 §3） |
| 无重叠 | scene 之间、同 scene 内 segment 之间无负向重叠 |
| 段长上限（schema） | 任意 `visual_segment.duration_seconds ≤ 15.0`（严格） |
| 段数下限 | `counts.visual_segments ≥ ceil(T/15)`（schema 地板；常与超长段共现） |
| 实体引用 | `present_entity_ids` / `loc_id` 等均能在实体表解析 |
| 实体时间 | `first/last_presence`、`state_changes.seconds` ∈ [0, T] 且 first ≤ last |

合并成功后：删除 kit 槽位与影片旁临时 `vlm_output.*.json`，只留正式 `vlm_output.json`（`AGENTS.md` 规则 3）。

## 2. 失败 kind → 动作

| 失败（扫描 kind） | 推荐动作 | 提示词模板 | 结果文件名惯例 |
|---|---|---|---|
| 覆盖率过低（经验：&lt;~50%，或大段缺失） | `FULL_RERUN` | `prompts/full_rerun.md` | `vlm_output.rerun.json` |
| 仅缺片头（&gt;~30s） | `CONTINUE_HEAD` | `prompts/continue_head_tail.md` | `vlm_output.continue_head.json` |
| 仅缺片尾（&gt;~30s） | `CONTINUE_TAIL` | 同上 | `vlm_output.continue_tail.json` |
| 中段内容洞（单/多） | `FILL_GAP` / `FILL_GAPS` | `prompts/fill_gap.md` | `vlm_output.fill_gap(s).json` |
| 区间重叠/乱序需整段重写 | `REPLACE_RANGE` | `prompts/replace_range.md` | `vlm_output.replace_range.json` |
| 全片已覆盖，但大量 15–20s 超限 | `REVISE_OVERLONG` | `prompts/revise_overlong.md` | `vlm_output.revise.json` |
| 个别 `>30s` 超长 | `SPLIT_OVERLONG_GT30` | `prompts/split_overlong_gt30.md` | `vlm_output.split_gt30.json` |
| 实体时间戳越界（可从 screenplay 派生） | **确定性修复**（无需 VLM） | — | 直接写回正式文件 |
| dangling ID / 脏粘贴截断 | 视情况：抽 JSON 清洗 / 局部 REPLACE / 重跑该次输出 | — | — |

组装方式见 `prompts/assemble.md`。A/B 类（rerun/continue/fill/replace/revise）附完整 **v5**；C 类（`>30s` 拆分）用精简 schema，**不**附完整 v5。

## 3. 策略边界（已拍板，复现时勿擅自加严）

1. **基座提示**：生产用 `prompt_qwen3_7_plus_quick_v5.md`，不是 v6。
2. **超长段**：
   - `>30s`：应用 VLM 目标化拆分（C 类）。
   - `15–30s`：可暂缓批量硬切；若日后要强制合规，优先考虑 **S2 确定性硬切**，而不是 S3 自动重标。
3. **≈1s 量级 scene 缝**：可忽略，不发起 FILL。
4. **续标上下文**：默认「原片 + `SEND_TO_VLM`」；不要附整份旧 `vlm_output.json`（易截断/诱导全片重写）。续标靠断点时间 + `EXISTING_ENTITIES` + 续号规则。
5. **`minimum_required_segments`**：是 S1 schema 地板，不是下游打分硬拒；但实跑中 `segs < min` 几乎总与超长或空洞共现，仍应在作者侧处理。

## 4. 与 S2 / S3 的分工（写论文时用）

| 层 | 抓什么 |
|---|---|
| S1 起草期审 | 整片覆盖、格式/schema、超长/大洞、实体本体 seed |
| S2 | 确定性归一 + structural lint；**不**语义拆段补洞 |
| S3 | 逐段视频 Presence + action 文本修复；不替代 S1 覆盖修复 |

## 5. 人工实体审（S1 第 3 步）

与格式审计正交：确认哪些实体进 benchmark roster、稳定 ID、exemplar。  
当前为半人工；代码固化程度见底稿 §7 / `pipeline_track_first/roster_seed.py`。
