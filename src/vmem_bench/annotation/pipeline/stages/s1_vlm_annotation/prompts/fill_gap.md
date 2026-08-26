# 补丁模板 · `FILL_GAP` / `FILL_GAPS`

> 与完整 v5 组装方式见 [`assemble.md`](assemble.md)。  
> 单洞用 `FILL_GAP`；多洞一次列出全部区间用 `FILL_GAPS`。

```markdown
# 补洞补丁 · `{{movie_id}}`

**问题**：{{problem_one_liner}}

## 硬约束

1. 上传**完整原片**（方便定位时间码；JSON 里**只输出**下面要求的时间区间）。
2. `video_duration_seconds` 仍写全片时长 **{{T}}**（与正式文件一致）。
3. 本次真正要标注的区间：{{ranges_text}}（合计约 {{span}}s）。
4. 本段 `visual_segments` 数量建议 ≥ **{{min_local}}**（= ceil(区间总长/15)）；全片 `minimum_required_segments` 字段可写 **{{min_full}}**，但以本段覆盖与 ≤15s 为准。
5. 任意 `duration_seconds <= 15.0`；区间内时间连续，无大缺口、无重叠。
6. **优先复用**同目录 `EXISTING_ENTITIES.json` 里已有 `char_id` / `prop_id` / `loc_id`；确有新实体再往后递增编号。
7. 新 scene / segment 编号请从 **`{{next_scene_id}}` / `{{next_seg_id}}`** 起递增（合并时会再统一重编号，但请不要与旧号冲突）。
8. 不要使用外部剧情知识；只根据当前视频可见/可听证据。
9. {{extra_note}}
   - 单洞示例：严丝合缝覆盖 `[t0,t1]`，首尾对齐邻接已有 segment。
   - 多洞示例：每个 listed range 都要完整覆盖，不要漏洞。

## 必须复用的已有实体（摘要）

{{entity_summary_or_see_EXISTING_ENTITIES}}

## 输出

只输出一个完整 JSON 对象（含实体表 + **仅覆盖上述区间**的 `screenplay.scenes` + self_check + counts）。
不要输出空洞以外已有 scenes 的复述。
写入 `vlm_output.fill_gap.json` 或 `vlm_output.fill_gaps.json`。
```
