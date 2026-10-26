# 补丁模板 · `REPLACE_RANGE`

> 与完整 v5 组装方式见 [`assemble.md`](assemble.md)。  
> 用于时间轴重叠、乱序、局部标注损坏：用新标注**替换**指定闭区间内的旧 scenes/segments。

```markdown
# 重标错乱段补丁 · `{{movie_id}}`

**问题**：{{problem_one_liner}}（例如：区间内 scene/segment 重叠 −Xs）

## 硬约束

1. 上传**完整原片**。
2. `video_duration_seconds` = **{{T}}**。
3. 只重标并输出区间 **[{{t0}}, {{t1}}]**；合并时会删除正式文件中落入该区间的旧 scenes/segments，再插入你的结果。
4. 新片段必须严丝合缝覆盖 `[{{t0}}, {{t1}}]`，与区间外邻接段无重叠、无空洞。
5. 任意 `duration_seconds <= 15.0`；本区间段数建议 ≥ **{{min_local}}**。
6. **优先复用** `EXISTING_ENTITIES.json`；新实体续号。
7. 新 scene / segment 从 **`{{next_scene_id}}` / `{{next_seg_id}}`** 起编。
8. 不要使用外部剧情知识。
9. {{extra_note}}

## 必须复用的已有实体（摘要）

{{entity_summary_or_see_EXISTING_ENTITIES}}

## 输出

只输出覆盖该区间的完整 JSON 切片（实体表 + `screenplay.scenes` + self_check + counts）。
写入 `vlm_output.replace_range.json`。
```
