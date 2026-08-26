# 补丁模板 · `CONTINUE_HEAD` / `CONTINUE_TAIL`

> 与完整 v5 组装方式见 [`assemble.md`](assemble.md)。  
> `CONTINUE_HEAD`：只标片头缺失；`CONTINUE_TAIL`：只标断点之后到片尾。

## CONTINUE_TAIL

```markdown
# 续标补丁（缺尾）· `{{movie_id}}`

本任务是对**已有半成品标注的续标**，不是整片重做。

## 硬约束

1. 上传**完整原片**（不要只切片段；若只能上传片段，仍须使用原片绝对时间戳）。
2. 只标注时间区间 **[{{t0}}, {{T}}]** 秒。
3. 顶层 `video_duration_seconds` 必须写成整片时长 **{{T}}**（不是区间长度）。
4. 所有 `start_seconds` / `end_seconds` / presence / state_changes 一律使用**原片绝对时间**。
5. 现有标注已覆盖到 {{t0}}s（最后 scene=`{{last_scene_id}}`，seg=`{{last_seg_id}}`）。不要重复标注 0–{{t0}}。
6. 新 scene 从 `{{next_scene_id}}` 起编；新 segment 从 `{{next_seg_id}}` 起编。
7. 优先复用下面已有实体 ID；只有本区间新出现、无法对齐到已有实体时，才新增并续号。
8. 输出仍是完整合法 JSON：含 `characters` / `props` / `locations` / `screenplay` / `self_check` / `counts`。
9. 续标区间的 `visual_segments` 必须时间连续覆盖该区间；每个 `duration_seconds <= 15.0`。
10. 本区间 segment 数量至少 `ceil(({{T}}-{{t0}})/15) = {{min_local}}`。

## 必须复用的已有实体

### characters
{{char_lines}}

### props
{{prop_lines}}

### locations
{{loc_lines}}

## 输出

只输出一个 JSON 对象。`screenplay.scenes` **只包含**本续标区间的 scenes（不要塞已有区间）。
写入 `vlm_output.continue_tail.json`。
```

## CONTINUE_HEAD

```markdown
# 续标补丁（缺头）· `{{movie_id}}`

本任务是对**已有半成品标注的续标**，不是整片重做。

## 硬约束

1. 上传**完整原片**。
2. 只标注时间区间 **[0.0, {{t1}}]** 秒。
3. `video_duration_seconds` = **{{T}}**。
4. 一律使用原片绝对时间。
5. 现有标注从 {{t1}}s 才开始。不要标注 {{t1}}s 之后的内容。
6. 本区间 scene/segment 可从 `scene_0001` / `seg_0001` 起编；合并时会整体重编号。
7. 优先复用下面已有实体 ID（片中后段已出现的角色/道具/地点）；仅当本区间出现无法对齐的新实体时再续号。
8. 输出完整合法 JSON；区间内连续覆盖；每段 `duration_seconds <= 15.0`。
9. 本区间 segment 数量至少 `ceil({{t1}}/15) = {{min_local}}`。

## 必须复用的已有实体

### characters
{{char_lines}}

### props
{{prop_lines}}

### locations
{{loc_lines}}

## 输出

`screenplay.scenes` 只含本区间。写入 `vlm_output.continue_head.json`。
```
