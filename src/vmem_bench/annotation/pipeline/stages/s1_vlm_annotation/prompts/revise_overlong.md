# 补丁模板 · `REVISE_OVERLONG`

> 与完整 v5 组装方式见 [`assemble.md`](assemble.md)。  
> 适用：正式稿**已全覆盖**，但存在大量 `duration_seconds > 15`（常见整片卡在 20s），且 `counts.visual_segments < minimum_required_segments`。  
> 目标：输出**修正后的完整 JSON**（不是 replacements 切片）。

```markdown
# 修订超长段补丁 · `{{movie_id}}`

## 现状（不要整片从零虚构另一套无关标注）

正式 `vlm_output.json` **已经覆盖全片**：

- 覆盖：`0 → {{T}}`
- 结构：{{n_scenes}} scenes / **{{n_segs}}** visual_segments
- 实体：characters/props/locations 已建立

但未通过 schema：

1. 有 **{{n_overlong}}** 个 `visual_segment` 的 `duration_seconds > 15.0`
2. `counts.visual_segments = {{n_segs}}` **<** `minimum_required_segments = {{min_req}}`（还差 **{{deficit}}**）

本任务：**输出一份修正后的完整 JSON**，修掉超长段并补足段数；时间轴与实体尽量继承现状。

## 硬约束

1. 上传**完整原片**。
2. `video_duration_seconds` = **{{T}}**
3. `minimum_required_segments` = **{{min_req}}**；最终 `counts.visual_segments` **必须 ≥ {{min_req}}**
4. **任意** `duration_seconds <= 15.0`
5. 继续覆盖全片 **[0.0, {{T}}]**，scene 之间不要制造大缺口/重叠
6. **优先复用** `EXISTING_ENTITIES.json`；确有新实体再递增
7. 对每个超长段：在动作阶段 / 构图重心 / 视线关系变化处拆成 **≥2** 个 ≤15s 的 `visual_segment`；若无明显视觉边界，`prev_boundary.kind` 用 `continuity`
8. 拆开后每一段的 `action` **只写该时间窗内**可见事实，不要把整段长文案原样复制到每一截
9. 不要使用外部剧情知识

## 超长段清单

见同目录 `OVERLONG_SEGMENTS.json`（或下列样例）：

{{overlong_samples}}

## 推荐做法

- 以现有正式稿的 scene 划分为骨架，**重点重写/拆分含超长 segment 的 scene**
- 其它已合格（≤15s）的片段可保持时间边界，只需保证最终整份 JSON 自洽
- 若一次输出太长：可分多次回复，但最终粘贴文件必须是**一个**完整合法 JSON

## 输出

只输出一个完整 JSON 对象（实体表 + 全片 screenplay + self_check + counts）。
写入 `vlm_output.revise.json`。
```
