# 补丁模板 · `SPLIT_OVERLONG_GT30`（精简，不附完整 v5）

> **不要**拼接完整 `prompt_qwen3_7_plus_quick_v5.md`。  
> 本文件填空后即为可发送的 `SEND_TO_VLM.md` 全文。  
> 策略：只强制处理 `duration_seconds > 30`；15–30s 可暂缓（见 `../audit_checklist.md` §3）。  
> 结果文件：`vlm_output.split_gt30.json`。合并器按 `replacements[].replace_segment_id` 替换正式稿中的对应 segment。

---

占位符填空后发给 VLM 的正文：

~~~
【请把本文件全文发给 VLM，并上传完整原片】
【任务类型：仅拆分超长 visual_segment（>30s）· 非全片重标 · 不附完整 v5 schema】

# 拆分任务 · `{{movie_id}}`

你只需要把下面列出的 **超长片段** 拆成多个 `duration_seconds <= 15.0` 的连续子片段。
不要重标整部电影，不要改动这些时间窗以外的任何内容。

## 硬约束

1. 上传完整原片（用于看清该时间段画面）。
2. 对每个 `replace_segment_id`：新子片段必须 **严丝合缝覆盖** 原 `[start_seconds, end_seconds]`，首尾对齐，中间无空洞、无重叠。
3. 每个子片段 `duration_seconds = end_seconds - start_seconds`，且 **<= 15.0**。
4. 优先在动作阶段 / 构图 / 机位 / 主体变化处切开；若无明显边界，用 `prev_boundary.kind = "continuity"`。
5. **实体 ID**：`present_entity_ids` / `loc_id` 优先使用下方已有表。只有画面中确实出现且表中没有的对象，才在 `new_entities` 里新增并递增编号。
6. 每个子片段的 `action` 只写 **该时间窗内** 可见事实；不要把原 30s+ 长文案原样复制到每一截。
7. 不要使用外部剧情知识；只根据当前视频。
8. 自然语言可用中文；字段名与枚举保持英文。

## 输出格式（只输出这一个 JSON）

{
  "movie_id": "{{movie_id}}",
  "replacements": [
    {
      "replace_segment_id": "seg_xxxx",
      "scene_id": "scene_xxxx",
      "start_seconds": 0.0,
      "end_seconds": 0.0,
      "visual_segments": [
        {
          "segment_id": "seg_xxxx_a",
          "start_seconds": 0.0,
          "end_seconds": 0.0,
          "duration_seconds": 0.0,
          "prev_boundary": {"kind": "continuity|visual_change|initial", "evidence": "...", "confidence": "high|medium|low"},
          "action": "...",
          "dialogue_or_audio": "",
          "camera": {"shot_size": "", "angle_or_view": "", "movement": "", "composition_note": ""},
          "present_entity_ids": []
        }
      ]
    }
  ],
  "new_entities": {
    "characters": [],
    "props": [],
    "locations": []
  }
}

说明：
- `replacements` 必须覆盖下列全部超长段（本片共 **{{n_overlong}}** 条）。
- `segment_id` 可用临时后缀（如 `_a/_b`）；合并时会重编号。
- 若无新实体，`new_entities` 三类都给 `[]`。

## 本片待拆列表

- 影片时长：{{T}}s
- 待拆区间：{{ranges_text}}
- 建议子段总数下限（粗算）：≥ {{min_parts}}

### 已有实体（必须优先复用这些 ID）

**characters**
{{char_lines}}

**props**
{{prop_lines}}

**locations**
{{loc_lines}}

## 原超长片段详情（请对照拆分）

{{per_segment_context_blocks}}

（实现时每个超长段一块：时间、scene、原 action、present_entity_ids、camera、邻段摘要、原 segment JSON。）
~~~
