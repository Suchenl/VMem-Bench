# 补丁模板 · `FULL_RERUN`（整片重跑）

> 与完整 v5 组装方式见 [`assemble.md`](assemble.md)。  
> 占位符用 `{{...}}`；生成 kit 时替换为具体数值。

```markdown
# 整片重跑补丁 · `{{movie_id}}`

本任务是**整片重新标注**（旧稿覆盖太少或不值得续标）。不要附带旧 `vlm_output.json`。

## 硬约束

1. 上传完整原片。
2. 严格按官方 base prompt 标注 **[0.0, {{T}}]** 全片。
3. `video_duration_seconds` = **{{T}}**
4. `minimum_required_segments` = **{{ceil_T_over_15}}**；最终 `counts.visual_segments` 必须 ≥ 该值。
5. 任意 `duration_seconds <= 15.0`；时间连续覆盖全片，无大缺口、无重叠。
6. 不要使用外部剧情知识；只根据当前视频可见/可听证据。

## 输出

只输出一个完整 JSON 对象（含实体表 + screenplay + self_check + counts）。
把结果写入 `vlm_output.rerun.json`。
```
