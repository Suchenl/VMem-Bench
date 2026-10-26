# 如何组装 `SEND_TO_VLM.md`

人只负责：打开组装好的文件 → 全文复制给起草 VLM → 上传**完整原片** → 把返回 JSON 写入约定空文件。

## A/B 类（附完整 v5）

用于：`FULL_RERUN` / `CONTINUE_*` / `FILL_*` / `REPLACE_RANGE` / `REVISE_OVERLONG`。

```text
【请把本文件全文发给 VLM，并上传完整原片】
【base prompt 版本：qwen3_7_plus_quick_v5】

<完整粘贴 prompt_qwen3_7_plus_quick_v5.md>

====================
【续标/重跑补丁：以下约束优先于上文“从头标到尾”的默认表述】
====================

<粘贴对应 prompts/*.md 填空后的补丁>
```

同目录建议附带（勿塞进 prompt 全文除非很小）：

- `EXISTING_ENTITIES.json`（续标/补洞：优先复用的 ID 表；补丁正文里也应列出摘要）
- `meta.json`（动作、annotate_range、merge_target、next_scene/seg）
- 空结果文件（如 `vlm_output.continue_tail.json` 初始为 `{}`）

## C 类（精简，不附完整 v5）

用于：`SPLIT_OVERLONG_GT30`。

直接使用 `prompts/split_overlong_gt30.md` 填空后的全文作为 `SEND_TO_VLM.md`（含输出 schema）。  
同目录附：`EXISTING_ENTITIES.json`、`OVERLONG_CONTEXT.json`（原超长段 + 邻段摘要）。

## 合并后

校验通过并写入正式 `vlm_output.json` 后，删除 kit 槽位与临时结果文件（`AGENTS.md` 规则 3）。
