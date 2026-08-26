# MemStrata-Bench 人机协同审核：结论与落地原则

## 结论

标注主链路应追求的不是“模型代替人做所有判断”，而是把人从浏览 JSON、重复查找证据和机械修复中移开：机器负责发现、聚合、排序和验证可形式化的错误；人只裁决模型无法可靠区分的视觉身份与叙事语义。

当前 Track-First 已具备该方向的基础：确定性 `present`、严格 lint、`auto_review`、人工 disposition 冻结门、多 crop 起草证据、review-only identity candidates，以及 location 不参与 headline asset-selection 评分。它尚未通过真实 BBB 重跑证明“最低人工成本”；该结论必须由候选数、人工分钟数、建议接受率、strict-lint 和抽样误差共同验证。

## 思考路径

1. **先区分错误性质。** 路径、引用、派生字段、空值和 schema 问题有唯一正确答案，应由确定性代码处理；“两个 crop 是否同一角色”“某变化是否不可逆”没有仅靠规则可得的答案，不能伪装成自动化。
2. **再区分机器动作。** 自动拒绝、自动生成可逆 patch、给人建议、交由人裁决的安全等级不同。高置信不是单一模型分数，而是多证据一致且没有反证。
3. **最后优化人的操作。** 人工审核的单位应是一个可以回答的决策，而非文件或 chunk。先给结论、再给最少但足够的证据、最后提供少量可撤销动作。

## 三层审核策略

### A. 确定性自动 gate

无需模型或人工。包括：schema/ID/路径完整性、`present` 与 `first_appearances` 的派生一致性、非法 representation 引用、空/占位 prompt、无效 state-event、冻结所需 disposition 缺失。任何失败均阻止 freeze，不静默降级。

### B. 高置信机器建议与有限自动修复

机器可生成 patch 或“接受建议”，但每次都保留前后差异、证据、阈值/模型版本，并可撤销。身份合并只有同时满足以下条件才有资格进入该层：

- 同 kind，静态属性无冲突；
- body、face、class、text 至少两路强一致，或规范名一致且视觉证据强一致；
- 不存在同帧共现等时间线反证；
- patch 后局部重算和 strict lint 通过。

即使达到条件，也应抽样人工审计（建议每片至少 5%，且每类至少若干项）并根据接受率/撤销率校准阈值。prompt 文本相似度和单次 VLM 判断只能产生 finding，不能单独自动改写 gold。

### C. 人工裁决

下列情况必须升级人工：多实例角色/相似道具的身份判断、模型或证据通道相互冲突、不可逆 state-event、影响大量未来 chunk 的 merge/split、以及自动化抽样审计。人工的最终决定写入 review patch + disposition，freeze 只接受 strict lint 为零且必审项均已处置的实例。

## Human-friendly 审核体验

统一队列中的每一项只提一个问题，并按下式排序：

`预计评分影响 × 受影响 chunk 数 × 证据分歧 × 审核优先级`。

### 身份卡

问题：“这两个实体是否相同？”展示双方 2–3 张高质量、多样的代表 crop、首末出现时间线、同屏反证、body/face/class/text 信号、受影响 chunk/prompt 摘要与推荐动作。动作限制为：合并、保持独立、需要更多证据。

### 状态卡

问题：“该变化是否不可逆并应废弃旧 representation？”展示事件前后 crop、发生 chunk 的 prompt、候选 `deprecates`、未来受影响 chunk。动作限制为：确认不可逆、非不可逆、证据不足。

### Prompt 卡

问题：“prompt 是否覆盖已确定的 present roster？”展示 sampled frames、给 VLM 的实体证据和当前 prompt。动作限制为：覆盖充分、漏实体、叙述冲突；只有后两项需要最小文本编辑。

别名候选应按连通簇一次呈现，避免审阅者反复判断 A-B、B-C、A-C。每次动作后应即时展示 patch、局部重算结果与剩余 blocker 数，并支持撤销。

### 当前已落地

dashboard 的 Review 模式已提供“审核收件箱”：`tmp/review_queue.json` 中的 identity、state、prompt 和 lint 项按确定性优先级呈现。identity 候选按连通簇合并，避免重复审 A-B、B-C；双实体卡显示代表 crop、时间跨度和多路相似度，审阅者必须填写理由，才能暂存“合并为左侧实体”或“保持独立”。状态卡展示不可逆事件、废弃 references 与影响范围，但不伪造未实现的 state-event 编辑功能。prompt/lint 卡可跳转至对应实体或 chunk 的完整编辑上下文。

所有动作只更新浏览器中的 review patch/disposition 草稿，仍须经过现有的保存、应用、lint 与 freeze 流程。状态事件使用 `state_event_reviews`：确认、驳回或受限编辑都需理由；原始事件与人工结论追加保存为 `tmp/state_event_review_pairs.jsonl`，而当前处置保存在 `tmp/state_event_dispositions.json`。应用后会重算 `forbidden` 与 state-change tags；freeze 拒绝任何尚未有人工决定的剩余事件。前端的“预览 Lint”会在临时副本应用当前 patch 并返回 strict-lint blocker；它不写入真实 gold。

## 模型使用边界

初版不需要新增模型：现有 DINO/face/class/text 信号和现有 VLM 足够生成队列与证据。后续可增加一个**受约束的证据裁判**：仅对已筛选的 identity/state 卡，基于固定 crop 与时间线输出“同一/不同/证据不足”及依据；不得发现新实体或直接改 gold。

若裁判结果用于自动 merge，它必须与起草/命名模型独立（不同模型家族或至少不同 endpoint），并先在人工已裁决样本上测量 precision、误合并率与节省分钟数。未达到预先设定的 precision 门槛时，裁判只能排序，不能自动执行。

## 近期实施顺序与验收

1. 先建立 `tmp/review_queue.json`：合并 auto-review、identity candidates、annotation QA 与 lint 的证据为单一、只读、可排序队列。
2. 在现有 review 页面展示身份/状态/prompt 决策卡与剩余 blocker 摘要；不改变 patch 或 freeze 语义。
3. patch 后只重算受影响实体/chunk，显示局部 lint 和派生字段差异。
4. 在真实 BBB 运行记录自动通过率、人工处置数/分钟、建议接受率、抽样误差和 freeze 成功率；用这些指标校准阈值。
5. 仅在用户批准模型与 precision 门槛后，引入独立证据裁判或自动 merge。
