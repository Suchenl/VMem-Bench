# SUT 记忆像素通道 · 设计方案（已实现）

> 状态：**已实现（口径 a；基线仍走回切，未切换）**。本文回答一个 Track A 契约层面的问题——
> *图像原生的记忆系统（MemStrata、帧检索基线）到底该被"降维成时间戳→源帧"打分，
> 还是直接用它自己 compose 出来的参考图打分？* 先由 owner 审定方向再动代码。
>
> 关联：`contract.py`（`RetrievedItem`）、`frame_materializer.py`、
> `docs/baselines/fairness_decisions.md`、`docs/benchmark/scoring_v2.md`、
> `docs/baselines/track_a.md`。

## 1. 现状（事实，已核对代码）

Track A 现在对**每个** causal SUT 走同一条"记忆→帧"归一化：

- `RetrievedItem`（`contract.py`）**只携带时间身份**：`evidence_kind`（`frame|latent|kv|slot|reference_image`）
  + `source_seconds` / `source_chunk_id` + latent/kv 调试句柄。注释明确：
  *"No present / roster / crop is ever included here."* —— **没有任何字段能让 SUT 传自己的像素**。
- `frame_materializer.py` 把每个 item 解析成一个绝对秒数,然后从**源视频**在该秒切一张
  **整帧(832×480)**,写 `visual_selections/<system>.json`,交给 `visual_coverage` 打分。
- 因果护栏:item 的源时刻必须严格早于当前 chunk 起点,否则丢弃。

**后果(以 MemStrata 为例)**:它内部维护的结构化记忆(每个实体的紧致裁剪图 + 状态 + 时间轴)
在打分前被丢掉,只剩"你指向了哪个过去时刻",然后 bench 在那个时刻切一张**整帧**当参考图。
即 MemStrata 事实上被当成"一个更聪明的选帧器"来评,它 compose 的裁剪级上下文从未进入评分。

## 2. 为什么当初这么设计(现有理由,不要轻易推翻)

这是**刻意的反作弊 + 可比性**设计:

1. **反作弊**:若允许 SUT 自带像素,系统可以塞入精挑细选/裁到只剩目标/接近 gold 的图,
   visual-coverage 会被人为拉高。让所有像素都由 bench 从源视频按 SUT 报出的时间戳自己切,
   杜绝这条作弊路径(scorer 还把参考图当作 **UNLABELED**,只判"里面能看到哪些 gold 实体")。
2. **异构可比**:KV/latent 系统(MemFlow/LongLive)本就没有像素,必须转帧;把所有系统统一
   降维成"过去时刻集合"才有同一把尺子。
3. **因果忠实**:时间戳 + 严格 <t 护栏能机械地拒未来泄露。

## 3. 张力(owner 提出的问题)

owner 的设计直觉:*转帧只对 KV/latent 这种"记忆非像素"的系统必要;图像原生的记忆(MemStrata)
本来就是像素,不该被强行转成整帧,应直接拿它 compose 的裁剪图打分。*

这和 §2 的反作弊规则**直接冲突**,是两个都合理的目标在打架:

| | 归一化转帧(现状) | 原生像素(提案) |
|---|---|---|
| 反作弊 | 强(bench 掌控所有像素) | 需新护栏(见 §4) |
| 异构可比 | 强(统一尺子) | 帧 vs 裁剪图,coverage 语义要重定 |
| 忠实反映方法 | 弱(丢掉 MemStrata 的裁剪级记忆) | 强 |
| precision 语义 | 整帧含无关实体→稀释 | 单实体裁剪图 precision 天然≈1,需重定义 |

**关键新增事实**:MemStrata 的裁剪图**是从它观测过的源视频抠的、不是 gold**,所以"用它自己的裁剪图"
本身不等于作弊——但一旦开了"SUT 传像素",就必须有护栏防止有人**曲线塞 gold / 过度裁剪刷分**。

## 4. 提案:给契约开一条"自带参考图"通道(带护栏)

### 4.1 契约改动(最小)

- `RetrievedItem` 增加**可选**字段 `image_path`(SUT 自己 compose 出的参考图,绝对路径),
  搭配已有的 `evidence_kind="reference_image"`。**仍必须同时带 `source_seconds`**(见护栏)。
- `frame_materializer`:
  - 若 item 带 `image_path` 且通过护栏 → **直接用该图**(link/copy),不再从源视频切帧;
  - 否则(latent/kv/纯时间戳)→ 走现状的"时间戳切源帧"回退。
  这样 KV/latent 系统零改动,图像原生系统才走新路径,**统一一条 materializer、按能力分流**。

### 4.2 反作弊护栏(必须同时落地,否则不许开)

1. **必带源时刻 + 因果护栏**:`image_path` 的 item 仍要报 `source_seconds`,`<t` 护栏照旧;
   缺时间戳的自带图一律丢弃。
2. **源自源视频的可验证性**:bench 侧做一次**廉价一致性校验**——该裁剪图应能在
   `source_seconds` 对应的源帧里找到(如裁剪图与源帧的一个子区域高相似 / 是其掩膜子区域)。
   校验不过 → 丢弃并计入 `extras.n_provenance_failed`(审计)。目的是拦"塞 gold / 塞非本片像素"。
3. **禁 gold 血缘**:沿用现有"SUT 侧零 gold"约束,自带图不得来自 gold 目录/标注。

### 4.3 评分口径(要和 owner 敲定,否则不公平)

visual-coverage 现在把参考图当 UNLABELED、判"能看到哪些 gold 实体"。若允许裁剪图:

- **recall** 语义不变(裁剪图里能不能看到该 gold 实体)。
- **precision / 冗余** 需重定义:单实体裁剪图天然 precision 高,不能让"交更碎的图"白拿分。
  候选口径(择一,需 owner 定):
  - (a) **不改指标**,但要求 SUT 每个 chunk 交的参考图集合覆盖它"想提供的全部实体",
    precision 仍按"集合里被 gold 命中的比例"算——裁剪图集合和整帧集合同尺子;
  - (b) 给裁剪图和整帧**分开报**两套 coverage,不混排,避免"crop 组 vs frame 组"错位比较;
  - (c) 保留整帧口径为**主表**(可比),自带裁剪图口径作**附录轴**(反映方法上限)。

### 4.4 一致适用(公平性硬要求)

同一规则对**所有**图像原生系统开放:MemStrata 和帧检索基线都可选择"交自己的图"。
不能只给 MemStrata 开小灶——否则违反 `fairness_decisions.md` 的一视同仁原则。
(帧检索基线本来交的就是整帧,天然落在回退路径;这是范式差异,不是偏袒。)

## 5. 建议

- 方向上倾向 **4.1 + 4.2 + 4.3(c)**:开像素通道 + 严格护栏,但**主表仍用整帧口径保证可比**,
  自带裁剪图口径作**附录**,专门展示"若按方法原生上下文评,MemStrata 上限有多高"。
  这样既回应 owner 的诉求(方法不再被整帧稀释),又不动主表的反作弊/可比基线。
- 需要 owner 拍板的开放项:
  1. 是否接受"SUT 自带像素"这一范式松动(4.2 护栏够不够);
  2. 评分口径选 (a)/(b)/(c) 哪个;
  3. 主表用哪套口径(建议整帧口径仍为主表)。

## 6. 影响面(改前清单)

- `contract.py`:`RetrievedItem` 加 `image_path` + `to_dict`;文档 §同步。
- `frame_materializer.py`:分流 + 护栏 + 新审计字段。
- `visual_coverage` scorer:按 §4.3 选定口径调整(可能新增一列/一附录轴)。
- 各 causal 适配器:仅**图像原生**的(memstrata / 帧检索)需要在 `RetrievedItem` 里回填 `image_path`;
  latent/kv 适配器不改。
- 文档:本文件转正 + `fairness_decisions.md` 增一条 D6 + `scoring_v2.md` 增口径说明。
- **不实现,直到 owner 审定本方案**。
