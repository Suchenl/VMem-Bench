# annotation_mode: screenplay_first_bounded_visual_segment_probe

你是一名专业的视频内容分析员和场记。请从头到尾观看用户上传的视频。
如果视频有音频，可以结合视觉证据和音频证据；所有描述都必须基于本次上传视频中实际可见/可听到的证据。

证据来源硬约束：不要联网检索，不要使用维基、IMDb、剧情简介、字幕站、影评或任何外部资料。即使你知道这部电影、角色或后续剧情，也不要使用记忆中的电影知识补全身份、关系、结局或时间线。如果某个信息没有在当前视频中明确出现，就不要写入标注。

本次任务是一个 benchmark-oriented visual segmentation probe：我要测试你是否能输出稳定、可复现、适合后续评测的视觉片段。请先输出实体表，再按叙事场景组织 screenplay，并在每个 scene 内输出 `visual_segments`。`visual_segment` 不是严格电影学意义上的 shot，也不要求逐个真实镜头切分；它是一个不超过 15.0 秒、语义自洽、尽量不跨明显视觉边界的评测片段。生成 segments 前，先计算 `minimum_required_segments = ceil(video_duration_seconds / 15.0)`，最终输出的 `counts.visual_segments` 必须大于或等于这个数；这是最低合法数量，不是目标数量，不要把视频机械切成一串接近 15.0 秒的片段。明显剪辑点、转场、构图/机位突变、场景变化通常是优先边界；但多个极短镜头如果共同表达一个连续动作或单一事件，可以合并为一个 `visual_segment`。长于 15.0 秒的连续内容必须在动作阶段、构图重心、视线关系、空间位置、叙事推进变化处，或直接按时间上限硬切拆分；如果是按照时间上限或连续动作阶段拆分，边界标为 `continuity`。任何 `duration_seconds > 15.0` 的 segment 都是无效输出，不能通过 `self_check`、`evidence` 或 `action` 解释保留。

最终输出为一个合法 JSON 对象。请严格返回如下顶层 JSON 结构：

{
  "video_duration_seconds": 0.0,
  "minimum_required_segments": "JSON 整数；等于 ceil(video_duration_seconds / 15.0)，例如 30.1 秒至少需要 3 个 visual_segment",
  "characters": [
    {
      "char_id": "稳定机器 ID，格式 char_001；全局唯一，供所有跨字段引用使用",
      "name": "稳定实体名；若当前视频中明确出现姓名/称呼则使用该称呼，否则自拟一个与外观、服装、位置、状态、关系和剧情功能无关的中性代号，不超过12个中文字符或6个英文词",
      "identity_scope": "individual",
      "description": "只写首次清晰出现时已经可见、可用于重新识别的特征；这些特征只代表首次观察，不代表永久属性",
      "first_presence_seconds": 0.0,
      "last_presence_seconds": 0.0,
      "state_changes": [
        {
          "seconds": 0.0,
          "state_change_kind": "destroyed | consumed | broken | acquired | attached | detached | appearance_changed",
          "description": "这个角色自身发生的不可逆或持续性状态变化"
        }
      ]
    }
  ],
  "props": [
    {
      "prop_id": "稳定机器 ID，格式 prop_001；全局唯一，供所有跨字段引用使用",
      "name": "稳定实体名；若当前视频中明确出现文字或称呼则使用该称呼，否则自拟一个与外观、位置、状态和剧情功能无关的中性代号，不超过12个中文字符或6个英文词",
      "identity_scope": "individual | category",
      "description": "只写首次清晰出现时已经可见、可用于重新识别的特征；这些特征只代表首次观察，不代表永久属性",
      "first_presence_seconds": 0.0,
      "last_presence_seconds": 0.0,
      "state_changes": [
        {
          "seconds": 0.0,
          "state_change_kind": "destroyed | consumed | broken | acquired | attached | detached | appearance_changed",
          "description": "这个道具自身发生的不可逆或持续性状态变化"
        }
      ]
    }
  ],
  "locations": [
    {
      "loc_id": "稳定机器 ID，格式 loc_001；全局唯一，供所有跨字段引用使用",
      "name": "稳定实体名；若当前视频中明确出现地点名称则使用该名称，否则自拟一个与外观、状态和剧情功能无关的中性代号，不超过12个中文字符或6个英文词",
      "identity_scope": "scene",
      "description": "基于视频证据写出的地点描述；只描述可见场景空间，不写剧情功能",
      "first_presence_seconds": 0.0,
      "last_presence_seconds": 0.0,
      "state_changes": [
        {
          "seconds": 0.0,
          "state_change_kind": "destroyed | consumed | broken | acquired | attached | detached | appearance_changed",
          "description": "这个地点自身发生的不可逆或持续性状态变化"
        }
      ]
    }
  ],
  "screenplay": {
    "scenes": [
      {
        "scene_id": "scene_0001",
        "start_seconds": 0.0,
        "end_seconds": 0.0,
        "loc_id": "必须来自 locations 中已有的 loc_id，或为空字符串",
        "scene_title": "简短场景标题",
        "scene_purpose": "这段场景在当前视频中已经呈现出的叙事作用",
        "prev_boundary": {
          "kind": "initial | change | uncertain",
          "evidence": "基于画面/声音说明为什么这里开始一个新的叙事场景；不确定则说明疑点"
        },
        "visual_segments": [
          {
            "segment_id": "seg_0001",
            "start_seconds": 0.0,
            "end_seconds": 0.0,
            "duration_seconds": "JSON 数字；必须严格等于 end_seconds - start_seconds，且 <= 15.0",
            "prev_boundary": {
              "kind": "initial | visual_change | continuity | uncertain；initial=当前 scene 的第一个片段；visual_change=存在明显视觉边界，例如剪辑点、转场、场景/地点变化、构图或机位突变、主要视觉焦点变化；continuity=没有明显视觉边界，只是同一连续内容内部按时间上限或动作阶段拆分；uncertain=边界依据不确定",
              "evidence": "说明为什么选择上面的 kind，并给出上一片段到本片段之间的边界依据；如果是 visual_change，要写清楚可见变化；如果是 continuity，要写清楚这是连续内容中的时间上限硬切或动作阶段拆分；不要写未修复的问题",
              "confidence": "high | medium | low"
            },
            "action": "用剧本动作行风格客观描述本 visual_segment 内可见内容，只写本片段内发生的事实；必须自然点名本片段 present_entity_ids 中的每个实体（用其 name，不用 ID）；对在全片首次出现的实体，把其 description 中可辨识的外观特征自然写进动作句，使读者仅凭本句即可想象其长相；对之前已出现过的实体只点名、不重复描述外观/服装/颜色等身份属性",
            "dialogue_or_audio": "本片段内可听到的对白、环境声或音乐；没有则写空字符串",
            "camera": {
              "shot_size": "wide | medium | close_up | extreme_close_up | insert | mixed | unknown",
              "angle_or_view": "正面/侧面/过肩/主观/高角度/低角度/混合等",
              "movement": "static | pan | tilt | tracking | handheld | zoom | mixed | unknown",
              "composition_note": "只描述本片段内的主要构图和视觉变化；不要声称这是严格单一镜头"
            },
            "present_entity_ids": ["上方 characters/props/locations 中已经存在的 char_id/prop_id/loc_id；本片段中可见或作为当前场景上下文明确成立的实体"]
          }
        ]
      }
    ]
  },
  "self_check": {
    "duration_check": "只能写 passed；如果存在任何 duration_seconds > 15.0 的 visual_segment，必须先回到 screenplay.scenes[*].visual_segments 拆分修复，不能输出 failed 或解释原因",
    "segment_count_check": "只能写 passed；如果 counts.visual_segments < minimum_required_segments，必须先继续拆分修复",
    "boundary_reason_check": "只能写 passed；如果大量片段被机械切成接近 15 秒、且忽略明显视觉边界，必须先修复",
    "low_confidence_segment_ids": ["prev_boundary.confidence 为 low 的 segment_id；否则空数组"],
    "notes": "简短说明你对整体 visual segment 划分可靠性的判断；不得报告未修复的时长超限、数量不足或建议后处理拆分"
  },
  "counts": {
    "characters": 0,
    "props": 0,
    "locations": 0,
    "scenes": 0,
    "visual_segments": 0
  }
}

标注规则：

1. 顶层第一个字段必须是 `video_duration_seconds`，第二个字段必须是 `minimum_required_segments`。所有时间戳都使用从视频开始算起的秒数，写成 JSON 数字，例如 `0.0`, `3.0`, `12.5`；任意时间戳都必须在 `[0.0, video_duration_seconds]` 内，且每个 visual_segment 的 `duration_seconds` 必须严格等于 `end_seconds - start_seconds`，并且 `duration_seconds <= 15.0`。
2. 所有 `counts.*` 都必须是整数，且分别等于对应数组长度：`characters`, `props`, `locations`, `screenplay.scenes`，以及所有 scene 内 `visual_segments` 的总数。
3. 在输出最终 JSON 前，必须先计算顶层 `minimum_required_segments = ceil(video_duration_seconds / 15.0)`，其中 `ceil` 表示向上取整。最终 `counts.visual_segments` 必须 `>= minimum_required_segments`；如果数量不够，必须继续把长连续段按时间上限拆成更多 `continuity` visual_segments，直到满足下限。但 `minimum_required_segments` 只是最低合法数，不是推荐片段数，更不是把视频切成固定 15 秒网格的目标。
4. `visual_segments` 是 benchmark 视觉片段，不是严格电影 shot。优先让片段语义自洽、时间可控、边界可解释；明显剪辑点、转场、场景变化、构图/机位突变通常是好边界，但多个极短镜头如果共同表达一个连续动作或单一事件，可以合并。
5. `prev_boundary.kind` 的含义必须稳定：`initial` 只用于每个 scene 的第一个 visual_segment；`visual_change` 用于剪辑点、转场、场景/地点变化、构图或机位突变、主要视觉焦点变化等明显视觉边界；`continuity` 用于没有明显视觉边界、只是在同一连续内容内部按 15.0 秒上限或动作阶段拆分；`uncertain` 只在边界依据确实不确定时使用，并把 `confidence` 写为 `low`。
6. 只有在同一个连续视觉内容内部没有明显视觉边界时，才允许因为 15.0 秒时间上限硬切出下一个 visual_segment；这种边界的 `prev_boundary.kind` 使用 `continuity`。即使只是因为达到 15.0 秒上限而硬切，也必须使用 `continuity`。
7. 不要为了让片段接近 15.0 秒而跨过场景变化或明显视觉边界；也不要为了追求逐镜头切分而产生大量信息过碎、难以评测的片段。片段应服务于 benchmark 的稳定标注和后续检索/记忆评测。
8. 实体表只负责建立稳定 ID、首次/末次出现时间和持续状态变化；不要为每个实体单独维护细粒度在场时间轴。每个 visual_segment 中出现或作为当前场景上下文明确成立的实体，统一写入该片段的 `present_entity_ids`。
9. 所有跨字段引用必须使用已有 `char_id`、`prop_id` 或 `loc_id`，不得用 `name`；引用数组中不允许出现空字符串，没有实体时使用空数组 `[]`。
10. `state_changes` 只描述实体自身发生的不可逆或持续性状态变化；普通动作、情绪变化、关系进展和镜头运动都不要写成 `state_changes`。
11. `screenplay.scenes` 是叙事场景，不等于地点实体；同一个 location 可以在多个 scene 中反复出现。
12. `action` 和 `camera.composition_note` 只描述本 visual_segment 内的事实与构图；如果某个片段跨越了明显场景变化或过多独立事件，必须先拆成多个 visual_segments。不要在 `action`、`camera` 或 `prev_boundary.evidence` 中写“应拆分但未拆分”“时长超限”“建议后处理拆分”等未修复问题。此外，`action` 必须用自然语言点名本片段 `present_entity_ids` 中的每个实体（用其 `name`，不用 ID），并写清主要动作、实体间关系与空间进展；某实体在全片首次出现的那个片段，`action` 要把它 `description` 里可辨识的外观特征自然融入动作句（这是外观唯一进入 `action` 的时机）；该实体之后再次出现的片段只点名、不再复述外观/服装/颜色等身份属性，以免泄漏后续记忆评测的答案。
13. 输出最终 JSON 前，请自检：片段时间是否连续或合理覆盖全片；是否有重叠或明显 gap；是否有 `duration_seconds > 15.0`；是否有 `counts.visual_segments < minimum_required_segments`；是否有大量片段被机械切成接近 15.0 秒；是否有片段跨越明显场景变化或过多独立事件。如果发现任何问题，必须先修改 `screenplay.scenes[*].visual_segments` 直到全部满足要求，然后才能输出最终 JSON。最终 `self_check.duration_check`、`self_check.segment_count_check` 和 `self_check.boundary_reason_check` 都必须是 `passed`。
14. 自然语言描述优先使用中文；契约字段和枚举值保持英文。

现在请为上传的视频生成 screenplay-first visual segment JSON 标注。
