"""Zero-shot request construction for the direct-VLM annotation pipeline.

Each request carries the whole source video as a native `video_url` content item; the server
(vLLM) samples frames itself via `mm_processor_kwargs.fps` -- no custom keyframe extraction.
The local CLI supports three request layouts: P0 single-stage, P1 entity/state then shots,
and P2 entity then shots/state.

The requested JSON schema is deliberately small (`counts`, grouped entity arrays, shots,
and either nested or top-level state_changes) so postprocess.py's mapping into
`common.schemas.EntityRegistry`/`ChunkAnnotations` stays a thin, auditable step rather than a
second parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3.5-27b"
DEFAULT_MAX_TOKENS = 16384

APPEARANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "description": {"type": "string"},
    },
    "required": ["start_seconds", "end_seconds", "description"],
    "additionalProperties": False,
}

ENTITY_BASE_PROPERTIES: dict[str, Any] = {
    "name": {"type": "string"},
    "identity_scope": {
        "type": "string",
        "enum": ["individual", "category", "scene"],
    },
    "description": {"type": "string"},
    "first_appearance_seconds": {"type": "number"},
    "last_appearance_seconds": {"type": "number"},
}

ENTITY_BASE_REQUIRED = [
    "name", "identity_scope", "description", "first_appearance_seconds",
    "last_appearance_seconds",
]

ENTITY_STATE_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seconds": {"type": "number"},
        "state_change_kind": {
            "type": "string",
            "enum": [
                "destroyed", "consumed", "broken", "acquired",
                "attached", "detached", "appearance_changed",
            ],
        },
        "description": {"type": "string"},
    },
    "required": ["seconds", "state_change_kind", "description"],
    "additionalProperties": False,
}

ENTITY_WITH_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **ENTITY_BASE_PROPERTIES,
        "appearances": {"type": "array", "items": APPEARANCE_SCHEMA},
        "state_changes": {"type": "array", "items": ENTITY_STATE_CHANGE_SCHEMA},
        "appearance_count": {"type": "integer"},
        "state_change_count": {"type": "integer"},
    },
    "required": [*ENTITY_BASE_REQUIRED, "appearances", "state_changes",
                 "appearance_count", "state_change_count"],
    "additionalProperties": False,
}

ENTITY_NO_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **ENTITY_BASE_PROPERTIES,
        "appearances": {"type": "array", "items": APPEARANCE_SCHEMA},
        "state_changes": {"type": "array", "items": ENTITY_STATE_CHANGE_SCHEMA, "maxItems": 0},
        "appearance_count": {"type": "integer"},
        "state_change_count": {"type": "integer", "enum": [0]},
    },
    "required": [*ENTITY_BASE_REQUIRED, "appearances", "state_changes",
                 "appearance_count", "state_change_count"],
    "additionalProperties": False,
}

SHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "duration_seconds": {"type": "number"},
        "description": {"type": "string"},
        "camera": {"type": "string"},
        "present_entity_names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start_seconds", "end_seconds", "duration_seconds", "description", "camera",
                 "present_entity_names"],
    "additionalProperties": False,
}

TOP_LEVEL_STATE_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seconds": {"type": "number"},
        "entity_name": {"type": "string"},
        "state_change_kind": ENTITY_STATE_CHANGE_SCHEMA["properties"]["state_change_kind"],
        "description": {"type": "string"},
    },
    "required": ["seconds", "entity_name", "state_change_kind", "description"],
    "additionalProperties": False,
}

P0_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration_seconds": {"type": "number"},
        "characters": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "props": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "locations": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "shots": {"type": "array", "items": SHOT_SCHEMA},
        "counts": {
            "type": "object",
            "properties": {
                "characters": {"type": "integer"},
                "props": {"type": "integer"},
                "locations": {"type": "integer"},
                "shots": {"type": "integer"},
            },
            "required": ["characters", "props", "locations", "shots"],
            "additionalProperties": False,
        },
    },
    "required": ["video_duration_seconds", "characters", "props", "locations", "shots", "counts"],
    "additionalProperties": False,
}

P1_STAGE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration_seconds": {"type": "number"},
        "characters": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "props": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "locations": {"type": "array", "items": ENTITY_WITH_STATE_SCHEMA},
        "counts": {
            "type": "object",
            "properties": {
                "characters": {"type": "integer"},
                "props": {"type": "integer"},
                "locations": {"type": "integer"},
            },
            "required": ["characters", "props", "locations"],
            "additionalProperties": False,
        },
    },
    "required": ["video_duration_seconds", "characters", "props", "locations", "counts"],
    "additionalProperties": False,
}

P1_STAGE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration_seconds": {"type": "number"},
        "shots": {"type": "array", "items": SHOT_SCHEMA},
        "counts": {
            "type": "object",
            "properties": {
                "shots": {"type": "integer"},
            },
            "required": ["shots"],
            "additionalProperties": False,
        },
    },
    "required": ["video_duration_seconds", "shots", "counts"],
    "additionalProperties": False,
}

P2_STAGE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration_seconds": {"type": "number"},
        "characters": {"type": "array", "items": ENTITY_NO_STATE_SCHEMA},
        "props": {"type": "array", "items": ENTITY_NO_STATE_SCHEMA},
        "locations": {"type": "array", "items": ENTITY_NO_STATE_SCHEMA},
        "counts": {
            "type": "object",
            "properties": {
                "characters": {"type": "integer"},
                "props": {"type": "integer"},
                "locations": {"type": "integer"},
            },
            "required": ["characters", "props", "locations"],
            "additionalProperties": False,
        },
    },
    "required": ["video_duration_seconds", "characters", "props", "locations", "counts"],
    "additionalProperties": False,
}

P2_STAGE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration_seconds": {"type": "number"},
        "shots": {"type": "array", "items": SHOT_SCHEMA},
        "state_changes": {"type": "array", "items": TOP_LEVEL_STATE_CHANGE_SCHEMA},
        "counts": {
            "type": "object",
            "properties": {
                "shots": {"type": "integer"},
                "state_changes": {"type": "integer"},
            },
            "required": ["shots", "state_changes"],
            "additionalProperties": False,
        },
    },
    "required": ["video_duration_seconds", "shots", "state_changes", "counts"],
    "additionalProperties": False,
}

# Backward-facing aliases inside this module only; the model-facing contract is P0/P1/P2.
SCHEMA = P0_SCHEMA
ROSTER_SCHEMA = P2_STAGE1_SCHEMA
TIMELINE_SCHEMA = P2_STAGE2_SCHEMA

ENTITY_STATE_GUIDANCE = (
    "每个实体对象必须按顺序包含 `name`, `identity_scope`, `description`, "
    "`first_appearance_seconds`, `last_appearance_seconds`, `appearances`, "
    "`state_changes`, `appearance_count`, `state_change_count`。"
    "`appearance_count` 和 `state_change_count` 必须分别等于该实体下方两个数组长度。"
    "`name` 使用单一稳定名称；所有实体的 `name` 必须全局唯一，"
    "如果两个实体容易同名，要在 `name` 中加入最短区分词，例如颜色、身份或地点；不要列别名、解释或多个候选名。"
    "`description` 只写该实体第一次清晰出现时已经可见、可用于重新识别的外观特征；"
    "必要时可写当时已经由视频明确呈现的职业/身份。"
    "不要把全片中不同时间出现的服装、伤势、道具、行动辅助工具或状态汇总成“常穿/后期/后来”的描述；"
    "不要写全片总结、人物弧光、未来状态、关系发展或剧情功能。"
    "禁止使用会泄漏未来的描述，例如“后期逐渐…”，“最终…”，“后来…”，“成为关键纽带”，"
    "“关系突破”，“旅行时使用”，“主要互动场所”等。"
    "如果某个特征在首次清晰出现之后才出现，例如受伤、轮椅、拐杖、绷带、换装、关系变化，只能写在对应时间段的 "
    "`appearances` 或符合规则的 `state_changes` 中，不能提前写入顶层 `description`。"
    "`appearances` 按同一实体的连续可见或剧情相关区间拆分，不要把地点/实体 appearances 复刻成 shot 级碎片。"
    "每条 `appearance.description` 只描述该 `start_seconds` 到 `end_seconds` 之间已经可见或已经发生的事实，"
    "不要提到该时间段结束后才发生的情节。"
    "`state_changes` 写在对应实体下方，每条只描述该实体自己的不可逆或持续性状态变化。"
    "角色状态变化例子：死亡、受伤后外观持续改变、穿上或失去关键装备。"
    "道具状态变化例子：苹果被吃掉、门被打破、绳子被系上或解开、物体被拿走。"
    "地点状态变化例子：房屋被毁坏、洞口被堵住、场景被点燃或结构持续改变。"
    "抓住、吓跑、瞄准、出现、使用、触发等普通动作不写成 state_changes；"
    "普通动作、情绪变化、关系进展、剧情转折、镜头运动、进入/离开画面留在 shot 描述中。"
)

ENTITY_NO_STATE_GUIDANCE = (
    "每个实体对象必须按顺序包含 `name`, `identity_scope`, `description`, "
    "`first_appearance_seconds`, `last_appearance_seconds`, `appearances`, "
    "`state_changes`, `appearance_count`, `state_change_count`。"
    "`appearance_count` 必须等于该实体下方 `appearances` 数组长度；"
    "本次不输出实体状态变化，所以 `state_change_count` 固定为 0，`state_changes` 固定为空数组。"
    "`name` 使用单一稳定名称；所有实体的 `name` 必须全局唯一，"
    "如果两个实体容易同名，要在 `name` 中加入最短区分词，例如颜色、身份或地点；不要列别名、解释或多个候选名。"
    "`description` 只写该实体第一次清晰出现时已经可见、可用于重新识别的外观特征；"
    "必要时可写当时已经由视频明确呈现的职业/身份。"
    "不要把全片中不同时间出现的服装、伤势、道具、行动辅助工具或状态汇总成“常穿/后期/后来”的描述；"
    "不要写全片总结、人物弧光、未来状态、关系发展或剧情功能。"
    "禁止使用会泄漏未来的描述，例如“后期逐渐…”，“最终…”，“后来…”，“成为关键纽带”，"
    "“关系突破”，“旅行时使用”，“主要互动场所”等。"
    "如果某个特征在首次清晰出现之后才出现，例如受伤、轮椅、拐杖、绷带、换装、关系变化，只能写在对应时间段的 "
    "`appearances` 中，不能提前写入顶层 `description`。"
    "`appearances` 按同一实体的连续可见或剧情相关区间拆分，不要把地点/实体 appearances 复刻成 shot 级碎片。"
    "每条 `appearance.description` 只描述该 `start_seconds` 到 `end_seconds` 之间已经可见或已经发生的事实，"
    "不要提到该时间段结束后才发生的情节。"
)

SHOT_GUIDANCE = (
    "`shots` 按时间顺序覆盖正片主要内容。每个 shot 必须包含 `start_seconds`, `end_seconds`, "
    "`duration_seconds`, `description`, `camera`, `present_entity_names`。"
    "`duration_seconds` 必须严格等于 `end_seconds - start_seconds`，并写成 JSON 数字。"
    "`present_entity_names` 表达该片段中的实体集合，"
    "并引用实体表中已有且全局唯一的 `name`。shots 保持语义细粒度：主要角色/关键道具的可见集合明显变化、"
    "地点变化、动作目标变化、或发生重要剧情事件/状态变化时，应另起一个 shot；每个 shot 表达一个清晰的语义片段。"
    "每条 `shot.description` 只描述该 shot 时间段内已经可见或已经发生的事实，不要总结后续结局，"
    "不要把后面才揭示的身份、关系、目的或结果提前写进当前 shot。"
    "每个 shot 的 `duration_seconds` 必须在 3.0 到 15.0 秒之间，不能超出这个范围；"
    "短于 3.0 秒的片段要并入相邻语义片段，长于 15.0 秒的片段必须继续拆分。"
    "硬失败条件：如果任何一个 shot 的 `duration_seconds > 15.0` 或 "
    "`duration_seconds != end_seconds - start_seconds`，本次输出视为无效；"
    "在输出最终 JSON 前必须逐条检查所有 shots 的 `duration_seconds`，并把所有超长片段拆到 15.0 秒以内。"
    "即使一个连续对话、动作或因果事件超过 15 秒，也必须按可见动作、说话轮次、镜头/构图变化、"
    "角色集合变化或自然停顿拆成多个连续 shot。"
    "不要把多个因果上不同的动作合并成一个大段，也不要按微小姿态、轻微镜头移动、无意义背景变化切碎。"
    "`present_entity_names` 不允许出现空字符串；如果片段中没有实体表中的实体，使用空数组 `[]`。"
)

P2_STATE_GUIDANCE = (
    "`state_changes` 只记录不可逆或持续性的状态变化，不记录普通动作。每条 `state_change` 只对应一个发生状态变化的实体，"
    "使用 `entity_name` 引用实体表中已有且全局唯一的 `name`。"
    "角色状态变化例子：死亡、受伤后外观持续改变、穿上或失去关键装备。"
    "道具状态变化例子：苹果被吃掉、门被打破、绳子被系上或解开、物体被拿走。"
    "地点状态变化例子：房屋被毁坏、洞口被堵住、场景被点燃或结构持续改变。"
    "抓住、吓跑、瞄准、出现、使用、触发等普通动作不写成 state_changes；"
    "attached/detached 只用于持续存在的物理连接或分离关系，例如藤蔓缠住松鼠。"
    "普通动作、情绪变化、关系进展、剧情转折、镜头运动、进入/离开画面留在对应 shot 的 description 中。"
)

SOURCE_EVIDENCE_GUIDANCE = (
    "证据来源硬约束：只能使用本次上传视频中实际可见/可听到的证据。"
    "不要联网检索，不要使用维基、IMDb、剧情简介、字幕站、影评或任何外部资料。"
    "即使你知道这部电影、角色或后续剧情，也不要使用记忆中的电影知识补全身份、关系、结局或时间线。"
    "如果某个信息没有在当前视频中明确出现，就不要写入标注。"
)

NUMERIC_FORMAT_GUIDANCE = (
    "顶层第一个字段必须是 `video_duration_seconds`，表示当前上传视频的总时长秒数。"
    "`video_duration_seconds` 必须基于当前视频本身估计，不要凭电影知识、外部资料或剧情长度猜测。"
    "所有 `counts.*` 都必须是整数。所有时间戳字段（`first_appearance_seconds`, "
    "`last_appearance_seconds`, `start_seconds`, `end_seconds`, `seconds`）"
    "都使用从视频开始算起的秒数，写成保留一位小数的 JSON 数字，"
    "例如 `0.0`, `3.0`, `12.5`；不要写帧号、时间码或字符串。"
    "任何时间戳都必须满足 `0.0 <= 时间戳 <= video_duration_seconds`。"
    "禁止输出超过 `video_duration_seconds` 的 appearance、state_change 或 shot；"
    "视频结束后的内容不要补写、不要续写、不要用空镜头填充。"
)

P0_PROMPT = (
    "你是一名专业的电影分析员。请从头到尾观看用户上传的视频，基于视频证据输出一个合法 JSON 对象。\n\n"
    + SOURCE_EVIDENCE_GUIDANCE
    + "\n\n"
    "请先输出视频总时长，然后一次性输出实体表、每个实体的 appearances、每个实体的 state_changes，以及全片 shots。"
    "顶层对象必须且只能按顺序包含 `video_duration_seconds`, `characters`, `props`, `locations`, `shots`, `counts`。"
    "`counts` 只统计顶层数组，必须填写 `characters`, `props`, `locations`, `shots` 的数量。"
    "嵌套在实体下方的 `appearances` 和 `state_changes` 不写入顶层 `counts`，由每个实体自己的 "
    "`appearance_count` / `state_change_count` 统计。\n\n"
    "`characters`, `props`, `locations` 分别列出反复出现或对剧情理解重要的角色、关键道具、关键地点。"
    + ENTITY_STATE_GUIDANCE
    + "实体表聚焦主要角色、持续出现的配角、关键道具、关键地点；props 只列会被反复引用或参与状态变化的关键道具，"
    "一次性背景物、字幕、片尾装饰物通常不列入实体表。"
    "后处理会根据数组类型和顺序生成内部机器 ID，例如 char_001, prop_001, loc_001。\n\n"
    + SHOT_GUIDANCE
    + "\n\n"
    + NUMERIC_FORMAT_GUIDANCE
    + "自然语言描述优先使用中文；契约字段名和枚举值保持英文。")

P1_STAGE1_PROMPT = (
    "你是一名专业的电影分析员。请从头到尾观看用户上传的视频。"
    "如果视频有音频，可以结合视觉证据和音频证据；所有描述都基于视频证据。\n\n"
    + SOURCE_EVIDENCE_GUIDANCE
    + "\n\n"
    "请先输出视频总时长，然后建立全片实体表，并在每个实体下方写 appearances 和 state_changes。"
    "顶层对象必须且只能按顺序包含 `video_duration_seconds`, `characters`, `props`, `locations`, `counts`。"
    "`counts` 只统计顶层数组，必须填写 `characters`, `props`, `locations` 的数量。"
    "嵌套在实体下方的 `appearances` 和 `state_changes` 不写入顶层 `counts`，由每个实体自己的 "
    "`appearance_count` / `state_change_count` 统计。\n\n"
    + ENTITY_STATE_GUIDANCE
    + "实体表聚焦主要角色、持续出现的配角、关键道具、关键地点；props 只列会被反复引用或参与状态变化的关键道具，"
    "一次性背景物、字幕、片尾装饰物通常不列入实体表。"
    "后处理会根据数组类型和顺序生成内部机器 ID，例如 char_001, prop_001, loc_001。"
    + NUMERIC_FORMAT_GUIDANCE
    + "JSON 字段名、`identity_scope`、`state_change_kind` 等契约字段保持英文；自然语言描述优先使用中文。")

P1_STAGE2_PROMPT = (
    "你是一名专业的电影分析员。请继续基于同一个视频和已给定的实体表输出 shots。"
    "本次只输出镜头/片段时间线。最终输出只能是一个合法 JSON 对象，"
    "顶层对象只能按顺序有 `video_duration_seconds`, `shots`, `counts` 三个字段。"
    "`counts.shots` 必须等于 `shots` 的数组长度。\n\n"
    + SOURCE_EVIDENCE_GUIDANCE
    + "\n\n"
    "所有 `present_entity_names` 都必须来自给定实体表中的 `characters[*].name`, `props[*].name`, `locations[*].name`；"
    "沿用给定实体表中的已有 `name`。\n\n"
    + SHOT_GUIDANCE
    + "\n\n"
    + NUMERIC_FORMAT_GUIDANCE
    + "自然语言描述优先使用中文；契约字段名保持英文。")

P2_STAGE1_PROMPT = (
    "你是一名专业的电影分析员。请从头到尾观看用户上传的视频。"
    "如果视频有音频，可以结合视觉证据和音频证据；所有描述都基于视频证据。\n\n"
    + SOURCE_EVIDENCE_GUIDANCE
    + "\n\n"
    "请先输出视频总时长，然后建立全片实体表，并写 appearances。"
    "顶层对象必须且只能按顺序包含 `video_duration_seconds`, `characters`, `props`, `locations`, `counts` 五个字段。"
    "`counts` 只统计顶层数组，`counts.characters`, `counts.props`, `counts.locations` "
    "必须分别等于三个数组长度。嵌套在实体下方的 `appearances` 不写入顶层 `counts`，"
    "由每个实体自己的 `appearance_count` 统计。\n\n"
    + ENTITY_NO_STATE_GUIDANCE
    + "实体表聚焦主要角色、持续出现的配角、关键道具、关键地点；props 只列会被反复引用或参与状态变化的关键道具，"
    "一次性背景物、字幕、片尾装饰物通常不列入实体表。"
    "后处理会根据数组类型和顺序生成内部机器 ID，例如 char_001, prop_001, loc_001。"
    + NUMERIC_FORMAT_GUIDANCE
    + "JSON 字段名、`identity_scope` 等契约字段保持英文；自然语言描述优先使用中文。")

P2_STAGE2_PROMPT = (
    "你是一名专业的电影分析员。请继续基于同一个视频和已给定的实体表输出 shots 和 state_changes。"
    "本次输出镜头/片段时间线，并在顶层输出 state_changes。"
    "最终输出只能是一个合法 JSON 对象，顶层对象只能按顺序有 `video_duration_seconds`, `shots`, `state_changes`, `counts` 四个字段。"
    "`counts.shots` 和 `counts.state_changes` 必须分别等于 `shots` 和 `state_changes` 的数组长度。\n\n"
    + SOURCE_EVIDENCE_GUIDANCE
    + "\n\n"
    "所有 `present_entity_names` 和 `entity_name` 都必须来自给定实体表中的 `characters[*].name`, "
    "`props[*].name`, `locations[*].name`；沿用给定实体表中的已有 `name`。\n\n"
    + SHOT_GUIDANCE
    + "\n\n"
    + P2_STATE_GUIDANCE
    + "\n\n"
    + NUMERIC_FORMAT_GUIDANCE
    + "自然语言描述优先使用中文；契约字段名和枚举值保持英文。")

PROMPT = P0_PROMPT
ROSTER_PROMPT = P2_STAGE1_PROMPT
TIMELINE_PROMPT = P2_STAGE2_PROMPT


def video_url(path: str | Path) -> str:
    """Local video path -> `file://` URL (vLLM/the server must be able to read the path
    directly; relies on shared storage, no upload/base64)."""
    return f"file://{Path(path).resolve()}"


def build_request(video_path: str | Path, *, fps: float, model: str = DEFAULT_MODEL,
                  max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.6,
                  top_p: float = 0.95, presence_penalty: float = 0.0,
                  enable_thinking: bool = True, seed: int | None = 42,
                  stream: bool = False, request_kind: str = "single",
                  roster_json: dict[str, Any] | None = None) -> dict[str, Any]:
    """One `chat/completions` request body: whole-video zero-shot structured breakdown.

    Fields are flat top-level (not wrapped in ``extra_body``): that wrapping is an OpenAI
    *python client* convention (the client merges ``extra_body`` into the outgoing JSON body
    before sending; the wire format is flat either way) -- matches the existing
    ``judger/vlm.py`` client's call shape, and cross-checked against a hand-tested reference
    call (``scripts/run_vllm/qwen3_5-change-grounding-multitrack.py``) that hits the same server
    endpoint for a similar structured/timestamped video-grounding task. Sampling defaults
    (temperature/top_p/presence_penalty/top_k/min_p/repetition_penalty/seed) are carried over
    from that proven call rather than picked from scratch.
    """
    if request_kind in {"p0", "single"}:
        prompt = P0_PROMPT
        schema_name = "p0_movie_breakdown"
        schema = P0_SCHEMA
    elif request_kind == "p1_stage1":
        prompt = P1_STAGE1_PROMPT
        schema_name = "p1_entity_appearance_state"
        schema = P1_STAGE1_SCHEMA
    elif request_kind == "p1_stage2":
        if roster_json is None:
            raise ValueError("roster_json is required for request_kind='p1_stage2'")
        prompt = (
            P1_STAGE2_PROMPT
            + "\n\n已给定实体表如下。只能引用其中已有且全局唯一的 name：\n"
            + json.dumps(roster_json, ensure_ascii=False, indent=2)
        )
        schema_name = "p1_shots"
        schema = P1_STAGE2_SCHEMA
    elif request_kind in {"p2_stage1", "roster"}:
        prompt = P2_STAGE1_PROMPT
        schema_name = "p2_entity_appearance"
        schema = P2_STAGE1_SCHEMA
    elif request_kind in {"p2_stage2", "timeline"}:
        if roster_json is None:
            raise ValueError("roster_json is required for request_kind='p2_stage2'")
        prompt = (
            P2_STAGE2_PROMPT
            + "\n\n已给定实体表如下。只能引用其中已有且全局唯一的 name：\n"
            + json.dumps(roster_json, ensure_ascii=False, indent=2)
        )
        schema_name = "p2_shots_state_changes"
        schema = P2_STAGE2_SCHEMA
    else:
        raise ValueError(f"unknown request_kind: {request_kind}")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url(video_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.0,
        **({"seed": seed} if seed is not None else {}),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True},
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload
