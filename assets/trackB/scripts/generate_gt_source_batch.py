#!/usr/bin/env python3
"""Deterministic Track B gt_source batch generator.

Builds high-quality, auditable story skeletons that follow the same hard-case
coverage pattern as ``0001_lighthouse_keeper``:
  multi-scene chronology, lookalike solo/co-present, state chains, removal +
  indirect reference, count memory, false friends, long-gap filler, temporal
  anchors.

Author only the semantic skeleton (entities / state_machines / lookalike_pairs /
scenes). Derived memory labels come from ``complete_gt.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # trackB root (scripts/ lives one level down)
OUT_DIR = HERE / "gt_source"


def _story_shell(
    story_id: str,
    title: str,
    premise: str,
    entities: dict[str, Any],
    state_machines: dict[str, list[str]],
    lookalike_pairs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "story_id": story_id,
        "title": title,
        "premise": premise,
        "_comment": "作者层：人只写 entities + 状态机 + 分场时间线（present + 手写 action + 语义事件）。op/gap/probe/forbidden 由 complete_gt.py 派生。",
        "segment_sec": 5.0,
        "gap_long_threshold": 30,
        "avoidance_probe_window": 4,
        "probe_target_default": 5,
        "entities": entities,
        "state_machines": state_machines,
        "lookalike_pairs": lookalike_pairs,
        "scenes": scenes,
    }


def _act(action: str, present: list[Any] | None = None, events: list[dict] | None = None,
         lookalike_present: dict | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"action": action}
    if present is not None:
        out["present"] = present
    if events:
        out["events"] = events
    if lookalike_present is not None:
        out["lookalike_present"] = lookalike_present
    return out


def _lp(a: str, b: str, members: list[str]) -> dict[str, Any]:
    return {"pair": [a, b], "members": members}


# ---------------------------------------------------------------------------
# Shared chronological template (mirrors 0001 probe coverage)
# ---------------------------------------------------------------------------

def build_from_blueprint(bp: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact blueprint into a full gt_source story.

    Blueprint keys (all Chinese strings except ids):
      story_id, title, premise, style_note
      names/looks for cast roles, locations, props, counts, false friends
      state labels + appearance texts
      lookalike features + note
      key action phrase fragments for each beat
    """
    N = bp["names"]  # short aliases used in actions
    E = bp["entities"]
    SM = bp["state_machines"]
    LA, LB = bp["lookalike"]  # eid pair

    # Convenience role map used by scene builder
    P = bp["roles"]  # maps role -> eid
    # roles expected:
    #  hero, companion, twin, prop_main, prop_side, prop_kill,
    #  loc_a, loc_b, loc_c, loc_d, memorial,
    #  count_a, count_b, count_c,
    #  ff_kill, ff_hero, ff_comp, ff_prop, ff_extra

    scenes: list[dict[str, Any]] = []

    # S1 — introduce hero + main loc + counts
    scenes.append({
        "id": "S1",
        "setting": bp["S1_setting"],
        "present": [P["hero"], P["loc_a"]],
        "actions": [
            bp["S1_a1"],
            _act(bp["S1_a2"], [P["hero"], P["loc_a"], {"eid": P["count_b"], "count": bp["count_b_n"]}]),
            _act(bp["S1_a3"], [P["hero"], P["loc_a"], {"eid": P["count_c"], "count": bp["count_c_n"]}]),
            bp["S1_a4"],
            bp["S1_a5"],
            _act(bp["S1_a6"], [P["loc_a"], P["prop_main"]]),
            bp["S1_a7"],
        ],
    })

    # S2 — side prop / log
    scenes.append({
        "id": "S2",
        "setting": bp["S2_setting"],
        "present": [P["hero"], P["loc_a"], P["prop_main"], P["prop_side"]],
        "actions": [bp["S2_a1"], bp["S2_a2"], bp["S2_a3"],
                    _act(bp["S2_a4"], [P["loc_a"], P["prop_main"]])],
    })

    # S3 — companion + killable prop + count_a
    scenes.append({
        "id": "S3",
        "setting": bp["S3_setting"],
        "present": [P["hero"], P["loc_b"], P["companion"], P["prop_kill"]],
        "actions": [
            bp["S3_a1"],
            _act(bp["S3_a2"], [P["loc_b"], P["companion"], P["prop_kill"]]),
            _act(bp["S3_a3"], [P["hero"], P["loc_b"], P["companion"], {"eid": P["count_a"], "count": bp["count_a_n"]}]),
            _act(bp["S3_a4"], [P["loc_b"], P["companion"]]),
            _act(bp["S3_a5"], [P["hero"], P["loc_b"], P["companion"]]),
            _act(bp["S3_a6"], [P["loc_b"], P["companion"], P["prop_kill"]]),
        ],
    })

    # S4 — twin alone
    scenes.append({
        "id": "S4",
        "setting": bp["S4_setting"],
        "present": [P["twin"], P["loc_c"]],
        "lookalike_present": _lp(LA, LB, [P["twin"]]),
        "actions": [bp["S4_a1"], bp["S4_a2"], bp["S4_a3"], bp["S4_a4"]],
    })

    # S5 — both lookalikes
    scenes.append({
        "id": "S5",
        "setting": bp["S5_setting"],
        "present": [P["companion"], P["twin"], P["loc_b"]],
        "lookalike_present": _lp(LA, LB, [P["companion"], P["twin"]]),
        "actions": [bp["S5_a1"], bp["S5_a2"], bp["S5_a3"], bp["S5_a4"]],
    })

    # S6 — side prop state change
    scenes.append({
        "id": "S6",
        "setting": bp["S6_setting"],
        "present": [P["hero"], P["loc_a"], P["prop_main"], P["prop_side"]],
        "actions": [
            bp["S6_a1"],
            _act(bp["S6_a2"], events=[{"type": "state_change", "eid": P["prop_side"], "to": SM[P["prop_side"]][1]}]),
            bp["S6_a3"],
            bp["S6_a4"],
            _act(bp["S6_a5"], [P["loc_a"], P["prop_main"]]),
        ],
    })

    # S6b — filler to stretch gaps
    scenes.append({
        "id": "S6b",
        "setting": bp["S6b_setting"],
        "present": [P["hero"], P["loc_a"], P["prop_main"]],
        "actions": bp["S6b_actions"],
    })

    # S7 — count recall + false friend of kill prop
    scenes.append({
        "id": "S7",
        "setting": bp["S7_setting"],
        "present": [P["hero"], P["loc_b"], P["prop_kill"]],
        "actions": [
            bp["S7_a1"],
            _act(bp["S7_a2"], [P["loc_b"], P["prop_kill"]]),
            _act(bp["S7_a3"], [P["hero"], P["loc_b"], {"eid": P["count_a"], "count": bp["count_a_n"]}]),
            _act(bp["S7_a4"], [P["loc_b"], {"eid": P["ff_kill"], "confusable_with": P["prop_kill"]}]),
            _act(bp["S7_a5"], [P["hero"], P["loc_b"]]),
            _act(bp["S7_a6"], [P["hero"], P["loc_b"]]),
        ],
    })

    # S8 — disaster: main prop transform + kill prop destroy
    scenes.append({
        "id": "S8",
        "setting": bp["S8_setting"],
        "present": [P["prop_main"], P["loc_a"]],
        "actions": [
            bp["S8_a1"],
            bp["S8_a2"],
            _act(bp["S8_a3"], events=[{"type": "state_change", "eid": P["prop_main"], "to": SM[P["prop_main"]][1]}]),
            _act(bp["S8_a4"], [P["prop_kill"]]),
            _act(bp["S8_a5"], [P["prop_kill"]],
                 events=[{"type": "remove", "eid": P["prop_kill"], "to": SM[P["prop_kill"]][1],
                          "reason": "destroyed", "shown": True}]),
            _act(bp["S8_a6"], [P["hero"], P["loc_a"], P["prop_main"]]),
        ],
    })

    # S9 — aftermath: indirect ref + count_b recall
    scenes.append({
        "id": "S9",
        "setting": bp["S9_setting"],
        "present": [P["hero"], P["loc_a"], P["prop_main"]],
        "actions": [
            _act(bp["S9_a1"], [P["hero"], P["loc_a"], P["prop_main"]]),
            _act(bp["S9_a2"], [P["hero"], P["loc_a"], P["prop_main"]]),
            _act(bp["S9_a3"], [P["hero"], P["loc_a"]]),  # may name destroyed prop
            _act(bp["S9_a4"], [P["hero"], P["loc_a"]]),
            _act(bp["S9_a5"], [P["hero"], P["loc_a"], {"eid": P["count_b"], "count": bp["count_b_n"]}]),
            _act(bp["S9_a6"], [P["loc_a"], P["prop_main"]]),
        ],
    })

    # S10 — repair main prop + ff_prop
    scenes.append({
        "id": "S10",
        "setting": bp["S10_setting"],
        "present": [P["hero"], P["loc_a"], P["prop_main"]],
        "actions": [
            bp["S10_a1"],
            _act(bp["S10_a2"], [P["hero"], P["loc_a"], {"eid": P["ff_prop"], "confusable_with": P["prop_main"]}]),
            bp["S10_a3"],
            _act(bp["S10_a4"], events=[{"type": "state_change", "eid": P["prop_main"], "to": SM[P["prop_main"]][2]}]),
            bp["S10_a5"],
        ],
    })

    # S11 — false friends: distinct E19 (ff_extra) + ff_hero
    # false_friend probe only fires on introduce, so ff_extra must be a NEW eid
    # (not a recall of ff_kill from S7).
    scenes.append({
        "id": "S11",
        "setting": bp["S11_setting"],
        "present": [P["hero"], P["loc_b"]],
        "actions": [
            _act(bp["S11_a1"], [P["loc_b"], {"eid": P["ff_extra"], "confusable_with": P["prop_kill"]}]),
            _act(bp["S11_a2"], [P["hero"], P["loc_b"], {"eid": P["ff_hero"], "confusable_with": P["hero"]}]),
            _act(bp["S11_a3"], [P["hero"], P["loc_b"],
                                {"eid": P["ff_extra"], "confusable_with": P["prop_kill"]},
                                {"eid": P["ff_hero"], "confusable_with": P["hero"]}]),
            _act(bp["S11_a4"], [P["hero"], P["loc_b"]]),
        ],
    })

    # S12 — ff_comp
    scenes.append({
        "id": "S12",
        "setting": bp["S12_setting"],
        "present": [P["hero"], P["loc_b"]],
        "actions": [
            _act(bp["S12_a1"], [P["loc_b"], {"eid": P["ff_comp"], "confusable_with": P["companion"]}]),
            _act(bp["S12_a2"], [P["hero"], P["loc_b"], {"eid": P["ff_comp"], "confusable_with": P["companion"]}]),
        ],
    })

    # S13 — hero age/state + count_c long recall
    hero_states = SM.get(P["hero"], [])
    s13_actions: list[Any] = []
    if len(hero_states) >= 2:
        s13_actions.append(_act(
            bp["S13_a1"], [P["hero"], P["loc_d"]],
            events=[{"type": "state_change", "eid": P["hero"], "to": hero_states[1]}]))
    else:
        s13_actions.append(_act(bp["S13_a1"], [P["hero"], P["loc_d"]]))
    s13_actions.extend([
        bp["S13_a2"],
        _act(bp["S13_a3"], [P["hero"], P["loc_d"], {"eid": P["count_c"], "count": bp["count_c_n"]}]),
        bp["S13_a4"],
    ])
    scenes.append({
        "id": "S13",
        "setting": bp["S13_setting"],
        "present": [P["hero"], P["loc_d"]],
        "actions": s13_actions,
    })

    # S14 — companion state + optional hero removal
    s14_actions: list[Any] = []
    companion_states = SM.get(P["companion"], [])
    if len(companion_states) >= 2:
        s14_actions.append(_act(
            bp["S14_a1"], [P["companion"], P["loc_d"]],
            events=[{"type": "state_change", "eid": P["companion"], "to": companion_states[1]}]))
    else:
        s14_actions.append(_act(bp["S14_a1"], [P["companion"], P["loc_d"]]))
    s14_actions.append(bp["S14_a2"])
    s14_actions.append(_act(bp["S14_a3"], [P["hero"], P["companion"]]))
    if len(hero_states) >= 3:
        s14_actions.append(_act(
            bp["S14_a4"], [P["hero"], P["companion"]],
            events=[{"type": "remove", "eid": P["hero"], "to": hero_states[2],
                     "reason": "deceased", "shown": True}]))
    else:
        # remove a secondary if hero has no death state: use twin? No — keep hero.
        # Instead transform twin if available, else just continue.
        twin_states = SM.get(P["twin"], [])
        if len(twin_states) >= 2:
            s14_actions.append(_act(
                bp["S14_a4"], [P["companion"], P["twin"], P["loc_d"]],
                events=[{"type": "state_change", "eid": P["twin"], "to": twin_states[1]}],
                lookalike_present=_lp(LA, LB, [P["companion"], P["twin"]])))
        else:
            s14_actions.append(_act(bp["S14_a4"], [P["hero"], P["companion"], P["loc_d"]]))
    scenes.append({
        "id": "S14",
        "setting": bp["S14_setting"],
        "present": [P["hero"], P["companion"], P["loc_d"]],
        "actions": s14_actions,
    })

    # S15 — memorial / indirect
    if len(hero_states) >= 3:
        scenes.append({
            "id": "S15",
            "setting": bp["S15_setting"],
            "present": [P["companion"], P["loc_d"], P["memorial"]],
            "actions": [
                _act(bp["S15_a1"], [P["companion"], P["loc_a"]]),
                _act(bp["S15_a2"], [P["companion"], P["loc_d"], P["memorial"]]),
                _act(bp["S15_a3"], [P["companion"], P["loc_d"], P["memorial"]]),
            ],
        })
    else:
        scenes.append({
            "id": "S15",
            "setting": bp["S15_setting"],
            "present": [P["companion"], P["loc_d"], P["memorial"]],
            "actions": [
                _act(bp["S15_a1"], [P["companion"], P["loc_a"], P["hero"]]),
                _act(bp["S15_a2"], [P["companion"], P["loc_d"], P["memorial"], P["hero"]]),
                _act(bp["S15_a3"], [P["companion"], P["loc_d"], P["memorial"]]),
            ],
        })

    # S16 — twin reunion
    twin_states = SM.get(P["twin"], [])
    s16_actions: list[Any] = []
    if len(twin_states) >= 2 and not (len(hero_states) < 3 and len(twin_states) >= 2):
        # if twin not already aged in S14
        s16_actions.append(_act(
            bp["S16_a1"],
            events=[{"type": "state_change", "eid": P["twin"], "to": twin_states[1]}]))
    else:
        s16_actions.append(bp["S16_a1"])
    s16_actions.extend([bp["S16_a2"], bp["S16_a3"]])
    scenes.append({
        "id": "S16",
        "setting": bp["S16_setting"],
        "present": [P["companion"], P["twin"], P["loc_d"]],
        "lookalike_present": _lp(LA, LB, [P["companion"], P["twin"]]),
        "actions": s16_actions,
    })

    # S17 — temporal flashbacks (name-free anchors); avoid removed entities
    flash_present = bp["flash_presents"]  # list of present lists for 5 flashes
    scenes.append({
        "id": "S17",
        "setting": bp["S17_setting"],
        "present": [],
        "_comment": "时序/意图定位：全程尽量 name-free；anchor.resolves_to 供判分。",
        "actions": [
            _act(bp["S17_a1"], flash_present[0]),
            _act(bp["S17_a2"], flash_present[1]),
            _act(bp["S17_a3"], flash_present[2]),
            _act(bp["S17_a4"], flash_present[3]),
            _act(bp["S17_a5"], flash_present[4]),
        ],
    })

    # S18 — long-gap recalls
    scenes.append({
        "id": "S18",
        "setting": bp["S18_setting"],
        "present": [P["companion"], P["loc_a"], P["prop_main"]],
        "actions": [
            _act(bp["S18_a1"], [P["companion"], P["loc_a"], {"eid": P["prop_side"]}]),
            _act(bp["S18_a2"], [P["loc_a"], {"eid": P["count_a"], "count": bp["count_a_n"]}]),
            _act(bp["S18_a3"], [P["companion"], P["loc_a"], P["prop_main"]]),
        ],
    })

    return _story_shell(
        bp["story_id"], bp["title"], bp["premise"],
        E, SM, bp["lookalike_pairs"], scenes,
    )


# ---------------------------------------------------------------------------
# Blueprint helpers for night-market / desert (hand-tuned) and catalog
# ---------------------------------------------------------------------------

def blueprint_0002() -> dict[str, Any]:
    """Expanded neon night-market courier — quality bar matched to 0001."""
    entities = {
        "E1": {"name": "林", "kind": "character", "states": {
            "机警青年": "二十多岁的女信使，短发，穿亮红色连帽雨衣，左肩背一只黑色防水邮包，眼神机警。",
            "疲惫带伤": "同一位女信使的疲惫之貌：短发湿贴额头、红雨衣撕开一道口子，左臂缠着临时绷带，仍背黑色邮包。",
            "离场隐退": "信使林已隐退离场，只留下空荡的红色雨衣搭在椅背，人影不再出现。",
        }},
        "E2": {"name": "阿宁", "kind": "character", "states": {
            "少年跑腿": "少年时的阿宁：十五六岁、瘦小的夜市跑腿女孩，穿深蓝雨披，帽檐压得很低，眼神倔强。",
            "成年接班": "成年后的阿宁：二十出头、身形挺拔的女信使，仍穿深蓝雨披，眼神沉稳坚定。",
        }},
        "E3": {"name": "封蜡木盒", "kind": "prop", "states": {
            "完好封蜡": "一只巴掌大的深色木盒，盒盖压着一枚暗红封蜡印，四角包铜。",
            "被撬开": "封蜡木盒被撬开，暗红封蜡印崩裂剥落、盒盖撬起，铜角歪斜。",
            "空盒封存": "木盒盒盖大开后又被线绳草草勒住，内里空空，封蜡碎屑粘在盒沿；此后只以空盒示人。",
        }},
        "E4": {"name": "派件簿", "kind": "prop", "states": {
            "完好": "一本油布封面的派件簿，夹着复写纸，写满霓虹巷名与签收栏。",
            "雨渍糊字": "同一本派件簿被雨水浸透，封面发皱、墨迹晕开成紫黑污斑。",
        }},
        "E5": {"name": "旧摩托", "kind": "prop", "states": {
            "完好": "一辆暗红旧摩托，油箱贴着褪色『林』字贴纸，后座绑着防水邮包架。",
            "撞毁": "暗红旧摩托被货车擦撞翻倒，油箱凹陷、车轮扭曲，成为无法骑行的残骸。",
        }},
        "E6": {"name": "阿豹", "kind": "character", "states": {
            "青年劲敌": "与林体态相近的男信使劲敌，同样穿亮红色连帽雨衣，下巴有一道疤、右手戴银戒，神情桀骜。",
            "改头换面": "成年后的阿豹仍穿红雨衣，下巴疤更深、银戒换成黑铁戒，神情阴沉。",
        }},
        "E7": {"name": "夜市摊", "kind": "location",
               "appearance": "霓虹招牌下的湿漉夜市大排档，红灯笼、蒸汽与塑料棚顶滴水。"},
        "E8": {"name": "天台", "kind": "location",
               "appearance": "堆满空调外机与水箱的旧楼天台，远处是霓虹城市天际线，地面积水倒映灯光。"},
        "E9": {"name": "地铁通道", "kind": "location",
               "appearance": "泛白日光灯管的地下通道，瓷砖墙贴满褪色广告，尽头有闸机。"},
        "E10": {"name": "雨巷驿站", "kind": "location",
                "appearance": "窄雨巷尽头的信使驿站，铁皮屋顶、黄灯窗口、墙上挂满钥匙牌。"},
        "E11": {"name": "铜牌信箱", "kind": "prop",
                "appearance": "钉在驿站墙上的黄铜信箱，箱面刻着信使之名与年份，边缘已生铜绿。"},
        "E12": {"name": "备用蜡印", "kind": "prop",
                "appearance": "三枚深红备用蜡印，用麻绳串着挂在夜市摊木梁上，印面刻着星形纹。"},
        "E13": {"name": "共享电单车", "kind": "prop",
                "appearance": "一辆崭新的青绿色共享电单车，车筐里插着广告牌，与暗红旧摩托毫不相同。"},
        "E14": {"name": "城管老周", "kind": "character",
                "appearance": "一名精瘦城管，络腮胡、深蓝制服雨衣、大盖帽，身形与林有几分相似却更粗壮。"},
        "E15": {"name": "红灯笼串", "kind": "prop",
                "appearance": "一排五盏湿漉漉的红灯笼，挂在夜市摊棚沿，灯笼穗滴着雨水。"},
        "E16": {"name": "铜锁", "kind": "prop",
                "appearance": "两把挂在驿站铁门上的黄铜挂锁，锁梁泛着幽光，钥匙孔塞着防雨蜡。"},
        "E17": {"name": "小桃", "kind": "character",
                "appearance": "夜市另一头的跑腿少女，系浅褐发带、穿灰蓝雨披，乍看与少年阿宁有几分神似，却下巴更圆、目光怯生。"},
        "E18": {"name": "仿制木盒", "kind": "prop",
                "appearance": "一只半人掌大的仿制木盒，外形酷似封蜡木盒，但无铜角、封蜡是鲜红廉价蜡。"},
        "E19": {"name": "买家周", "kind": "character",
                "appearance": "戴金丝眼镜的中年男子，穿米色风衣，左手拎鳄纹皮箱，笑容温和却藏锋。"},
        "E20": {"name": "黄牌助力车", "kind": "prop",
                "appearance": "一辆挂着黄牌的旧助力车，车筐里塞着雨衣，外形近似旧摩托却绝非同一辆。"},
    }
    state_machines = {
        "E1": ["机警青年", "疲惫带伤", "离场隐退"],
        "E2": ["少年跑腿", "成年接班"],
        "E3": ["完好封蜡", "被撬开", "空盒封存"],
        "E4": ["完好", "雨渍糊字"],
        "E5": ["完好", "撞毁"],
        "E6": ["青年劲敌", "改头换面"],
    }
    lookalike_pairs = [{
        "pair": ["E1", "E6"],
        "features": {
            "E1": "林：女信使，亮红连帽雨衣，左肩黑邮包；下巴无疤、右手不戴戒指。",
            "E6": "阿豹：男劲敌，同穿亮红连帽雨衣、体态相近，下巴有疤、右手戴银戒。",
        },
        "note": "红雨衣错认：单人在场不得画出另一方；同段共屏须绑定疤/戒/邮包。",
    }]
    roles = {
        "hero": "E1", "companion": "E2", "twin": "E6",
        "prop_main": "E3", "prop_side": "E4", "prop_kill": "E5",
        "loc_a": "E8", "loc_b": "E7", "loc_c": "E9", "loc_d": "E10",
        "memorial": "E11",
        "count_a": "E12", "count_b": "E15", "count_c": "E16",
        "ff_kill": "E13", "ff_hero": "E14", "ff_comp": "E17",
        "ff_prop": "E18", "ff_extra": "E13",
    }
    # Note: lookalike pair is E1/E6 (林/阿豹). Companion E2 is apprentice 阿宁.
    # For lookalike scenes S4/S5 we need LA=E1 LB=E6, but S4 shows twin alone,
    # S5 shows companion+twin — that would be E2+E6 which is WRONG for lookalike.
    # Fix: make companion = lookalike A? In 0001, companion E2 and twin E6 ARE the lookalike pair.
    # So for 0002, lookalike should be 林(E1) and 阿豹(E6), but then companion for apprentice
    # should not be in the lookalike pair scenes as "companion".
    #
    # Remap to match 0001 structure:
    #   hero = older mentor? Actually 0001: hero=Elias, companion=Mara, twin=Lina (Mara/Lina lookalike)
    #   For 0002: hero=林, companion=阿宁? But lookalike is 林/阿豹.
    #
    # Better remap for 0002:
    #   hero = 买家周? No.
    #   Structure like 0001: companion/twin = lookalike pair.
    #   So: companion=林(E1), twin=阿豹(E6), hero= somehow else?
    #   Or: hero=林 with states, lookalike pair companion=阿宁 and someone? Theme says red raincoat lookalikes.
    #
    # Custom scenes for 0002 instead of generic blueprint builder.
    return {
        "story_id": "0002_night_market_courier",
        "title": "霓虹夜市信使",
        "premise": "霓虹雨夜的都市长剧。红雨衣信使林与劲敌阿豹极易错认；封蜡木盒完好→被撬→空盒；旧摩托毁于车祸后须规避；学徒阿宁长时缺席后接班；数量记忆（蜡印/灯笼/铜锁）、假朋友与时序闪回贯穿全片。仅作人读上下文，不喂给 SUT。",
        "entities": entities,
        "state_machines": state_machines,
        "lookalike_pairs": lookalike_pairs,
        "roles": roles,
        # Will use handcrafted scenes via build_0002_hand()
    }


def build_0002_hand() -> dict[str, Any]:
    """Hand-authored expanded 0002 matching 0001 coverage."""
    bp = blueprint_0002()
    E, SM = bp["entities"], bp["state_machines"]
    la = _lp("E1", "E6", ["E1"])
    lb = _lp("E1", "E6", ["E6"])
    both = _lp("E1", "E6", ["E1", "E6"])

    scenes = [
        {
            "id": "S1", "setting": "天台接件（引入五盏灯笼·两把铜锁）",
            "present": ["E1", "E8"],
            "actions": [
                "雨夜，红衣信使林踏上天台，积水倒映着远处霓虹。",
                _act("夜市摊棚沿挂着一串五盏红灯笼串，林抬头数了数，确认自己没有走错巷。",
                     ["E1", "E7", {"eid": "E15", "count": 5}]),
                _act("雨巷驿站铁门上挂着两把铜锁，林用钥匙拧开其中一把。",
                     ["E1", "E10", {"eid": "E16", "count": 2}]),
                "林在天台甩去帽檐的雨水，护紧左肩黑色邮包。",
                "林检查邮包夹层，确认封蜡木盒还在。",
                _act("封蜡木盒在天台积水旁静静搁着，暗红封蜡印完好。", ["E8", "E3"]),
                "林靠着水箱小憩，红雨衣在风里轻轻鼓动。",
            ],
        },
        {
            "id": "S2", "setting": "登记派件簿",
            "present": ["E1", "E8", "E3", "E4"],
            "actions": [
                "林翻开派件簿，用复写纸登记天台这一单。",
                "林在派件簿上写下买家的代号与交接时间。",
                _act("戴金丝眼镜的买家周短暂现身天台边缘，只点了点头便消失在雨里。",
                     ["E1", "E8", "E19"]),
                "林合上派件簿，塞回邮包侧袋。",
                _act("霓虹映在封蜡木盒的铜角上，一闪一闪。", ["E8", "E3"]),
            ],
        },
        {
            "id": "S3", "setting": "夜市迎学徒（引入三枚蜡印）",
            "present": ["E1", "E7", "E2", "E5"],
            "lookalike_present": la,
            "actions": [
                "清晨的夜市摊，林迎向骑着暗红旧摩托赶来的少年阿宁。",
                _act("阿宁跳下旧摩托，把头盔挂在车把上。", ["E7", "E2", "E5"]),
                _act("林指着木梁上串着的三枚备用蜡印，教阿宁辨认星形纹。",
                     ["E1", "E7", "E2", {"eid": "E12", "count": 3}]),
                _act("阿宁蹲在夜市摊的水洼旁，好奇地看蒸汽升腾。", ["E7", "E2"]),
                _act("林向阿宁讲解雨巷里哪些招牌是暗号。", ["E1", "E7", "E2"]),
                _act("阿宁回望停着的旧摩托，眼里满是向往。", ["E7", "E2", "E5"]),
            ],
        },
        {
            "id": "S4", "setting": "地铁通道（只有阿豹）",
            "present": ["E6", "E9"],
            "lookalike_present": lb,
            "actions": [
                "泛白的地铁通道里，红雨衣的阿豹从广告墙后回头，下巴疤一闪。",
                "阿豹在地铁通道尽头的闸机前停步，右手银戒敲着栏杆。",
                "阿豹与通道里的流浪歌手擦肩，神情桀骜。",
                "阿豹提着空邮包穿过地铁通道的人群。",
            ],
        },
        {
            "id": "S5", "setting": "红雨衣擦肩共屏",
            "present": ["E1", "E6", "E9"],
            "lookalike_present": both,
            "actions": [
                "地铁通道里，林与阿豹并肩而过，一个背黑邮包、一个戴银戒。",
                "林下意识护住邮包，阿豹轻嗤一声继续前行。",
                "阿豹回头盯了林一眼，两人红雨衣在灯管下几乎同色。",
                "林与阿豹在通道岔口分开，各自走向相反方向。",
            ],
        },
        {
            "id": "S6", "setting": "派件簿雨渍",
            "present": ["E1", "E8", "E3", "E4"],
            "lookalike_present": la,
            "actions": [
                "数周后的暴雨夜，林独自在天台值守，封蜡木盒压在膝上。",
                _act("林翻开派件簿，封面已被雨水渍得发皱、字迹糊成污斑。",
                     events=[{"type": "state_change", "eid": "E4", "to": "雨渍糊字"}]),
                "林擦掉封蜡木盒铜角上的水珠。",
                "林凭着天台栏杆，望向被雨幕吞没的霓虹。",
                _act("封蜡木盒在积水旁映出破碎的灯影。", ["E8", "E3"]),
            ],
        },
        {
            "id": "S6b", "setting": "雨季流转（填充·拉长间隔）",
            "present": ["E1", "E8"],
            "lookalike_present": la,
            "actions": [
                "雨季漫长，林日复一日踏上天台交接。",
                "又一个台风夜，林裹紧红雨衣守在水箱旁。",
                "清明时节，林在天台望着远处未熄的霓虹。",
                "盛夏正午，林擦拭被烈日晒烫的空调外机。",
                "秋雨绵绵，封蜡木盒的蜡印在潮气里微微发黏。",
                "又一个除夕，林独自在天台守到天明。",
            ],
        },
        {
            "id": "S7", "setting": "夜市巡查（回忆蜡印·共享电单车假朋友）",
            "present": ["E1", "E7", "E5"],
            "lookalike_present": la,
            "actions": [
                "拂晓，林沿夜市摊巡查，暗红旧摩托泊在摊旁。",
                _act("林蹲下检查旧摩托的链条松紧。", ["E7", "E5"]),
                _act("林走到木梁下，先前那三枚备用蜡印仍串在原处。",
                     ["E1", "E7", {"eid": "E12", "count": 3}]),
                _act("一辆青绿色共享电单车停在摊前，车筐插着广告牌，并非林的旧摩托。",
                     ["E7", {"eid": "E13", "confusable_with": "E5"}]),
                _act("林站在夜市摊记录昨夜未签收的单号。", ["E1", "E7"]),
                _act("林把湿透的雨披晾在夜市摊的竹竿上。", ["E1", "E7"]),
            ],
        },
        {
            "id": "S8", "setting": "雨夜车祸（木盒被撬·旧摩托撞毁）",
            "present": ["E3", "E8"],
            "actions": [
                "暴雨骤起，天台积水漫过脚踝，封蜡木盒几乎漂起。",
                "远处货车喇叭刺耳，巷口传来金属刮擦声。",
                _act("阿豹摸上天台，用小刀撬开封蜡木盒，暗红封蜡崩裂。",
                     ["E6", "E3", "E8"],
                     events=[{"type": "state_change", "eid": "E3", "to": "被撬开"}],
                     lookalike_present=lb),
                _act("巷口，暗红旧摩托在雨幕中剧烈颠簸。", ["E5", "E7"]),
                _act("一记擦撞将旧摩托掀翻，油箱凹陷、车轮扭曲。",
                     ["E5", "E7"],
                     events=[{"type": "remove", "eid": "E5", "to": "撞毁",
                              "reason": "destroyed", "shown": True}]),
                _act("天台上，林扑向被撬开的封蜡木盒，死死按住盒盖。",
                     ["E1", "E8", "E3"], lookalike_present=la),
            ],
        },
        {
            "id": "S9", "setting": "灾后（旧摩托间接指代·灯笼回忆）",
            "present": ["E1", "E8"],
            "lookalike_present": la,
            "actions": [
                _act("暴雨次日，林站在天台，凝视着被撬开的封蜡木盒。", ["E1", "E8", "E3"]),
                _act("林用手指抚过封蜡木盒崩裂的蜡屑，神情黯然。", ["E1", "E8", "E3"]),
                _act("林在天台的派件墙旁，记下旧摩托失事的日期。", ["E1", "E8"]),
                _act("驿站墙上挂着旧摩托的照片，林久久凝望。", ["E1", "E10"]),
                _act("林取下棚沿五盏红灯笼中的一盏，就着微光独坐。",
                     ["E1", "E7", {"eid": "E15", "count": 5}]),
                _act("被撬开的封蜡木盒仍敞着口，空空如也。", ["E8", "E3"]),
            ],
        },
        {
            "id": "S10", "setting": "空盒封存（第三态·仿制木盒假朋友）",
            "present": ["E1", "E8", "E3"],
            "lookalike_present": la,
            "actions": [
                "数周后，林用线绳把空木盒勒紧，准备封存证据。",
                _act("林从纸箱取出一只仿制木盒，它酷似封蜡木盒、却无铜角。",
                     ["E1", "E8", {"eid": "E18", "confusable_with": "E3"}]),
                "林小心地把碎蜡扫进纸袋，贴上封存标签。",
                _act("线绳勒紧，封蜡木盒成为空盒封存之态，内里空空。",
                     events=[{"type": "state_change", "eid": "E3", "to": "空盒封存"}]),
                "林把空盒封存的封蜡木盒锁进天台铁柜，长出一口气。",
            ],
        },
        {
            "id": "S11", "setting": "假朋友：黄牌助力车与城管",
            "present": ["E1", "E7"],
            "lookalike_present": la,
            "actions": [
                _act("一辆黄牌助力车停进夜市摊，与已撞毁的旧摩托毫不相同。",
                     ["E7", {"eid": "E20", "confusable_with": "E5"}]),
                _act("一名络腮胡的城管老周走入夜市摊，与林并肩交谈，两人身形相仿。",
                     ["E1", "E7", {"eid": "E14", "confusable_with": "E1"}]),
                _act("城管老周指着黄牌助力车，向林说明新的停放规定。",
                     ["E1", "E7", {"eid": "E20", "confusable_with": "E5"},
                      {"eid": "E14", "confusable_with": "E1"}]),
                _act("黄牌助力车被推走，林独自留在夜市摊目送。", ["E1", "E7"]),
            ],
        },
        {
            "id": "S12", "setting": "假朋友：神似阿宁的小桃",
            "present": ["E1", "E7"],
            "lookalike_present": la,
            "actions": [
                _act("夜市另一头来了个系浅褐发带的跑腿少女小桃，乍看竟有几分像少年阿宁。",
                     ["E7", {"eid": "E17", "confusable_with": "E2"}]),
                _act("小桃学着在摊前系缆绳式的邮包带，林却摇摇头——她终究不是阿宁。",
                     ["E1", "E7", {"eid": "E17", "confusable_with": "E2"}]),
            ],
        },
        {
            "id": "S13", "setting": "疲惫带伤（铜锁长程回忆）",
            "present": ["E1", "E10"],
            "lookalike_present": la,
            "actions": [
                _act("半年后的黄昏，左臂缠绷带的林疲惫地立在雨巷驿站门口。",
                     ["E1", "E10"],
                     events=[{"type": "state_change", "eid": "E1", "to": "疲惫带伤"}]),
                "林仰望驿站黄灯窗口，目光悠远。",
                _act("驿站铁门上那两把黄铜挂锁在雨里轻响，林驻足倾听。",
                     ["E1", "E10", {"eid": "E16", "count": 2}]),
                "林把手掌贴在驿站冰凉的铁皮墙上，似在告别。",
            ],
        },
        {
            "id": "S14", "setting": "阿宁接班·林隐退",
            "present": ["E1", "E2", "E10"],
            "lookalike_present": la,
            "actions": [
                _act("白日，成年的阿宁沿雨巷归来，快步走向驿站。",
                     ["E2", "E10"],
                     events=[{"type": "state_change", "eid": "E2", "to": "成年接班"}]),
                "阿宁在驿站门前与带伤的林重逢，两人短暂拥抱。",
                _act("驿站里，阿宁接过林递来的空盒封存钥匙。", ["E1", "E2", "E10"]),
                _act("林把红色雨衣脱下搭在椅背，转身隐入雨巷，不再出现。",
                     ["E1", "E2", "E10"],
                     events=[{"type": "remove", "eid": "E1", "to": "离场隐退",
                              "reason": "deceased", "shown": True}]),
            ],
        },
        {
            "id": "S15", "setting": "遗物·铜牌信箱（间接指代）",
            "present": ["E2", "E10", "E11"],
            "actions": [
                _act("驿站椅背上仍搭着林的红色雨衣，阿宁伸手轻轻抚过。", ["E2", "E10"]),
                _act("驿站墙边，成年阿宁伫立在一块新钉上的铜牌信箱前。",
                     ["E2", "E10", "E11"]),
                _act("阿宁的指尖抚过铜牌信箱上刻着的林之名。", ["E2", "E10", "E11"]),
            ],
        },
        {
            "id": "S16", "setting": "阿豹改头换面重逢（长程+共屏）",
            "present": ["E2", "E6", "E10"],
            "lookalike_present": _lp("E1", "E6", ["E6"]),
            "actions": [
                _act("改头换面的阿豹从地铁方向赶来，黑铁戒在雨里一闪。",
                     ["E6", "E10"],
                     events=[{"type": "state_change", "eid": "E6", "to": "改头换面"}]),
                "阿宁拦住阿豹，一人深蓝雨披、一人红雨衣，对峙在驿站门口。",
                "阿豹冷笑着离开，阿宁握紧空盒封存的钥匙。",
            ],
        },
        {
            "id": "S17", "setting": "回忆闪回（时序定位）",
            "present": [],
            "actions": [
                _act("镜头回到多年前——那个在夜市摊学认蜡印的少年，身影再度浮现。",
                     [{"eid": "E2", "state": "少年跑腿",
                       "anchor": {"type": "temporal", "phrase": "在夜市摊学认蜡印的少年",
                                  "resolves_to": "E2"}}]),
                _act("重现开场的那一刻——天台上，一只封蜡完好的木盒第一次被取出。",
                     ["E8", {"eid": "E3", "state": "完好封蜡",
                             "anchor": {"type": "temporal",
                                        "phrase": "开场第一次被取出的封蜡完好木盒",
                                        "resolves_to": "E3"}}]),
                _act("闪回到当年的地铁通道——那个下巴有疤、右手戴银戒的红雨衣身影一闪而过。",
                     [{"eid": "E6", "state": "青年劲敌",
                       "anchor": {"type": "temporal",
                                  "phrase": "当年通道里戴银戒的红雨衣身影",
                                  "resolves_to": "E6"}}]),
                _act("眼前又浮现夜市棚沿那一排五盏湿漉漉的红灯笼。",
                     [{"eid": "E15", "count": 5,
                       "anchor": {"type": "temporal",
                                  "phrase": "夜市棚沿那一排五盏红灯笼",
                                  "resolves_to": "E15"}}]),
                _act("回到初见时那片积水的高台，霓虹倒映，一如往昔。",
                     [{"eid": "E8",
                       "anchor": {"type": "temporal",
                                  "phrase": "初见时那片积水的天台",
                                  "resolves_to": "E8"}}]),
            ],
        },
        {
            "id": "S18", "setting": "收束（派件簿长程·蜡印长程·买家回归）",
            "present": ["E2", "E8", "E3"],
            "actions": [
                _act("阿宁重新翻开那本雨渍糊字的派件簿，走上天台。",
                     ["E2", "E8", {"eid": "E4"}]),
                _act("夜市木梁边，先前那三枚备用蜡印依旧串在原处。",
                     ["E7", {"eid": "E12", "count": 3}]),
                _act("买家周终于现身天台，阿宁把空盒封存的封蜡木盒拍在积水旁的木箱上。",
                     ["E2", "E8", "E3", "E19"]),
            ],
        },
    ]
    return _story_shell(
        "0002_night_market_courier", bp["title"], bp["premise"],
        E, SM, bp["lookalike_pairs"], scenes,
    )


def build_0003_hand() -> dict[str, Any]:
    """Hand-authored expanded 0003 desert archaeologist."""
    entities = {
        "E1": {"name": "艾拉", "kind": "character", "states": {
            "青壮探险": "三十岁出头的女考古学家，戴宽檐帆布帽与护目镜，穿卡其探险装，脖挂放大镜，手上常沾沙尘。",
            "风沙灼伤": "同一位考古学家的风沙灼伤之貌：帆布帽边缘焦黑、护目镜裂了一道细纹，面颊晒伤脱皮，仍挂着放大镜。",
            "离队静养": "艾拉已离队静养，营地只留她的帆布帽与放大镜，人影不再出现。",
        }},
        "E2": {"name": "娜迪娅", "kind": "character", "states": {
            "少年学徒": "少年时的娜迪娅：十四五岁、身形清瘦的沙漠学徒少女，系靛蓝头巾，穿粗棉沙色短袍，眉眼清亮。",
            "成年助手": "成年后的娜迪娅：二十余岁、身形挺拔的女助手，仍系靛蓝头巾、穿沙色短袍，眉眼坚定。",
        }},
        "E3": {"name": "楔形泥碑", "kind": "prop", "states": {
            "完好": "一块砖红色泥板，正面刻满楔形文字，边缘有古老磕痕，约两掌大。",
            "碎裂": "楔形泥碑跌落岩石，砖红碑面迸裂成数块残片。",
            "拼合复原": "碎裂残片被逐块拼合，砖红碑面重现完整楔形文字，但可见拼缝。",
        }},
        "E4": {"name": "伪造羊皮地图", "kind": "prop", "states": {
            "伪图": "一张做旧的浅黄羊皮地图，墨线标注绿洲与遗址，右下角有一枚可疑的红蜡伪印。",
            "证伪受潮": "羊皮地图右下角红蜡印被识破为伪造，图面受潮起皱、墨线晕开。",
        }},
        "E5": {"name": "驮运骆驼", "kind": "prop", "states": {
            "健壮": "一峰健壮的沙色骆驼，鞍囊绣着蓝色几何纹，脖铃清脆。",
            "倒毙": "沙色骆驼倒毙于沙丘背风处，鞍囊散落，脖铃不再作响。",
        }},
        "E6": {"name": "向导哈桑", "kind": "character", "states": {
            "青年向导": "沉稳的沙漠向导，披沙色粗布斗篷，缠深蓝头巾，左颊有风沙皱纹，牵一峰骆驼。",
            "鬓白归来": "多年后的哈桑仍披沙色斗篷、缠深蓝头巾，鬓角已白，左颊皱纹更深。",
        }},
        "E7": {"name": "蒙面劫匪", "kind": "character", "states": {
            "青年劫匪": "同披沙色斗篷的蒙面人，体态与哈桑相近，缠赭红头巾、腰间别弯刀，眼神凶悍。",
            "伤疤劫匪": "成年后的劫匪仍缠赭红头巾、蒙面持弯刀，面罩下露出一道新伤疤。",
        }},
        "E8": {"name": "遗址石门", "kind": "location",
               "appearance": "半埋于沙丘的古遗址石门，门楣刻浮雕，两侧断柱，落日把阴影拉长。"},
        "E9": {"name": "沙暴荒漠", "kind": "location",
               "appearance": "无垠的赭黄沙丘，风起时黄沙蔽日、能见度极低。"},
        "E10": {"name": "绿洲营地", "kind": "location",
                "appearance": "棕榈环绕的小绿洲，几顶米色帐篷、一汪浅水、拴着骆驼。"},
        "E11": {"name": "发掘帐篷", "kind": "location",
                "appearance": "遗址旁的临时发掘帐篷，帆布门掀开，桌上摊着测绘工具。"},
        "E12": {"name": "纪念石板", "kind": "prop",
                "appearance": "一块立在绿洲边的砂岩石板，表面刻着考古队长之名与年份，边缘被风沙磨圆。"},
        "E13": {"name": "陶油灯", "kind": "prop",
                "appearance": "三盏手掌大的陶油灯，蒙着沙尘，摆在遗址石门内侧的壁龛上。"},
        "E14": {"name": "出租越野车", "kind": "prop",
                "appearance": "一辆崭新的沙色出租越野车，车顶绑着备胎，与驮运骆驼毫不相同。"},
        "E15": {"name": "营地医生", "kind": "character",
                "appearance": "一名精瘦的营地医生，络腮胡、卡其背心、帆布帽，身形与艾拉有几分相似。"},
        "E16": {"name": "测绘旗", "kind": "prop",
                "appearance": "一排五面红色测绘小旗，插在石门前的沙地上，旗面印着白色坐标编号。"},
        "E17": {"name": "铜铃", "kind": "prop",
                "appearance": "两只悬在帐篷檐下的黄铜铃，铃体泛着幽光，风起时轻响报风。"},
        "E18": {"name": "莎娜", "kind": "character",
                "appearance": "绿洲另一侧的放羊少女，系浅褐头巾、穿灰扑扑的粗布袍，乍看与少年娜迪娅有几分神似，却目光怯生。"},
        "E19": {"name": "备用泥板", "kind": "prop",
                "appearance": "一块半人掌大的备用空白泥板，色泽与楔形泥碑相近，但无文字、边缘更整齐。"},
        "E20": {"name": "沙色辎重车", "kind": "prop",
                "appearance": "一辆沙色辎重拖车，车斗蒙着帆布，外形近似驮运役畜的补给单元却绝非骆驼。"},
    }
    state_machines = {
        "E1": ["青壮探险", "风沙灼伤", "离队静养"],
        "E2": ["少年学徒", "成年助手"],
        "E3": ["完好", "碎裂", "拼合复原"],
        "E4": ["伪图", "证伪受潮"],
        "E5": ["健壮", "倒毙"],
        "E6": ["青年向导", "鬓白归来"],
        "E7": ["青年劫匪", "伤疤劫匪"],
    }
    lookalike_pairs = [{
        "pair": ["E6", "E7"],
        "features": {
            "E6": "哈桑：沙色斗篷、深蓝头巾，未蒙面、不持械，左颊有风沙皱纹。",
            "E7": "蒙面劫匪：同披沙色斗篷、体态相近，赭红头巾、蒙面、腰别弯刀。",
        },
        "note": "沙色斗篷错认：单人在场不得画出另一方；共屏须绑定头巾色/蒙面/弯刀。",
    }]
    ha = _lp("E6", "E7", ["E6"])
    hb = _lp("E6", "E7", ["E7"])
    both = _lp("E6", "E7", ["E6", "E7"])

    scenes = [
        {
            "id": "S1", "setting": "石门日常（引入五面旗·两只铜铃）",
            "present": ["E1", "E8"],
            "actions": [
                "落日下，考古学家艾拉拂去遗址石门浮雕上的黄沙。",
                _act("石门前沙地插着一排五面红色测绘旗，艾拉逐一核对编号。",
                     ["E1", "E8", {"eid": "E16", "count": 5}]),
                _act("发掘帐篷檐下悬着两只黄铜铃，艾拉伸手让它们轻轻鸣响。",
                     ["E1", "E11", {"eid": "E17", "count": 2}]),
                "艾拉绕着石门检查每一道裂缝。",
                "艾拉凭着断柱眺望渐暗的沙海。",
                _act("完好的楔形泥碑搁在石门内侧壁龛旁，楔形文字清晰。", ["E8", "E3"]),
                "艾拉在石门阴影里坐下，就着余晖小憩。",
            ],
        },
        {
            "id": "S2", "setting": "展阅伪图",
            "present": ["E1", "E8", "E3", "E4"],
            "actions": [
                "艾拉摊开伪造羊皮地图，对照石门方位。",
                "艾拉在伪造羊皮地图上用铅笔标出一条可疑路线。",
                "艾拉卷起伪造羊皮地图，塞回帆布筒。",
                _act("落日把楔形泥碑的影子投在石门内侧。", ["E8", "E3"]),
            ],
        },
        {
            "id": "S3", "setting": "迎学徒（引入三盏陶油灯）",
            "present": ["E1", "E9", "E2", "E5"],
            "actions": [
                "清晨的沙暴荒漠边缘，艾拉迎向牵着驮运骆驼走来的少年娜迪娅。",
                _act("娜迪娅把驮运骆驼拴在断柱旁，拍去鞍囊上的沙。", ["E9", "E2", "E5"]),
                _act("艾拉指着石门壁龛里的三盏陶油灯，教娜迪娅辨认灯盏纹样。",
                     ["E1", "E8", "E2", {"eid": "E13", "count": 3}]),
                _act("娜迪娅蹲在沙丘背风处，好奇地拨弄干枯的灌木。", ["E9", "E2"]),
                _act("艾拉向娜迪娅讲解沙丘走向与风向。", ["E1", "E9", "E2"]),
                _act("娜迪娅回望健壮的驮运骆驼，眼里满是向往。", ["E9", "E2", "E5"]),
            ],
        },
        {
            "id": "S4", "setting": "绿洲（只有哈桑）",
            "present": ["E6", "E10"],
            "lookalike_present": ha,
            "actions": [
                "棕榈环绕的绿洲营地里，深蓝头巾的向导哈桑从帐篷后回头张望。",
                "哈桑在绿洲浅水边给骆驼打水。",
                "哈桑整理米色帐篷的绳索，左颊皱纹在落日下更深。",
                "哈桑牵着骆驼穿过绿洲营地。",
            ],
        },
        {
            "id": "S5", "setting": "向导与劫匪共屏对峙",
            "present": ["E6", "E7", "E9"],
            "lookalike_present": both,
            "actions": [
                "沙暴荒漠上，哈桑与蒙面劫匪遥遥对峙，一个深蓝头巾、一个赭红头巾。",
                "哈桑把骆驼护在身后，劫匪腰间弯刀出鞘半寸。",
                "劫匪低喝一声退入沙尘，哈桑仍立在原处。",
                "哈桑与劫匪各自走向相反的沙丘。",
            ],
        },
        {
            "id": "S6", "setting": "伪图受潮证伪",
            "present": ["E1", "E11", "E3", "E4"],
            "actions": [
                "数月后的夜里，艾拉独自在发掘帐篷整理测绘。",
                _act("艾拉借放大镜识破伪造羊皮地图的红蜡伪印，图面已受潮起皱。",
                     events=[{"type": "state_change", "eid": "E4", "to": "证伪受潮"}]),
                "艾拉擦拭楔形泥碑上的沙尘。",
                "艾拉掀开帐篷门帘，望向漆黑的沙海。",
                _act("楔形泥碑在油灯旁静静躺着。", ["E11", "E3"]),
            ],
        },
        {
            "id": "S6b", "setting": "沙季流转（填充）",
            "present": ["E1", "E8", "E3"],
            "actions": [
                "沙季漫长，艾拉日复一日清理石门浮雕。",
                "又一个沙暴夜，艾拉裹紧卡其外套守在石门内侧。",
                "春旱时节，艾拉在石门前望着远处海市蜃楼。",
                "盛夏正午，艾拉擦拭被烈日晒烫的断柱。",
                "秋风起时，楔形泥碑的砖红色在沙尘里显得黯淡。",
                "又一个岁末，艾拉独自在石门下守到天明。",
            ],
        },
        {
            "id": "S7", "setting": "巡查（回忆陶油灯·越野车假朋友）",
            "present": ["E1", "E9", "E5"],
            "actions": [
                "拂晓，艾拉沿沙暴荒漠边缘巡查，健壮的驮运骆驼跟在身后。",
                _act("艾拉检查驮运骆驼鞍囊的缝线。", ["E9", "E5"]),
                _act("艾拉走进石门，先前那三盏陶油灯仍摆在壁龛上。",
                     ["E1", "E8", {"eid": "E13", "count": 3}]),
                _act("一辆沙色出租越野车驶过沙丘，车顶绑着备胎，并非驮运骆驼。",
                     ["E9", {"eid": "E14", "confusable_with": "E5"}]),
                _act("艾拉站在沙丘上记录风向变化。", ["E1", "E9"]),
                _act("艾拉把测绳晾在沙丘背风处的木架上。", ["E1", "E9"]),
            ],
        },
        {
            "id": "S8", "setting": "沙暴之夜（碑碎·骆驼倒毙）",
            "present": ["E3", "E8"],
            "actions": [
                "沙暴骤起，遗址石门在狂风中震颤。",
                "黄沙灌进石门，能见度骤降。",
                _act("艾拉踉跄中让楔形泥碑跌落岩石，砖红碑面迸裂。",
                     ["E1", "E3", "E9"],
                     events=[{"type": "state_change", "eid": "E3", "to": "碎裂"}]),
                _act("远处沙丘上，驮运骆驼在风暴中剧烈颠簸。", ["E5", "E9"]),
                _act("一记塌陷的沙丘吞没驮运骆驼，它倒毙不起。",
                     ["E5", "E9"],
                     events=[{"type": "remove", "eid": "E5", "to": "倒毙",
                              "reason": "destroyed", "shown": True}]),
                _act("石门里，艾拉扑向碎裂的楔形泥碑残片，死死护住。",
                     ["E1", "E8", "E3"]),
            ],
        },
        {
            "id": "S9", "setting": "灾后（骆驼间接指代·测绘旗回忆）",
            "present": ["E1", "E8", "E3"],
            "actions": [
                _act("沙暴次日，艾拉站在石门里，凝视着碎裂的楔形泥碑。", ["E1", "E8", "E3"]),
                _act("艾拉用手指抚过楔形泥碑的拼缝缺口，神情黯然。", ["E1", "E8", "E3"]),
                _act("艾拉在发掘帐篷的日志旁，记下驮运骆驼倒毙的日期。", ["E1", "E11"]),
                _act("帐篷墙上挂着驮运骆驼的旧照片，艾拉久久凝望。", ["E1", "E11"]),
                _act("艾拉拔起沙地上五面测绘旗中的一面，就着残旗独坐。",
                     ["E1", "E8", {"eid": "E16", "count": 5}]),
                _act("碎裂的楔形泥碑残片仍散在壁龛下。", ["E8", "E3"]),
            ],
        },
        {
            "id": "S10", "setting": "拼合复原（备用泥板假朋友）",
            "present": ["E1", "E11", "E3"],
            "actions": [
                "数周后，艾拉在发掘帐篷铺开残片，开始拼合楔形泥碑。",
                _act("艾拉从木箱取出一块备用泥板，它酷似楔形泥碑、却无文字。",
                     ["E1", "E11", {"eid": "E19", "confusable_with": "E3"}]),
                "艾拉小心地把残片一块块对齐。",
                _act("最后一道拼缝合拢，楔形泥碑成为拼合复原之态。",
                     events=[{"type": "state_change", "eid": "E3", "to": "拼合复原"}]),
                "拼合复原的楔形泥碑躺在桌上，艾拉欣慰地望着它。",
            ],
        },
        {
            "id": "S11", "setting": "假朋友：辎重车与营地医生",
            "present": ["E1", "E9"],
            "actions": [
                _act("一辆沙色辎重车驶入沙丘边缘，与已倒毙的驮运骆驼毫不相同。",
                     ["E9", {"eid": "E20", "confusable_with": "E5"}]),
                _act("一名络腮胡的营地医生跳下沙丘，与艾拉并肩交谈，两人身形相仿。",
                     ["E1", "E9", {"eid": "E15", "confusable_with": "E1"}]),
                _act("营地医生指着沙色辎重车，向艾拉说明新的补给安排。",
                     ["E1", "E9", {"eid": "E20", "confusable_with": "E5"},
                      {"eid": "E15", "confusable_with": "E1"}]),
                _act("沙色辎重车鸣笛驶离，艾拉独自留在沙丘目送。", ["E1", "E9"]),
            ],
        },
        {
            "id": "S12", "setting": "假朋友：神似娜迪娅的莎娜",
            "present": ["E1", "E10"],
            "actions": [
                _act("绿洲另一侧来了个系浅褐头巾的放羊少女莎娜，乍看竟有几分像少年娜迪娅。",
                     ["E10", {"eid": "E18", "confusable_with": "E2"}]),
                _act("莎娜学着在浅水边打水，艾拉却摇摇头——她终究不是娜迪娅。",
                     ["E1", "E10", {"eid": "E18", "confusable_with": "E2"}]),
            ],
        },
        {
            "id": "S13", "setting": "风沙灼伤（铜铃长程回忆）",
            "present": ["E1", "E10"],
            "actions": [
                _act("多年后的黄昏，面颊晒伤的艾拉立在绿洲营地棕榈下。",
                     ["E1", "E10"],
                     events=[{"type": "state_change", "eid": "E1", "to": "风沙灼伤"}]),
                "艾拉仰望帐篷顶，目光悠远。",
                _act("帐篷檐下那两只黄铜铃在风里轻响，艾拉驻足倾听。",
                     ["E1", "E10", {"eid": "E17", "count": 2}]),
                "艾拉把手掌贴在帐篷冰凉的桅杆上，似在告别。",
            ],
        },
        {
            "id": "S14", "setting": "娜迪娅归来·艾拉离队",
            "present": ["E1", "E2", "E10"],
            "actions": [
                _act("白日，成年的娜迪娅沿沙道归来，快步走向绿洲营地。",
                     ["E2", "E10"],
                     events=[{"type": "state_change", "eid": "E2", "to": "成年助手"}]),
                "娜迪娅在营地与灼伤的艾拉重逢，两人相拥。",
                _act("帐篷内，娜迪娅握着艾拉满是沙尘的手。", ["E1", "E2", "E11"]),
                _act("艾拉把帆布帽与放大镜留在桌上，离队静养，不再出现。",
                     ["E1", "E2", "E11"],
                     events=[{"type": "remove", "eid": "E1", "to": "离队静养",
                              "reason": "deceased", "shown": True}]),
            ],
        },
        {
            "id": "S15", "setting": "遗物·纪念石板（间接指代）",
            "present": ["E2", "E10", "E12"],
            "actions": [
                _act("帐篷桌上仍放着艾拉的帆布帽，娜迪娅伸手轻轻抚过。", ["E2", "E11"]),
                _act("绿洲边，成年娜迪娅伫立在一块新立的纪念石板前。",
                     ["E2", "E10", "E12"]),
                _act("娜迪娅的指尖抚过纪念石板上刻着的艾拉之名。", ["E2", "E10", "E12"]),
            ],
        },
        {
            "id": "S16", "setting": "哈桑鬓白归来（长程+与劫匪共屏）",
            "present": ["E2", "E6", "E10"],
            "lookalike_present": ha,
            "actions": [
                _act("鬓白的向导哈桑牵着新骆驼归来，沙色斗篷与深蓝头巾一如从前。",
                     ["E6", "E10"],
                     events=[{"type": "state_change", "eid": "E6", "to": "鬓白归来"}]),
                _act("伤疤蒙面劫匪远在沙丘上闪过赭红头巾，随即消失。",
                     ["E7", "E9"],
                     events=[{"type": "state_change", "eid": "E7", "to": "伤疤劫匪"}],
                     lookalike_present=hb),
                "娜迪娅与哈桑在绿洲营地木桌旁相对而坐。",
            ],
        },
        {
            "id": "S17", "setting": "回忆闪回（时序定位）",
            "present": [],
            "actions": [
                _act("镜头回到多年前——那个在石门壁龛前学认油灯的少女，身影再度浮现。",
                     [{"eid": "E2", "state": "少年学徒",
                       "anchor": {"type": "temporal", "phrase": "在石门壁龛前学认油灯的少女",
                                  "resolves_to": "E2"}}]),
                _act("重现开场的那一刻——石门里，一块完好的砖红泥板第一次被捧出。",
                     ["E8", {"eid": "E3", "state": "完好",
                             "anchor": {"type": "temporal",
                                        "phrase": "开场第一次被捧出的完好泥碑",
                                        "resolves_to": "E3"}}]),
                _act("闪回到当年的绿洲——那个缠深蓝头巾、未蒙面的向导身影一闪而过。",
                     [{"eid": "E6", "state": "青年向导",
                       "anchor": {"type": "temporal",
                                  "phrase": "当年绿洲里缠深蓝头巾的向导",
                                  "resolves_to": "E6"}}]),
                _act("眼前又浮现石门前那一排五面红色小旗。",
                     [{"eid": "E16", "count": 5,
                       "anchor": {"type": "temporal",
                                  "phrase": "石门前那一排五面测绘小旗",
                                  "resolves_to": "E16"}}]),
                _act("回到初见时那座半埋的石门洞口，落日拉长阴影，一如往昔。",
                     [{"eid": "E8",
                       "anchor": {"type": "temporal",
                                  "phrase": "初见时那座半埋的石门",
                                  "resolves_to": "E8"}}]),
            ],
        },
        {
            "id": "S18", "setting": "收束（地图长程·油灯长程·拼合碑）",
            "present": ["E2", "E11", "E3"],
            "actions": [
                _act("娜迪娅重新展开那张证伪受潮的伪造羊皮地图，走进发掘帐篷。",
                     ["E2", "E11", {"eid": "E4"}]),
                _act("石门壁龛边，先前那三盏陶油灯依旧摆在原处。",
                     ["E8", {"eid": "E13", "count": 3}]),
                _act("娜迪娅与哈桑在木桌上护住拼合复原的楔形泥碑，落日再度照进石门。",
                     ["E2", "E6", "E8", "E3"], lookalike_present=ha),
            ],
        },
    ]
    return _story_shell(
        "0003_desert_archaeologist",
        "沙海残碑",
        "黄沙落日下的考古长剧。楔形泥碑完好→碎裂→拼合；伪图受潮证伪；驮运骆驼倒毙后须规避；向导哈桑与蒙面劫匪沙色斗篷错认；学徒娜迪娅长时缺席后接班；数量记忆、假朋友与时序闪回贯穿。仅作人读上下文，不喂给 SUT。",
        entities, state_machines, lookalike_pairs, scenes,
    )


def main() -> None:
    from generate_trackb_catalog import CATALOG_IDS, build_catalog_story

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="story_ids to generate")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "0002_night_market_courier": build_0002_hand,
        "0003_desert_archaeologist": build_0003_hand,
    }
    for sid in CATALOG_IDS:
        builders[sid] = (lambda s=sid: build_catalog_story(s))

    targets = a.only or sorted(builders)
    for sid in targets:
        if sid not in builders:
            raise SystemExit(f"unknown story_id: {sid}")
        path = out_dir / f"{sid}.json"
        if a.skip_existing and path.exists() and sid.startswith("0001"):
            continue
        if sid == "0001_lighthouse_keeper":
            print(f"[skip] {sid} (frozen exemplar, not overwritten)")
            continue
        story = builders[sid]()
        path.write_text(json.dumps(story, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        nseg = sum(len(s["actions"]) for s in story["scenes"])
        print(f"[write] {path.name}  scenes={len(story['scenes'])} segs≈{nseg} ents={len(story['entities'])}")


if __name__ == "__main__":
    main()
