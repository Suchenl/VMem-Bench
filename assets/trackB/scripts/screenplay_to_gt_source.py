#!/usr/bin/env python3
"""Convert MemStrata production screenplays into Track B v2 gt_source files.

This is a one-way bootstrap helper for upgrading the old Track B samples. It
keeps the benchmark-facing author layer small: entities, state machines,
lookalike pairs, scenes, per-shot present sets, and semantic events.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # trackB root (scripts/ lives one level down)
REPO = HERE.parents[3]
SCREENPLAY_DIR = REPO / "methods" / "MemStrata" / "production" / "screenplay" / "products" / "cn"
OUT_DIR = HERE / "gt_source"

EID_TAG_RE = re.compile(r"[（(]E\d+[）)]")


def _kind(entity_type: str) -> str:
    if entity_type == "object":
        return "prop"
    return entity_type


def _clean_action(text: str) -> str:
    return EID_TAG_RE.sub("", text).replace("  ", " ").strip()


def _normalize_entity_names(story_id: str, text: str) -> str:
    aliases = {
        "0002_night_market_courier": {
            "木盒": "封蜡木盒",
        },
        "0003_desert_archaeologist": {
            "泥碑": "楔形泥碑",
            "羊皮地图": "伪造羊皮地图",
        },
    }
    for src, dst in aliases.get(story_id, {}).items():
        if dst not in text:
            text = text.replace(src, dst)
    return text


def _state_label(raw: str) -> str:
    labels = {
        "pried": "被撬开",
        "empty": "空盒",
        "shattered": "碎裂",
        "reassembled": "拼合复原",
        "discredited": "证伪",
        "burned": "焚毁",
    }
    return labels.get(raw, raw)


def _initial_label(story_id: str, eid: str) -> str:
    labels = {
        ("0002_night_market_courier", "E3"): "完好封蜡",
        ("0003_desert_archaeologist", "E2"): "完好",
        ("0003_desert_archaeologist", "E3"): "伪图",
    }
    return labels.get((story_id, eid), "初始")


def _state_changes(overrides: dict[str, Any]) -> dict[str, dict[str, str]]:
    by_shot: dict[str, dict[str, str]] = {}
    for item in overrides.get("state_changes", []):
        by_shot.setdefault(item["from_shot"], {})[item["eid"]] = _state_label(item["label"])
    return by_shot


def _state_defs(story_id: str, screenplay: dict[str, Any], overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    changed: dict[str, list[tuple[str, str]]] = {}
    for item in overrides.get("state_changes", []):
        changed.setdefault(item["eid"], []).append((_state_label(item["label"]), item["desc"]))

    entities: dict[str, Any] = {}
    machines: dict[str, list[str]] = {}
    for ent in screenplay["main_entities"]:
        eid = ent["entity_id"]
        base = {"name": ent["name"], "kind": _kind(ent["entity_type"])}
        if eid in changed:
            initial = _initial_label(story_id, eid)
            states = {initial: ent["appearance"]}
            for label, desc in changed[eid]:
                states[label] = desc
            base["states"] = states
            machines[eid] = list(states)
        else:
            base["appearance"] = ent["appearance"]
        entities[eid] = base
    return entities, machines


def _shot_present(shot: dict[str, Any]) -> list[str]:
    present: list[str] = []
    for aid in shot.get("active_characters", []):
        if aid not in present:
            present.append(aid)
    for asset in shot.get("planned_assets", []):
        eid = asset.get("planned_asset_id")
        if not eid or asset.get("operation") == "avoid":
            continue
        if asset.get("required", True) and eid not in present:
            present.append(eid)
    return present


def _lookalike_present(pairs: list[dict[str, Any]], present: list[str]) -> dict[str, Any] | None:
    present_set = set(present)
    for pair in pairs:
        members = [eid for eid in pair["pair"] if eid in present_set]
        if members:
            return {"pair": pair["pair"], "members": members}
    return None


def convert(story_id: str) -> dict[str, Any]:
    screenplay_path = SCREENPLAY_DIR / f"{story_id}.json"
    overrides_path = SCREENPLAY_DIR / f"{story_id}.overrides.json"
    screenplay = json.loads(screenplay_path.read_text(encoding="utf-8"))
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    entities, state_machines = _state_defs(story_id, screenplay, overrides)
    shot_changes = _state_changes(overrides)
    lookalike_pairs = overrides.get("lookalike_pairs", [])

    shots_by_scene: dict[str, list[dict[str, Any]]] = {}
    for shot in screenplay["production_screenplay"]["shots"]:
        shots_by_scene.setdefault(shot["scene_id"], []).append(shot)

    scenes: list[dict[str, Any]] = []
    for scene in screenplay["production_screenplay"]["scenes"]:
        scene_shots = shots_by_scene.get(scene["scene_id"], [])
        actions: list[Any] = []
        for shot in scene_shots:
            action = _normalize_entity_names(
                story_id,
                _clean_action(" ".join(shot.get("visual_track", {}).get("actions", []))),
            )
            present = _shot_present(shot)
            item: dict[str, Any] = {"action": action, "present": present}
            events: list[dict[str, Any]] = []
            for eid, label in shot_changes.get(shot["shot_id"], {}).items():
                event = {"type": "state_change", "eid": eid, "to": label}
                if label == "焚毁":
                    event = {"type": "remove", "eid": eid, "to": label, "reason": "destroyed", "shown": True}
                events.append(event)
            if events:
                item["events"] = events
            lp = _lookalike_present(lookalike_pairs, present)
            if lp:
                item["lookalike_present"] = lp
            actions.append(item)

        scenes.append({
            "id": scene["scene_id"],
            "setting": scene.get("scene_title", scene["scene_id"]),
            "present": scene.get("entities_present", []),
            "actions": actions,
        })

    return {
        "story_id": story_id,
        "title": screenplay.get("story_name", story_id),
        "premise": screenplay.get("story_overview", ""),
        "_comment": "Generated from MemStrata production screenplay plus overrides; hand-audit before freezing.",
        "segment_sec": 5.0,
        "gap_long_threshold": 30,
        "avoidance_probe_window": 4,
        "probe_target_default": 5,
        "entities": entities,
        "state_machines": state_machines,
        "lookalike_pairs": lookalike_pairs,
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stories", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for story_id in args.stories:
        out = args.out_dir / f"{story_id}.json"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        data = convert(story_id)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
