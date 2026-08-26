#!/usr/bin/env python3
"""Audit Track B story quality beyond schema validity.

This is a lightweight gate for draft assets. It does not replace human review,
but catches the common failure mode of template-filled stories that pass the
GT builder while lacking concrete visual detail.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # trackB root (scripts/ lives one level down)
GT_DIR = HERE / "gt"
SUT_DIR = HERE / "sut_prompts"

PROBE_MIN = {
    "lookalike_disambiguation": 5,
    "state_change": 5,
    "persist_state": 5,
    "count_memory": 5,
    "false_friend": 5,
    "deprecation_avoidance": 5,
    "reference_indirect": 5,
    "long_gap_reappearance": 5,
    "temporal_reference": 5,
}

TEMPLATE_PATTERNS = [
    r"一辆/一件",
    r"一本/一卷",
    r"与[^，。]{0,20}相符的",
    r"开始例行检查",
    r"主场日常",
    r"主场中央静静",
    r"主场角落坐下小憩",
    r"核心器物",
    r"信物",
    r"载具",
    r"驿馆檐下",
    r"驿馆梁下",
    r"驼峰",
    r"月牙烙印",
    r"青铜狮钮印",
    r"狮鬃",
    r"沙窝",
    r"鼻铃",
    r"重新重新",
    r"过关章程",
    r"认真记下当日状况",
    r"关键记号",
    r"一件[^，。]{0,20}细节不同",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def story_ids(paths: list[str] | None) -> list[str]:
    if paths:
        return [Path(p).stem.replace("_name_anchored", "") for p in paths]
    return [p.stem for p in sorted(GT_DIR.glob("*.json"))]


def audit_one(story_id: str) -> list[str]:
    issues: list[str] = []
    gt_path = GT_DIR / f"{story_id}.json"
    sut_path = SUT_DIR / f"{story_id}_name_anchored.json"
    if not gt_path.exists():
        return [f"{story_id}: missing gt"]
    gt = load_json(gt_path)
    nseg = gt.get("summary", {}).get("n_segments", len(gt.get("segments", [])))
    if not 50 <= nseg <= 200:
        issues.append(f"{story_id}: n_segments={nseg} outside [50,200]")
    probes = gt.get("summary", {}).get("probe_counts", {})
    for key, minimum in PROBE_MIN.items():
        if probes.get(key, 0) < minimum:
            issues.append(f"{story_id}: probe {key}={probes.get(key, 0)} < {minimum}")
    if gt.get("warnings"):
        issues.append(f"{story_id}: gt warnings={len(gt['warnings'])}")
    if sut_path.exists():
        sut = load_json(sut_path)
        if sut.get("warnings"):
            issues.append(f"{story_id}: sut warnings={len(sut['warnings'])}")
    else:
        issues.append(f"{story_id}: missing sut prompt")

    text = json.dumps(gt, ensure_ascii=False)
    hits = sum(len(re.findall(pat, text)) for pat in TEMPLATE_PATTERNS)
    if story_id not in {"0001_lighthouse_keeper", "0002_night_market_courier", "0003_desert_archaeologist"}:
        if hits > 8:
            issues.append(f"{story_id}: template-pattern hits={hits} > 8")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stories", nargs="*", help="story IDs or filenames; default: all")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on any issue")
    args = parser.parse_args()

    ids = story_ids(args.stories)
    all_issues: list[str] = []
    segs: list[int] = []
    for sid in ids:
        gt_path = GT_DIR / f"{sid}.json"
        if gt_path.exists():
            gt = load_json(gt_path)
            segs.append(gt.get("summary", {}).get("n_segments", len(gt.get("segments", []))))
        all_issues.extend(audit_one(sid))

    if segs:
        buckets = {
            "50-80": sum(50 <= x <= 80 for x in segs),
            "81-120": sum(81 <= x <= 120 for x in segs),
            "121-160": sum(121 <= x <= 160 for x in segs),
            "161-200": sum(161 <= x <= 200 for x in segs),
        }
        print(f"stories={len(ids)} segments min={min(segs)} max={max(segs)} mean={mean(segs):.1f} buckets={buckets}")
    if all_issues:
        print("ISSUES:")
        for issue in all_issues:
            print(f"- {issue}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("OK")


if __name__ == "__main__":
    main()
