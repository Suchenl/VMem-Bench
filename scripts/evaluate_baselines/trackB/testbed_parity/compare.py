#!/usr/bin/env python3
"""Compare testbed parity runs (run_parity.sh) with the frozen Track-B paper artifacts, shot by shot.

Per shot: generation mode (the paper forced ``recompose_keyframe`` on every shot; the testbed always
composes a keyframe), keyframe type (FLUX text-to-image vs multi-image), the reference ids handed to
FLUX (retrieved frame ids / last-seen source stills), the FLUX seed, and the keyframe sha256 (paper
keyframes are content-addressed: ``objects/sha256/<sha>.png``). For last_seen also the plan bytes.

    compare.py --testbed-root <out_root> --frozen none=<run> --frozen last_seen=<run> \
        --frozen frame_text=<run> [-n 5] [--story 0001_lighthouse_keeper] [--out results.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return p.stem if "/objects/sha256/" in str(path) and len(p.stem) == 64 else None


def frozen_view(run: Path, n: int) -> list[dict]:
    prog = {int(s["segment_id"]): s for s in json.loads((run / "progress.json").read_text())["segments"]}
    out = []
    for i in range(n):
        s = prog.get(i, {})
        task = run / "generator_logs" / f"segment_{i:03d}" / "media_generation_task.json"
        ctl = json.loads(task.read_text()).get("controls", {}) if task.is_file() else {}
        refs = [str(c.get("asset_id", "")).removeprefix("retr:") for c in ctl.get("source_crops") or []]
        out.append({"shot": i, "mode": s.get("used_mode"), "keyframe_mode": s.get("fused"), "ref_ids": refs,
                    "flux_seed": 2026 + i, "keyframe_sha": _sha(s.get("keyframe")),
                    "clip_sha": _sha(s.get("video"))})
    return out


def testbed_view(run: Path, n: int) -> list[dict]:
    prog = {int(s["segment_id"]): s for s in json.loads((run / "progress.json").read_text())["segments"]}
    out = []
    for i in range(n):
        s = prog.get(i, {})
        meta = s.get("backend_meta") or {}
        out.append({"shot": i, "mode": "recompose_keyframe" if s.get("status") == "ok" else None,
                    "keyframe_mode": s.get("keyframe_mode"), "ref_ids": [r["id"] for r in s.get("references") or []],
                    "flux_seed": meta.get("flux_seed"), "keyframe_sha": _sha(s.get("keyframe")),
                    "clip_sha": _sha(s.get("video")),
                    "timing_s": s.get("timing_s")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testbed-root", type=Path, required=True)
    ap.add_argument("--frozen", action="append", default=[], metavar="VARIANT=RUN_DIR")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--story", default="0001_lighthouse_keeper")
    ap.add_argument("--run-tag", default="parity_testbed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = {}
    for item in args.frozen:
        variant, _, frozen = item.partition("=")
        budget = 0 if variant == "none" else 3
        tb = args.testbed_root / f"{variant}_b{budget}" / args.story / "name_anchored" / args.run_tag / "seed_2026"
        if not (tb / "progress.json").is_file():
            res[variant] = {"status": "testbed run missing", "testbed": str(tb)}
            continue
        a, b = testbed_view(tb, args.n), frozen_view(Path(frozen), args.n)
        rows = []
        for x, y in zip(a, b):
            same = {k: x[k] == y[k] for k in ("mode", "keyframe_mode", "ref_ids", "flux_seed", "keyframe_sha",
                                             "clip_sha")}
            rows.append({"shot": x["shot"], "testbed": x, "frozen": y, "same": same})
        entry = {
            "testbed": str(tb), "frozen": frozen, "shots": rows,
            "decisions_equal": all(r["same"]["mode"] and r["same"]["keyframe_mode"] and r["same"]["flux_seed"]
                                   for r in rows),
            "ref_ids_equal_shots": [r["shot"] for r in rows if r["same"]["ref_ids"]],
            "t2i_keyframes_byte_identical": [r["shot"] for r in rows
                                            if r["frozen"]["keyframe_mode"] == "flux_t2i" and r["same"]["keyframe_sha"]],
            "t2i_shots": [r["shot"] for r in rows if r["frozen"]["keyframe_mode"] == "flux_t2i"],
        }
        if variant == "last_seen":
            pa, pb = tb / "input" / "last_seen_plan.json", Path(frozen) / "input" / "oracle_reference_plan.json"
            entry["plan_identical"] = pa.read_bytes() == pb.read_bytes()
        res[variant] = entry
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    for v, r in res.items():
        print(v, {k: r.get(k) for k in ("decisions_equal", "ref_ids_equal_shots", "t2i_shots",
                                        "t2i_keyframes_byte_identical", "plan_identical", "status")})
        for row in r.get("shots", []):
            diff = [k for k, ok in row["same"].items() if not ok]
            print("   shot", row["shot"], "differs:" if diff else "equal", ",".join(diff),
                  "| testbed", row["testbed"]["keyframe_mode"], row["testbed"]["ref_ids"],
                  "| frozen", row["frozen"]["keyframe_mode"], row["frozen"]["ref_ids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
