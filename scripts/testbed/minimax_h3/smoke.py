#!/usr/bin/env python
"""GPU smoke for the MiniMax-H3 Turbo backend: reference fidelity, zero-reference, seed determinism, timing.

    python scripts/testbed/minimax_h3/smoke.py --endpoints <endpoint_dir|urls> \
        --prompts assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
        --ref-dir <run>/pipeline/retrieval_read/oracle_frames --out <dir>

Each ``--case`` is ``name:segment_index:ref_shots:seed[:geometry[:ref_short_edge]]``; ``ref_shots`` is ``-``
or ``+``-joined shot ids whose ``c%05d_mid.png`` stills in ``--ref-dir`` become the references, and
``geometry`` is a preset name (default: the backend default). ``--cases-json`` instead takes a list of
``{"name", "prompts", "segment", "references": [paths], "seed", "geometry"?, "ref_short_edge"?}`` so cases
can mix stories and arbitrary reference stills. ``--pair a,b`` adds a pixel comparison of two cases
(``render_sha_equal`` tells bitwise reproducibility). Writes ``<out>/<name>/seg_*.mp4``, a contact sheet
per case (references | frames 0/30/60/90/120), and ``<out>/smoke_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from vmem_bench.testbed.backends.base import RealizationRequest  # noqa: E402
from vmem_bench.testbed.backends.minimax_h3 import MiniMaxH3TurboBackend  # noqa: E402
from vmem_bench.testbed.backends.minimax_h3.geometry import DEFAULT_GEOMETRY  # noqa: E402

DEFAULT_CASES = [
    "t2v_s0_a:0:-:2026",
    "t2v_s0_b:0:-:2026",
    "t2v_s0_seed2027:0:-:2027",
    "ref1_s1:1:0:2026",
    "ref2_s3:3:0+2:2026",
    "ref2_s3_rerun:3:0+2:2026",
    "ref3_s25:25:9+10+15:2026",
    "t2v_s3:3:-:2026",
    "t2v_s25:25:-:2026",
    "ref2_s3_704p:3:0+2:2026:704x1280",
]
DEFAULT_PAIRS = ["t2v_s0_a,t2v_s0_b", "ref2_s3,ref2_s3_rerun", "t2v_s0_a,t2v_s0_seed2027",
                 "ref2_s3,t2v_s3", "ref3_s25,t2v_s25"]
SHEET_FRAMES = (0, 30, 60, 90, 120)


def read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(f) for f in reader]
    finally:
        reader.close()


def contact_sheet(refs: list[Path], frames: list[np.ndarray], out: Path, cell_h: int = 240, title: str = "") -> None:
    from PIL import Image, ImageDraw

    def fit(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.resize((max(1, round(w * cell_h / h)), cell_h), Image.Resampling.LANCZOS)

    cells = [("ref%d" % (i + 1), fit(Image.open(p).convert("RGB"))) for i, p in enumerate(refs)]
    cells += [(f"f{k}", fit(Image.fromarray(frames[k]))) for k in SHEET_FRAMES if k < len(frames)]
    top = 36 if title else 20
    sheet = Image.new("RGB", (sum(c.width for _, c in cells) + 8 * len(cells), cell_h + top), "white")
    x = 0
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 2), title, fill="black")
    for label, c in cells:
        sheet.paste(c, (x, top))
        draw.text((x + 4, top - 16), label, fill="black")
        x += c.width + 8
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def frame_stats(refs: list[Path], frames: list[np.ndarray]) -> dict:
    """Cheap health signals: darkest frame mean, and how close the last frame is to any reference."""
    from PIL import Image

    means = [float(f.mean()) for f in frames]
    out = {"min_frame_mean": round(min(means), 2), "n_dark_frames": sum(m < 8 for m in means)}
    if refs:
        h, w = frames[-1].shape[:2]
        last, first = frames[-1].astype(np.int16), frames[0].astype(np.int16)
        mads_last, mads_first = [], []
        for p in refs:
            r = np.asarray(Image.open(p).convert("RGB").resize((w, h), Image.Resampling.BILINEAR), dtype=np.int16)
            mads_last.append(float(np.abs(last - r).mean()))
            mads_first.append(float(np.abs(first - r).mean()))
        out["last_vs_ref_min_mad"] = round(min(mads_last), 2)
        out["first_vs_ref_min_mad"] = round(min(mads_first), 2)
    return out


def load_cases(args) -> list[dict]:
    if args.cases_json:
        return json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
    cases = []
    for spec in args.case or DEFAULT_CASES:
        parts = spec.split(":")
        shots = parts[2]
        cases.append({
            "name": parts[0], "prompts": args.prompts, "segment": int(parts[1]), "seed": int(parts[3]),
            "references": [] if shots == "-" else [str(Path(args.ref_dir) / f"c{int(s):05d}_mid.png")
                                                   for s in shots.split("+")],
            "geometry": parts[4] if len(parts) > 4 and parts[4] else None,
            "ref_short_edge": int(parts[5]) if len(parts) > 5 else None,
        })
    return cases


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoints", required=True)
    p.add_argument("--prompts")
    p.add_argument("--ref-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--case", action="append", default=None)
    p.add_argument("--cases-json")
    p.add_argument("--pair", action="append", default=None, help="name_a,name_b pixel comparison")
    p.add_argument("--no-cache-check", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    backends: dict[tuple[str, int], MiniMaxH3TurboBackend] = {}
    prompts: dict[str, list] = {}
    rows, clips = [], {}
    for case in load_cases(args):
        name, seg, seed = case["name"], int(case["segment"]), int(case["seed"])
        geometry, edge = case.get("geometry") or DEFAULT_GEOMETRY, int(case.get("ref_short_edge") or 768)
        be = backends.get((geometry, edge))
        if be is None:
            be = backends[(geometry, edge)] = MiniMaxH3TurboBackend(args.endpoints, geometry=geometry,
                                                                    ref_image_short_edge=edge)
            be.start()
        if case["prompts"] not in prompts:
            prompts[case["prompts"]] = json.loads(Path(case["prompts"]).read_text(encoding="utf-8"))["segments"]
        refs = tuple(Path(r) for r in case["references"])
        req = RealizationRequest(segment_id=seg, prompt=prompts[case["prompts"]][seg]["prompt"], references=refs,
                                 seed=seed, out_dir=out_root / name)
        t0 = time.time()
        res = be.realize(req)
        wall = time.time() - t0
        frames = read_frames(res.clip)
        clips[name] = frames
        sheet = out_root / "sheets" / f"{name}.png"
        contact_sheet(list(refs), frames, sheet, title=f"{name} | {Path(case['prompts']).stem[:28]} seg {seg} | "
                                                      f"{len(refs)} refs | seed {seed} | {geometry}")
        (out_root / name / "prompt.txt").write_text(res.meta["prompt"], encoding="utf-8")
        row = {k: res.meta.get(k) for k in ("elapsed_sec", "generate_sec", "queue_wait_sec", "peak_vram_gib",
                                            "reserved_vram_gib", "peak_reserved_vram_gib", "rendered_frames",
                                            "delivered_frames", "delivered_size", "render_frames_sha256",
                                            "cached", "job_index", "geometry", "vram", "host")}
        row.update(name=name, story=Path(case["prompts"]).stem, segment=seg, n_refs=len(refs), seed=seed,
                   ref_short_edge=edge, wall_sec=round(wall, 2), clip=str(res.clip), sheet=str(sheet),
                   **frame_stats(list(refs), frames))
        rows.append(row)
        print(json.dumps(row), flush=True)

    def diff(a: str, b: str) -> dict:
        if a not in clips or b not in clips:
            return {}
        x, y = np.stack(clips[a]).astype(np.int16), np.stack(clips[b]).astype(np.int16)
        d = np.abs(x - y)
        by = {r["name"]: r for r in rows}
        return {"pair": [a, b], "mean_abs_diff": round(float(d.mean()), 4), "max_abs_diff": int(d.max()),
                "identical_pixels_frac": round(float((d == 0).mean()), 4),
                "render_sha_equal": by[a]["render_frames_sha256"] == by[b]["render_frames_sha256"]}

    pairs = args.pair if args.pair is not None else ([] if args.cases_json else DEFAULT_PAIRS)
    summary = {"cases": rows, "determinism": [d for d in (diff(*s.split(",")) for s in pairs) if d]}
    if rows and not args.no_cache_check:
        # Same request, same out -> server cache.
        first = load_cases(args)[0]
        be = backends[(first.get("geometry") or DEFAULT_GEOMETRY, int(first.get("ref_short_edge") or 768))]
        req = RealizationRequest(segment_id=int(first["segment"]),
                                 prompt=prompts[first["prompts"]][int(first["segment"])]["prompt"],
                                 references=tuple(Path(r) for r in first["references"]), seed=int(first["seed"]),
                                 out_dir=out_root / first["name"])
        t0 = time.time()
        summary["cache_hit"] = {"cached": be.realize(req).meta.get("cached"), "wall_sec": round(time.time() - t0, 3)}
    (out_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["determinism"], indent=1))
    print("cache:", summary.get("cache_hit"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
