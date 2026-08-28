#!/usr/bin/env python3
"""Package a TrackB long video with per-segment memory annotations.

For every segment of a generated ``long_video.mp4`` this overlays:
  * a black caption bar under the video with the segment prompt, where the
    entities that must be remembered are highlighted, plus an explicit
    "memory anchors" line (recall gaps, counts, state changes, avoidance);
  * a translucent tag box in the top-right corner listing the hard cases
    (memory probes) that this segment stress-tests.

The annotations switch at every segment boundary. Output is meant for paper
figures and the project page.

Segment metadata comes from the frozen TrackB ground truth
(``assets/trackB/en/gt/<story>.json``) and the SUT prompt stream
(``assets/trackB/en/sut_prompts/<story>_<register>.json``). Segment->frame
mapping comes from ``input/trackb_generation_params.json`` (frames_per_segment).

Runs under a Python that has cv2 + PIL:
  python3 annotate_long_video.py \
      --run-dir <trackB run dir> --tag-lang en

or standalone:
  ... annotate_long_video.py --video long_video.mp4 --story 0003_desert_archaeologist \
      --frames-per-segment 39 [--register name_anchored]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

BENCH_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = BENCH_ROOT / "assets" / "trackB"
FONT_DIR = BENCH_ROOT / "assets" / "_fonts"
# Track B assets are split into parallel zh/ and en/ trees (gt + sut_prompts).
GT_DIR = {"zh": ASSET_ROOT / "zh" / "gt", "en": ASSET_ROOT / "en" / "gt"}
SUT_DIR = {"zh": ASSET_ROOT / "zh" / "sut_prompts", "en": ASSET_ROOT / "en" / "sut_prompts"}
FONT_REGULAR = FONT_DIR / "NotoSansCJKsc-Regular.otf"
FONT_BOLD = FONT_DIR / "NotoSansCJKsc-Bold.otf"

# --- palette (RGB) -------------------------------------------------------
CLR_BAR_BG = (14, 14, 16)
CLR_PROMPT = (232, 232, 236)
CLR_ENTITY = (255, 196, 74)      # amber: entities to remember
CLR_SEGLBL = (150, 205, 255)     # soft blue: segment id
CLR_ANCHOR_LABEL = (150, 156, 168)
CLR_DIVIDER = (60, 62, 70)

# hard-case probes -> (english label, chinese label, tag color RGB).
# continuity / first_appearance are trivial and intentionally omitted.
PROBE_LABELS = {
    "long_gap_reappearance": ("Long-gap recall", "长时重现", (255, 128, 128)),
    "lookalike_disambiguation": ("Look-alike", "相似体区分", (255, 170, 90)),
    "false_friend": ("False friend", "假朋友", (255, 210, 92)),
    "state_change": ("State change", "状态切换", (140, 220, 160)),
    "persist_state": ("State persist", "状态保持", (120, 200, 220)),
    "count_memory": ("Count memory", "数量记忆", (170, 190, 255)),
    "deprecation_avoidance": ("Avoidance", "废弃规避", (220, 150, 240)),
    "reference_indirect": ("Indirect ref", "间接指代", (200, 200, 210)),
    "temporal_reference": ("Temporal ref", "时序指代", (250, 190, 150)),
}
PROBE_ORDER = list(PROBE_LABELS.keys())


# ---------------------------------------------------------------------------
# metadata loading
# ---------------------------------------------------------------------------
def load_gt(story: str, lang: str = "zh") -> dict:
    return json.loads((GT_DIR[lang] / f"{story}.json").read_text(encoding="utf-8"))


def load_prompts(story: str, register: str, lang: str = "zh") -> dict:
    return json.loads(
        (SUT_DIR[lang] / f"{story}_{register}.json").read_text(encoding="utf-8")
    )


PAREN_RE = re.compile(r"（[^（）]*）")


def clean_prompt(text: str) -> str:
    """Drop parenthetical entity descriptions so the caption stays readable."""
    prev = None
    while prev != text:
        prev = text
        text = PAREN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def match_term(name: str, prompt: str) -> str | None:
    """Longest contiguous substring (len>=2) of an entity name present in the
    prompt. Handles role-prefixed names (向导哈桑 -> 哈桑) and full names alike."""
    n = len(name)
    for length in range(n, 1, -1):
        for s in range(0, n - length + 1):
            sub = name[s : s + length]
            if sub in prompt:
                return sub
    return None


def _name_map(gt: dict | None) -> dict:
    if not gt:
        return {}
    return {eid: e.get("name", eid) for eid, e in gt.get("entities", {}).items()}


def _prompt_by_sid(prompts: dict | None) -> dict:
    if not prompts:
        return {}
    return {p["segment_id"]: p.get("prompt", "") for p in prompts.get("segments", [])}


def build_segment_meta(gt: dict, prompts_zh: dict, n_segments: int, tag_lang: str,
                       gt_en: dict | None = None, prompts_en: dict | None = None) -> list[dict]:
    """Structure (cast/probes/gaps) comes from the language-independent zh GT;
    display text is carried for both languages."""
    long_gap = int(gt.get("params", {}).get("gap_long_threshold", 10))
    gt_segs = {s["segment_id"]: s for s in gt["segments"]}
    names = {"zh": _name_map(gt), "en": _name_map(gt_en)}
    pz, pe = _prompt_by_sid(prompts_zh), _prompt_by_sid(prompts_en)
    order = [s["segment_id"] for s in prompts_zh["segments"][:n_segments]]
    out: list[dict] = []
    for i, sid in enumerate(order):
        g = gt_segs.get(sid, {})
        prompt = {"zh": clean_prompt(pz.get(sid, g.get("action", ""))),
                  "en": clean_prompt(pe[sid]) if sid in pe else None}
        anchors = []  # (kind, eid, suffix)
        for c in g.get("cast", []):
            eid, op, gap = c["eid"], c.get("op"), c.get("gap")
            if c.get("count"):
                anchors.append(("count", eid, f"×{c['count']}"))
            elif op == "transform":
                anchors.append(("state", eid, "→"))
            elif isinstance(gap, int) and gap >= long_gap:
                anchors.append(("gap", eid, f"↺{gap}"))
            elif op == "recall_after_gap":
                anchors.append(("gap", eid, "↺"))
        for f in g.get("forbidden", []):
            anchors.append(("avoid", f["eid"], "⊘"))
        tags = []
        for p in PROBE_ORDER:
            if p in g.get("memory_probes", []):
                en, zh, clr = PROBE_LABELS[p]
                label = {"en": en, "zh": zh, "both": f"{en} · {zh}"}[tag_lang]
                tags.append((label, clr))
        hl = {}
        for lang in ("zh", "en"):
            txt = prompt[lang]
            hl[lang] = sorted(
                {t for eid in {c["eid"] for c in g.get("cast", [])}
                 if (nm := names[lang].get(eid)) and txt and (t := match_term(nm, txt))},
                key=len, reverse=True) if txt else []
        out.append({
            "seg_idx": i, "seg_id": sid, "prompt": prompt, "hl": hl,
            "anchors": anchors, "names": names, "tags": tags,
        })
    return out


# ---------------------------------------------------------------------------
# text layout helpers (PIL)
# ---------------------------------------------------------------------------
def tokenize_highlight(text: str, entities: list[str]) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_entity) runs by matching entity names."""
    if not entities:
        return [(text, False)]
    pattern = re.compile("|".join(re.escape(e) for e in entities if e))
    runs: list[tuple[str, bool]] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            runs.append((text[pos : m.start()], False))
        runs.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], False))
    return runs


def layout_runs(runs, font, max_w):
    """Greedy character-level wrap of highlighted runs. Returns list of lines;
    each line is a list of (char, is_entity)."""
    lines: list[list[tuple[str, bool]]] = [[]]
    w = 0
    for chunk, is_ent in runs:
        for ch in chunk:
            if ch == "\n":
                lines.append([])
                w = 0
                continue
            cw = font.getlength(ch)
            if w + cw > max_w and lines[-1]:
                lines.append([])
                w = 0
            lines[-1].append((ch, is_ent))
            w += cw
    return lines


def draw_lines(draw, lines, x, y, font, lh, normal_clr, ent_clr):
    for line in lines:
        cx = x
        for ch, is_ent in line:
            draw.text((cx, y), ch, font=font, fill=ent_clr if is_ent else normal_clr)
            cx += font.getlength(ch)
        y += lh
    return y


# ---------------------------------------------------------------------------
# per-segment overlay rendering (done once per segment)
# ---------------------------------------------------------------------------
CHIP_CLR = {"gap": (255, 150, 150), "count": (170, 190, 255),
            "state": (150, 220, 170), "avoid": (225, 150, 240)}


def bar_height(caption_lang: str, scale: float) -> int:
    return int((150 if caption_lang != "both" else 220) * scale)


def _draw_prompt(d, text, hl_terms, x, y, font, max_w, lh, ent_clr):
    runs = tokenize_highlight(text, hl_terms)
    lines = layout_runs(runs, font, max_w)[:2]
    return draw_lines(d, lines, x, y, font, lh, CLR_PROMPT, ent_clr)


def render_caption_bar(meta, width, bar_h, scale, caption_lang):
    """Bottom caption bar as an (bar_h, width, 3) uint8 RGB array.

    caption_lang: 'zh' | 'en' | 'both' (en primary on top, zh secondary below).
    """
    img = Image.new("RGB", (width, bar_h), CLR_BAR_BG)
    d = ImageDraw.Draw(img)
    pad = int(28 * scale)
    max_w = width - 2 * pad
    f_lbl = ImageFont.truetype(str(FONT_BOLD), int(26 * scale))
    f_small = ImageFont.truetype(str(FONT_REGULAR), int(24 * scale))

    # which languages to show, dropping any without text
    want = ["en", "zh"] if caption_lang == "both" else [caption_lang]
    langs = [l for l in want if meta["prompt"].get(l)]
    if not langs:  # requested lang missing -> fall back to whatever exists
        langs = [l for l in ("en", "zh") if meta["prompt"].get(l)][:1]

    y = int(14 * scale)
    d.text((pad, y), f"SEG {meta['seg_idx'] + 1:02d}", font=f_lbl, fill=CLR_SEGLBL)
    y += int(34 * scale)

    for idx, lang in enumerate(langs):
        primary = idx == 0
        fsize = int((30 if primary else 24) * scale)
        f = ImageFont.truetype(str(FONT_REGULAR), fsize)
        ent_clr = CLR_ENTITY if primary else tuple(int(c * 0.8) for c in CLR_ENTITY)
        y = _draw_prompt(d, meta["prompt"][lang], meta["hl"][lang], pad, y, f,
                         max_w, int((fsize + 8)), ent_clr)
        y += int(4 * scale)

    if meta["anchors"]:
        nm = meta["names"]["en" if (langs and langs[0] == "en") else "zh"]
        y += int(2 * scale)
        d.line([(pad, y), (width - pad, y)], fill=CLR_DIVIDER, width=max(1, int(scale)))
        y += int(10 * scale)
        d.text((pad, y), "memory", font=f_small, fill=CLR_ANCHOR_LABEL)
        cx = pad + f_small.getlength("memory") + int(18 * scale)
        for kind, eid, suffix in meta["anchors"]:
            name = nm.get(eid, eid)
            txt = f"{suffix}{name}" if kind == "avoid" else (
                f"{name}{suffix}new" if kind == "state" else f"{name}{suffix}")
            cw = f_small.getlength(txt)
            if cx + cw > width - pad:
                break
            d.text((cx, y), txt, font=f_small, fill=CHIP_CLR.get(kind, CLR_PROMPT))
            cx += cw + int(22 * scale)
    return np.asarray(img)


def render_tag_box(meta, vid_w, scale):
    """Top-right hard-case box as (rgba array, x, y) or None."""
    if not meta["tags"]:
        return None
    f = ImageFont.truetype(str(FONT_BOLD), int(24 * scale))
    pad = int(14 * scale)
    gap = int(9 * scale)
    dot = int(11 * scale)
    lh = int(34 * scale)
    tw = max(f.getlength(t) for t, _ in meta["tags"])
    box_w = int(dot + 10 * scale + tw + 2 * pad)
    box_h = int(len(meta["tags"]) * lh + 2 * pad - (lh - int(28 * scale)))
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=int(10 * scale), fill=(12, 12, 16, 190))
    y = pad
    for label, clr in meta["tags"]:
        cy = y + int(14 * scale)
        d.ellipse([pad, cy - dot // 2, pad + dot, cy + dot // 2], fill=clr + (255,))
        d.text((pad + dot + int(10 * scale), y), label, font=f, fill=(240, 240, 244, 255))
        y += lh
    margin = int(18 * scale)
    x = vid_w - box_w - margin
    return np.asarray(img), x, margin


def alpha_paste(dst, rgba, x, y):
    h, w = rgba.shape[:2]
    H, W = dst.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        w = min(w, W - x)
        h = min(h, H - y)
        rgba = rgba[:h, :w]
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    dst[y : y + h, x : x + w] = (
        rgba[:, :, :3].astype(np.float32) * a + dst[y : y + h, x : x + w].astype(np.float32) * (1 - a)
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# main annotate
# ---------------------------------------------------------------------------
def annotate(video: Path, segs: list[dict], out: Path, fps: float, caption_lang: str):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = max(0.7, w / 960.0)
    bar_h = bar_height(caption_lang, scale)
    n_seg = len(segs)
    # decoded RGB frames are evenly split across the generated segments
    # (each segment has the same latent length, VAE-decoded at the same ratio).
    per_seg = max(1.0, total / n_seg)

    # pre-render overlays once per segment (RGB caption bar + RGBA tag box)
    bars = [render_caption_bar(m, w, bar_h, scale, caption_lang) for m in segs]
    tags = [render_tag_box(m, w, scale) for m in segs]

    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h + bar_h))
    fi = 0
    while True:
        ok, frame = cap.read()  # BGR
        if not ok:
            break
        seg = min(int(fi / per_seg), n_seg - 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if tags[seg] is not None:
            rgba, tx, ty = tags[seg]
            alpha_paste(rgb, rgba, tx, ty)
        canvas = np.vstack([rgb, bars[seg]])
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        fi += 1
    cap.release()
    writer.release()
    return {"frames": fi, "segments": n_seg, "size": [w, h + bar_h], "out": str(out)}


def resolve_from_run_dir(run_dir: Path):
    manifest = json.loads((run_dir / "trackb_manifest.json").read_text(encoding="utf-8"))
    story = manifest["story_id"]
    register = manifest.get("register", "name_anchored")
    n_segments = int(manifest["n_segments"])
    review = run_dir / "review"
    cands = sorted(review.glob("long_video.mp4")) or sorted(
        review.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not cands:
        raise SystemExit(f"no mp4 under {review}")
    return story, register, n_segments, cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="TrackB run dir (auto-resolves video + params)")
    ap.add_argument("--video", type=Path, help="explicit long_video.mp4 (overrides run-dir)")
    ap.add_argument("--story", help="story id, e.g. 0003_desert_archaeologist")
    ap.add_argument("--register", default="name_anchored")
    ap.add_argument("--n-segments", type=int, help="number of generated segments in the video")
    ap.add_argument("--caption-lang", choices=["zh", "en", "both"], default="zh",
                    help="bottom caption language (both = en primary + zh secondary)")
    ap.add_argument("--tag-lang", choices=["en", "zh", "both"], default="en")
    ap.add_argument("--fps", type=float, default=0.0, help="output fps (0 = read from input, fallback 16)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    story = args.story
    register = args.register
    n_segments = args.n_segments
    video = args.video
    if args.run_dir:
        s, r, n, v = resolve_from_run_dir(args.run_dir)
        story = story or s
        register = args.register if args.register != "name_anchored" else r
        n_segments = n_segments or n
        video = video or v
    if not (story and n_segments and video):
        raise SystemExit("need --run-dir OR (--video --story --n-segments)")

    cap = cv2.VideoCapture(str(video))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = args.fps or (in_fps if in_fps and in_fps > 0 else 16.0)

    gt = load_gt(story, "zh")
    prompts_zh = load_prompts(story, register, "zh")
    gt_en = prompts_en = None
    if args.caption_lang in ("en", "both"):
        try:
            gt_en = load_gt(story, "en")
            prompts_en = load_prompts(story, register, "en")
        except FileNotFoundError:
            print(f"[warn] en assets for {story} not found; falling back to zh captions")
    segs = build_segment_meta(gt, prompts_zh, n_segments, args.tag_lang,
                              gt_en=gt_en, prompts_en=prompts_en)

    out = args.out or (video.parent / f"{video.stem}_annotated.mp4")
    info = annotate(video, segs, out, fps, args.caption_lang)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
