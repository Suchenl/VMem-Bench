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
(``assets/trackB/gt/<story>.json``) and the SUT prompt stream
(``assets/trackB/sut_prompts/<story>_<register>.json``). Segment->frame
mapping comes from ``input/trackb_generation_params.json`` (frames_per_segment).

Runs under the vace env (has cv2 + PIL):
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
    # generic recall: any entity brought back after being away (not a special
    # hard case, but still exercises earlier memory).
    "recall": ("Recall", "记忆回调", (150, 200, 255)),
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


CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LAT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9''\-]*")
STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "on", "at", "by", "with", "for",
    "from", "up", "its", "his", "her", "their", "two", "three", "one", "that",
    "this", "these", "those", "is", "are", "as", "or", "s",
}


def _is_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s))


def _match_cjk(name: str, prompt: str) -> str | None:
    """Longest contiguous substring (len>=2) of a CJK entity name present in the
    prompt. Handles role-prefixed names (向导哈桑 -> 哈桑) and full names alike."""
    n = len(name)
    for length in range(n, 1, -1):
        for s in range(0, n - length + 1):
            sub = name[s : s + length]
            if sub in prompt:
                return sub
    return None


def highlight_terms(name: str, prompt: str) -> list[str]:
    """Surface strings inside ``prompt`` to highlight for entity ``name``.

    CJK: single longest contiguous substring (word boundaries do not apply).
    Latin: only *whole-word* matches — the full phrase if present, else the
    longest contiguous word n-gram, else significant single words — so we never
    colour a mid-word fragment like the ``inting`` inside "painting"."""
    if not name or not prompt:
        return []
    if _is_cjk(name):
        t = _match_cjk(name, prompt)
        return [t] if t else []
    full = re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", re.I)
    m = full.search(prompt)
    if m:
        return [m.group(0)]
    words = _LAT_WORD_RE.findall(name)
    for size in range(len(words), 0, -1):
        hits: list[str] = []
        for start in range(0, len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            if len(phrase) < 3:
                continue
            pm = re.compile(r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])",
                            re.I).search(prompt)
            if pm:
                hits.append(pm.group(0))
        if hits:
            return list(dict.fromkeys(hits))
    hits = []
    for wd in words:
        if len(wd) >= 4 and wd.lower() not in STOPWORDS:
            wm = re.compile(r"(?<![A-Za-z])" + re.escape(wd) + r"(?![A-Za-z])",
                            re.I).search(prompt)
            if wm:
                hits.append(wm.group(0))
    return list(dict.fromkeys(hits))


def _name_map(gt: dict | None) -> dict:
    if not gt:
        return {}
    return {eid: e.get("name", eid) for eid, e in gt.get("entities", {}).items()}


def _prompt_by_sid(prompts: dict | None) -> dict:
    if not prompts:
        return {}
    return {p["segment_id"]: p.get("prompt", "") for p in prompts.get("segments", [])}


# ---------------------------------------------------------------------------
# guided mode: per-segment Test/Hard explanation + first-appearance anchor,
# auto-derived from the GT so the reader sees what memory is under test and how
# far back it was established (mirrors the main-paper fig:trackb-money panel).
# ---------------------------------------------------------------------------
GUIDED_PRIORITY = [
    "count_memory", "long_gap_reappearance", "state_change",
    "lookalike_disambiguation", "deprecation_avoidance", "reference_indirect",
]


_RECALL_OPS = {"recall", "recall_after_gap", "transform", "persist", "reappear"}


def _guided_meta(g: dict, seg_idx: int, first_seen: dict, name_of) -> dict | None:
    """Build a grounded Test/Hard explanation for this segment.

    A specific hard-case probe (count / long-gap / state-change / look-alike /
    avoidance / indirect) is preferred. Otherwise we still annotate ANY segment
    that recalls an earlier entity (an entity brought back after being away
    >=1 segment) as a generic ``recall`` — so every segment that exercises prior
    memory is labelled, not just the special hard cases. Only true fresh-intro
    segments (nothing to recall) return None."""
    probes = g.get("memory_probes", [])
    cast = g.get("cast", [])
    forbidden = g.get("forbidden", [])
    probe = next((p for p in GUIDED_PRIORITY if p in probes), None)
    eid = None
    count = gap = None
    if probe == "count_memory":
        e = next((c for c in cast if c.get("count")), None)
        if e:
            eid, count, gap = e["eid"], e.get("count"), e.get("gap")
    elif probe == "state_change":
        e = next((c for c in cast if c.get("op") == "transform"), None)
        eid = e["eid"] if e else None
    elif probe == "deprecation_avoidance":
        eid = forbidden[0]["eid"] if forbidden else None
    elif probe == "long_gap_reappearance":
        cand = [c for c in cast if isinstance(c.get("gap"), int)]
        e = max(cand, key=lambda c: c["gap"], default=None) or \
            next((c for c in cast if c.get("op") == "recall_after_gap"), None)
        if e:
            eid, gap = e["eid"], e.get("gap")

    if probe is None:
        # generic recall: ANY entity established in an earlier segment (so the
        # model must remember it), preferring the longest recency gap. Only true
        # fresh-intro segments (no earlier entity) fall through to None.
        cand = [c for c in cast if first_seen.get(c["eid"], seg_idx) < seg_idx]
        e = max(cand, key=lambda c: c.get("gap") or 0, default=None)
        if e is None:
            return None
        eid, gap, probe = e["eid"], (e.get("gap") or 0), "recall"

    if eid is None:
        eid = cast[0]["eid"] if cast else (forbidden[0]["eid"] if forbidden else None)
    if eid is None:
        return None
    name = name_of(eid)
    est = first_seen.get(eid, seg_idx)
    if not isinstance(gap, int):
        gap = max(0, seg_idx - est)
    templates = {
        "long_gap_reappearance": (f"Recall {name} on return.",
                                  f"Absent ~{gap} segments, far beyond any recency window."),
        "count_memory": (f"Recall the exact count of {name}" + (f" (\u00d7{count})." if count else "."),
                         f"Gone ~{gap} segments; the exact count must return."),
        "state_change": (f"{name} changes state here.",
                         "The new state must persist afterwards, not revert."),
        "lookalike_disambiguation": (f"Recall {name}, not the look-alike.",
                                     "A near-identical entity competes in memory."),
        "deprecation_avoidance": (f"Do NOT reintroduce {name}.",
                                  "Removed earlier; it must stay gone."),
        "reference_indirect": (f"Resolve the indirect reference to {name}.",
                               "Named only indirectly in this segment."),
        "recall": (f"Recall {name} on its return." if gap >= 1 else f"Keep {name} consistent.",
                   f"Last seen ~{gap} segment(s) ago; identity must carry over." if gap >= 1
                   else "Carried over from earlier; identity/state must stay stable."),
    }
    test, hard = templates[probe]
    return {"probe": probe, "eid": eid, "name": name, "establish_idx": est,
            "gap": gap, "test": test, "hard": hard}


def build_segment_meta(gt: dict, prompts_zh: dict, n_segments: int, tag_lang: str,
                       gt_en: dict | None = None, prompts_en: dict | None = None) -> list[dict]:
    """Structure (cast/probes/gaps) comes from the language-independent zh GT;
    display text is carried for both languages."""
    long_gap = int(gt.get("params", {}).get("gap_long_threshold", 10))
    gt_segs = {s["segment_id"]: s for s in gt["segments"]}
    names = {"zh": _name_map(gt), "en": _name_map(gt_en)}
    pz, pe = _prompt_by_sid(prompts_zh), _prompt_by_sid(prompts_en)
    order = [s["segment_id"] for s in prompts_zh["segments"][:n_segments]]
    # first-appearance segment per entity (for the guided "in memory" anchor)
    first_seen: dict = {}
    for i, sid in enumerate(order):
        for c in gt_segs.get(sid, {}).get("cast", []):
            first_seen.setdefault(c["eid"], i)
    # guided text uses English names when available, else Chinese
    g_lang = "en" if names["en"] else "zh"

    def _name_of(eid):
        return names[g_lang].get(eid) or names["zh"].get(eid, eid)

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
            terms: set[str] = set()
            if txt:
                for eid in {c["eid"] for c in g.get("cast", [])}:
                    nm = names[lang].get(eid)
                    if nm:
                        terms.update(highlight_terms(nm, txt))
            hl[lang] = sorted(terms, key=len, reverse=True)
        out.append({
            "seg_idx": i, "seg_id": sid, "prompt": prompt, "hl": hl,
            "anchors": anchors, "names": names, "tags": tags,
            "guided": _guided_meta(g, i, first_seen, _name_of),
        })
    return out


# ---------------------------------------------------------------------------
# text layout helpers (PIL)
# ---------------------------------------------------------------------------
def tokenize_highlight(text: str, entities: list[str]) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_entity) runs by matching entity names.

    Latin terms are matched on word boundaries so we never highlight a mid-word
    fragment; CJK terms match verbatim (no word boundaries in CJK)."""
    parts = []
    for e in sorted({e for e in entities if e}, key=len, reverse=True):
        esc = re.escape(e)
        if not _is_cjk(e) and re.match(r"[A-Za-z]", e):
            esc = r"(?<![A-Za-z])" + esc + r"(?![A-Za-z])"
        parts.append(esc)
    if not parts:
        return [(text, False)]
    pattern = re.compile("|".join(parts))
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


def bar_height(caption_lang: str, scale: float, guided: bool = False) -> int:
    # Constant across the whole clip. The guided Test/Hard explanation floats
    # over the video (see render_guided_panel), so it needs no extra bar height
    # and non-probe segments are not padded with empty black.
    base = 150 if caption_lang != "both" else 220
    return int(base * scale)


def _draw_prompt(d, text, hl_terms, x, y, font, max_w, lh, ent_clr):
    runs = tokenize_highlight(text, hl_terms)
    lines = layout_runs(runs, font, max_w)[:2]
    return draw_lines(d, lines, x, y, font, lh, CLR_PROMPT, ent_clr)


def render_caption_bar(meta, width, bar_h, scale, caption_lang, guided=False):
    """Bottom caption bar as an (bar_h, width, 3) uint8 RGB array.

    caption_lang: 'zh' | 'en' | 'both' (en primary on top, zh secondary below).
    guided: also draw the Test/Hard memory-probe explanation block.
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


def render_guided_panel(gm: dict, vid_w: int, scale: float):
    """Floating lower-third Test/Hard panel (RGBA) drawn over the video for a
    memory-decisive segment. Returns (sprite, x_from_left) or None. The caller
    positions it near the bottom of the video area."""
    if not gm:
        return None
    pad = int(16 * scale)
    panel_w = int(min(vid_w - 2 * int(18 * scale), vid_w * 0.66))
    inner_w = panel_w - 2 * pad
    f_tag = ImageFont.truetype(str(FONT_BOLD), int(23 * scale))
    f_key = ImageFont.truetype(str(FONT_BOLD), int(21 * scale))
    f_val = ImageFont.truetype(str(FONT_REGULAR), int(22 * scale))
    dot = int(12 * scale)
    en_lbl, _zh, clr = PROBE_LABELS.get(gm["probe"], (gm["probe"], "", (200, 200, 210)))

    # measure wrapped Test/Hard lines first so the panel is exactly as tall as
    # its content (no wasted space, no clipping).
    rows = []
    for key, kclr, text, hl in (
        ("Test", (120, 205, 165), gm["test"], [gm["name"]]),
        ("Hard", (240, 150, 120), gm["hard"], []),
    ):
        kw = f_key.getlength(key) + int(10 * scale)
        lines = layout_runs(tokenize_highlight(text, hl), f_val, inner_w - kw)[:2]
        rows.append((key, kclr, kw, lines))
    lh = int(26 * scale)
    body_h = sum(len(r[3]) * lh + int(3 * scale) for r in rows)
    panel_h = pad + int(30 * scale) + body_h + pad

    img = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, panel_w - 1, panel_h - 1], radius=int(12 * scale),
                        fill=(10, 10, 14, 205), outline=clr + (235,), width=max(1, int(scale)))
    y = pad
    d.ellipse([pad, y + int(5 * scale), pad + dot, y + int(5 * scale) + dot], fill=clr + (255,))
    d.text((pad + dot + int(9 * scale), y), en_lbl, font=f_tag, fill=(238, 238, 242, 255))
    y += int(30 * scale)
    for key, kclr, kw, lines in rows:
        d.text((pad, y), key, font=f_key, fill=kclr + (255,))
        yy = y
        for line in lines:
            cx = pad + kw
            for ch, is_ent in line:
                d.text((cx, yy), ch, font=f_val,
                       fill=(CLR_ENTITY + (255,)) if is_ent else (216, 218, 224, 255))
                cx += f_val.getlength(ch)
            yy += lh
        y = yy + int(3 * scale)
    return np.asarray(img), int(18 * scale)


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
def _mmss(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _make_inset_sprite(frame_rgb, inw: int, inh: int, scale: float):
    """Boxed 'in memory' inset: the establish (first-appearance) frame with a
    white gap, amber border, and an 'in memory' chip on its bottom edge."""
    f = ImageFont.truetype(str(FONT_BOLD), int(20 * scale))
    pad = 3
    img = Image.new("RGBA", (inw + 2 * pad, inh + 2 * pad), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    thumb = Image.fromarray(frame_rgb).resize((inw, inh), Image.LANCZOS)
    img.paste(thumb, (pad, pad))
    d.rectangle([0, 0, inw + 2 * pad - 1, inh + 2 * pad - 1], outline=(250, 210, 90), width=2)
    txt = "in memory"
    tw = d.textlength(txt, font=f)
    d.rectangle([pad, inh + 2 * pad - int(24 * scale), pad + tw + int(12 * scale),
                 inh + 2 * pad - 1], fill=(20, 18, 10, 235))
    d.text((pad + int(5 * scale), inh + 2 * pad - int(23 * scale)), txt, font=f, fill=(255, 224, 130, 255))
    return np.asarray(img)


def _make_stamp_sprite(text: str, scale: float):
    """Top-left MM:SS -> MM:SS memory-gap stamp on a dark chip."""
    f = ImageFont.truetype(str(FONT_BOLD), int(22 * scale))
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw = tmp.textlength(text, font=f)
    W, H = int(tw + 20 * scale), int(30 * scale)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=int(6 * scale), fill=(12, 12, 14, 225))
    d.text((int(9 * scale), int(4 * scale)), text, font=f, fill=(240, 240, 244, 255))
    return np.asarray(img)


def annotate(video: Path, segs: list[dict], out: Path, fps: float, caption_lang: str,
             guided: bool = False):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 16.0
    scale = max(0.7, w / 960.0)
    bar_h = bar_height(caption_lang, scale, guided)
    n_seg = len(segs)
    # decoded RGB frames are evenly split across the generated segments
    # (each segment has the same latent length, VAE-decoded at the same ratio).
    per_seg = max(1.0, total / n_seg)
    spf = per_seg / (in_fps if in_fps > 0 else 16.0)  # seconds per segment

    # pre-render overlays once per segment (RGB caption bar + RGBA tag box)
    bars = [render_caption_bar(m, w, bar_h, scale, caption_lang, guided) for m in segs]
    tags = [None] * n_seg if guided else [render_tag_box(m, w, scale) for m in segs]

    # guided extras: per-segment 'in memory' inset (decoded from THIS video at the
    # entity's first-appearance segment) + a MM:SS->MM:SS memory-gap stamp.
    extras: list[list] = [[] for _ in segs]
    if guided:
        inw = int(w * 0.28)
        inh = max(1, round(inw * h / w))
        need = sorted({m["guided"]["establish_idx"] for m in segs if m.get("guided")})
        est_frame = {}
        cap2 = cv2.VideoCapture(str(video))
        for est in need:
            fn = min(total - 1, max(0, int(est * per_seg + per_seg * 0.5)))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ok2, fr = cap2.read()
            if ok2:
                est_frame[est] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cap2.release()
        margin = int(14 * scale)
        for i, m in enumerate(segs):
            gm = m.get("guided")
            if not gm:
                continue
            has_gap = gm["establish_idx"] < i  # entity was established earlier
            fr = est_frame.get(gm["establish_idx"])
            if fr is not None and has_gap:
                spr = _make_inset_sprite(fr, inw, inh, scale)
                extras[i].append((spr, w - spr.shape[1] - margin, margin))
            if has_gap:
                stamp = f"{_mmss(gm['establish_idx'] * spf)} \u2192 {_mmss(i * spf)}"
                ss = _make_stamp_sprite(stamp, scale)
                extras[i].append((ss, int(6 * scale), int(6 * scale)))
            panel = render_guided_panel(gm, w, scale)
            if panel is not None:
                pspr, px = panel
                py = max(0, h - pspr.shape[0] - int(16 * scale))
                extras[i].append((pspr, px, py))

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
        for spr, sx, sy in extras[seg]:
            alpha_paste(rgb, spr, sx, sy)
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
    ap.add_argument("--guided", action="store_true",
                    help="add per-probe Test/Hard explanation, memory-gap stamp, and "
                         "'in memory' first-appearance inset (mirrors fig:trackb-money)")
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
    info = annotate(video, segs, out, fps, args.caption_lang, guided=args.guided)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
