#!/usr/bin/env python3
"""Freeze the closed-loop failure-attribution sample and build the annotation bundle.

    python3 scripts/human_attribution/freeze_manifest.py [--out results/human_attribution/v1]

Scope: the four shared-generator retrieval rollouts (one generator, one prompt stream),
the only closed-loop setting whose intermediates are all inspectable:
earlier appearance -> retrieved references -> composed keyframe -> generated clip.

Unit kinds (sampled from the frozen judge's per-entity verdicts; the judge verdict is
used only for sampling and is never shown to the annotator):
  miss      required recall entity absent from the judge's present set A
  wrong     entity present but wrong state (transform/persist) or wrong look-alike member
  forbidden deprecated (destroyed/deceased) entity present in A
  control   required recall entity present in A (keeps the annotator from expecting failures)

Everything is deterministic given --seed. The sealed manifest (systems, judge verdicts,
source paths, media sha256) stays outside the bundle; the bundle only carries blinded
cards and media. Re-running with the same inputs reproduces the same files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TESTBED = ROOT / "results" / "trackB" / "shared_generator_testbed"
GT_DIR = ROOT / "assets" / "trackB" / "en" / "gt"
APP_DIR = Path(__file__).resolve().parent / "app"

# clip_shift: production clip index = GT index + clip_shift
SYSTEMS = {
    "text_frame": {"link": "text_frame", "tag": "prod_retr_20260831", "scores": None, "clip_shift": 0},
    "seg_uniform": {"link": "seg_uniform", "tag": "prod_retr_20260831", "scores": None, "clip_shift": 0},
    "seg_dinokey": {"link": "seg_dinokey", "tag": "prod_retr_20260831", "scores": None, "clip_shift": 0},
    "seg_frame": {"link": "seg_frame_clean", "tag": "prod_retr_segframe_clean_20260921",
                  "scores": "seg_frame_clean_scores", "clip_shift": -1},
}
RECALL_KINDS = {"character", "prop"}
RECALL_OPS = {"recall", "recall_after_gap"}
PER_SYSTEM = {  # stratum -> units per system
    "miss_near": 8, "miss_mid": 8, "miss_long": 8, "miss_temporal": 4,
    "wrong_state": 6, "wrong_lookalike": 4, "forbidden": 10, "control": 8,
}
FALLBACK = {"miss_temporal": "miss_mid", "wrong_lookalike": "wrong_state", "miss_long": "miss_mid"}
REPEAT_FRACTION = 0.10
N_PRACTICE = 5


def seg_num(sid: str) -> int:
    return int(sid.split("_")[-1])


def rebase(path: str, run: Path, tag: str) -> Path | None:
    """Map a recorded absolute path onto this run dir using the part after `<tag>/`."""
    p = Path(path)
    if p.exists():
        return p
    marker = f"/{tag}/"
    if marker in path:
        q = run / path.split(marker, 1)[1]
        if q.exists():
            return q
    return None


def current_look(ent: dict, eid: str, segments: list[dict], upto: int) -> str:
    """Appearance the entity should have at segment `upto`: its latest non-flashback state so far."""
    look = None
    for seg in segments[: upto - 1]:
        for c in seg.get("cast") or []:
            st = c.get("state")
            if c.get("eid") == eid and st and not st.get("flashback") and st.get("appearance"):
                look = st["appearance"]
    states = ent.get("states") or {}
    return look or ent.get("appearance") or (next(iter(states.values())) if states else "")


def entity_card(ent: dict, eid: str, state: dict | None, look: str) -> dict:
    return {
        "name": ent.get("name", eid),
        "kind": ent.get("kind", ""),
        "appearance": look,
        "required_state": (state or {}).get("appearance"),
    }


def last_appearance(segments: list[dict], upto: int, eid: str) -> str | None:
    for seg in reversed(segments[: upto - 1]):
        if any(c.get("eid") == eid for c in seg.get("cast") or []):
            return seg["segment_id"]
    return None


def collect(system: str, cfg: dict) -> tuple[list[dict], dict]:
    base = TESTBED / cfg["link"]
    units, excluded = [], defaultdict(int)
    for story_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        story = story_dir.name
        gt = json.loads((GT_DIR / f"{story}.json").read_text())
        ents, segments = gt["entities"], gt["segments"]
        run = story_dir / "name_anchored" / cfg["tag"]
        det_path = (TESTBED / cfg["scores"] / story / "details.json") if cfg["scores"] else run / "_trackB_score" / "details.json"
        details = {d["segment_id"]: d for d in json.loads(det_path.read_text())}

        def clip(sid: str) -> Path:
            return run / "review" / "segments" / f"seg_{seg_num(sid) + cfg['clip_shift']:03d}.mp4"

        for seg in segments:
            sid = seg["segment_id"]
            d = details.get(sid)
            if d is None or d.get("parse_error"):
                excluded["no_judge"] += 1
                continue
            A = set(d.get("A") or [])
            voted = d.get("voted") or {}
            rec_path = run / "pipeline" / f"segment_{seg_num(sid) - 1:03d}" / "pipeline_record.json"
            if not rec_path.is_file():
                excluded["no_pipeline_record"] += 1
                continue
            ctl = json.loads(rec_path.read_text())["generation"]["task"]["controls"]
            refs = [rebase(c["image"], run, cfg["tag"]) for c in ctl.get("source_crops") or []]
            kf_raw = (ctl.get("keyframe_record") or {}).get("keyframe")
            kf = rebase(kf_raw, run, cfg["tag"]) if kf_raw else None
            cur = clip(sid)
            if not refs or any(r is None for r in refs) or kf is None or not cur.is_file():
                excluded["missing_intermediate"] += 1
                continue
            common = {"system": system, "story_id": story, "segment_id": sid, "intent": seg.get("action", ""),
                      "clip": str(cur), "keyframe": str(kf), "refs": [str(r) for r in refs]}

            la_members = {e: la for la in seg.get("lookalike_active") or [] for e in la.get("present_members", [])}
            for c in seg.get("cast") or []:
                eid, op, gap = c.get("eid"), c.get("op"), c.get("gap")
                ent = ents.get(eid, {})
                if ent.get("kind") not in RECALL_KINDS or op == "introduce":
                    continue
                prev = c.get("last_seen") or last_appearance(segments, seg_num(sid), eid)
                if not prev or not clip(prev).is_file():
                    continue
                state = c.get("state") if c.get("state") and not c["state"].get("flashback") else None
                look = current_look(ent, eid, segments, seg_num(sid))
                u = dict(common, eid=eid, earlier_segment=prev, earlier_clip=str(clip(prev)),
                         card=entity_card(ent, eid, state, look), gap=gap, op=op)
                if eid in la_members:
                    la = la_members[eid]
                    other = next(x for x in la["pair"] if x != eid)
                    u["card"]["lookalike_note"] = la.get("features")
                    u["card"]["lookalike_other"] = ents.get(other, {}).get("name", other)
                present = eid in A
                is_recall = op in RECALL_OPS
                if not present and is_recall and gap:
                    if (c.get("anchor") or {}).get("type") == "temporal":
                        units.append(dict(u, kind="miss", stratum="miss_temporal"))
                    elif gap >= 30:
                        units.append(dict(u, kind="miss", stratum="miss_long"))
                    elif gap >= 5:
                        units.append(dict(u, kind="miss", stratum="miss_mid"))
                    elif gap >= 1:
                        units.append(dict(u, kind="miss", stratum="miss_near"))
                elif present and state and op == "persist" and voted.get(eid, {}).get("state") != "changed":
                    units.append(dict(u, kind="wrong", stratum="wrong_state"))
                elif present and eid in la_members and voted.get(eid, {}).get("instance_match") not in (None, eid):
                    units.append(dict(u, kind="wrong", stratum="wrong_lookalike"))
                elif present and is_recall and gap and gap >= 1:
                    units.append(dict(u, kind="control", stratum="control"))
            for f in seg.get("forbidden") or []:
                eid = f.get("eid")
                if f.get("reason") not in ("destroyed", "deceased") or eid not in A:
                    continue
                prev = last_appearance(segments, seg_num(sid), eid)
                if not prev or not clip(prev).is_file():
                    continue
                ent = ents.get(eid, {})
                units.append(dict(common, eid=eid, kind="forbidden", stratum="forbidden", reason=f.get("reason"),
                                  earlier_segment=prev, earlier_clip=str(clip(prev)),
                                  card=entity_card(ent, eid, None, current_look(ent, eid, segments, seg_num(sid)))))
    return units, dict(excluded)


def localize(u: dict, lang: str, cache: dict) -> dict:
    """Rebuild the annotator-facing card and intent from the `lang` gold (same ids and structure as en)."""
    if lang == "en":
        return u
    if u["story_id"] not in cache:
        cache[u["story_id"]] = json.loads((GT_DIR.parents[1] / lang / "gt" / f"{u['story_id']}.json").read_text())
    gt = cache[u["story_id"]]
    ents, segments = gt["entities"], gt["segments"]
    seg = next(s for s in segments if s["segment_id"] == u["segment_id"])
    eid, n = u["eid"], seg_num(u["segment_id"])
    ent = ents.get(eid, {})
    c = next((c for c in seg.get("cast") or [] if c.get("eid") == eid), {})
    state = c.get("state") if c.get("state") and not c["state"].get("flashback") else None
    if u["kind"] == "forbidden":
        state = None
    card = entity_card(ent, eid, state, current_look(ent, eid, segments, n))
    la = next((la for la in seg.get("lookalike_active") or [] if eid in la.get("present_members", [])), None)
    if la and u["kind"] != "forbidden":
        other = next(x for x in la["pair"] if x != eid)
        card["lookalike_note"] = la.get("features")
        card["lookalike_other"] = ents.get(other, {}).get("name", other)
    return dict(u, card=card, intent=seg.get("action", ""), card_en=u["card"], intent_en=u["intent"], lang=lang)


def sample(pool: list[dict], rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for u in pool:
        by[(u["system"], u["stratum"])].append(u)
    picked = []
    for system in SYSTEMS:
        taken_ids = set()
        for stratum, n in PER_SYSTEM.items():
            cands = sorted(by[(system, stratum)], key=lambda u: (u["story_id"], u["segment_id"], u["eid"]))
            rng.shuffle(cands)
            take = cands[:n]
            short = n - len(take)
            if short and stratum in FALLBACK:
                extra = [u for u in sorted(by[(system, FALLBACK[stratum])], key=lambda u: (u["story_id"], u["segment_id"], u["eid"]))]
                rng.shuffle(extra)
                take += [u for u in extra if id(u) not in taken_ids and u not in take][:short]
            for u in take:
                taken_ids.add(id(u))
                pool_n = len(by[(system, u["stratum"])])
                picked.append(dict(u, inclusion_prob=round(min(1.0, n / pool_n), 6) if pool_n else None))
    return picked


def to_jpg(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((832, 832))
        im.save(dst, "JPEG", quality=88)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_media(u: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    files = {"earlier": dest / "earlier.mp4", "clip": dest / "clip.mp4", "keyframe": dest / "keyframe.jpg"}
    shutil.copyfile(u["earlier_clip"], files["earlier"])
    shutil.copyfile(u["clip"], files["clip"])
    to_jpg(Path(u["keyframe"]), files["keyframe"])
    for i, r in enumerate(u["refs"]):
        files[f"ref_{i}"] = dest / f"ref_{i}.jpg"
        to_jpg(Path(r), files[f"ref_{i}"])
    return {k: sha(v) for k, v in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "human_attribution" / "v3")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--lang", choices=("zh", "en"), default="zh",
                    help="language of annotator-facing text; sampling always uses the en gold the judge scored")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool, excluded = [], {}
    for system, cfg in SYSTEMS.items():
        units, exc = collect(system, cfg)
        pool += units
        excluded[system] = exc
    picked = sample(pool, rng)

    practice_pool = [u for u in pool if u["kind"] in ("miss", "control") and u not in picked]
    practice_pool.sort(key=lambda u: (u["system"], u["story_id"], u["segment_id"], u["eid"]))
    rng.shuffle(practice_pool)
    practice = practice_pool[:N_PRACTICE]

    rng.shuffle(picked)
    n_rep = round(REPEAT_FRACTION * len(picked))
    rep_src = rng.sample(range(len(picked) // 2), n_rep)  # repeat items first shown in the first half
    order = [dict(u, base_unit=f"u{i}", is_repeat=False) for i, u in enumerate(picked)]
    for i in rep_src:
        pos = rng.randrange(len(order) // 2 + 1, len(order) + 1)
        order.insert(pos, dict(picked[i], base_unit=f"u{i}", is_repeat=True))

    out = args.out
    bundle = out / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "media").mkdir(parents=True)

    sealed_units, cards = [], []
    cache: dict = {}
    items = [("p", i, localize(dict(u, base_unit=None, is_repeat=False), args.lang, cache)) for i, u in enumerate(practice)]
    items += [("u", i, localize(u, args.lang, cache)) for i, u in enumerate(order)]
    for prefix, i, u in items:
        disp = hashlib.sha1(f"{args.seed}|{prefix}|{i}|{u['system']}|{u['story_id']}|{u['segment_id']}|{u['eid']}".encode()).hexdigest()[:10]
        media_sha = build_media(u, bundle / "media" / disp)
        cards.append({
            "display_id": disp, "practice": prefix == "p",
            "target": "must_be_absent" if u["kind"] == "forbidden" else "must_appear",
            "intent": u["intent"], "card": u["card"],
        })
        sealed_units.append(dict(u, display_id=disp, practice=prefix == "p", position=len(sealed_units),
                                 media_sha256=media_sha))

    data_js = "window.STUDY = " + json.dumps({"version": out.name, "seed": args.seed, "items": cards}, ensure_ascii=False) + ";\n"
    (bundle / "data.js").write_text(data_js)
    for f in ("index.html", "app.js", "INSTRUCTIONS.md"):
        shutil.copyfile(APP_DIR / f, bundle / f)

    counts = defaultdict(int)
    for u in picked:
        counts[f"{u['system']}:{u['stratum']}"] += 1
    pool_counts = defaultdict(int)
    for u in pool:
        pool_counts[f"{u['system']}:{u['stratum']}"] += 1
    sealed = {
        "study": "closed-loop failure attribution (supplementary human study)",
        "seed": args.seed, "lang": args.lang, "systems": list(SYSTEMS), "per_system_plan": PER_SYSTEM, "fallback": FALLBACK,
        "repeat_fraction": REPEAT_FRACTION, "n_scored_unique": len(picked), "n_scored_items": len(order),
        "n_practice": len(practice), "pool_counts": dict(sorted(pool_counts.items())),
        "sample_counts": dict(sorted(counts.items())), "excluded": excluded,
        "data_js_sha256": hashlib.sha256(data_js.encode()).hexdigest(),
        "units": sealed_units,
    }
    (out / "sealed_manifest.json").write_text(json.dumps(sealed, indent=2, ensure_ascii=False) + "\n")

    zpath = out / "bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, Path("vmem_human_study") / p.relative_to(bundle))
    print(json.dumps({k: sealed[k] for k in ("n_scored_unique", "n_scored_items", "n_practice", "sample_counts", "excluded", "data_js_sha256")}, indent=1))
    print(f"bundle: {bundle}\nzip: {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
