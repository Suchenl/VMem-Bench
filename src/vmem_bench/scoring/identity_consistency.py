"""VMem-Bench Track B identity consistency (metric ``trackB-identity-1.0``).

The frozen presence scorer (``end2end_coverage``, ``trackB-end2end-1.0``) decides whether a clip shows
something that matches an entity's gold *text* description. It never compares a returning entity with
its own earlier rendering. This scorer does: for every return of a character/prop it shows a pinned
VLM two 3-frame strips — A = the clip of the entity's previous appearance, B = the current clip, both
from the same system — plus the entity's name, kind and gold description(s), and asks (strict JSON)
whether the entity is visible in A, visible in B, and, if in both, whether it is the SAME
individual/object. The system/condition name is never shown.

Units (deterministic from gold, ``build_units``), per cast entry of the configured kinds:
  main            op in {recall, recall_after_gap, persist}, gap >= 1
  gap0            seeded 20% sample of the same ops with gap == 0 (continuity reference)
  transform_ge30  op == transform, gap >= 30 (state changed; identity should hold)
The previous appearance is the latest earlier GT segment whose cast lists the entity.

Controls on the control condition's clips (``build_controls``): ``ctl_pos`` = same clip, disjoint
frame sets; ``ctl_neg`` = entity X's clip vs a clip of a different same-kind entity Y of the same story
in which the frozen presence judge voted X absent (prompt describes X).

Metrics per condition and cell (micro-pooled): consistent_rate = #(visible_B and same) / #units;
identity_given_both = #same / #(visible_A and visible_B); visible_B_rate (+ visible_A, both, unsure).
Differences vs a baseline condition use a story-clustered paired bootstrap (same draws for every
condition). Parse errors count as a unit with nothing visible.

Clip layout is pluggable: a conditions JSON maps condition -> ``clip_template`` (``{story}``,
``{idx:03d}`` and any ``--var name=value``), ``file_index_offset`` (int or ``"auto"``) and optional
``details_template`` (frozen judge details.json, used for control presence).

CLI (``python -m vmem_bench.scoring.identity_consistency <cmd>``): ``units``, ``strips``, ``judge``,
``aggregate``. See docs/benchmark/trackB_scoring.md.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "trackb_identity.json"
METRIC_VERSION = "trackB-identity-1.0"

PROMPT = """You compare two images taken from generated video clips of the same film. Each image is a horizontal strip of 3 frames sampled from one clip. Image A comes from an earlier clip, image B from a later clip.

Entity: {name} ({kind})
Description of the entity in A: {desc_a}
Description of the entity in B: {desc_b}{state_note}

Camera angle, framing, pose, expression, lighting and background may differ between A and B; ignore those differences.

Questions:
1. visible_A: is this entity (matching the name, kind and description) visible in image A? true, false or "unsure".
2. visible_B: the same question for image B.
3. same: only if the entity is visible in both images, is it the SAME individual/object in A and B, i.e. the same identity (for a character: face, hair, build, clothing; for a prop: shape, color, material, markings)? true, false or "unsure". Use null if it is not visible in both.

Answer with strict JSON only, no other text:
{{"visible_A": ..., "visible_B": ..., "same": ...}}"""
STATE_NOTE = ("\nThe described state changes between A and B (for example age, damage or repair); a change "
              "that matches the descriptions is expected and does not by itself mean a different identity.")

FIELDS = ("n", "vis_A", "vis_B", "both", "same", "consistent", "unsure_vis_B", "unsure_same", "parse")
BOOT_METRICS = ("consistent_rate", "identity_given_both", "visible_B_rate")


# ------------------------------------------------------------------------------------------ config
def load_config(path: Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def gap_bin(gap: int) -> str:
    return "0" if gap == 0 else "1-4" if gap <= 4 else "5-29" if gap <= 29 else ">=30"


def seg_num(seg_id: str) -> int:
    return int(str(seg_id).rsplit("_", 1)[-1])


def load_gold(gt_dir: Path, stories: list[str]) -> dict[str, dict]:
    return {s: json.loads((Path(gt_dir) / f"{s}.json").read_text()) for s in stories}


def default_stories(gt_dir: Path, n: int = 30) -> list[str]:
    return sorted(p.stem for p in Path(gt_dir).glob("*.json"))[:n]


# ------------------------------------------------------------------------------------------ units
def describe(ent: dict, label: str | None) -> str:
    states = ent.get("states") or {}
    if label and label in states:
        v = states[label]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return str(ent.get("appearance") or "")


def build_units(gold: dict[str, dict], cfg: dict, *, kinds: list[str] | None = None,
                rng: random.Random | None = None) -> tuple[list[dict], dict]:
    """All identity units of ``gold`` (story -> GT json), in story then segment order.

    ``rng`` drives the gap-0 sample; one draw per eligible gap-0 entry in document order, so the
    unit set is a pure function of gold + seed + kinds.
    """
    kinds = list(kinds or cfg["unit_kinds"])
    ops = set(cfg["unit_ops"])
    rate = float(cfg["gap0_sample_rate"])
    tmin = int(cfg["transform_min_gap"])
    rng = rng or random.Random(int(cfg["unit_seed"]))
    units: list[dict] = []
    audit: Counter = Counter()
    for story in sorted(gold):
        g = gold[story]
        ents = g["entities"]
        state = {eid: e.get("initial_state") for eid, e in ents.items()}
        last_seen: dict[str, int] = {}
        seg_state: dict[tuple[str, int], str | None] = {}
        for seg in g["segments"]:
            n = seg_num(seg["segment_id"])
            for c in seg.get("cast") or []:
                st = c.get("state")
                if isinstance(st, dict) and st.get("label"):
                    state[c["eid"]] = st["label"]
                seg_state[(c["eid"], n)] = state.get(c["eid"])
            for c in seg.get("cast") or []:
                eid, op = c["eid"], c["op"]
                ent = ents[eid]
                prev = last_seen.get(eid)
                if ent.get("kind") not in kinds or op == "introduce" or prev is None:
                    continue
                gap = int(c.get("gap") or 0)
                if c.get("last_seen") and seg_num(c["last_seen"]) != prev:
                    audit["last_seen_mismatch"] += 1
                if op in ops and gap >= 1:
                    stratum = "main"
                elif op in ops and gap == 0:
                    stratum = "gap0" if rng.random() < rate else None
                elif op == "transform" and gap >= tmin:
                    stratum = f"transform_ge{tmin}"
                else:
                    stratum = None
                if not stratum:
                    continue
                sa, sb = seg_state.get((eid, prev)), seg_state[(eid, n)]
                units.append({
                    "uid": f"{story}|{n:03d}|{eid}", "story": story, "eid": eid, "name": ent.get("name"),
                    "kind": ent["kind"], "op": op, "gap": gap, "gap_bin": gap_bin(gap), "stratum": stratum,
                    "cur_gt": n, "prev_gt": prev, "state_A": sa, "state_B": sb,
                    "desc_A": describe(ent, sa), "desc_B": describe(ent, sb),
                })
            for c in seg.get("cast") or []:
                last_seen[c["eid"]] = n
    return units, dict(audit)


def presence_from_details(details: list[dict]) -> dict[int, set[str]]:
    """GT seg -> entities the frozen presence judge voted present (from its details.json)."""
    out = {}
    for seg in details:
        voted = seg.get("voted") or {}
        out[seg_num(seg["segment_id"])] = {e for e, v in voted.items() if str(v.get("present")).lower() == "true"}
    return out


def build_controls(units: list[dict], gold: dict[str, dict], presence: dict[str, dict[int, set[str]]],
                   cfg: dict, *, kinds: list[str] | None = None, seed: int | None = None) -> list[dict]:
    """Positive (same clip, disjoint strips) and negative (different same-kind entity) controls."""
    kinds = list(kinds or cfg["unit_kinds"])
    n_pos, n_neg = int(cfg["n_controls"]["ctl_pos"]), int(cfg["n_controls"]["ctl_neg"])
    crng = random.Random(int(cfg["unit_seed"]) + 1 if seed is None else seed)
    stories = sorted(gold)
    pos_pool = [u for u in units if u["stratum"] in ("main", "gap0") and u["kind"] in kinds
                and u["eid"] in presence[u["story"]].get(u["cur_gt"], set())]
    pos = crng.sample(pos_pool, min(n_pos, len(pos_pool)))
    controls = [{**u, "uid": f"pos|{u['uid']}", "stratum": "ctl_pos", "prev_gt": u["cur_gt"],
                 "variant_A": "posA", "variant_B": "posB", "desc_A": u["desc_B"], "state_A": u["state_B"]}
                for u in pos]
    neg_pool = []
    for s in stories:
        ents = gold[s]["entities"]
        seen: dict[str, list[int]] = {}
        for n, ps in sorted(presence[s].items()):
            for e in sorted(ps):
                if ents.get(e, {}).get("kind") in kinds:
                    seen.setdefault(e, []).append(n)
        for x in sorted(seen):
            for y in sorted(seen):
                if x != y and ents[x]["kind"] == ents[y]["kind"]:
                    neg_pool.append((s, x, y))
    crng.shuffle(neg_pool)
    n_done = 0
    for s, x, y in neg_pool:
        if n_done == n_neg:
            break
        ents = gold[s]["entities"]
        xs = [n for n, ps in sorted(presence[s].items()) if x in ps]
        ys = [n for n, ps in sorted(presence[s].items()) if y in ps and x not in ps]
        if not ys:
            continue
        na, nb = crng.choice(xs), crng.choice(ys)
        d = describe(ents[x], ents[x].get("initial_state"))
        controls.append({"uid": f"neg|{s}|{x}@{na:03d}|{y}@{nb:03d}", "story": s, "eid": x, "eid_B": y,
                         "name": ents[x].get("name"), "kind": ents[x]["kind"], "stratum": "ctl_neg",
                         "prev_gt": na, "cur_gt": nb, "variant_A": "main", "variant_B": "main",
                         "desc_A": d, "desc_B": d, "state_A": None, "state_B": None, "gap": None, "gap_bin": None})
        n_done += 1
    return controls


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------ clips
class ClipLayout:
    """Condition -> clip path, from a conditions JSON plus template variables."""

    def __init__(self, conditions: dict[str, dict], variables: dict[str, str] | None = None) -> None:
        self.conditions = {k: v for k, v in conditions.items() if not k.startswith("_")}
        self.variables = dict(variables or {})

    @classmethod
    def from_file(cls, path: Path, variables: dict[str, str] | None = None) -> "ClipLayout":
        return cls(json.loads(Path(path).read_text()), variables)

    def _fmt(self, template: str, **kw: Any) -> str:
        return template.format(**self.variables, **kw)

    def offset(self, cond: str, story: str) -> int:
        off = self.conditions[cond].get("file_index_offset", 0)
        if off == "auto":
            probe = Path(self._fmt(self.conditions[cond]["clip_template"], story=story, idx=0))
            return -1 if probe.is_file() else 0
        return int(off)

    def clip(self, cond: str, story: str, gt_seg: int) -> Path:
        c = self.conditions[cond]
        return Path(self._fmt(c["clip_template"], story=story, idx=gt_seg + self.offset(cond, story)))

    def details(self, cond: str, story: str) -> Path | None:
        t = self.conditions[cond].get("details_template")
        return Path(self._fmt(t, story=story)) if t else None

    def label(self, cond: str) -> str:
        return str(self.conditions[cond].get("label", cond))


def parse_vars(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def strip_path(out: Path, cond: str, story: str, gt_seg: int, variant: str) -> Path:
    return Path(out) / "strips" / cond / story / f"gt{gt_seg:03d}_{variant}.jpg"


def read_frames(clip: Path, fracs: list[float]):
    """BGR frames at the given fractions of the clip's frame count (None on failure)."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round(f * (n - 1))))))
        ok, img = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(img)
    cap.release()
    return frames


def make_strip(clip: Path, fracs: list[float], out: Path, height: int = 256, quality: int = 90) -> str | None:
    """Write the 3-frame strip; returns an error string or None."""
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    if out.is_file() and out.stat().st_size > 0:
        return None
    if not clip.is_file():
        return "missing clip"
    frames = read_frames(clip, fracs)
    if frames is None:
        return "read failed"
    frames = [cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height), interpolation=cv2.INTER_AREA)
              for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), np.hstack(frames), [cv2.IMWRITE_JPEG_QUALITY, quality])
    os.replace(tmp, out)
    return None


def strip_jobs(units: list[dict], controls: list[dict], conditions: list[str], control_cond: str) -> list[tuple]:
    need = set()
    for u in units:
        for c in conditions:
            need.add((c, u["story"], u["prev_gt"], "main"))
            need.add((c, u["story"], u["cur_gt"], "main"))
    for u in controls:
        need.add((control_cond, u["story"], u["prev_gt"], u["variant_A"]))
        need.add((control_cond, u["story"], u["cur_gt"], u["variant_B"]))
    return sorted(need)


# ------------------------------------------------------------------------------------------ judge
def norm_answer(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    return {"true": True, "yes": True, "false": False, "no": False, "null": None, "none": None}.get(s, "unsure")


def parse_answer(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not all(k in d for k in ("visible_A", "visible_B", "same")):
        return None
    return {k: norm_answer(d[k]) for k in ("visible_A", "visible_B", "same")}


def build_prompt(u: dict) -> str:
    return PROMPT.format(name=u["name"], kind=u["kind"], desc_a=u["desc_A"], desc_b=u["desc_B"],
                         state_note=STATE_NOTE if u["desc_A"] != u["desc_B"] else "")


def judge_tasks(units: list[dict], controls: list[dict], conditions: list[str], control_cond: str) -> list[dict]:
    out = [{"cond": c, "unit": u, "va": "main", "vb": "main"} for c in conditions for u in units]
    out += [{"cond": u["stratum"], "clip_cond": control_cond, "unit": u, "va": u["variant_A"], "vb": u["variant_B"]}
            for u in controls]
    return sorted(out, key=lambda t: (t["cond"], t["unit"]["uid"]))


def result_path(out: Path, t: dict, sub: str = "judgments") -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "_", t["unit"]["uid"])
    return Path(out) / sub / t["cond"] / t["unit"]["story"] / f"{safe}.json"


def _b64(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def judge_call(endpoint: str, model: str, t: dict, out: Path, jcfg: dict) -> dict:
    u = t["unit"]
    cc = t.get("clip_cond", t["cond"])
    pa, pb = strip_path(out, cc, u["story"], u["prev_gt"], t["va"]), strip_path(out, cc, u["story"], u["cur_gt"], t["vb"])
    body = {"model": model, "temperature": float(jcfg["temperature"]), "max_tokens": int(jcfg["max_tokens"]),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Image A:"}, {"type": "image_url", "image_url": {"url": _b64(pa)}},
                {"type": "text", "text": "Image B:"}, {"type": "image_url", "image_url": {"url": _b64(pb)}},
                {"type": "text", "text": build_prompt(u)}]}]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    raw = resp["choices"][0]["message"]["content"]
    ans = parse_answer(raw)
    return {"task": {"cond": t["cond"], "clip_cond": cc, "uid": u["uid"], "va": t["va"], "vb": t["vb"],
                     "strip_A": str(pa), "strip_B": str(pb)},
            "answer": ans, "parse_error": ans is None, "raw": raw, "endpoint": endpoint,
            "latency_s": round(time.time() - t0, 2)}


def run_judge(tasks: list[dict], out: Path, endpoints: list[str], jcfg: dict, *, per_endpoint: int = 12,
              sub: str = "judgments", call: Callable = judge_call) -> dict:
    """Resume-safe: tasks whose result file exists are skipped. Returns counters."""
    todo = [t for t in tasks if not result_path(out, t, sub).is_file()]
    q: queue.Queue = queue.Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    stats = {"todo": len(todo), "done": 0, "err": 0, "parse": 0}
    model = jcfg["model"]

    def worker(ep: str) -> None:
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            res, err = None, ""
            for attempt in range(3):
                try:
                    res = call(ep, model, t, out, jcfg)
                    break
                except Exception as exc:  # noqa: BLE001
                    err = repr(exc)
                    time.sleep(5 * (attempt + 1))
            with lock:
                if res is None:
                    stats["err"] += 1
                    print(f"ERR {t['cond']} {t['unit']['uid']} {ep}: {err}", flush=True)
                    continue
                p = result_path(out, t, sub)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, ensure_ascii=False))
                os.replace(tmp, p)
                stats["done"] += 1
                stats["parse"] += bool(res["parse_error"])

    threads = [threading.Thread(target=worker, args=(e,)) for e in endpoints for _ in range(per_endpoint)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return stats


# ------------------------------------------------------------------------------------------ aggregate
def tally(ans: dict | None) -> dict[str, int]:
    t = dict.fromkeys(FIELDS, 0)
    t["n"] = 1
    if ans is None:
        t["parse"] = 1
        return t
    va, vb, same = ans["visible_A"] is True, ans["visible_B"] is True, ans["same"] is True
    t["vis_A"], t["vis_B"] = int(va), int(vb)
    t["both"] = int(va and vb)
    t["same"] = int(va and vb and same)
    t["consistent"] = int(vb and same)
    t["unsure_vis_B"] = int(ans["visible_B"] == "unsure")
    t["unsure_same"] = int(va and vb and ans["same"] == "unsure")
    return t


def _r(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def metrics(t: dict[str, int]) -> dict[str, float | None]:
    return {"consistent_rate": _r(t["consistent"], t["n"]), "identity_given_both": _r(t["same"], t["both"]),
            "visible_B_rate": _r(t["vis_B"], t["n"]), "visible_A_rate": _r(t["vis_A"], t["n"]),
            "both_visible_rate": _r(t["both"], t["n"]), "unsure_visible_B_rate": _r(t["unsure_vis_B"], t["n"]),
            "unsure_same_given_both": _r(t["unsure_same"], t["both"])}


def cells(u: dict) -> list[str]:
    s = u["stratum"]
    out = [f"{s}|all", f"{s}|{u['kind']}|{u['gap_bin']}", f"{s}|{u['kind']}"]
    if s == "main":
        out.append(f"main|all|{u['gap_bin']}")
    return out


def add(dst: dict, t: dict) -> None:
    for k, v in t.items():
        dst[k] = dst.get(k, 0) + v


def bootstrap_draws(stories: list[str], n_boot: int, seed: int) -> list[list[str]]:
    """Story-clustered resamples; identical draws for every condition (paired)."""
    sts = sorted(stories)
    rng = random.Random(seed)
    return [[rng.choice(sts) for _ in sts] for _ in range(n_boot)]


def percentile_ci(ds: list[float], n_boot: int) -> tuple[float | None, float | None]:
    """Same percentile indices as the paper's memory_gain_bootstrap.py (scaled to the valid draws)."""
    ds = sorted(ds)
    if not ds:
        return None, None
    return ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]


def pooled_table(per: dict[str, dict], conditions: list[str], baseline: str, all_cells: list[str],
                 stories: list[str], boot: dict, metric_fn: Callable, boot_metrics: tuple[str, ...],
                 fields: tuple[str, ...]) -> dict:
    """per[cond][cell][story] = tallies. Point estimates on all stories + paired bootstrap diffs."""
    sts = sorted(stories)
    draws = bootstrap_draws(sts, int(boot["n_boot"]), int(boot["seed"]))

    def pooled(c: str, cell: str, ss: list[str]) -> dict:
        t = dict.fromkeys(fields, 0)
        for s in ss:
            add(t, per[c][cell].get(s, {}))
        return t

    table = {}
    for cell in all_cells:
        row = {}
        for c in conditions:
            t = pooled(c, cell, sts)
            row[c] = {"counts": t, **metric_fn(t)}
        for c in conditions:
            if c == baseline:
                continue
            for m in boot_metrics:
                a, b = row[c][m], row[baseline][m]
                if a is None or b is None:
                    continue
                ds = []
                for d in draws:
                    x, y = metric_fn(pooled(c, cell, d))[m], metric_fn(pooled(baseline, cell, d))[m]
                    if x is not None and y is not None:
                        ds.append(x - y)
                lo, hi = percentile_ci(ds, int(boot["n_boot"]))
                row[c].setdefault(f"diff_vs_{baseline}", {})[m] = {
                    "diff": round(a - b, 4), "lo": None if lo is None else round(lo, 4),
                    "hi": None if hi is None else round(hi, 4), "n_draws": len(ds)}
        table[cell] = row
    return table


def load_judgments(out: Path, cond: str, sub: str = "judgments") -> dict[str, dict]:
    res = {}
    for p in sorted((Path(out) / sub / cond).glob("*/*.json")):
        r = json.loads(p.read_text())
        res[r["task"]["uid"]] = r
    return res


def aggregate(out: Path, units: list[dict], conditions: list[str], stories: list[str], cfg: dict,
              *, baseline: str = "no_memory", retest_sub: str = "judgments_rep1") -> dict:
    per = {c: defaultdict(lambda: defaultdict(dict)) for c in conditions}
    missing = {}
    for c in conditions:
        res = load_judgments(out, c)
        missing[c] = sum(u["uid"] not in res for u in units)
        for u in units:
            r = res.get(u["uid"])
            if r is None:
                continue
            t = tally(r["answer"])
            for cell in cells(u):
                add(per[c][cell][u["story"]], t)
    all_cells = sorted({cell for u in units for cell in cells(u)})
    table = pooled_table(per, conditions, baseline, all_cells, stories, cfg["bootstrap"], metrics, BOOT_METRICS, FIELDS)

    ctl = {}
    for name in ("ctl_pos", "ctl_neg"):
        t = dict.fromkeys(FIELDS, 0)
        for r in load_judgments(out, name).values():
            add(t, tally(r["answer"]))
        ctl[name] = {"counts": t, **metrics(t)}
    pos, neg = ctl["ctl_pos"], ctl["ctl_neg"]
    ctl["summary"] = {
        "pos_same_given_both": pos["identity_given_both"],
        "pos_consistent_rate": pos["consistent_rate"],
        "neg_not_consistent_rate": round(1 - neg["counts"]["consistent"] / neg["counts"]["n"], 4) if neg["counts"]["n"] else None,
        "neg_different_given_both": round(1 - neg["counts"]["same"] / neg["counts"]["both"], 4) if neg["counts"]["both"] else None,
        "neg_visible_B_rate": neg["visible_B_rate"],
    }
    retest = {}
    for c in list(conditions) + ["ctl_pos", "ctl_neg"]:
        a, b = load_judgments(out, c), load_judgments(out, c, retest_sub)
        common = sorted(set(a) & set(b))
        if common:
            same_ans = sum(a[k]["answer"] == b[k]["answer"] for k in common)
            cons = sum(tally(a[k]["answer"])["consistent"] == tally(b[k]["answer"])["consistent"] for k in common)
            retest[c] = {"n": len(common), "exact_answer_agreement": round(same_ans / len(common), 4),
                         "consistent_flag_agreement": round(cons / len(common), 4)}
    return {"metric_version": METRIC_VERSION, "missing_judgments": missing, "bootstrap": cfg["bootstrap"],
            "controls": ctl, "test_retest": retest, "cells": table}


# ------------------------------------------------------------------------------------------ CLI
def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True, type=Path, help="run root (units/, strips/, judgments/)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def cmd_units(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    stories = a.stories.split(",") if a.stories else default_stories(a.gt_dir, a.n_stories)
    gold = load_gold(a.gt_dir, stories)
    units, audit = build_units(gold, cfg)
    layout = ClipLayout.from_file(a.conditions, parse_vars(a.var))
    cc = cfg["control_condition"]
    presence = {s: presence_from_details(json.loads(layout.details(cc, s).read_text())) for s in stories}
    controls = build_controls(units, gold, presence, cfg)
    (a.out / "units").mkdir(parents=True, exist_ok=True)
    uw, cw = jsonl(units), jsonl(controls)
    (a.out / "units" / "units.jsonl").write_text(uw)
    (a.out / "units" / "controls.jsonl").write_text(cw)
    strata = Counter((u["stratum"], u["kind"], u["gap_bin"]) for u in units)
    manifest = {"metric_version": METRIC_VERSION, "seed": cfg["unit_seed"], "stories": stories,
                "n_units": len(units), "n_controls": Counter(c["stratum"] for c in controls),
                "strata": {"|".join(k): v for k, v in sorted(strata.items())}, "audit": audit,
                "units_sha256": hashlib.sha256(uw.encode()).hexdigest(),
                "controls_sha256": hashlib.sha256(cw.encode()).hexdigest(),
                "state_changed_units": sum(u["state_A"] != u["state_B"] for u in units)}
    (a.out / "units" / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


def _strip_job(args: tuple) -> dict | None:
    layout, out, cfg, (cond, story, gt, variant) = args
    clip = layout.clip(cond, story, gt)
    err = make_strip(clip, cfg["strip"]["variants"][variant], strip_path(out, cond, story, gt, variant),
                     int(cfg["strip"]["height"]), int(cfg["strip"]["jpeg_quality"]))
    return {"job": [cond, story, gt, variant], "clip": str(clip), "error": err} if err else None


def cmd_strips(a: argparse.Namespace) -> int:
    from multiprocessing import Pool

    cfg = load_config(a.config)
    layout = ClipLayout.from_file(a.conditions, parse_vars(a.var))
    units, controls = read_jsonl(a.out / "units" / "units.jsonl"), read_jsonl(a.out / "units" / "controls.jsonl")
    jobs = strip_jobs(units, controls, a.cond, cfg["control_condition"])
    print(f"strips needed: {len(jobs)}", flush=True)
    with Pool(a.procs) as pool:
        errs = [e for e in pool.imap(_strip_job, [(layout, a.out, cfg, j) for j in jobs], chunksize=8) if e]
    (a.out / "strips").mkdir(parents=True, exist_ok=True)
    (a.out / "strips" / "missing.jsonl").write_text(jsonl(errs))
    print(f"done; errors={len(errs)}", flush=True)
    return 0 if not errs else 1


def cmd_judge(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    units, controls = read_jsonl(a.out / "units" / "units.jsonl"), read_jsonl(a.out / "units" / "controls.jsonl")
    if a.only == "ctl":
        units = []
    elif a.only == "units":
        controls = []
    eps = [e for e in re.split(r"[,\s]+", a.endpoints) if e]
    if a.endpoint_glob:
        eps += sorted(p.read_text().strip() for p in Path("/").glob(a.endpoint_glob.lstrip("/")))
    stats = run_judge(judge_tasks(units, controls, a.cond, cfg["control_condition"]), a.out, eps, cfg["judge"],
                      per_endpoint=a.per_endpoint, sub=a.sub)
    print(json.dumps(stats), flush=True)
    return 0 if stats["err"] == 0 else 1


def cmd_aggregate(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    units = read_jsonl(a.out / "units" / "units.jsonl")
    manifest = json.loads((a.out / "units" / "manifest.json").read_text())
    res = aggregate(a.out, units, a.cond, manifest["stories"] if "stories" in manifest else
                    sorted({u["story"] for u in units}), cfg, baseline=a.baseline)
    res = {"judge": cfg["judge"], "units_manifest": manifest, **res}
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.json_out}; missing={res['missing_judgments']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    conds_default = "no_memory,text_frame,seg_uniform,seg_dinokey,seg_frame_clean,retr_oracle"

    p = sub.add_parser("units", help="build units + controls from gold")
    _common(p)
    p.add_argument("--gt-dir", required=True, type=Path)
    p.add_argument("--stories", default="", help="comma list (default: first --n-stories GT files)")
    p.add_argument("--n-stories", type=int, default=30)
    p.add_argument("--conditions", required=True, type=Path, help="conditions JSON (clip layout)")
    p.add_argument("--var", action="append", default=[], help="template variable name=value")
    p.set_defaults(fn=cmd_units)

    p = sub.add_parser("strips", help="extract 3-frame strips")
    _common(p)
    p.add_argument("--conditions", required=True, type=Path)
    p.add_argument("--var", action="append", default=[])
    p.add_argument("--cond", default=conds_default)
    p.add_argument("--procs", type=int, default=32)
    p.set_defaults(fn=cmd_strips)

    p = sub.add_parser("judge", help="pairwise VLM identity judge")
    _common(p)
    p.add_argument("--endpoints", default="", help="comma/space separated /v1 base URLs")
    p.add_argument("--endpoint-glob", default="", help="glob of files each holding one /v1 URL")
    p.add_argument("--cond", default=conds_default)
    p.add_argument("--only", choices=["", "ctl", "units"], default="")
    p.add_argument("--per-endpoint", type=int, default=12)
    p.add_argument("--sub", default="judgments", help="results subdir (judgments_rep1 = test-retest pass)")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("aggregate", help="pool judgments + paired bootstrap")
    _common(p)
    p.add_argument("--cond", default=conds_default)
    p.add_argument("--baseline", default="no_memory")
    p.add_argument("--json-out", required=True, type=Path)
    p.set_defaults(fn=cmd_aggregate)

    a = ap.parse_args(argv)
    if hasattr(a, "cond") and isinstance(a.cond, str):
        a.cond = [c for c in a.cond.split(",") if c]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
