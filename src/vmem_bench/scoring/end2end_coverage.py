"""VMem-Bench Track B — end-to-end generated-video judge (VLM, blinded mixed roster).

Scores a SUT's GENERATED video (per-segment) against the frozen per-segment GT built by
``assets/trackB/complete_gt.py`` (``gt_version: trackB-gt-2.0``). Unlike Track A
(``visual_coverage.py``, which judges the reference images the SUT *selected*), Track B
judges what the SUT actually *rendered*, so it measures memory→generation 落地 fidelity.

WHAT CHANGED vs the old ``shots``-based scorer (which read ``trackb_gt.py`` output):
  * GT is now a per-segment ``cast`` list with derived memory labels. Each cast entry
    carries ``op`` (introduce/recall/recall_after_gap/transform/persist), ``gap``,
    optional ``state`` (label + per-state ``appearance``; ``flashback`` marks a
    temporal-to-past-state look, judged for presence only), ``count`` (quantity memory), ``confusable_with``
    (false-friend target) and ``anchor`` (temporal/intent reference). ``forbidden`` entries
    carry ``reason`` + ``grounded_by`` (+ ``reference_indirect``). ``present_required`` /
    ``continuity`` / ``decoys`` are no longer pre-baked — we derive present from ``cast`` and
    sample decoys deterministically at score time.
  * Scoring is now PER-CAPABILITY (one number per memory ability, each with enough
    opportunities thanks to the builder's balance check) PLUS a gap-stratified recall decay
    curve. The headline recall is unweighted (pooled micro over character+prop present).

BLINDED MIXED ROSTER per segment = present(cast) ∪ forbidden ∪ false-friend targets ∪ decoys,
shuffled, with NO hint of which is which. This keeps precision genuinely <1 (the SUT can be
caught drawing forbidden/absent/look-alike-old entities) and makes avoidance measurable.

The VLM is the primary judge for presence / state / instance / count (no cropping — cropping a
generated video is error-prone). Per-entity JSON keyed by entity_id, strict parse + one retry
+ k-vote; parse failure is recorded, never silently False.

CLI::

    python -m vmem_bench.scoring.end2end_coverage \
        --gt   assets/trackB/gt/0001_lighthouse_keeper.json \
        --run  production/outputs/0001_lighthouse_keeper/memstrata/optCA \
        --out  production/outputs/0001_lighthouse_keeper/memstrata/optCA/_trackB_score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from vmem_bench.scoring.judge_service import (
    DEFAULT_API,
    DEFAULT_MODEL,
    PooledJudgeCaller,
    build_judge_api,
    call_judge,
)
from vmem_bench.scoring.visual_coverage import _txt, _vid

KINDS = ("character", "prop", "location")
RECALL_KINDS = ("character", "prop")           # headline recall pools these (location is near-free)
GAP_BUCKETS = ("0", "1-4", "5-29", ">=30")     # decay-curve strata
# blinded-roster decoy sampling: clearly-absent entities mixed in per segment
DECOY_MIN, DECOY_MAX = 3, 5


# ------------------------------------------------------------------------- VLM
def _judge_call(
    api: str | PooledJudgeCaller,
    model: str,
    content: list,
    temperature: float,
    max_tokens: int = 1536,
) -> str:
    return call_judge(api, model, content, temperature=temperature, max_tokens=max_tokens)


def _extract_json(s: str) -> dict | None:
    """Robust: grab the outermost {...} and json.loads; tolerate ``'`` and trailing commas."""
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    for cand in (blob, blob.replace("'", '"')):
        cand2 = re.sub(r",\s*([}\]])", r"\1", cand)  # drop trailing commas
        try:
            obj = json.loads(cand2)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _norm_present(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return "true"
    if s in ("false", "no", "0"):
        return "false"
    return "abstain"


def _norm_count(v) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"-?\d+", str(v))
    return int(m.group(0)) if m else None


def _seeded_sample(candidates: list[str], k: int, salt: str) -> list[str]:
    """Deterministic decoy sample: reproducible across machines (hashlib seed, not hash())."""
    if not candidates or k <= 0:
        return []
    seed = int(hashlib.md5(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pool = sorted(candidates)
    rng.shuffle(pool)
    return sorted(pool[:min(k, len(pool))])


def _gap_bucket(gap: int) -> str:
    if gap <= 0:
        return "0"
    if gap <= 4:
        return "1-4"
    if gap < 30:
        return "5-29"
    return ">=30"


# --------------------------------------------------------------- segment prep
def _prep_segment(seg: dict, ents: dict, story_id: str) -> dict:
    """Derive the blinded roster + per-entity judging plan from a GT segment.

    Returns a dict with the shuffled ``roster_ids``, per-id ``meta`` (kind + which
    questions to ask), and the resolved expectation sets used by :func:`score_segment`.
    The judge is NOT told which entities are present/forbidden/false-friend/decoy.
    """
    cast = seg.get("cast", [])
    cast_by_eid = {c["eid"]: c for c in cast}
    present = set(cast_by_eid)

    forb_entries = seg.get("forbidden", [])
    forbidden = {f["eid"] for f in forb_entries}

    # false-friend: try to catch the SUT rendering the OLD look-alike identity. Force the
    # confusable target into the roster (blinded) whenever it is NOT actually present.
    ff_entries = [c for c in cast if c.get("confusable_with") and c["op"] == "introduce"]
    ff_targets = {c["confusable_with"] for c in ff_entries} - present

    # decoys: clearly-absent entities (not present, not forbidden, not a forced FF target)
    all_ids = set(ents)
    cand = sorted(all_ids - present - forbidden - ff_targets)
    decoys = set(_seeded_sample(cand, DECOY_MAX, f"{story_id}:{seg['segment_id']}"))
    if len(decoys) < min(DECOY_MIN, len(cand)):
        decoys |= set(cand[:DECOY_MIN])

    roster_ids = sorted(present | forbidden | ff_targets | decoys)
    rng = random.Random(int.from_bytes(
        hashlib.md5(f"{story_id}:{seg['segment_id']}:roster".encode()).digest()[:8], "big"))
    rng.shuffle(roster_ids)

    # questions to ask per entity (state / instance / count) — only where GT expects them.
    # Only real state changes (transform/persist) are judged; flashback state entries
    # carry a past-state appearance for presence only, not a change judgment.
    state_expected = {c["eid"]: {"label": c["state"]["label"],
                                 "desc": c["state"].get("appearance", c["state"].get("desc", "")),
                                 "op": c["op"]}
                      for c in cast if c.get("state") and c["op"] in ("transform", "persist")}
    count_expected = {c["eid"]: int(c["count"]) for c in cast if c.get("count") is not None}
    lookalike_active = seg.get("lookalike_active", [])
    la_members = {m for la in lookalike_active for m in la.get("present_members", [])}
    la_feat: dict[str, str] = {}
    for la in lookalike_active:
        for eid, f in (la.get("features") or {}).items():
            la_feat[eid] = f

    meta = {}
    for eid in roster_ids:
        e = ents.get(eid, {"name": eid, "kind": "prop", "appearance": ""})
        meta[eid] = {"kind": e.get("kind", "prop"),
                     "ask_state": eid in state_expected,
                     "ask_instance": eid in la_members,
                     "ask_count": eid in count_expected}
    return {
        "roster_ids": roster_ids, "meta": meta,
        "present": present, "cast_by_eid": cast_by_eid,
        "forbidden_entries": forb_entries, "forbidden": forbidden,
        "ff_entries": ff_entries, "ff_targets": ff_targets, "decoys": decoys,
        "state_expected": state_expected, "count_expected": count_expected,
        "lookalike_active": lookalike_active, "la_feat": la_feat,
    }


def _base_appearance(e: dict) -> str:
    """A stateful entity has no top-level appearance — resolve its initial-state look
    (used for decoys/forbidden/first sight where no per-segment state is carried)."""
    states = e.get("states")
    if states:
        return states.get(e.get("initial_state"), next(iter(states.values()), ""))
    return e.get("appearance", "")


def _seg_appearance(eid: str, prep: dict, ents: dict) -> str:
    """Per-segment appearance: the entity's CURRENT (or flashback) state look if the
    cast entry carries one, else its base/initial appearance. One state, one look."""
    c = prep["cast_by_eid"].get(eid)
    if c and (c.get("state") or {}).get("appearance"):
        return c["state"]["appearance"]
    return _base_appearance(ents.get(eid, {}))


def _roster_text(prep: dict, ents: dict) -> str:
    lines = []
    for eid in prep["roster_ids"]:
        e = ents.get(eid, {"name": eid, "kind": "prop", "appearance": ""})
        desc = (_seg_appearance(eid, prep, ents) or "")[:120]
        extra = ""
        if prep["meta"][eid]["ask_state"]:
            extra += f"｜状态判定：若呈现为『{prep['state_expected'][eid]['desc']}』记 changed，否则 default"
        if prep["meta"][eid]["ask_count"]:
            extra += "｜数量判定：数一数画面里该实体有几个"
        if eid in prep["la_feat"]:
            extra += f"｜区分特征：{prep['la_feat'][eid]}"
        lines.append(f"- {eid} | {e['name']}（{e.get('kind','prop')}）：{desc}{extra}")
    return "\n".join(lines) or "（无）"


def _build_prompt(roster_txt: str, prompt: str, lookalike_active: list) -> str:
    la_note = ""
    if lookalike_active:
        pairs = "；".join("/".join(la["pair"]) for la in lookalike_active)
        la_note = (f"\n注意：清单中存在**相似易混实体对**（{pairs}）。对这些实体，"
                   "额外给出 instance_match=你认为画面里那个实例最像清单中的哪个 id。")
    return (
        "下面是一段视频片段，以及一份**候选实体清单**（其中**有些出现在视频里、有些没有**，"
        "不要假设都出现）。\n\n候选实体清单（带描述）:\n" + roster_txt +
        f"\n\n该片段剧情提示（仅供参考，不代表清单里都在场）：{prompt}\n\n"
        "请**逐条**判断清单里每个 entity_id：\n"
        "1) present：该实体是否**真的出现在这段视频画面里**（true / false / abstain）。"
        "看不清或无法确定填 \"abstain\"，不要猜。若为场景/地点(location)，present 指视频**发生在该场景**里。\n"
        "2) state：仅对标注了『状态判定』的实体，给 default 或 changed。\n"
        "3) count：仅对标注了『数量判定』的实体，给一个整数，表示画面里该实体出现的**数量**。\n"
        "4) instance_match：仅对相似易混实体，给你判断的最像的 id。\n"
        + la_note +
        "\n另给出 extra：画面里明显出现、但**不在清单里**的显著对象（自由文本，可空）。\n"
        '严格只输出一个 JSON 对象：{"items":[{"entity_id":"E1","present":true,"state":"changed","count":3,"instance_match":"E1"},'
        '{"entity_id":"E5","present":false}],"extra":["..."]}')


# --------------------------------------------------------------------- voting
def _vote(per_call_items: list[dict], roster_ids: list[str]) -> tuple[dict, float]:
    """Majority vote across k calls, per entity. Returns (voted[eid]->item, mean_agreement)."""
    voted, agrees = {}, []
    for eid in roster_ids:
        preses = [_norm_present(c.get(eid, {}).get("present", "abstain")) for c in per_call_items]
        cnt = Counter(preses)
        top, n = cnt.most_common(1)[0]
        agrees.append(n / len(preses) if preses else 1.0)
        states = [c[eid].get("state") for c in per_call_items if eid in c and c[eid].get("state")]
        insts = [c[eid].get("instance_match") for c in per_call_items if eid in c and c[eid].get("instance_match")]
        counts = [_norm_count(c[eid].get("count")) for c in per_call_items if eid in c and _norm_count(c[eid].get("count")) is not None]
        voted[eid] = {"present": top,
                      "state": Counter(states).most_common(1)[0][0] if states else None,
                      "instance_match": Counter(insts).most_common(1)[0][0] if insts else None,
                      "count": Counter(counts).most_common(1)[0][0] if counts else None}
    return voted, round(mean(agrees), 4) if agrees else 1.0


# --------------------------------------------------------------------- scoring
@dataclass
class SegScore:
    segment_id: str
    scene_id: str
    n_roster: int
    acc: dict = field(default_factory=lambda: defaultdict(int))  # summed tallies
    abstain: int = 0
    parse_error: bool = False
    vote_agreement: float | None = None
    score_ms: float | None = None


def _tally(seg: dict, prep: dict, voted: dict) -> tuple[SegScore, set, dict]:
    """Pure scoring: turn a per-entity ``voted`` dict into capability tallies.

    Separated from the VLM call so it can be unit-tested with simulated labels
    (perfect / adversarial) without a real video. Returns (SegScore, A, detail-A).
    """
    roster_ids, meta = prep["roster_ids"], prep["meta"]
    A = {eid for eid in roster_ids if voted[eid]["present"] == "true"}
    n_abstain = sum(1 for eid in roster_ids if voted[eid]["present"] == "abstain")
    sc = SegScore(seg["segment_id"], seg.get("scene_id", ""), len(roster_ids), abstain=n_abstain)
    acc = sc.acc
    cast_by_eid = prep["cast_by_eid"]

    # ---- recall (headline char+prop) + per-kind + gap-decay curve + op capabilities ----
    for eid, c in cast_by_eid.items():
        kd = meta.get(eid, {}).get("kind", "prop")
        hit = 1 if eid in A else 0
        acc[f"recall.{kd}.tot"] += 1
        acc[f"recall.{kd}.hit"] += hit
        op = c.get("op")
        gap = c.get("gap")
        # capability by op (char+prop only; location recall is near-free, tracked separately)
        if kd in RECALL_KINDS:
            if op == "introduce":
                acc["cap.first_appearance.tot"] += 1
                acc["cap.first_appearance.hit"] += hit
            elif op == "recall" and (gap or 0) < 30:
                acc["cap.continuity.tot"] += 1
                acc["cap.continuity.hit"] += hit
            if gap is not None and gap >= 30:
                acc["cap.long_gap.tot"] += 1
                acc["cap.long_gap.hit"] += hit
            # gap-decay curve (only re-appearances have a gap; introduce excluded)
            if gap is not None:
                b = _gap_bucket(gap)
                acc[f"gap.{b}.tot"] += 1
                acc[f"gap.{b}.hit"] += hit
        # ---- state: change (transform) vs persistence (persist) ----
        # flashback entries carry a PAST-state look for presence only — not a state judgment.
        if c.get("state") and not c["state"].get("flashback"):
            key = "state_change" if op == "transform" else "persist_state"
            # only scorable when the judge saw it present
            if eid in A:
                acc[f"cap.{key}.tot"] += 1
                if voted[eid]["state"] == "changed":
                    acc[f"cap.{key}.ok"] += 1
            else:
                acc[f"cap.{key}.missed"] += 1  # present-miss: can't judge state, tracked
        # ---- count memory ----
        if c.get("count") is not None and eid in A:
            acc["cap.count.tot"] += 1
            got = voted[eid]["count"]
            if got == int(c["count"]):
                acc["cap.count.exact"] += 1
            elif got is not None and abs(got - int(c["count"])) == 1:
                acc["cap.count.offby1"] += 1
        # ---- temporal / intent reference (name-free in SUT prompt) ----
        if (c.get("anchor") or {}).get("type") == "temporal":
            acc["cap.temporal.tot"] += 1
            acc["cap.temporal.hit"] += hit

    # ---- precision over STORY roster (decoys excluded), non-location ----
    story = prep["present"] | prep["forbidden"] | prep["ff_targets"]
    A_story_np = {e for e in (A & story) if meta.get(e, {}).get("kind") != "location"}
    acc["prec.hit"] += len(A_story_np & prep["present"])
    acc["prec.tot"] += len(A_story_np)

    # ---- avoidance, split by reason (deprecation / reference_indirect / lookalike_absent) ----
    for f in prep["forbidden_entries"]:
        eid = f["eid"]
        viol = 1 if eid in A else 0
        acc["avoid.opp"] += 1
        acc["avoid.viol"] += viol
        if f.get("reason") == "lookalike_absent":
            acc["cap.lookalike_absent.opp"] += 1
            acc["cap.lookalike_absent.viol"] += viol
        else:  # destroyed / deceased
            acc["cap.deprecation.opp"] += 1
            acc["cap.deprecation.viol"] += viol
            if f.get("reference_indirect"):
                acc["cap.reference_indirect.opp"] += 1
                acc["cap.reference_indirect.viol"] += viol

    # ---- look-alike instance discrimination (per present member) ----
    for la in prep["lookalike_active"]:
        pair = la["pair"]
        for eid in la.get("present_members", []):
            if eid in A:
                acc["cap.lookalike.tot"] += 1
                im = voted[eid]["instance_match"]
                if im == eid:
                    acc["cap.lookalike.correct"] += 1
                elif im in pair:
                    acc["cap.lookalike.wrong"] += 1

    # ---- false-friend (new look-alike recognised as itself, old one NOT hallucinated) ----
    # correct-id is over every FF introduce; confusion is only testable when the look-alike
    # target is ABSENT (if it is legitimately co-present, seeing it is correct, not confusion).
    for c in prep["ff_entries"]:
        eid, tgt = c["eid"], c["confusable_with"]
        acc["cap.false_friend.tot_id"] += 1
        if eid in A:
            acc["cap.false_friend.correct_id"] += 1
        if tgt not in prep["present"]:
            acc["cap.false_friend.tot_conf"] += 1
            if tgt in A:  # SUT rendered the OLD identity -> confusion
                acc["cap.false_friend.confused"] += 1

    # ---- decoy noise-floor FPR ----
    acc["decoy.tot"] += len(prep["decoys"])
    acc["decoy.fp"] += len(prep["decoys"] & A)
    acc["roster.tot"] += len(roster_ids)
    return sc, A


def score_segment(seg: dict, ents: dict, video: Path, api: str | PooledJudgeCaller, model: str, k: int,
                  vote_temp: float, story_id: str, prompt_text: str) -> tuple[SegScore, dict]:
    prep = _prep_segment(seg, ents, story_id)
    roster_ids = prep["roster_ids"]
    content = [_txt(_build_prompt(_roster_text(prep, ents), prompt_text, prep["lookalike_active"]))]
    content += [_vid(str(video.resolve()))]

    t0 = time.perf_counter()
    per_call, raws = [], []
    for j in range(max(1, k)):
        temp = 0.0 if j == 0 else vote_temp
        raw = _judge_call(api, model, content, temperature=temp)
        obj = _extract_json(raw)
        if obj is None:  # one strict retry
            raw = _judge_call(api, model, content + [_txt("上次输出无法解析。请**只**输出一个合法 JSON 对象，不要任何解释。")],
                              temperature=temp)
            obj = _extract_json(raw)
        raws.append(raw[:800])
        items = {}
        if obj and isinstance(obj.get("items"), list):
            for it in obj["items"]:
                if isinstance(it, dict) and it.get("entity_id"):
                    items[str(it["entity_id"])] = it
        per_call.append(items)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)

    parse_error = all(len(c) == 0 for c in per_call)
    voted, agreement = _vote(per_call, roster_ids)
    sc, A = _tally(seg, prep, voted)
    sc.parse_error = parse_error
    sc.vote_agreement = agreement
    sc.score_ms = ms

    detail = {"segment_id": seg["segment_id"], "scene_id": seg.get("scene_id", ""),
              "roster": roster_ids, "present": sorted(prep["present"]),
              "forbidden": sorted(prep["forbidden"]), "ff_targets": sorted(prep["ff_targets"]),
              "decoys": sorted(prep["decoys"]), "A": sorted(A),
              "voted": voted, "agreement": agreement, "parse_error": parse_error, "raw": raws}
    return sc, detail


# ------------------------------------------------------------------------ run
def _wilson(k: int, n: int) -> list[float] | None:
    """95% Wilson interval for a proportion (avoidance opportunities are few)."""
    if n == 0:
        return None
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4)]


def _seg_index(segment_id: str) -> int:
    m = re.search(r"(\d+)", segment_id)
    return int(m.group(1)) if m else -1


def _video_for(run: Path, seg_id: str, prog: dict) -> Path | None:
    idx = _seg_index(seg_id)
    for cand in (run / "review" / "segments" / f"{seg_id}.mp4",
                 run / "review" / "segments" / f"seg_{idx:03d}.mp4"):
        if cand.is_file():
            return cand
    pv = prog.get(idx)
    if pv and Path(pv).is_file():
        return Path(pv)
    return None


def _ratio(num_key: str, den_key: str, acc: dict) -> float | None:
    den = acc.get(den_key, 0)
    return round(acc.get(num_key, 0) / den, 4) if den else None


def run(gt_path: Path, run_dir: Path, out_dir: Path, api: str | PooledJudgeCaller, model: str, k: int,
        vote_temp: float, prompts_path: Path | None = None, limit: int | None = None,
        workers: int = 1):
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if gt.get("gt_version", "").startswith("trackB-gt-0"):
        raise SystemExit(
            f"{gt_path} is an OLD shots-based GT (gt_version={gt.get('gt_version')}). "
            "Rebuild with assets/trackB/complete_gt.py (gt_version trackB-gt-2.0).")
    ents = gt["entities"]
    story_id = gt["story_id"]
    segments = gt["segments"]
    if limit:
        segments = segments[:limit]

    # SUT-facing prompts: prefer the frozen sut_prompts file, else fall back to GT action prose.
    prompt_by_seg: dict[str, str] = {}
    if prompts_path and prompts_path.is_file():
        pj = json.loads(prompts_path.read_text(encoding="utf-8"))
        for s in pj.get("segments", pj.get("prompts", [])):
            prompt_by_seg[str(s.get("segment_id"))] = str(s.get("prompt", ""))

    prog = {}
    pj = run_dir / "progress.json"
    if pj.is_file():
        for c in json.loads(pj.read_text()).get("chunks", []):
            prog[int(c["chunk_id"])] = c.get("video")

    def _score_one(seg: dict) -> tuple[str, SegScore | None, dict | None]:
        seg_id = seg["segment_id"]
        video = _video_for(run_dir, seg_id, prog)
        if video is None:
            print(f"{seg_id}: NO VIDEO (skipped)")
            return seg_id, None, None
        prompt_text = prompt_by_seg.get(seg_id, seg.get("action", ""))
        workload_ctx = (
            api.workload(
                job_id=f"trackB:{story_id}:{run_dir.name}",
                movie_id=story_id,
                dataset="trackB",
                segment_id=seg_id,
                run_dir=str(run_dir),
            )
            if hasattr(api, "workload")
            else nullcontext()
        )
        with workload_ctx:
            sc, det = score_segment(seg, ents, video, api, model, k, vote_temp, story_id, prompt_text)
        return seg_id, sc, det

    def _print_segment(seg: dict, sc: SegScore) -> None:
        a = sc.acc
        rc = _ratio("recall.character.hit", "recall.character.tot", a)
        rp = _ratio("recall.prop.hit", "recall.prop.tot", a)
        print(f"{sc.segment_id} [{sc.scene_id}] {'|'.join(seg.get('memory_probes', []))[:40]}: "
              f"rec(c={rc},p={rp}) prec={a.get('prec.hit',0)}/{a.get('prec.tot',0)} "
              f"avoid={a.get('avoid.viol',0)}v/{a.get('avoid.opp',0)} "
              f"decoyFP={a.get('decoy.fp',0)}/{a.get('decoy.tot',0)} "
              f"agree={sc.vote_agreement} {'PARSE_ERR' if sc.parse_error else ''}")

    if workers <= 0:
        workers = int(getattr(api, "size", 1) or 1)
    workers = max(1, min(int(workers), max(1, len(segments))))
    scored: dict[str, tuple[SegScore, dict]] = {}
    seg_by_id = {seg["segment_id"]: seg for seg in segments}
    if workers == 1:
        for seg in segments:
            seg_id, sc, det = _score_one(seg)
            if sc is None or det is None:
                continue
            scored[seg_id] = (sc, det)
            _print_segment(seg, sc)
    else:
        print(f"[end2end_coverage] scoring {len(segments)} segments with workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_seg = {pool.submit(_score_one, seg): seg for seg in segments}
            for future in as_completed(future_to_seg):
                seg_id, sc, det = future.result()
                if sc is None or det is None:
                    continue
                scored[seg_id] = (sc, det)
                _print_segment(seg_by_id[seg_id], sc)

    ordered_ids = [seg["segment_id"] for seg in segments if seg["segment_id"] in scored]
    scores = [scored[seg_id][0] for seg_id in ordered_ids]
    details = [scored[seg_id][1] for seg_id in ordered_ids]

    # ---- pooled aggregation ----
    G: dict = defaultdict(int)
    for s in scores:
        for kk, vv in s.acc.items():
            G[kk] += vv

    recall_by_kind = {kd: _ratio(f"recall.{kd}.hit", f"recall.{kd}.tot", G) for kd in KINDS}
    cp_hit = G.get("recall.character.hit", 0) + G.get("recall.prop.hit", 0)
    cp_tot = G.get("recall.character.tot", 0) + G.get("recall.prop.tot", 0)
    recall_cp = round(cp_hit / cp_tot, 4) if cp_tot else None
    prec = _ratio("prec.hit", "prec.tot", G)
    f1 = None
    if recall_cp is not None and prec is not None:
        f1 = 0.0 if (recall_cp + prec) == 0 else round(2 * recall_cp * prec / (recall_cp + prec), 4)
    avoid_ok = round(1 - G.get("avoid.viol", 0) / G["avoid.opp"], 4) if G.get("avoid.opp") else None

    def _avoid(cap: str) -> dict:
        opp, viol = G.get(f"cap.{cap}.opp", 0), G.get(f"cap.{cap}.viol", 0)
        return {"avoidance_ok": round(1 - viol / opp, 4) if opp else None,
                "violations": viol, "opportunities": opp,
                "wilson95_ok": _wilson(opp - viol, opp) if opp else None}

    capabilities = {
        "first_appearance": {"recall": _ratio("cap.first_appearance.hit", "cap.first_appearance.tot", G),
                             "hit": G.get("cap.first_appearance.hit", 0), "tot": G.get("cap.first_appearance.tot", 0)},
        "continuity": {"recall": _ratio("cap.continuity.hit", "cap.continuity.tot", G),
                       "hit": G.get("cap.continuity.hit", 0), "tot": G.get("cap.continuity.tot", 0)},
        "long_gap_reappearance": {"recall": _ratio("cap.long_gap.hit", "cap.long_gap.tot", G),
                                  "hit": G.get("cap.long_gap.hit", 0), "tot": G.get("cap.long_gap.tot", 0)},
        "state_change": {"correct": _ratio("cap.state_change.ok", "cap.state_change.tot", G),
                         "ok": G.get("cap.state_change.ok", 0), "tot": G.get("cap.state_change.tot", 0),
                         "present_missed": G.get("cap.state_change.missed", 0)},
        "persist_state": {"correct": _ratio("cap.persist_state.ok", "cap.persist_state.tot", G),
                          "ok": G.get("cap.persist_state.ok", 0), "tot": G.get("cap.persist_state.tot", 0),
                          "present_missed": G.get("cap.persist_state.missed", 0)},
        "lookalike_disambiguation": {
            "correct_rate": _ratio("cap.lookalike.correct", "cap.lookalike.tot", G),
            "wrong_rate": _ratio("cap.lookalike.wrong", "cap.lookalike.tot", G),
            "correct": G.get("cap.lookalike.correct", 0), "wrong": G.get("cap.lookalike.wrong", 0),
            "tot": G.get("cap.lookalike.tot", 0)},
        "reference_indirect": _avoid("reference_indirect"),
        "deprecation_avoidance": _avoid("deprecation"),
        "lookalike_absent_avoidance": _avoid("lookalike_absent"),
        "count_memory": {
            "exact_rate": _ratio("cap.count.exact", "cap.count.tot", G),
            "off_by_one_rate": _ratio("cap.count.offby1", "cap.count.tot", G),
            "exact": G.get("cap.count.exact", 0), "off_by_one": G.get("cap.count.offby1", 0),
            "tot": G.get("cap.count.tot", 0)},
        "false_friend": {
            "correct_id_rate": _ratio("cap.false_friend.correct_id", "cap.false_friend.tot_id", G),
            "confusion_rate": _ratio("cap.false_friend.confused", "cap.false_friend.tot_conf", G),
            "correct_id": G.get("cap.false_friend.correct_id", 0),
            "confused": G.get("cap.false_friend.confused", 0),
            "tot_id": G.get("cap.false_friend.tot_id", 0),
            "tot_confusion": G.get("cap.false_friend.tot_conf", 0)},
        "temporal_reference": {"recall": _ratio("cap.temporal.hit", "cap.temporal.tot", G),
                               "hit": G.get("cap.temporal.hit", 0), "tot": G.get("cap.temporal.tot", 0)},
    }
    recall_by_gap = {b: {"recall": _ratio(f"gap.{b}.hit", f"gap.{b}.tot", G),
                         "hit": G.get(f"gap.{b}.hit", 0), "tot": G.get(f"gap.{b}.tot", 0)}
                     for b in GAP_BUCKETS}

    summary = {
        "story_id": story_id, "run_dir": str(run_dir), "model": model,
        "metric_version": "trackB-end2end-1.0", "gt_version": gt.get("gt_version"),
        "k_votes": k, "vote_temp": vote_temp,
        "n_segments_scored": len(scores),
        "n_parse_errors": sum(1 for s in scores if s.parse_error),
        # HEADLINE (unweighted, pooled micro)
        "headline": {
            "recall_char_prop": recall_cp,
            "precision": prec,
            "f1": f1,
            "avoidance_ok": avoid_ok,
            "avoidance_detail": {"violations": G.get("avoid.viol", 0), "opportunities": G.get("avoid.opp", 0)},
        },
        # PER-CAPABILITY (each memory ability scored separately)
        "capabilities": capabilities,
        # gap-stratified recall decay curve (char+prop re-appearances)
        "recall_by_gap": recall_by_gap,
        "recall_by_kind": recall_by_kind,
        # noise floor / reliability
        "noise_floor": {
            "decoy_fpr": _ratio("decoy.fp", "decoy.tot", G),
            "decoy_detail": {"false_present": G.get("decoy.fp", 0), "tot": G.get("decoy.tot", 0)},
            "vote_self_consistency": round(mean([s.vote_agreement for s in scores if s.vote_agreement is not None]), 4) if scores else None,
            "abstain_rate": round(sum(s.abstain for s in scores) / max(1, G.get("roster.tot", 0)), 4),
        },
        "score_ms_mean": round(mean([s.score_ms for s in scores if s.score_ms]), 1) if scores else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score.json").write_text(json.dumps(
        {"summary": summary,
         "per_segment": [{"segment_id": s.segment_id, "scene_id": s.scene_id,
                          "n_roster": s.n_roster, "abstain": s.abstain,
                          "parse_error": s.parse_error, "vote_agreement": s.vote_agreement,
                          "score_ms": s.score_ms, "acc": dict(s.acc)} for s in scores]},
        ensure_ascii=False, indent=2))
    (out_dir / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print("\n=== TRACK B SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSCORE:   {out_dir}/score.json")
    print(f"DETAILS: {out_dir}/details.json")
    print("TRACKB_END2END_DONE")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VMem-Bench Track B end-to-end generated-video judge")
    ap.add_argument("--gt", required=True, type=Path, help="complete_gt.py output (gt_version trackB-gt-2.0)")
    ap.add_argument("--run", required=True, type=Path, help="SUT run dir (progress.json + review/segments)")
    ap.add_argument("--prompts", type=Path, default=None,
                    help="frozen SUT prompts json (assets/trackB/en/sut_prompts/<story>_name_anchored.json)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--api-list", default="", help="comma/whitespace separated /v1 base URLs or chat endpoints")
    ap.add_argument("--fleet", action="store_true", help="resolve online VLM endpoints from the annotation fleet registry")
    ap.add_argument("--fleet-root", type=Path, default=None)
    ap.add_argument("--fleet-role", default="reviewer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=1, help="judge votes for self-consistency (k>1 needs --vote-temp>0)")
    ap.add_argument("--vote-temp", type=float, default=0.0, help="temperature for votes 2..k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="segment workers; 0 = endpoint-pool size when pooled, otherwise 1",
    )
    ap.add_argument(
        "--endpoint-slots",
        type=int,
        default=1,
        help="concurrent judge requests per endpoint (default 1). Raise ONLY when the "
             "vLLM replicas run a matching MAX_NUM_SEQS; scheduling-only, does not "
             "change prompts, sampling, or metrics",
    )
    a = ap.parse_args(argv)
    out = a.out or (a.run / "_trackB_score")
    api = build_judge_api(
        api=a.api,
        api_list=a.api_list,
        use_fleet=a.fleet,
        fleet_root=a.fleet_root,
        fleet_role=a.fleet_role,
        model=a.model,
        stage="trackB_end2end_coverage",
        endpoint_slots=a.endpoint_slots,
    )
    run(a.gt, a.run, out, api, a.model, a.k, a.vote_temp, a.prompts, a.limit, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
