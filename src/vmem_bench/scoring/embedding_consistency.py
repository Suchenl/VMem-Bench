"""VMem-Bench Track B embedding appearance consistency (metric ``trackB-embed-1.0``).

Deterministic companion to the VLM pairwise identity judge (``identity_consistency``,
``trackB-identity-1.0``), on the same units, strata, clip layout and bootstrap.

For each unit side (A = previous appearance, B = current clip of the same condition) three frames
are decoded at 15/50/85% of the clip (controls: disjoint 10/30/50% vs 55/75/95% sets):

* characters and props (``dino_crop``): GroundingDINO (grounding-dino-base) localizes the entity with
  the text query ``"<name>, <first 12 words of the gold description>."``; per frame the top box
  (score = max over text tokens of sigmoid(logit)) counts as a detection if score >= box_threshold;
  the padded crop is embedded with the pinned ``dinov3`` scoring provider (CLS, L2-normalized).
* characters additionally (``face``): the pinned ``arcface`` provider (InsightFace buffalo_l) on the
  character crop; the largest face found inside the crop is the character's face in that frame.
* locations: whole frame through the pinned ``megaloc`` provider (VPR, primary, ``location_megaloc``)
  and through ``dinov3`` (secondary, ``location_dino``). Whole-frame similarity is NOT used for
  characters/props: memory that copies a reference frame's composition would inflate it without the
  entity being the same.

similarity(A, B) = max over frame pairs (a in A, b in B) that both carry the modality of cos(a, b).
A unit is ``consistent`` for a modality iff both sides carry it and similarity >= the modality's
threshold. Thresholds are chosen once (``calibrate``: Youden-optimal on positive vs negative control
similarities) and stored in the metric config before any condition is aggregated.

Controls per modality (control condition, frozen presence judge): positive = the same clip, two
disjoint frame sets; negative = entity X in a clip where it is present vs a different same-kind entity
Y of the same story in a clip where Y is present and X is not (each side localized with its own query).
Characters/props reuse the VLM identity controls; locations get their own.

CLI (``python -m vmem_bench.scoring.embedding_consistency <cmd>``): ``units``, ``extract`` (GPU,
sharded, resume-safe), ``calibrate``, ``aggregate``. See docs/benchmark/trackB_scoring.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vmem_bench.scoring import identity_consistency as ic

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "trackb_embed.json"
METRIC_VERSION = "trackB-embed-1.0"
MODALITIES = ("dino_crop", "face", "location_megaloc", "location_dino")
FIELDS = ("n", "det_A", "det_B", "both", "sim_sum", "cons")
BOOT_METRICS = ("consistent_rate", "mean_similarity", "consistent_given_both")


def load_config(path: Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def identity_config(cfg: dict) -> dict:
    return ic.load_config(CONFIG_DIR / cfg["identity_config"])


# ------------------------------------------------------------------------------------------ units
def build_all(gold: dict[str, dict], presence: dict[str, dict[int, set[str]]], cfg: dict
              ) -> tuple[list[dict], list[dict], dict]:
    """Character/prop units + controls identical to the VLM metric; location units/controls added
    with their own seeds so the character/prop sets do not change."""
    icfg = identity_config(cfg)
    units, audit = ic.build_units(gold, icfg)
    controls = ic.build_controls(units, gold, presence, icfg)
    loc_units, loc_audit = ic.build_units(gold, icfg, kinds=["location"],
                                          rng=random.Random(int(cfg["location_unit_seed"])))
    loc_controls = ic.build_controls(loc_units, gold, presence, icfg, kinds=["location"],
                                     seed=int(cfg["location_control_seed"]))
    for c in loc_controls:
        c["uid"] = "loc" + c["uid"]
    # The shared controls hold few character negatives; faces are character-only, so add a
    # character-only control set (own seed) to calibrate the face and character-crop thresholds.
    chr_controls = ic.build_controls(units, gold, presence, icfg, kinds=["character"],
                                     seed=int(cfg["character_control_seed"]))
    for c in chr_controls:
        c["uid"] = "chr" + c["uid"]
    audit = dict(Counter(audit) + Counter(loc_audit))
    return units + loc_units, controls + loc_controls + chr_controls, audit


def short_desc(text: str, n_words: int) -> str:
    words = str(text or "").replace("\n", " ").split()
    return " ".join(words[:n_words]).rstrip(",.;:")


def grounding_query(name: str, desc: str, gcfg: dict) -> str:
    return gcfg["query"].format(name=name or "", short_desc=short_desc(desc, int(gcfg["short_desc_words"]))).lower()


def sides(u: dict) -> list[dict]:
    """The two views a unit/control needs: (clip seg, frame set, entity, query description)."""
    if u["stratum"] == "ctl_pos":
        return [{"gt": u["cur_gt"], "variant": "posA", "eid": u["eid"], "name": u["name"], "desc": u["desc_B"]},
                {"gt": u["cur_gt"], "variant": "posB", "eid": u["eid"], "name": u["name"], "desc": u["desc_B"]}]
    if u["stratum"] == "ctl_neg":
        return [{"gt": u["prev_gt"], "variant": "main", "eid": u["eid"], "name": u["name"], "desc": u["desc_A"]},
                {"gt": u["cur_gt"], "variant": "main", "eid": u["eid_B"], "name": u.get("name_B"),
                 "desc": u.get("desc_Bent", "")}]
    return [{"gt": u["prev_gt"], "variant": "main", "eid": u["eid"], "name": u["name"], "desc": u["desc_A"]},
            {"gt": u["cur_gt"], "variant": "main", "eid": u["eid"], "name": u["name"], "desc": u["desc_B"]}]


def attach_neg_b(controls: list[dict], gold: dict[str, dict]) -> None:
    """Negative controls localize Y in B with Y's own name/description."""
    for c in controls:
        if c["stratum"] == "ctl_neg":
            ent = gold[c["story"]]["entities"][c["eid_B"]]
            c["name_B"] = ent.get("name")
            c["desc_Bent"] = ic.describe(ent, ent.get("initial_state"))


def clip_jobs(units: list[dict], controls: list[dict], conditions: list[str], control_cond: str) -> list[dict]:
    """One job per (condition, story, gt seg, frame set): the entities to localize there."""
    jobs: dict[tuple, dict] = {}

    def need(cond: str, story: str, s: dict, kind: str) -> None:
        key = (cond, story, s["gt"], s["variant"])
        j = jobs.setdefault(key, {"cond": cond, "story": story, "gt": s["gt"], "variant": s["variant"],
                                  "entities": {}, "faces": False})
        if kind in ("character", "prop"):
            j["entities"][s["eid"]] = {"name": s["name"], "desc": s["desc"], "kind": kind}
            j["faces"] = j["faces"] or kind == "character"

    for u in units:
        for c in conditions:
            for s in sides(u):
                need(c, u["story"], s, u["kind"])
    for u in controls:
        for s in sides(u):
            need(control_cond, u["story"], s, u["kind"])
    return [jobs[k] for k in sorted(jobs)]


def clip_result_path(out: Path, cond: str, story: str, gt: int, variant: str) -> Path:
    return Path(out) / "embed" / "clips" / cond / story / f"gt{gt:03d}_{variant}.npz"


# ------------------------------------------------------------------------------------------ models
def resolve_model(root: str | Path, model_id: str) -> Path:
    """Local dir of a model under PUBLIC_MODELS_ROOT; accepts HF-cache layouts (snapshots/<rev>)."""
    p = Path(root) / model_id
    if (p / "config.json").is_file():
        return p
    snaps = sorted((p / "snapshots").glob("*/config.json"))
    if snaps:
        return snaps[-1].parent
    raise FileNotFoundError(f"model {model_id} not found under {root}")


class Extractor:
    """GroundingDINO localization + the pinned scoring providers of ``vmem_bench.scoring.embedder``
    (``dinov3`` crops and frames, ``arcface`` faces inside character crops, ``megaloc`` places)."""

    def __init__(self, cfg: dict, models_root: str, device: str, face_root: str | None) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from vmem_bench.scoring.embedder import build_scoring_embedder

        self.cfg, self.device, self.torch = cfg, device, torch
        gpath = resolve_model(models_root, cfg["grounding"]["model"])
        self.gproc = AutoProcessor.from_pretrained(gpath)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gpath).to(device).eval()
        self.dino = build_scoring_embedder("dinov3")
        self.dino._device = device
        self.megaloc = build_scoring_embedder("megaloc")
        self.megaloc._device = device
        self.face = None
        self.face_error = None
        try:
            from vmem_bench.scoring.embedder import ArcFaceScoringEmbedder

            gpu = int(device.split(":")[-1]) if device.startswith("cuda") else -1
            self.face = ArcFaceScoringEmbedder(cfg["face"]["pack"], root=face_root or None,
                                               det_size=int(cfg["face"]["det_size"]), ctx_id=gpu)
            self.face._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            self.face, self.face_error = None, repr(exc)
        self.provenance = {"grounding_dir": str(gpath), "dinov3": self.dino.name, "megaloc": self.megaloc.name,
                           "arcface": getattr(self.face, "name", None), "face_error": self.face_error}

    def embed(self, images_rgb: list) -> Any:
        import numpy as np
        from PIL import Image

        return np.asarray(self.dino.embed_pils([Image.fromarray(im) for im in images_rgb]), dtype=np.float32)

    def place(self, images_rgb: list) -> Any:
        import numpy as np
        from PIL import Image

        return np.asarray(self.megaloc.embed_pils([Image.fromarray(im) for im in images_rgb]), dtype=np.float32)

    def ground(self, frames_rgb: list, query: str) -> tuple[Any, Any]:
        """Top box per frame: (scores[k], boxes[k,4] as x0,y0,x1,y1 pixels)."""
        import numpy as np

        with self.torch.no_grad():
            inp = self.gproc(images=frames_rgb, text=[query] * len(frames_rgb), return_tensors="pt").to(self.device)
            outp = self.gdino(**inp)
            logits = outp.logits.sigmoid().float()          # [k, Q, L]
            am = inp["attention_mask"].bool()
            mask = self.torch.zeros(logits.shape[0], logits.shape[-1], dtype=self.torch.bool, device=logits.device)
            mask[:, : min(am.shape[1], logits.shape[-1])] = am[:, : logits.shape[-1]]
            tok = logits.masked_fill(~mask[:, None, :], 0.0).amax(-1)  # [k, Q]
            best = tok.argmax(-1)
            scores = tok.gather(1, best[:, None])[:, 0]
            boxes = outp.pred_boxes.float()[self.torch.arange(len(frames_rgb)), best]  # cx,cy,w,h
        out = []
        for f, b in zip(frames_rgb, boxes.cpu().numpy()):
            h, w = f.shape[:2]
            cx, cy, bw, bh = b
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        return scores.cpu().numpy().astype(np.float32), np.asarray(out, dtype=np.float32)

    def crops(self, frames: list, boxes: Any) -> list:
        g = self.cfg["grounding"]
        pad, mn = float(g["crop_pad"]), int(g["min_crop_px"])
        out = []
        for f, (x0, y0, x1, y1) in zip(frames, boxes):
            h, w = f.shape[:2]
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            a0, b0 = int(max(0, x0 - px)), int(max(0, y0 - py))
            a1, b1 = int(min(w, x1 + px)), int(min(h, y1 + py))
            if a1 - a0 < mn or b1 - b0 < mn:
                a0, b0, a1, b1 = 0, 0, w, h
            out.append(f[b0:b1, a0:a1])
        return out

    def faces(self, crops_bgr: list) -> tuple[Any, Any]:
        """ArcFace on each character crop: (found[k] bool, emb[k,512]; zeros where no face)."""
        import numpy as np

        found, embs = [], []
        for c in crops_bgr:
            v = self.face.embed_bgr(c, min_det_score=float(self.cfg["face"]["min_det_score"])) if self.face else None
            found.append(v is not None)
            embs.append(np.asarray(v if v is not None else np.zeros(512), dtype=np.float32))
        return np.asarray(found), np.stack(embs)

    def run_job(self, job: dict, clip: Path) -> dict:
        import cv2
        import numpy as np

        frames_bgr = ic.read_frames(clip, self.cfg["frames"][job["variant"]]) if clip.is_file() else None
        if frames_bgr is None:
            return {"error": np.asarray("missing or unreadable clip")}
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        res: dict[str, Any] = {"frame_dino": self.embed(rgb), "frame_megaloc": self.place(rgb),
                               "frame_hw": np.asarray(rgb[0].shape[:2])}
        for eid, e in sorted(job["entities"].items()):
            q = grounding_query(e["name"], e["desc"], self.cfg["grounding"])
            scores, boxes = self.ground(rgb, q)
            res[f"score__{eid}"], res[f"box__{eid}"] = scores, boxes
            res[f"crop__{eid}"] = self.embed(self.crops(rgb, boxes))
            if e["kind"] == "character":
                res[f"faceok__{eid}"], res[f"face__{eid}"] = self.faces(self.crops(frames_bgr, boxes))
        return res


def run_extract(jobs: list[dict], layout: ic.ClipLayout, out: Path, cfg: dict, extractor: Any,
                shard: int = 0, nshards: int = 1) -> dict:
    """Resume-safe: a job whose npz exists is skipped. Deterministic shard = index mod nshards."""
    import numpy as np

    stats = Counter()
    for i, job in enumerate(jobs):
        if i % nshards != shard:
            continue
        p = clip_result_path(out, job["cond"], job["story"], job["gt"], job["variant"])
        if p.is_file():
            stats["skipped"] += 1
            continue
        res = extractor.run_job(job, layout.clip(job["cond"], job["story"], job["gt"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, **res)
        os.replace(tmp, p)
        stats["error" if "error" in res else "done"] += 1
    return dict(stats)


# ------------------------------------------------------------------------------------------ similarity
def side_vectors(npz: dict, eid: str | None, modality: str, cfg: dict) -> list:
    """Per-frame embeddings of one side that carry the modality (possibly empty)."""
    if npz is None or "error" in npz:
        return []
    if modality == "location_megaloc":
        return list(npz["frame_megaloc"])
    if modality == "location_dino":
        return list(npz["frame_dino"])
    thr = float(cfg["grounding"]["box_threshold"])
    scores = npz.get(f"score__{eid}")
    if scores is None:
        return []
    if modality == "dino_crop":
        crops = npz[f"crop__{eid}"]
        return [crops[i] for i in range(len(scores)) if scores[i] >= thr]
    ok, faces = npz.get(f"faceok__{eid}"), npz.get(f"face__{eid}")
    if ok is None:
        return []
    return [faces[i] for i in range(len(scores)) if scores[i] >= thr and bool(ok[i])]


def pair_similarity(va: list, vb: list) -> float | None:
    """max over frame pairs of cosine (vectors are L2-normalized)."""
    if not va or not vb:
        return None
    import numpy as np

    return float(np.max(np.asarray(va) @ np.asarray(vb).T))


def unit_modalities(u: dict, cfg: dict) -> list[str]:
    return [m for m, spec in cfg["modalities"].items() if u["kind"] in spec["kinds"]]


class NpzCache:
    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self._c: dict[Path, Any] = {}

    def get(self, cond: str, story: str, gt: int, variant: str) -> dict | None:
        import numpy as np

        p = clip_result_path(self.out, cond, story, gt, variant)
        if p not in self._c:
            self._c[p] = dict(np.load(p)) if p.is_file() else None
        return self._c[p]


def unit_record(u: dict, cond: str, modality: str, cache: NpzCache, cfg: dict) -> dict:
    sa, sb = sides(u)
    va = side_vectors(cache.get(cond, u["story"], sa["gt"], sa["variant"]), sa["eid"], modality, cfg)
    vb = side_vectors(cache.get(cond, u["story"], sb["gt"], sb["variant"]), sb["eid"], modality, cfg)
    return {"det_A": bool(va), "det_B": bool(vb), "sim": pair_similarity(va, vb)}


# ------------------------------------------------------------------------------------------ calibration
def roc_auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUC (ties count 1/2)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(wins / (len(pos) * len(neg)), 4)


def youden_threshold(pos: list[float], neg: list[float]) -> dict:
    """Threshold t maximizing TPR - FPR with rule sim >= t; ties -> the lowest such t."""
    best = None
    for t in sorted(set(pos) | set(neg)):
        tpr = sum(p >= t for p in pos) / len(pos)
        fpr = sum(n >= t for n in neg) / len(neg)
        j = tpr - fpr
        if best is None or j > best["youden_j"] + 1e-12:
            best = {"threshold": round(t, 6), "tpr": round(tpr, 4), "fpr": round(fpr, 4), "youden_j": round(j, 4)}
    return best or {}


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    q = lambda f: round(s[min(len(s) - 1, int(f * (len(s) - 1)))], 4)  # noqa: E731
    return {"n": len(s), "mean": round(sum(s) / len(s), 4), "p10": q(0.1), "p25": q(0.25), "p50": q(0.5),
            "p75": q(0.75), "p90": q(0.9)}


def calibrate(controls: list[dict], cache: NpzCache, cfg: dict, control_cond: str) -> dict:
    cal = {}
    for m in MODALITIES:
        kinds = cfg["modalities"][m]["kinds"]
        pos, neg, cnt = [], [], Counter()
        for u in controls:
            if u["kind"] not in kinds:
                continue
            r = unit_record(u, control_cond, m, cache, cfg)
            cnt[f"{u['stratum']}_n"] += 1
            cnt[f"{u['stratum']}_both"] += r["sim"] is not None
            if r["sim"] is not None:
                (pos if u["stratum"] == "ctl_pos" else neg).append(r["sim"])
        y = youden_threshold(pos, neg) if pos and neg else {}
        cal[m] = {"counts": dict(cnt), "auc": roc_auc(pos, neg), "youden": y,
                  "pos": _quantiles(pos), "neg": _quantiles(neg),
                  "neg_below_threshold": (round(sum(n < y["threshold"] for n in neg) / len(neg), 4) if y else None)}
    return cal


# ------------------------------------------------------------------------------------------ aggregate
def metrics(t: dict) -> dict:
    r = lambda a, b: (round(a / b, 4) if b else None)  # noqa: E731
    return {"consistent_rate": r(t["cons"], t["n"]), "consistent_given_both": r(t["cons"], t["both"]),
            "mean_similarity": r(t["sim_sum"], t["both"]), "detected_A_rate": r(t["det_A"], t["n"]),
            "detected_B_rate": r(t["det_B"], t["n"]), "both_detected_rate": r(t["both"], t["n"])}


def aggregate(units: list[dict], conditions: list[str], stories: list[str], cache: NpzCache, cfg: dict,
              *, baseline: str = "no_memory") -> dict:
    thr = cfg["thresholds"]
    if any(thr[m] is None for m in MODALITIES):
        raise RuntimeError("thresholds not calibrated: run `calibrate` first (thresholds are fixed on controls)")
    out = {}
    for m in MODALITIES:
        mu = [u for u in units if m in unit_modalities(u, cfg)]
        if not mu:
            continue
        per = {c: defaultdict(lambda: defaultdict(dict)) for c in conditions}
        for c in conditions:
            for u in mu:
                r = unit_record(u, c, m, cache, cfg)
                both = r["sim"] is not None
                t = {"n": 1, "det_A": int(r["det_A"]), "det_B": int(r["det_B"]), "both": int(both),
                     "sim_sum": r["sim"] if both else 0.0, "cons": int(both and r["sim"] >= float(thr[m]))}
                for cell in ic.cells(u):
                    ic.add(per[c][cell][u["story"]], t)
        all_cells = sorted({cell for u in mu for cell in ic.cells(u)})
        out[m] = {"threshold": thr[m], "cells": ic.pooled_table(per, conditions, baseline, all_cells, stories,
                                                                 cfg["bootstrap"], metrics, BOOT_METRICS, FIELDS)}
    return out


# ------------------------------------------------------------------------------------------ CLI
def _load_run(a: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    units = ic.read_jsonl(a.out / "embed" / "units.jsonl")
    controls = ic.read_jsonl(a.out / "embed" / "controls.jsonl")
    manifest = json.loads((a.out / "embed" / "manifest.json").read_text())
    return units, controls, manifest


def cmd_units(a: argparse.Namespace) -> int:
    import hashlib

    cfg = load_config(a.config)
    icfg = identity_config(cfg)
    stories = a.stories.split(",") if a.stories else ic.default_stories(a.gt_dir, a.n_stories)
    gold = ic.load_gold(a.gt_dir, stories)
    layout = ic.ClipLayout.from_file(a.conditions, ic.parse_vars(a.var))
    cc = icfg["control_condition"]
    presence = {s: ic.presence_from_details(json.loads(layout.details(cc, s).read_text())) for s in stories}
    units, controls, audit = build_all(gold, presence, cfg)
    attach_neg_b(controls, gold)
    d = a.out / "embed"
    d.mkdir(parents=True, exist_ok=True)
    uw, cw = ic.jsonl(units), ic.jsonl(controls)
    (d / "units.jsonl").write_text(uw)
    (d / "controls.jsonl").write_text(cw)
    manifest = {"metric_version": METRIC_VERSION, "stories": stories, "n_units": len(units),
                "strata": {"|".join(k): v for k, v in sorted(Counter((u["stratum"], u["kind"], u["gap_bin"]) for u in units).items())},
                "n_controls": {"|".join(k): v for k, v in sorted(Counter((c["stratum"], c["kind"]) for c in controls).items())},
                "audit": audit, "units_sha256": hashlib.sha256(uw.encode()).hexdigest(),
                "controls_sha256": hashlib.sha256(cw.encode()).hexdigest()}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_extract(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    units, controls, _ = _load_run(a)
    layout = ic.ClipLayout.from_file(a.conditions, ic.parse_vars(a.var))
    jobs = clip_jobs(units, controls, a.cond, identity_config(cfg)["control_condition"])
    if a.controls_only:
        cc = identity_config(cfg)["control_condition"]
        keys = {(cc, u["story"], s["gt"], s["variant"]) for u in controls for s in sides(u)}
        jobs = [j for j in jobs if (j["cond"], j["story"], j["gt"], j["variant"]) in keys]
    ex = Extractor(cfg, a.models_root, a.device, a.face_root)
    print(json.dumps({"jobs": len(jobs), "shard": a.shard, "nshards": a.nshards, **ex.provenance}), flush=True)
    stats = run_extract(jobs, layout, a.out, cfg, ex, a.shard, a.nshards)
    print(json.dumps(stats), flush=True)
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    _, controls, manifest = _load_run(a)
    cal = calibrate(controls, NpzCache(a.out), cfg, identity_config(cfg)["control_condition"])
    (a.out / "embed" / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))
    if a.write_config:
        cfg["thresholds"] = {m: cal[m]["youden"].get("threshold") for m in MODALITIES}
        cfg["calibration"] = {"controls_sha256": manifest["controls_sha256"], "rule": "Youden-optimal sim >= t",
                              "auc": {m: cal[m]["auc"] for m in MODALITIES}}
        Path(a.config).write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"thresholds written to {a.config}: {cfg['thresholds']}")
    return 0


def cmd_aggregate(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    units, _, manifest = _load_run(a)
    cache = NpzCache(a.out)
    res = aggregate(units, a.cond, manifest["stories"], cache, cfg, baseline=a.baseline)
    cal_p = a.out / "embed" / "calibration.json"
    out = {"metric_version": METRIC_VERSION, "config": cfg, "units_manifest": manifest,
           "calibration": json.loads(cal_p.read_text()) if cal_p.is_file() else None, "modalities": res}
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    conds_default = "no_memory,text_frame,seg_uniform,seg_dinokey,seg_frame_clean,retr_oracle"

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    p = sub.add_parser("units")
    common(p)
    p.add_argument("--gt-dir", required=True, type=Path)
    p.add_argument("--stories", default="")
    p.add_argument("--n-stories", type=int, default=30)
    p.add_argument("--conditions", required=True, type=Path)
    p.add_argument("--var", action="append", default=[])
    p.set_defaults(fn=cmd_units)

    p = sub.add_parser("extract", help="GPU feature extraction (sharded, resume-safe)")
    common(p)
    p.add_argument("--conditions", required=True, type=Path)
    p.add_argument("--var", action="append", default=[])
    p.add_argument("--cond", default=conds_default)
    p.add_argument("--models-root", default=os.environ.get("PUBLIC_MODELS_ROOT", ""))
    p.add_argument("--face-root", default="", help="InsightFace root containing models/<pack> (default: weights root human_face_embedding)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--controls-only", action="store_true")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("calibrate", help="control AUCs + Youden thresholds")
    common(p)
    p.add_argument("--write-config", action="store_true", help="store thresholds in --config")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("aggregate")
    common(p)
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
