"""trackB-embed-1.0: views, face assignment, similarity, calibration, aggregation (no models)."""

from __future__ import annotations

import numpy as np
import pytest

from vmem_bench.scoring import embedding_consistency as ec
from vmem_bench.scoring import identity_consistency as ic


def _unit(uid: str, story: str, kind: str = "character", stratum: str = "main") -> dict:
    return {"uid": uid, "story": story, "eid": "E1", "name": "Ana", "kind": kind, "stratum": stratum,
            "gap_bin": "1-4", "prev_gt": 1, "cur_gt": 3, "desc_A": "a girl in red", "desc_B": "a girl in red"}


def _v(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _npz(scores, boxes, crops, faces=None) -> dict:
    d = {"frame_dino": np.stack([_v(1, 0), _v(1, 0), _v(1, 0)]), "frame_megaloc": np.stack([_v(0, 1)] * 3),
         "score__E1": np.asarray(scores, dtype=np.float32), "box__E1": np.asarray(boxes, dtype=np.float32),
         "crop__E1": np.stack(crops)}
    if faces is not None:  # per frame: embedding or None (no face inside the crop)
        d["faceok__E1"] = np.asarray([f is not None for f in faces])
        d["face__E1"] = np.stack([f if f is not None else np.zeros(2, np.float32) for f in faces])
    return d


CFG = ec.load_config()


def test_query_and_views() -> None:
    q = ec.grounding_query("Ana", "A lean girl of fourteen, deep-blue headscarf, coarse cotton clothes, bright eyes, extra words",
                           {"query": "{name}, {short_desc}.", "short_desc_words": 5})
    assert q == "ana, a lean girl of fourteen."
    u = _unit("s|003|E1", "s")
    jobs = ec.clip_jobs([u], [], ["c1", "c2"], "c1")
    assert [(j["cond"], j["gt"], j["variant"]) for j in jobs] == [("c1", 1, "main"), ("c1", 3, "main"),
                                                                 ("c2", 1, "main"), ("c2", 3, "main")]
    assert all(j["faces"] and "E1" in j["entities"] for j in jobs)
    loc = {**_unit("s|003|E9", "s", kind="location"), "eid": "E9"}
    assert all(not j["entities"] for j in ec.clip_jobs([loc], [], ["c1"], "c1"))


def test_side_vectors_threshold_and_face_assignment() -> None:
    thr = CFG["grounding"]["box_threshold"]
    boxes = [[0, 0, 10, 10]] * 3
    d = _npz([thr + 0.1, thr - 0.1, thr + 0.2], boxes, [_v(1, 0), _v(0, 1), _v(1, 1)],
             faces=[_v(0, 1), _v(1, 0), None])
    assert len(ec.side_vectors(d, "E1", "dino_crop", CFG)) == 2
    fv = ec.side_vectors(d, "E1", "face", CFG)
    assert len(fv) == 1 and np.allclose(fv[0], _v(0, 1))  # frame 1 below box threshold, frame 2 no face
    assert np.allclose(ec.side_vectors(d, "E1", "location_megaloc", CFG)[0], _v(0, 1))
    assert np.allclose(ec.side_vectors(d, "E1", "location_dino", CFG)[0], _v(1, 0))
    assert ec.side_vectors(None, "E1", "dino_crop", CFG) == []
    assert ec.side_vectors({"error": np.asarray("x")}, "E1", "location_megaloc", CFG) == []
    assert ec.side_vectors(_npz([1, 1, 1], boxes, [_v(1, 0)] * 3), "E1", "face", CFG) == []


def test_pair_similarity_max_over_pairs() -> None:
    assert ec.pair_similarity([], [_v(1, 0)]) is None
    s = ec.pair_similarity([_v(1, 0), _v(0, 1)], [_v(0, 1)])
    assert s == pytest.approx(1.0)


def test_auc_and_youden() -> None:
    assert ec.roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ec.roc_auc([0.5], [0.5]) == 0.5
    y = ec.youden_threshold([0.9, 0.8, 0.4], [0.1, 0.5])
    assert y["threshold"] == 0.8 and y["youden_j"] == pytest.approx(0.6667, abs=1e-4)


class _Cache:
    def __init__(self, table: dict) -> None:
        self.table = table

    def get(self, cond, story, gt, variant):
        return self.table.get((cond, story, gt, variant))


def test_aggregate_requires_calibration_and_is_deterministic() -> None:
    units = [_unit(f"s{i}|003|E1", f"s{i}") for i in range(5)]
    same = _npz([0.9] * 3, [[0, 0, 10, 10]] * 3, [_v(1, 0)] * 3)
    diff = _npz([0.9] * 3, [[0, 0, 10, 10]] * 3, [_v(0, 1)] * 3)
    table = {}
    for u in units:
        table[("nm", u["story"], 1, "main")] = same
        table[("nm", u["story"], 3, "main")] = diff
        table[("mem", u["story"], 1, "main")] = same
        table[("mem", u["story"], 3, "main")] = same
    cfg = {**CFG, "bootstrap": {"n_boot": 100, "seed": 3}, "thresholds": {m: None for m in ec.MODALITIES}}
    with pytest.raises(RuntimeError):
        ec.aggregate(units, ["nm", "mem"], [u["story"] for u in units], _Cache(table), cfg, baseline="nm")
    cfg["thresholds"] = {m: 0.5 for m in ec.MODALITIES}
    r1 = ec.aggregate(units, ["nm", "mem"], [u["story"] for u in units], _Cache(table), cfg, baseline="nm")
    r2 = ec.aggregate(units, ["nm", "mem"], [u["story"] for u in units], _Cache(table), cfg, baseline="nm")
    assert r1 == r2
    cell = r1["dino_crop"]["cells"]["main|all"]
    assert cell["nm"]["consistent_rate"] == 0.0 and cell["mem"]["consistent_rate"] == 1.0
    assert cell["mem"]["diff_vs_nm"]["consistent_rate"] == {"diff": 1.0, "lo": 1.0, "hi": 1.0, "n_draws": 100}
    assert "location_megaloc" not in r1 and "location_dino" not in r1  # no location units
    assert r1["face"]["cells"]["main|all"]["nm"]["both_detected_rate"] == 0.0  # no faces stored


def test_calibrate_uses_controls_per_modality() -> None:
    pos = {**_unit("pos|s|003|E1", "s", stratum="ctl_pos"), "variant_A": "posA", "variant_B": "posB"}
    neg = {**_unit("neg|s|E1@001|E2@003", "s", stratum="ctl_neg"), "eid_B": "E1", "name_B": "Bo", "desc_Bent": "a boy"}
    a = _npz([0.9] * 3, [[0, 0, 10, 10]] * 3, [_v(1, 0)] * 3)
    b = _npz([0.9] * 3, [[0, 0, 10, 10]] * 3, [_v(1, 0.1)] * 3)
    c = _npz([0.9] * 3, [[0, 0, 10, 10]] * 3, [_v(0, 1)] * 3)
    cache = _Cache({("nm", "s", 3, "posA"): a, ("nm", "s", 3, "posB"): b,
                    ("nm", "s", 1, "main"): a, ("nm", "s", 3, "main"): c})
    cal = ec.calibrate([pos, neg], cache, CFG, "nm")
    assert cal["dino_crop"]["auc"] == 1.0 and cal["dino_crop"]["counts"] == {"ctl_pos_n": 1, "ctl_pos_both": 1,
                                                                             "ctl_neg_n": 1, "ctl_neg_both": 1}
    assert cal["face"]["auc"] is None


def test_embed_config_is_versioned() -> None:
    assert CFG["metric_version"] == ec.METRIC_VERSION == "trackB-embed-1.0"
    assert ic.load_config(ec.CONFIG_DIR / CFG["identity_config"])["metric_version"] == "trackB-identity-1.0"
