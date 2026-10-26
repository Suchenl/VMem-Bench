"""Machine-first review: humans only see the gray zone.

Deterministic per-entity suspicion scoring, two-tier merge split (auto vs gray),
optional SigLIP name/cover agreement, and optional VLM entity audit. Safe auto
tier applies MERGES only via ``review.apply_patch``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline_track_first.consolidation import Registry
from vmem_bench.annotation.pipeline_track_first import review as review_mod
from vmem_bench.common.paths import MovieDirs
from vmem_bench.common.schemas import Entity
from vmem_bench.common.vecmath import cosine_similarity

logger = logging.getLogger(__name__)

# Suspicion score weights (ponytail constants).
_W_DISPERSION = 2.0
_W_SINGLETON = 1.0
_W_QA_FLAGGED = 0.5
_W_SHORT_SCREEN = 0.5
_W_GRAY_MERGE = 1.5
_W_KIND_SUSPECT = 1.0
_W_NAME_DISAGREE = 1.5
_W_VLM_INCOHERENT = 2.0


def entity_dispersion(entity: Entity, embeddings: dict[str, list[float]]) -> float | None:
    """Min pairwise cosine among the entity's stored body vectors; None if <2 vectors."""
    vecs = [embeddings[r.embedding_key] for r in entity.representations
            if r.embedding_key and r.embedding_key in embeddings]
    if len(vecs) < 2:
        return None
    best = 1.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            best = min(best, cosine_similarity(vecs[i], vecs[j]))
    return best


def suspicion_signals(registry: Registry) -> dict[str, dict[str, Any]]:
    """Per entity_id: dispersion, n_reps, singleton, qa_flagged, screen_time_seconds."""
    out: dict[str, dict[str, Any]] = {}
    for eid, ent in registry.entities.items():
        n_reps = len(ent.representations)
        qa_flagged = sum(1 for r in ent.representations if r.qa.get("flagged"))
        out[eid] = {
            "dispersion": entity_dispersion(ent, registry.embeddings),
            "n_reps": n_reps,
            "singleton": n_reps <= 1,
            "qa_flagged": qa_flagged,
            "screen_time_seconds": ent.screen_time_seconds,
        }
    return out


# Head nouns that strongly signal a place; a "prop" named this way (or a "location" named like a
# carryable object) is a deterministic kind-mixture suspect. Small on purpose: only unambiguous
# words, because these checks add review reasons, never mutate gold.
_LOCATION_HEAD_WORDS = {
    "field", "path", "sky", "meadow", "forest", "creek", "river", "cave", "background",
    "ground", "hill", "clearing", "grassland", "canopy", "valley", "lake", "beach", "road",
}
_OBJECT_HEAD_WORDS = {
    "glider", "rock", "boulder", "stick", "apple", "fruit", "nut", "arrow", "bow", "leaf",
}


def kind_mixture_reasons(registry: Registry) -> dict[str, list[str]]:
    """Deterministic kind-vs-name sanity checks: per entity_id, human-readable suspicions."""
    out: dict[str, list[str]] = {}
    character_words = {
        w for eid, ent in registry.entities.items() if ent.kind == "character"
        for w in str(ent.name or eid).lower().replace("_", " ").split()}
    for eid, ent in registry.entities.items():
        words = str(ent.name or eid).lower().replace("_", " ").split()
        head = words[-1] if words else ""
        flags: list[str] = []
        if ent.kind == "prop" and head in _LOCATION_HEAD_WORDS:
            flags.append(f"kind_suspect: prop named like a location ({head})")
        if ent.kind == "location" and head in _OBJECT_HEAD_WORDS:
            flags.append(f"kind_suspect: location named like an object ({head})")
        if ent.kind != "character":
            hits = sorted(set(words) & character_words - {"the", "a", "of", "with"})
            if hits:
                flags.append(f"name_mentions_character_word: {','.join(hits)}")
        if flags:
            out[eid] = flags
    return out


def suspicion_score(sig: dict[str, Any], *, dispersion_floor: float = 0.35) -> float:
    """Weighted suspicion from signal dict. Weights are module-level ponytail constants."""
    score = 0.0
    disp = sig.get("dispersion")
    if disp is not None and disp < dispersion_floor:
        score += _W_DISPERSION
    if sig.get("singleton"):
        score += _W_SINGLETON
    score += _W_QA_FLAGGED * float(sig.get("qa_flagged") or 0)
    st = sig.get("screen_time_seconds")
    if st is not None and st < 1.0:
        score += _W_SHORT_SCREEN
    return score


def _species_guard_ok(a, b) -> bool:
    """Deterministic third vote: same kind, and species must agree when both sides declare one."""
    if a.kind != b.kind:
        return False
    sa = str((a.static_attributes or {}).get("species") or "")
    sb = str((b.static_attributes or {}).get("species") or "")
    return not (sa and sb and sa != sb)


def _entity_vote_crops(entity, out: Path, limit: int = 3) -> list[Path]:
    reps = sorted((r for r in entity.representations if r.crop_path),
                  key=lambda r: -float((r.qa or {}).get("grounding_score", 0.0)))
    crops: list[Path] = []
    for rep in reps:
        p = Path(rep.crop_path)
        p = p if p.is_absolute() else out / p
        if p.is_file():
            crops.append(p)
        if len(crops) >= limit:
            break
    return crops


def body_similarity_pairs(registry: Registry, *, body_floor: float) -> list[dict[str, Any]]:
    """Same-kind pairs nominated by VISUAL similarity alone (mean body cosine >= floor).

    Text nomination breaks exactly when naming is forced to be distinct per entity, so the
    VLM-vote candidate pool must also accept the visual cue on its own."""
    from vmem_bench.annotation.pipeline_track_first.reid import _entity_signature
    entities = list(registry.entities.values())
    sigs = {e.entity_id: _entity_signature(e, registry.embeddings) for e in entities}
    pairs: list[dict[str, Any]] = []
    ids = sorted(registry.entities)
    for i, a_id in enumerate(ids):
        a = registry.entities[a_id]
        for b_id in ids[i + 1:]:
            b = registry.entities[b_id]
            if a.kind != b.kind or sigs[a_id] is None or sigs[b_id] is None:
                continue
            body_cos = cosine_similarity(sigs[a_id], sigs[b_id])
            if body_cos < body_floor:
                continue
            keep, merge = ((a_id, b_id) if (a.first_chunk, a_id) <= (b.first_chunk, b_id)
                           else (b_id, a_id))
            pairs.append({"keep": keep, "merge": merge, "text_cos": None,
                          "body_cos": round(float(body_cos), 4), "kind": a.kind})
    return pairs


def vlm_confirmed_merges(registry: Registry, out: Path, candidates: list[dict[str, Any]],
                         adjudicator: Any,
                         preconfirmed: list[tuple[str, str]] | None = None,
                         ) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Three-vote auto-merge: embedding support (the candidate list) ∧ VLM same-individual
    verdict ∧ deterministic species/kind guard. Returns (merge pairs, full vote log).

    Confirmed pairs are union-found into components so chained duplicates collapse to one
    surviving entity (most representations wins) instead of conflicting pairwise merges."""
    votes: list[dict[str, Any]] = []
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keep_id, merge_id in (preconfirmed or []):
        if keep_id in registry.entities and merge_id in registry.entities:
            ra, rb = find(keep_id), find(merge_id)
            if ra != rb:
                parent[rb] = ra

    for p in candidates:
        keep_id, merge_id = str(p.get("keep")), str(p.get("merge"))
        a, b = registry.entities.get(keep_id), registry.entities.get(merge_id)
        if a is None or b is None:
            continue
        vote = {"keep": keep_id, "merge": merge_id,
                "text_cos": p.get("text_cos"), "body_cos": p.get("body_cos")}
        if not _species_guard_ok(a, b):
            vote["verdict"] = "guard_reject"
            votes.append(vote)
            continue
        a_crops, b_crops = _entity_vote_crops(a, out), _entity_vote_crops(b, out)
        if not a_crops or not b_crops:
            vote["verdict"] = "no_crops"
            votes.append(vote)
            continue
        try:
            result = adjudicator.judge_same_individual_pair(
                a_crops, b_crops, a.name or keep_id, b.name or merge_id)
        except Exception as exc:  # noqa: BLE001 — a VLM hiccup degrades to the human queue
            vote["verdict"] = f"vlm_error: {exc}"
            votes.append(vote)
            continue
        vote["vlm_same"] = bool(result.get("same"))
        vote["vlm_reason"] = str(result.get("reason") or "")
        vote["verdict"] = "auto_merge" if vote["vlm_same"] else "vlm_reject"
        votes.append(vote)
        if vote["vlm_same"]:
            ra, rb = find(keep_id), find(merge_id)
            if ra != rb:
                parent[rb] = ra

    components: dict[str, list[str]] = {}
    for eid in list(parent):
        components.setdefault(find(eid), []).append(eid)
    merges: list[list[str]] = []
    for members in components.values():
        if len(members) < 2:
            continue
        survivor = max(members,
                       key=lambda e: (len(registry.entities[e].representations), e))
        merges.extend([survivor, other] for other in members if other != survivor)
    return merges, votes


def split_merge_tiers(
    proposals: list[dict[str, Any]],
    *,
    auto_text: float,
    auto_body: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split proposals into (auto, gray). Auto requires high text AND body agreement."""
    auto: list[dict[str, Any]] = []
    gray: list[dict[str, Any]] = []
    for p in proposals:
        text_cos = float(p.get("text_cos") or 0.0)
        body_cos = p.get("body_cos")
        if (text_cos >= auto_text
                and body_cos is not None
                and float(body_cos) >= auto_body):
            auto.append(p)
        else:
            gray.append(p)
    return auto, gray


def name_agreement(classifier: Any, cover_path: Path, name: str, description: str) -> float | None:
    """Top SigLIP prob of cover vs ``name. description``; None on any failure."""
    try:
        ranked = classifier.classify(cover_path, [f"{name}. {description[:100]}"])
        if not ranked:
            return None
        return float(ranked[0][1])
    except Exception:  # noqa: BLE001 — optional hook must never fail the review
        return None


def _entity_crop_paths(entity: Entity, out: Path) -> list[Path]:
    paths: list[Path] = []
    for r in entity.representations:
        if not r.crop_path:
            continue
        p = Path(r.crop_path)
        if not p.is_absolute():
            p = out / p
        if p.is_file():
            paths.append(p)
    return paths


def _cover_path(entity: Entity, out: Path) -> Path | None:
    crops = _entity_crop_paths(entity, out)
    return crops[0] if crops else None


def run_auto_review(
    out: Path,
    *,
    apply_safe: bool = True,
    classifier: Any = None,
    namer_vlm: Any = None,
    top_k_model_audit: int = 10,
    auto_text: float = 0.92,
    auto_body: float = 0.75,
    text_embed_fn: Any = None,
    merge_text_threshold: float = 0.85,
    merge_body_threshold: float = 0.5,
    # Softer floor for the VLM-vote candidate pool: embedding only nominates, it never decides.
    vlm_merge_text_floor: float = 0.80,
    vlm_merge_body_floor: float = 0.35,
    dispersion_floor: float = 0.35,
    must_review_score: float = 1.0,
    auto_drop_singletons: bool = True,
) -> dict[str, Any]:
    """Score entities, split merge tiers, optionally apply safe merges, write report + review.html."""
    out = Path(out)
    dirs = MovieDirs(out, write=True)      # write targets: always the new scheme
    dirs_read = MovieDirs(out)             # reads: fall back to legacy locations
    gold_dir = dirs.gold
    dirs.tmp.mkdir(parents=True, exist_ok=True)

    registry, _chunks = review_mod._load(gold_dir)
    embeddings = review_mod._load_embeddings(gold_dir)
    reg = Registry()
    reg.entities = {e.entity_id: e for e in registry.entities}
    reg.embeddings = embeddings

    signals = suspicion_signals(reg)
    scores: dict[str, float] = {
        eid: suspicion_score(sig, dispersion_floor=dispersion_floor)
        for eid, sig in signals.items()
    }
    reasons: dict[str, list[str]] = {eid: [] for eid in scores}
    for eid, sig in signals.items():
        disp = sig.get("dispersion")
        if disp is not None and disp < dispersion_floor:
            reasons[eid].append(f"dispersion={disp:.3f}<{dispersion_floor}")
        if sig.get("singleton"):
            reasons[eid].append("singleton")
        if sig.get("qa_flagged"):
            reasons[eid].append(f"qa_flagged={sig['qa_flagged']}")
        st = sig.get("screen_time_seconds")
        if st is not None and st < 1.0:
            reasons[eid].append(f"screen_time={st}<1.0")

    for eid, flags in kind_mixture_reasons(reg).items():
        if eid in scores:
            scores[eid] += _W_KIND_SUSPECT
            reasons[eid].extend(flags)

    if text_embed_fn is not None:
        from vmem_bench.annotation.pipeline_track_first.entity_merge import propose_entity_merges
        proposals = propose_entity_merges(
            reg, text_embed_fn,
            text_threshold=merge_text_threshold,
            body_threshold=merge_body_threshold)
    else:
        mp = dirs_read.merge_proposals
        if mp.is_file():
            proposals = json.loads(mp.read_text(encoding="utf-8"))
        else:
            proposals = []

    auto_tier, gray_tier = split_merge_tiers(
        proposals, auto_text=auto_text, auto_body=auto_body)

    for p in gray_tier:
        for eid in (p.get("keep"), p.get("merge")):
            if eid in scores:
                scores[eid] += _W_GRAY_MERGE
                reasons[eid].append("gray_merge")

    ranked_ids = sorted(scores, key=lambda e: (-scores[e], e))
    audit_ids = ranked_ids[: max(0, top_k_model_audit)]

    if classifier is not None:
        for eid in audit_ids:
            ent = reg.entities.get(eid)
            if ent is None:
                continue
            cover = _cover_path(ent, out)
            if cover is None:
                continue
            agree = name_agreement(classifier, cover, ent.name, ent.description)
            if agree is not None and agree < 0.15:
                scores[eid] += _W_NAME_DISAGREE
                reasons[eid].append(f"name_agreement={agree:.3f}<0.15")

    if namer_vlm is not None:
        for eid in audit_ids:
            ent = reg.entities.get(eid)
            if ent is None:
                continue
            crops = _entity_crop_paths(ent, out)
            if not crops:
                continue
            try:
                audit = namer_vlm.audit_entity(crops, ent.name, ent.description)
            except Exception:  # noqa: BLE001
                audit = None
            if audit is not None and audit.get("coherent") is False:
                scores[eid] += _W_VLM_INCOHERENT
                note = str(audit.get("note") or "incoherent")
                reasons[eid].append(f"vlm_incoherent: {note}")

    must_review = [eid for eid in sorted(scores, key=lambda e: (-scores[e], e))
                   if scores[eid] >= must_review_score]
    auto_ok = [eid for eid in sorted(scores) if eid not in set(must_review)]

    # Three-vote auto-merge (approved design): embedding nominates (soft floor), the VLM
    # adjudicator votes on crops, the species/kind guard has veto power. Confirmed components
    # are applied through the normal patch mechanism so every merge is provenanced + undoable.
    vlm_votes: list[dict[str, Any]] = []
    merges: list[list[str]] = [[p["keep"], p["merge"]] for p in auto_tier]
    if (text_embed_fn is not None and namer_vlm is not None
            and hasattr(namer_vlm, "judge_same_individual_pair")):
        from vmem_bench.annotation.pipeline_track_first.entity_merge import propose_entity_merges as _propose
        floor_pool = _propose(reg, text_embed_fn, text_threshold=vlm_merge_text_floor,
                              body_threshold=vlm_merge_body_floor)
        # Text-only nomination misses same-individual splits with (deliberately) distinct
        # names; the visual cue nominates independently and the VLM+guard votes decide.
        seen_pairs = {(p["keep"], p["merge"]) for p in floor_pool}
        for p in body_similarity_pairs(reg, body_floor=0.55):
            if (p["keep"], p["merge"]) not in seen_pairs:
                floor_pool.append(p)
        auto_pairs = {(p["keep"], p["merge"]) for p in auto_tier}
        candidates = [p for p in floor_pool
                      if (p["keep"], p["merge"]) not in auto_pairs]
        merges, vlm_votes = vlm_confirmed_merges(
            reg, out, candidates, namer_vlm,
            preconfirmed=[(p["keep"], p["merge"]) for p in auto_tier])
        (dirs.tmp / "auto_review_vlm_merges.json").write_text(
            json.dumps({"votes": vlm_votes, "merges": merges}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    applied_merges: list[dict[str, Any]] = []
    merged_away: set[str] = set()
    if apply_safe and merges:
        patch = {
            "schema_version": "2.0.0",
            "merges": merges,
            "splits": [],
            "renames": {},
            "drops": [],
            "field_edits": [],
        }
        patch_path = dirs.auto_review_patch
        patch_path.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        review_mod.apply_patch(gold_dir, patch_path)
        applied_merges = [{"keep": k, "merge": m} for k, m in merges]
        merged_away = {m for _k, m in merges}
        registry, _chunks = review_mod._load(gold_dir)
        # Entities merged away must not spawn review cards or ranks.
        for eid in merged_away:
            scores.pop(eid, None)
            reasons.pop(eid, None)
        must_review = [eid for eid in must_review if eid not in merged_away]
        auto_ok = [eid for eid in auto_ok if eid not in merged_away]

    # Long-tail guard: a single-representation blink-and-gone entity with no state events is
    # detector noise, not cast. Auto-drop via the patch mechanism (undoable, provenanced) —
    # BBB v13 shipped 8 such fragments to the human queue for no decision value.
    drop_candidates: list[str] = []
    for eid, sig in signals.items():
        st = sig.get("screen_time_seconds")
        ent = reg.entities.get(eid)
        has_events = bool(ent is not None and ent.state_events)
        if sig.get("singleton") and st is not None and st < 1.0 and not has_events:
            drop_candidates.append(eid)
            reasons[eid].append("candidate_drop: singleton with <1s screen time")
    if apply_safe and auto_drop_singletons and drop_candidates:
        drop_patch = {"schema_version": "2.0.0", "merges": [], "splits": [], "renames": {},
                      "drops": sorted(drop_candidates), "field_edits": []}
        drop_path = dirs.tmp / "auto_review_drop_patch.json"
        drop_path.write_text(json.dumps(drop_patch, indent=2), encoding="utf-8")
        review_mod.apply_patch(gold_dir, drop_path)
        registry, _chunks = review_mod._load(gold_dir)
        for eid in drop_candidates:
            scores.pop(eid, None); reasons.pop(eid, None)
        must_review = [e for e in must_review if e not in set(drop_candidates)]
        auto_ok = [e for e in auto_ok if e not in set(drop_candidates)]

    queue = [
        {"entity_id": eid, "score": scores[eid], "reasons": reasons[eid],
         **({"recommendation": "candidate_drop"} if eid in set(drop_candidates) else {})}
        for eid in sorted(scores, key=lambda e: (-scores[e], e))
    ]
    queue_dict = {q["entity_id"]: {"score": q["score"], "reasons": q["reasons"]} for q in queue}
    # A low minimum pairwise similarity is evidence of a contaminated identity cluster, but never
    # enough to automatically split it: a human/model reviewer must select the offending reps.
    split_candidates = [
        {
            "entity_id": eid,
            "dispersion": round(float(sig["dispersion"]), 4),
            "representation_ids": [
                rep.representation_id for rep in reg.entities[eid].representations
            ],
        }
        for eid, sig in signals.items()
        if (sig.get("dispersion") is not None
            and float(sig["dispersion"]) < dispersion_floor
            and eid in reg.entities)
    ]

    gray_tier = [p for p in gray_tier if p.get("merge") not in merged_away
                 and p.get("keep") not in merged_away]
    report = {
        "applied_merges": applied_merges,
        "gray_merges": gray_tier,
        "vlm_merge_votes": vlm_votes,
        "queue": queue,
        "split_candidates": split_candidates,
        "must_review": must_review,
        "auto_ok": auto_ok,
        "stats": {
            "n_entities": len(scores),
            "n_must_review": len(must_review),
            "n_auto_ok": len(auto_ok),
            "n_applied_merges": len(applied_merges),
        },
    }
    dirs.auto_review_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Queue generation is a read-only convenience artifact; never let it fail annotation.
    try:
        from vmem_bench.annotation.pipeline_track_first.review_queue import write_review_queue
        write_review_queue(out)
    except Exception:  # noqa: BLE001
        logger.exception("review queue generation failed")
    review_mod.generate_review_html(out, machine_queue=queue_dict)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MemStrata machine-assisted auto review")
    parser.add_argument("--out", type=Path, required=True, help="annotated movie output dir")
    parser.add_argument("--no-apply-safe", dest="apply_safe", action="store_false", default=True,
                        help="do not auto-apply the safe merge tier")
    parser.add_argument("--siglip", action="store_true", default=False,
                        help="enable optional SigLIP name/cover agreement audit")
    parser.add_argument("--top-k", type=int, default=10, dest="top_k",
                        help="top-K entities by suspicion for optional model audits")
    parser.add_argument("--text-embed-base-url", default=None,
                        help="OpenAI-compatible /v1 endpoint for text embeddings; enables "
                             "fresh merge proposals + the auto-merge tier")
    parser.add_argument("--text-embed-model", default="qwen3-embedding-4b")
    parser.add_argument("--vlm-base-url", default=None,
                        help="OpenAI-compatible VLM endpoint for the three-vote auto-merge and "
                             "entity audits (use the LARGE judgment model, e.g. qwen3-vl-32b)")
    parser.add_argument("--vlm-model", default="qwen3-vl-32b")
    args = parser.parse_args(argv)

    text_embed_fn = None
    if args.text_embed_base_url:
        from vmem_bench.services.clients import EmbedClient
        text_embed_fn = EmbedClient(args.text_embed_base_url, args.text_embed_model).embed

    role = None
    if args.vlm_base_url:
        from vmem_bench.judger.vlm import VlmJudger
        from vmem_bench.annotation.pipeline_track_first.vlm_roles import AnnotatorRole
        role = AnnotatorRole(VlmJudger(base_url=args.vlm_base_url, model=args.vlm_model))

    classifier = None
    if args.siglip:
        try:
            from vmem_bench.annotation.pipeline_track_first.crop_classify import SiglipCropClassifier
            classifier = SiglipCropClassifier()
        except Exception:  # noqa: BLE001
            logger.exception("SigLIP classifier unavailable; continuing without it")
            classifier = None

    report = run_auto_review(
        args.out, apply_safe=args.apply_safe, classifier=classifier, namer_vlm=role,
        top_k_model_audit=args.top_k, text_embed_fn=text_embed_fn)
    stats = report["stats"]
    print(f"auto_review stats={stats} must_review={report['must_review']}")


if __name__ == "__main__":
    main()
