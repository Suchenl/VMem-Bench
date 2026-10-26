# Track B scoring protocol (presence, identity consistency, embedding consistency)

Track B scores what a system *rendered*: a long film generated one segment at a time from the
frozen prompt stream (`assets/trackB/en/sut_prompts/<story>_name_anchored.json`). Gold
(`assets/trackB/en/gt/<story>.json`, `gt_version trackB-gt-2.0`) is hidden from ordinary systems.
The privileged Last-seen control uses gold IDs to select earlier generated
frames; the generator receives no gold pixels or descriptions. Three metric families answer three different questions:

| metric_version | module | question |
|---|---|---|
| `trackB-end2end-1.0` (frozen) | `vmem_bench.scoring.end2end_coverage` | Does each segment show what the gold says should be there (and nothing forbidden)? |
| `trackB-identity-1.0` | `vmem_bench.scoring.identity_consistency` | When an entity returns, does a VLM judge it to be the SAME individual/object as at its previous appearance? |
| `trackB-embed-1.0` | `vmem_bench.scoring.embedding_consistency` | How similar are previous/current localized appearances or scenes? Pinned proxies: DINOv3 crops, ArcFace faces, MegaLoc and whole-frame DINOv3 locations. |

**Important:** presence recall does *not* measure identity consistency. An entity counts as
present whenever the clip shows something that matches its gold text description; two different
renderings that both match "a fisherwoman in a deep-blue headscarf" both count. Report the identity
metrics whenever a claim is about *keeping the same look*. Embedding similarity is a proxy,
not proof of the correct individual or requested state; old-appearance resemblance
can be undesirable after a state change.

## 1. Presence / description judge (`trackB-end2end-1.0`, frozen)

**What the judge sees.** One call per GT segment (k=1, temperature 0 in the paper runs): the segment's
clip as a video, the SUT prompt of that segment (as a hint), and a *blinded mixed roster*: every
entity in the segment's gold `cast` (present), every `forbidden` entity, false-friend targets, and
3–5 deterministic decoys (seeded by `story:segment`), shuffled, each with id, name, kind and the gold
appearance text of its current (or flashback) state (first 120 characters). Nothing marks which
entries are present.

**What it decides.** For every roster id: `present` true/false/abstain (for a location: the clip takes
place there); `state` default/changed where gold expects a state judgment (transform/persist, not
flashbacks); `count` where gold has a count; `instance_match` for active look-alike pairs.

**Metrics** (micro-pooled over segments, then stories): recall over character+prop cast (headline),
per-kind recall, recall by gap `0 / 1-4 / 5-29 / >=30`, capability tallies (first appearance,
continuity, long gap, temporal reference, state change, persist state, count, look-alike instance,
false friend), precision over the story roster excluding decoys and locations, avoidance-OK
= 1 − forbidden-present / forbidden opportunities (deprecation, reference-indirect, look-alike-absent),
decoy FPR and abstain rate as noise floors. A reply that cannot be parsed after one strict retry is
recorded as `parse_error` (the segment's entities count as not present); it is never silently dropped.

**Segment mapping.** The scorer reads `review/segments/seg_###.mp4` of the run. Runs whose segments are
0-indexed (MemStrata production runner) are scored through a `+1` view: GT `seg_k` -> production
`seg_{k-1}`.

**What it does NOT check.** Whether a returning entity looks like its earlier rendering (identity), how
long it is on screen, where it is, or image quality. A system that draws a fresh, description-matching
stranger every time gets full recall.

## 2. Units shared by the identity metrics

Built deterministically from gold (`identity_consistency.build_units`, seed `unit_seed` 20260926),
one unit per cast entry:

| stratum | rule | kinds |
|---|---|---|
| `main` | op ∈ {recall, recall_after_gap, persist} and gap >= 1 (absent in the immediately previous segment, seen before) | character, prop (+ location for embeddings) |
| `gap0` | the same ops with gap == 0, seeded 20% sample (one draw per eligible entry in document order); continuity reference | same |
| `transform_ge30` | op == transform and gap >= 30 (state changed, e.g. aging; identity should still hold) | same |

`introduce` and short-gap `transform` entries are excluded (the look is supposed to change).
The previous appearance `prev_gt` is the latest earlier GT segment whose cast lists the entity (it
equals gold `last_seen` for every unit of the 30-story set). The gold description passed to judges
is the entity's state description at A (`prev_gt`) and at B (`cur_gt`). Location units use a separate
seed (`location_unit_seed`), so adding them does not change the character/prop units.

Cells: `<stratum>|all`, `<stratum>|<kind>`, `<stratum>|<kind>|<gap bin>`, and `main|all|<gap bin>`.

**Clip layout** is pluggable (`configs/trackb_conditions.example.json`): per condition a
`clip_template` with `{story}`, `{idx}` and `--var name=value` variables, a `file_index_offset`
(`idx = GT segment + offset`; `0` for gold-numbered runs, `-1` for the +1 view, `"auto"` = −1 iff
`seg_000.mp4` exists) and, for the control condition, the frozen judge's `details_template`.

## 3. VLM identity consistency (`trackB-identity-1.0`)

Config: `src/vmem_bench/scoring/configs/trackb_identity.json`.

* **Strips.** For each clip, frames at 15/50/85% of the frame count, each resized to height 256, side by
  side, JPEG q=90 (controls: 10/30/50% vs 55/75/95%).
* **Judge.** Qwen3-VL-32B-Instruct (same weights as the presence judge), temperature 0, one call per
  unit × condition, images inline. The prompt (verbatim in `identity_consistency.PROMPT`) gives image A
  (previous appearance), image B (current), the entity name, kind and the two descriptions; when the
  descriptions differ it adds that a described state change is expected. It says camera, framing, pose,
  expression, lighting and background may differ and must be ignored, and asks for strict JSON:
  `visible_A`, `visible_B` (true/false/"unsure") and `same` (true/false/"unsure", null unless visible in
  both; identity = face, hair, build, clothing for characters; shape, color, material, markings for
  props). The condition name is never shown.
* **Parsing.** First `{...}` block; yes/no/true/false/null normalized; anything else -> `"unsure"`; a
  reply without the three keys is a parse error and counts as a unit with nothing visible.
* **Metrics** per condition and cell (micro counts): `consistent_rate = #(visible_B ∧ same) / #units`
  (appears and looks the same), `identity_given_both = #same / #(visible_A ∧ visible_B)`,
  `visible_B_rate`; also `visible_A_rate`, `both_visible_rate`, unsure rates.
* **Which number to report.** Identity is reported as `dino_crop` `mean_similarity` (Section 4) with
  `identity_given_both` as the VLM check. `consistent_rate` is not a return-level identity measure: it
  also counts whether the *previous* appearance A was rendered recognisably, which a memory changes through
  the earlier segments of its own rollout. On `transform_ge30` (paper run) B is visible in 0.98–1.00 of
  units and every both-visible pair is called the same in every condition, so the whole spread of
  `consistent_rate` (0.11 no memory vs 0.44–0.56) comes from A.
* **Controls** (control condition = no memory): `ctl_pos` 200 units whose current clip the frozen judge
  marked present, A/B = the two disjoint strips of that clip; `ctl_neg` 200 pairs X ≠ Y of the same kind
  and story, A = a clip where X is voted present, B = a clip where Y is present and X is not, prompt
  describes X. Reported: `pos_same_given_both`, `neg_not_consistent_rate`, `neg_different_given_both`.
* **Test-retest.** Optionally a full second pass (`--sub judgments_rep1`); exact-answer agreement is
  reported per condition.

## 4. Embedding consistency (`trackB-embed-1.0`)

Config: `src/vmem_bench/scoring/configs/trackb_embed.json` (models, thresholds, seeds).

* **Frames.** 3 full-resolution frames per side (same fractions as the strips).
All embeddings come from the pinned scoring providers of `vmem_bench.scoring.embedder`
(`SCORING_PROVIDERS`), the same ones Track A uses; only the detector is loaded separately.

* **Characters/props — `dino_crop`.** GroundingDINO (`IDEA-Research/grounding-dino-base`) with query
  `"<name>, <first 12 words of the gold description>."` (lower-cased). Per frame the top box, scored by
  max over text tokens of sigmoid(logit); a detection needs score >= `box_threshold` (0.35). The box is
  padded by 5% and embedded with the `dinov3` provider (ViT-B/16 CLS token, L2-normalized).
* **Characters — `face`.** The `arcface` provider (InsightFace `buffalo_l`, det size 640, det score
  >= 0.5) on the character crop of each frame; the largest face found inside the crop is used.
* **Locations — `location_megaloc` (primary) and `location_dino` (secondary).** The whole frame (a
  location is the whole scene) through the `megaloc` provider (MegaLoc visual place recognition,
  trained for same-place retrieval across viewpoint and lighting; 322×322 input) and through `dinov3`.
  Both are calibrated on the location controls and reported side by side. Whole-frame similarity is
  deliberately NOT used for characters/props: a memory that copies a reference frame's composition
  would score high without the entity being the same.
* **Similarity.** `sim(A,B) = max` over frame pairs (a ∈ A, b ∈ B) that both carry the modality of the
  cosine. A side without the modality makes the unit "not both detected".
* **Threshold.** Chosen once per modality on the calibration controls (Youden-optimal `sim >= t`,
  ties -> lowest t) with `calibrate --write-config`, stored in the config together with the control-set
  hash and AUCs, before any condition is aggregated; `aggregate` refuses to run without thresholds.
* **Metrics.** `consistent_rate = #(both detected ∧ sim >= t) / #units`, `consistent_given_both`,
  `mean_similarity` (over both-detected units), `detected_A/B_rate`, `both_detected_rate`.
* **Controls per modality.** Positive = same clip, disjoint frame sets; negative = X in a clip where the
  frozen judge voted X present vs a different same-kind entity Y of the same story in a clip where Y is
  present and X is not, each side localized with its own entity's query. Characters/props reuse the VLM
  controls plus a character-only set (seed `character_control_seed`) for the face threshold; locations
  get their own set. Reported per modality (`dino_crop`, `face`, `location_megaloc`, `location_dino`): ROC-AUC (Mann-Whitney) and pos/neg similarity quantiles.

## 5. Pooling and uncertainty

All rates are micro-pooled counts over units of the evaluated stories (the paper uses the first 30
stories). Differences vs the no-memory baseline use a story-clustered paired bootstrap: 2000 resamples
of the story list with replacement (`random.Random(20260925)`, `rng.choice` per slot), the same draws
for every condition; the CI is the 2.5% / 97.5% percentile with the indices of
`assets/_paper/.../scripts/memory_gain_bootstrap.py` (`int(0.025·n)`, `int(0.975·n) − 1` over the draws
where both rates are defined).

## 6. Known limitations

* Three frames per clip can miss a brief appearance (in the calibration set 31/200 positive clips showed
  the entity in neither half-strip although the presence judge saw it in the video).
* The VLM separates look-alikes poorly: on the negative control it called only 4 of 18 both-visible
  different-entity pairs "different" (many are the benchmark's deliberate look-alikes). Treat
  `identity_given_both` as upper-biased; `consistent_rate` is dominated by visibility.
* `consistent_rate` couples visibility in A and B with identity: a condition that renders the entity
  recognisably in both clips gains even if identity discrimination is weak.
* Descriptions come from gold (same text the presence judge uses); the detector query uses a 12-word
  prefix, so detections can latch onto a similar object.
* Embedding similarity measures appearance, not identity semantics: clothing/pose changes lower it and
  same-costume look-alikes raise it; thresholds are calibrated on one control set.
* Batched vLLM serving is not bit-deterministic; the test-retest pass bounds this (< 0.7% of answers).

## 7. Installing the scoring models

Every model the scorers use is listed, with the SHA256 of every file, in
`src/vmem_bench/scoring/configs/scoring_models_manifest.json` (recorded from the maintainers' copies):

| model | provider / use | root | layout |
|---|---|---|---|
| DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated) | `dinov3`: crops, frames | `PUBLIC_MODELS_ROOT` | `facebook/dinov3-vitb16-pretrain-lvd1689m/` |
| GroundingDINO base (`IDEA-Research/grounding-dino-base`) | entity localization | `PUBLIC_MODELS_ROOT` | `IDEA-Research/grounding-dino-base/` |
| InsightFace `buffalo_l` (GitHub release v0.7) | `arcface`: faces | `MEMSTRATA_WEIGHTS_ROOT` | `human_face_embedding/models/buffalo_l/` |
| MegaLoc code (GitHub `gmberton/MegaLoc`) + weights (HF `gberton/MegaLoc`) | `megaloc`: locations | `MEMSTRATA_WEIGHTS_ROOT` | `torch_hub/gmberton_MegaLoc_main/`, `location_embedding/MegaLoc/model.safetensors` |
| Qwen3-VL-32B-Instruct (`Qwen/Qwen3-VL-32B-Instruct`) | all VLM judges (vLLM, served name `qwen3-vl-32b`) | `PUBLIC_MODELS_ROOT` | `Qwen/Qwen3-VL-32B-Instruct/` |
| SigLIP2 base (`google/siglip2-base-patch16-224`, optional) | `siglip2` | `PUBLIC_MODELS_ROOT` | `google/siglip2-base-patch16-224/` |

```bash
python3 -m pip install -e ".[scoring]"                    # torch, transformers>=4.56, insightface, onnxruntime-gpu, ...
export MEMSTRATA_WEIGHTS_ROOT=/path/to/model_weights
export PUBLIC_MODELS_ROOT=/path/to/hf_models              # default: the weights root
export HF_TOKEN=...                                       # DINOv3 is gated on Hugging Face
scripts/install_scoring_models.sh --dry-run --with-optional   # what would be fetched, and where
scripts/install_scoring_models.sh [--only name,...] [--with-optional]
vmem-bench-install-models verify                          # re-hash every local file against the manifest
```

Present files are skipped; downloaded files are hash-verified and a mismatch fails the command. The
judge additionally needs vLLM (`start_reviewer_pool.sh` serves it on TP=2 A800/H800 pairs).

## 8. How to run


```bash
export PYTHONPATH=src
C=src/vmem_bench/scoring/configs/trackb_conditions.example.json
V="--var results=$PWD/results --var nomem_root=<no-memory run root> --var oracle_root=<oracle run root>"
# VLM identity (trackB-identity-1.0)
python -m vmem_bench.scoring.identity_consistency units --out RUN --gt-dir assets/trackB/en/gt --conditions $C $V
python -m vmem_bench.scoring.identity_consistency strips --out RUN --conditions $C $V --procs 48
python -m vmem_bench.scoring.identity_consistency judge --out RUN --endpoints http://host:port/v1,...
python -m vmem_bench.scoring.identity_consistency aggregate --out RUN --json-out identity.json
# embeddings (trackB-embed-1.0); extract on GPUs, one shard per GPU
python -m vmem_bench.scoring.embedding_consistency units --out RUN --gt-dir assets/trackB/en/gt --conditions $C $V
python -m vmem_bench.scoring.embedding_consistency extract --out RUN --conditions $C $V \
    --models-root $PUBLIC_MODELS_ROOT --face-root $MEMSTRATA_WEIGHTS_ROOT/human_face_embedding --shard i --nshards N
python -m vmem_bench.scoring.embedding_consistency calibrate --out RUN --write-config
python -m vmem_bench.scoring.embedding_consistency aggregate --out RUN --json-out embed.json
```

All steps are resume-safe (existing per-task files are skipped) and deterministic given the configs.
