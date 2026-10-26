# CVPR 2027 reproduction checklist

Audited 2026-10-23. This records existing artifacts; no generation, scoring,
bootstrap, or model download was run during the audit.

## Choose the reproduction level

1. **Render the paper tables:** use the manuscript's saved analysis JSON and
   deterministic renderers. This needs no GPU or local video.
2. **Recompute statistics from existing scores:** provide the per-segment
   `score.json` snapshots at the expected result-index paths. This is distinct
   from calling a judge again.
3. **Regenerate and rescore video:** provide generator/judge weights, dependencies,
   and hardware. Use the fixed cohort and seed below. Identical seeds do not
   guarantee identical pixels or subsequent retrieval paths.

## Inputs and provenance checked locally

| Item | Audited state | Evidence |
|---|---|---|
| Evaluated cohort | 30 identical story IDs across all six pooled seed-2026 cells | `docs/cvpr27_story_ids.txt`; manuscript `tables/experiments/trackb_story_cohort.json` |
| Gold and name-anchored prompts | All 30 gold and 30 prompt files match the pinned generation checkout byte for byte | Per-file SHA256 in `trackb_story_coverage.json` |
| Generation code | 180 saved manifests identify clean commit `0b41d3df8d5c2b53ecc491ad00c769db2fa24a8b` | `paper_input_audit.json` |
| Completed segments | Each condition records 2,824 successful and zero failed segments, seed 2026 | Same 180 manifests; metadata audit, not video decoding |
| Score snapshots | 180 `score.json` files exist and have recorded SHA256 | `paper_input_audit.json`; invalid judge outputs are handled by the published protocol, not assumed absent |
| Scoring model inventory | File-level model manifest exists and its own SHA256 is recorded | `src/vmem_bench/scoring/configs/scoring_models_manifest.json` |
| Main analysis inputs | Hashes recorded for main control/bootstrap, identity, prompt intervention, second judge, quality, and H3 JSON snapshots | `paper_input_audit.json` |
| Public availability | **Not verified by this audit** | `results/trackB/testbed_rerun_20260927` is a local symlink, not a Git-tracked download bundle |

The bundled `assets/trackB/freeze_manifest.json` currently certifies one sample;
it must not be cited as a 30-story freeze manifest. The evaluated cohort's current
60 input hashes are recorded separately in the manuscript coverage snapshot.
Genre labels were not inferred from titles; the stress coverage is deliberately
constructed and does not establish representativeness of movie genres.

## Current shared-generator conditions

Run tag: `testbed_rerun_20260927`; register: `name_anchored`; measured seed: **2026**.
The 2027 and 2028 queue cells are not evidence of completed multi-seed experiments.

| Paper row | CLI variant | Budget / result cell |
|---|---|---|
| Text-frame | `frame_text` | 3 / `frame_text_b3__s2026` |
| Seg+uniform | `seg_uniform` | 3 / `seg_uniform_b3__s2026` |
| Seg+DINO-key | `seg_dinokey` | 3 / `seg_dinokey_b3__s2026` |
| Seg+frame | `seg_framererank` | 3 / `seg_framererank_b3__s2026` |
| No memory | `none` | 0 / `none_b0__s2026` |
| Last-seen diagnostic control | `last_seen` | at most 3 / `last_seen_b3__s2026` |

Last-seen alone uses gold IDs to construct its plan; no gold pixels or descriptions
reach the generator. It is not a deployable system or matched-budget upper bound.
Native-generator systems, eight-story H3 checks, and the earlier-rollout author
audit are separate blocks and must not be pooled into this cohort's main effect.

For new evaluations, use current `main`, which retains the main generation
pipeline and adds later backend/judge support. To reconstruct the software version
used for the recorded rollouts, optionally unpack `reproduction/generation-source.tar.gz`
from the public release and follow `src/vmem_bench/testbed/README.md`. Copy the
current cohort list into that snapshot (the later checklist/renderers are not part
of the historical generation revision).
Select the exact cohort rather than `all`:

```bash
mapfile -t paper_stories < docs/cvpr27_story_ids.txt
PYTHONPATH=src python -m vmem_bench.testbed.run \
  --variant seg_dinokey --budget 3 --backend flux2_wan22 --seed 2026 \
  --run-tag cvpr27_reproduction --out-root ./outputs/cvpr27_reproduction \
  --stories "${paper_stories[@]}"
```

## Scoring and snapshot layout

The paper's local result index is:

```text
results/trackB/testbed_rerun_20260927/
  gen/<variant>_b<budget>/<story>/name_anchored/testbed_rerun_20260927/seed_2026/
  view/<cell>/<story>/review/segments/seg_<gold-id>.mp4
  scores/<cell>/<story>/score.json
```

Gold uses `seg_001` for the first segment; original generated clips use
`seg_000`. Score the gold-numbered **view** (GT segment k -> production k-1),
not an unadjusted original segment directory. For the presence judge, use
`vmem_bench.scoring.end2end_coverage` with explicit `--gt`, `--prompts`,
`--run <gold-numbered-view>`, `--out`, `--api`, `--model`, and `--k 1`.
The pinned presence judge is Qwen3-VL-32B-Instruct, one call per segment,
temperature 0; the second judge is InternVL3.5-30B-A3B on the same views.
Exact scorer inputs, identity calibration, parse-error handling, and model
installation are documented in `docs/benchmark/trackB_scoring.md`.
Bootstrap uses 2,000 paired story draws, seed 20260925, conditional on the one
observed rollout per story/condition. It does not measure seed variability.

## Render existing analysis snapshots

From `assets/_paper/VMem-Bench/cvpr_27_submission/`:

```bash
python3 scripts/make_trackb_main_table.py
python3 scripts/make_nolook_table.py
python3 scripts/make_main_identity_table.py
python3 scripts/make_main_quality_table.py
python3 scripts/make_main_sensitivity_table.py
python3 scripts/make_second_judge_table.py
python3 scripts/make_identity_tables.py
python3 scripts/make_quality_table.py --aggregate tables/experiments/trackb_quality_control.json
```

These renderers read saved values and intervals; they do not recreate measured
results from video. `scripts/memory_gain_bootstrap.py` is a different operation:
it reads raw saved scores through the local result index and recomputes CIs.

Metadata-only audit commands, from the same manuscript directory:

```bash
python3 scripts/summarize_story_coverage.py \
  --gold-root ../../../trackB/en/gt --prompts-root ../../../trackB/en/sut_prompts
python3 scripts/audit_paper_inputs.py \
  --run-root ../../../../results/trackB/testbed_rerun_20260927
```

The coverage script also accepts reference gold/prompt roots for byte comparison
against the pinned checkout. The input audit records file fingerprints and saved
manifest fields; it does not independently prove generator fidelity or re-audit
all adapters.

## Remaining distribution checks

Before claiming a self-contained public reproduction release, verify the actual
remote cohort, code revision, model inventory, score snapshots, and manifests
are downloadable and match these fingerprints. Preserve the same directory
contract or document a relocation. This checklist does not establish external download availability.
