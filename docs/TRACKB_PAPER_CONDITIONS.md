# Track-B paper conditions

## Current CVPR 2027 results

The current shared-generator rows use the unified `testbed_rerun_20260927`
batch at seed 2026, with all six conditions produced by clean commit
`0b41d3df8d5c2b53ecc491ad00c769db2fa24a8b`. Each condition covers the same
30 stories and 2,824 segments. See [the audited checklist](CVPR27_REPRODUCTION_CHECKLIST.md)
for the exact roster, inputs, hashes, scoring view, and table-rendering commands.
The three-seed examples and older `prod_retr_*` provenance below describe
planned coverage and historical batches, respectively; they do not describe
additional evaluated seeds in the current manuscript.

## Historical ICLR-27 conditions and parity notes

Every Track-B row of the ICLR-27 paper that runs on the shared generator (FLUX.2 Klein keyframe ->
Wan2.2-TI2V-5B-Turbo) is a condition of the benchmark's own testbed, `vmem_bench.testbed`
(package README: [`src/vmem_bench/testbed/README.md`](../src/vmem_bench/testbed/README.md)). The testbed
reads `sut_prompts` only (plus, for `last_seen`, gold to build an id-only plan) and imports no method.

## Commands

```bash
export PYTHONPATH=src PUBLIC_MODELS_ROOT=<weights root>
export VMEM_WAN22_TURBO_REPO=<upstream Wan2.2-TI2V-5B-Turbo checkout>
export VMEM_FLUX_PYTHON=<python whose diffusers has Flux2KleinKVPipeline>
export VMEM_FLUX_GPU=<g> VMEM_WAN_GPU=<g> CUDA_VISIBLE_DEVICES=<encoder gpu>
python -m vmem_bench.testbed.run --variant <variant> --budget 3 --backend flux2_wan22 \
  --seed 2026 2027 2028 --run-tag <tag> --out-root <dir> --stories all
```

| paper row | `--variant` | published artifacts (run tag) | frozen provenance (VMem-Bench / MemStrata) |
|---|---|---|---|
| text-frame | `frame_text` | `prod_retr_20260831` | `930e9be` / `a2d8193` |
| seg-uniform | `seg_uniform` | `prod_retr_20260831` | `930e9be` / `a2d8193` |
| seg-DINO-key | `seg_dinokey` | `prod_retr_20260831` | `930e9be` / `a2d8193` |
| seg-frame (clean rerun) | `seg_framererank` | `prod_retr_segframe_clean_20260921` | `930e9be` / `a2d8193` + GPU-isolation patch (= `e6fa7f7`) |
| no memory (K=0) | `none` | `prod_retr_nomem_20260925` | `9a03a1e` / `67be432` |
| last-seen references | `last_seen` (deprecated alias: `oracle`) | `prod_retr_oracle_20260925` | `972a350` / `6c14286` |

**Seeds.** The Track-B standard is 3 seeds per condition (`--seed 2026 2027 2028`). Seed S drives FLUX
with `S + shot` and Wan with `S`; the published runs are exactly seed 2026 (their keyframe composer set
`2026 + shot`, their Wan `base_seed` was 2026 — checked at every frozen MemStrata commit above).

**Encoders / GPUs.** Retrieval encoders (Qwen3-Embedding-4B, SigLIP2-base-512, DINOv3-ViT-B/16) run in the
runner process (`CUDA_VISIBLE_DEVICES`); FLUX and Wan run in resident servers on `VMEM_FLUX_GPU` /
`VMEM_WAN_GPU` (co-resident on one 80 GB card works). The encoders do not fit next to FLUX+Wan.

## What the testbed reproduces

The frozen artifacts were produced by `scripts/evaluate_baselines/trackB/baseline_runners/memstrata/run.py
--retrieval-variant <v>` through `memstrata.production.run` (MemStrata tags
`paper/iclr27-trackb-retrieval-a2d8193`, `paper/iclr27-trackb-segframe-e6fa7f7`,
`paper/iclr27-trackb-nomem-67be432`, `paper/iclr27-trackb-lastseen-6c14286`; VMem-Bench
`paper/iclr27-trackb-retrieval-930e9be`, `paper/iclr27-trackb-nomem-9a03a1e`,
`paper/iclr27-trackb-lastseen-972a350`). That path stays runnable from those tags; the testbed carries
the same behaviour:

* retrieval ranking/sampling/diversity and the generated-history store are the same code (the single
  implementation in `vmem_bench.testbed.retrieval`, checked against the frozen MemStrata
  implementation by `tests/test_testbed_retrieval_equivalence.py`, which runs when `MEMSTRATA_SRC` points
  at a checkout of tag `paper/iclr27-trackb-retrieval-a2d8193` and is skipped otherwise);
* the same last-seen rule and plan bytes (`input/last_seen_plan.json` = frozen `oracle_reference_plan.json`);
* per shot a fresh FLUX keyframe (the paper forced `recompose_keyframe` on every shot), multi-image from the
  references or text-to-image without, 832x480, 4 steps; Wan I2V from it at 704x1280 -> 480x832, 121 frames
  at 24 fps; byte-identical prompt formatting for both models.

Differences to the frozen producer: the testbed has no generation router (its suggestion was overridden by
`--force-recompose` in every paper run), no MemStrata bank/decomposer (off in those runs), and a
**selection failure aborts** the run instead of silently continuing with zero references (the original
seg-frame rollout's 1,687 retrieval CUDA OOMs); segment files are `review/segments/seg_###.mp4` as before.

## Parity gate (2026-09-27, `0001_lighthouse_keeper`, first 5 shots, seed 2026)

Scripts: `scripts/evaluate_baselines/trackB/testbed_parity/` (`run_parity.sh`, `compare.py`, `replay.py`).

| condition | keyframe type | reference ids | text-to-image keyframes | plan |
|---|---|---|---|---|
| `none` vs `prod_retr_nomem_20260925` | 5/5 equal | none (5/5) | 5/5 byte-identical | — |
| `last_seen` vs `prod_retr_oracle_20260925` | 5/5 equal | 5/5 equal | shot 0 byte-identical | identical bytes |
| `frame_text` vs `prod_retr_20260831` | 5/5 equal | shot 0 (empty) equal; shots 1–4 differ | shot 0 byte-identical | — |

Wan2.2-Turbo is not bit-deterministic: shot-0 clips generated from the byte-identical keyframe, prompt and
seed differ between any two runs (PSNR 21.5–22.8 dB, including between two frozen paper runs), so a
retrieval over the first clip can already reorder near-tied frames (text-frame shot 1: SigLIP2 scores
0.088/0.087/0.084 on the frozen clip). On each clip the testbed and the MemStrata code select identically,
and replaying the testbed's selection on the frozen runs' own clips (`replay.py`, teacher forcing) reproduces
every frozen reference id:

| replay on `0001_lighthouse_keeper` | shots | equal reference ids |
|---|---|---|
| `frame_text` vs `prod_retr_20260831` | 0–9 | 10/10 |
| `seg_uniform` vs `prod_retr_20260831` (`--emulate-resumes`, resumed at shots 2, 56) | 0–59 | 60/60 |
| `seg_dinokey` vs `prod_retr_20260831` (`--emulate-resumes`, resumed at shot 41) | 0–44 | 45/45 |
| `seg_framererank` vs `prod_retr_segframe_clean_20260921` | 0–9 | 10/10 |

A rerun therefore reproduces the decision path given the clips, not the pixels; the published numbers are
the artifacts of the run tags above, and new results use 3 seeds.

**Frozen-artifact caveat (resume).** When a frozen retrieval run was interrupted and resumed, MemStrata
reopened the story with an empty retrieval store (clock 0; the shots closed before the resume were never
re-observed), so from that shot on its memory could not see the earlier film. Stories with at least one
resume (from `logs/memstrata_run.log`): text-frame 2/30, seg-uniform 21/30, seg-DINO-key 6/30,
seg-frame clean rerun 0/30; last-seen reads its own `progress.json` and no-memory has no store, so neither
is affected. The testbed's `--resume` re-observes every closed clip.

Scoring: `python -m vmem_bench.scoring.end2end_coverage` (`trackB-end2end-1.0`) through the +1 view (GT
`seg_k` -> production `seg_{k-1}`), plus the identity metrics of
[`docs/benchmark/trackB_scoring.md`](benchmark/trackB_scoring.md).
