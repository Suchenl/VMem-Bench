# `vmem_bench.testbed` — Track-B shared-generator testbed

The testbed is part of the benchmark: it runs every Track-B memory condition on one shared
generation pipeline, so rows differ only in which references reach the generator. Methods under test
(MemStrata, native systems) plug in through adapters under `scripts/evaluate_baselines/`; this package
never imports them.

```text
testbed/
├── retrieval/        # the ONE similarity-retrieval implementation (Track A imports it too)
│   ├── retrievers.py #   frame_text, seg_uniform, seg_dinokey, seg_framererank (+ recency, bm25_desc, random)
│   └── encoders.py   #   hash | qwen3_embedding | siglip2 | dinov3, lazy and process-cached
├── store.py          # TrackBGeneratedStore: stills from this run's own clips on a causal clock
├── controls.py       # none (budget 0) and last_seen (gold-selected own-history stills; alias: oracle)
├── runner.py         # causal closed loop: select -> realize -> observe
├── run.py            # CLI: python -m vmem_bench.testbed.run
└── backends/
    ├── base.py       # RealizationRequest / RealizationResult / RealizationBackend contract
    ├── flux2_wan22.py, flux_server.py, wan22_turbo_server.py, job_queue.py, prompt_format.py
    ├── minimax_h3/   # MiniMax-H3-Turbo Ref2V backend
    └── synthetic.py  # CPU smoke backend (solid-colour clips)
```

## Conditions

| `--variant` | paper row | references per segment |
|---|---|---|
| `frame_text` | text-frame | top-K frames of the own history by SigLIP2 text→frame similarity |
| `seg_uniform` | seg-uniform | top segments by Qwen3-Embedding text similarity, uniform 1 fps stills |
| `seg_dinokey` | seg-DINO-key | top segments, DINOv3 farthest-point keyframes (3/segment at 2 fps) |
| `seg_framererank` | seg-frame | RRF of segment rank and SigLIP2 frame rank |
| `none` | no memory (K=0) | none: every keyframe is FLUX text-to-image |
| `last_seen` | last-seen references | gold-selected control: middle frames of the own shots where the needed entities were last seen |

`--budget` (default 3) caps the references; `none` forces 0. `last_seen` reads Track-B gold to build an
id-only plan (`input/last_seen_plan.json`, same rule and bytes as the frozen
`oracle_reference_plan.json`); it is a control, never a system under test.

## Running

```bash
export PYTHONPATH=src PUBLIC_MODELS_ROOT=<weights root>
export VMEM_WAN22_TURBO_REPO=<upstream Wan2.2-TI2V-5B-Turbo checkout>
export VMEM_FLUX_PYTHON=<python whose diffusers has Flux2KleinKVPipeline>
export VMEM_FLUX_GPU=0 VMEM_WAN_GPU=0 CUDA_VISIBLE_DEVICES=1   # FLUX+Wan on GPU 0, retrieval encoders on GPU 1
python -m vmem_bench.testbed.run --variant frame_text --budget 3 --backend flux2_wan22 \
  --seed 2026 --run-tag trackb_testbed --out-root <dir> --stories all
```

* `--seed`: the main paper measured **2026 only**, one rollout per story/condition.
  Additional seeds (e.g. `--seed 2026 2027 2028`) are a prospective robustness protocol, not completed paper evidence. For `flux2_wan22` the FLUX keyframe of segment *i* uses `seed + i` and Wan uses `seed` (the paper's
  keyframe composer used `2026 + shot` and Wan `base_seed = 2026`).
* `--stories all` or story ids; `--segments N` limits to the first N segments (probes only).
* `--backend-opt key=value` passes backend constructor options (e.g. `flux_gpu=0`, `wan_min_free_mib=40000`).
* `--resume` continues after the closed prefix of `progress.json`, re-observing the closed clips so the
  store equals the uninterrupted run; `--overwrite` restarts.
* CPU smoke: `--backend synthetic --text-provider hash --frame-provider hash --keyframe-provider hash`.

## Causality and failure policy

The store only ever holds clips this run produced, appended after each segment is realized; the
retrievers additionally gate on the causal clock. Before realizing, the runner checks every reference
is a still under this run's directory from a strictly earlier segment (`CausalityError` otherwise). The
backend receives only the prompt, the reference paths, the seed and an output directory.

A backend failure skips that segment (recorded `status: failed`; three in a row abort), as the paper
producer did. A selection failure (e.g. an encoder CUDA OOM) aborts the run instead of silently
degrading to zero references — the frozen producer swallowed those, which is how the original seg-frame
rollout ended up with 1,687 empty retrievals; fix and `--resume`.

## Artifacts (scorer-compatible)

`<out_root>/<variant>_b<budget>/<story>/<register>/<run_tag>/seed_<seed>/`

| path | content |
|---|---|
| `review/segments/seg_###.mp4` | clip of production shot ### (0-indexed = GT `seg_{###+1}`) |
| `review/keyframes/seg_###.png` | keyframe handed to the video model (flux2_wan22) |
| `progress.json` | per segment: `references` (id, source segment, still, score, arm), `seed`, `backend_meta` (keyframe mode, FLUX/Wan seeds, prompts, timings), `timing_s` {select, realize, observe}, `video`, `status` |
| `run_manifest.json` | config, backend, seed, gold use, VMem-Bench commit, wall time |
| `input/` | prompt stream copy; `last_seen_plan.json` / `last_seen_audit.json` for last_seen |
| `retrieval_read/` | materialized stills (`retrieval_frames/c#####_<sec>.png`, `last_seen_frames/`) |

Score directly: `python -m vmem_bench.scoring.end2end_coverage --run <run dir> ...` (0-indexed runs use the
+1 view: GT `seg_k` -> `seg_{k-1}`), and the identity scorer with
`clip_template = <run dir>/review/segments/seg_{idx:03d}.mp4`, `file_index_offset: "auto"`.

## Parity with the frozen paper runs

`scripts/evaluate_baselines/trackB/testbed_parity/` runs a condition on the first shots of a story and
compares it shot by shot with supplied frozen artifacts. Those video artifacts are not bundled in the code repository; see `docs/benchmark/trackB_scoring.md` for the scoring contract.
