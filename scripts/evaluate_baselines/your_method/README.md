# Evaluate your method on VMem-Bench (BYOM template)

This folder is a **copy-me starting point** for putting *your own* memory / long-video
method through VMem-Bench, on the same footing as MemStrata and every baseline. It is
stdlib-only and imports nothing from `vmem_bench` — you wire your method to the bench
purely by emitting the artifact files the scorers read.

> Read `docs/benchmark/running_eval.md` §0–§1 first. The three iron rules (gold is
> S4 text only; reference images are **SUT-produced**, the bench never hands you
> pixels; the SUT never sees `present`/`first_appearances`/roster) are what make the
> numbers meaningful. Breaking any of them invalidates your result.

## The contract in one table

| Phase | Bench → your method | Your method → bench |
|---|---|---|
| compose context | this chunk's **prompt text** | a set of **reference images** you recalled from your memory |
| observe memory | this chunk's **real segment video** (stands in for your generator's pixels) | *(nothing; you update your own memory)* |

Strict order per chunk: **`compose` (recall from current memory) → `observe` (write
the real segment into memory).** Never look at chunk *t*'s video while composing *t*.

## Files

| File | What it is |
|---|---|
| `sut_interface.py` | `Segment` / `Reference` / `Method` contract + writers for the exact scorer artifacts. Copy as-is. |
| `example_method.py` | `RecentFramesMethod`: a trivial GPU-free recency baseline. **Replace with your method.** |
| `run_tracka_example.py` | Drives the Track A loop over a gold movie and writes `visual_selections/<system>.json`. |

## 60-second CPU dry run (no GPU, no source video, no VLM)

```bash
cd VMem-Bench
python3 -m pip install -e ".[dev]"
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
    --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

This runs the full compose→observe loop and writes a schema-valid
`benchmark_run/visual_selections/your_method-recency.json` (placeholder frames, since
you have no source video yet). It proves your plumbing before you spend a GPU. The CI
test `tests/test_evaluate_your_method_template.py` validates that output through the
real scorer reader, so the schema can't drift.

## Wire in your method

Implement the two hooks in a class that satisfies `Method`:

```python
from sut_interface import Method, Reference, Segment

class MyMethod:
    def compose(self, seg: Segment) -> list[Reference]:
        # recall from YOUR memory (built by prior observes). Return the images you
        # would condition your generator on. Empty is fine when you have nothing.
        return [Reference(crop_abspath="/abs/path/to/frame.png"), ...]

    def observe(self, seg: Segment) -> None:
        # your perception + memory write over seg.video_path (detect/crop/embed/store)
        ...
```

Each `Reference` is EITHER a materialized image (`crop_abspath`) OR a timestamp into
the source video (`source_seconds`) that the Track A frame materializer cuts for you.
The scorer is name-blind and judges the pixels; `entity_id` is optional bookkeeping.

## Score it (Track A — the memory headline)

Real scoring needs the **source video** (obtain per `docs/DATA.md`) and a pinned VLM
judge endpoint (`qwen3-vl-32b`, OpenAI-compatible; see `docs/benchmark/running_eval.md` §6):

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.visual_coverage \
    --movie  assets/trackA/BlenderOpenMovies/charge \
    --system your_method-recency \
    --video  /path/to/charge_source.mp4 \
    --api    http://127.0.0.1:8110 \
    --limit  5
```

Headline metrics: `recall` (continuity only), `precision`, `f1`, plus
`redundancy_*` / `selection_efficiency`. Always report alongside the noise floor
(`docs/benchmark/scoring_v2.md`).

## Track B (does memory survive into *generated* pixels)

Track B judges what your method **renders**. Produce one video per story segment and
lay out a run dir with `write_trackb_run(...)`, then:

```bash
PYTHONPATH=src python3 -m vmem_bench.scoring.end2end_coverage \
    --gt      assets/trackB/en/gt/0001_lighthouse_keeper.json \
    --prompts assets/trackB/en/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
    --run     <your_run_dir> \
    --api     http://127.0.0.1:8110
```

`segment_id`s in your run dir must match the GT segment ids. The SUT prompts are the
only text you may feed your generator; the GT (`cast`, `forbidden`, decoys) is
scorer-side only.

## Fairness checklist (self-audit before you report)

- Identical inputs to every SUT; no per-case branches or eval-derived lexicons.
- Every reference image is one **your** method produced from prior observations.
- You never read `present` / `first_appearances` / roster ids.
- Same preprocessing for all systems (or none). Compare at matched context budget.
