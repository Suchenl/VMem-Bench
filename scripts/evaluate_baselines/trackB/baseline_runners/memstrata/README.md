# MemStrata Track-B adapter (method under test)

`run.py` runs **MemStrata as a system under test** on the Track-B prompt streams: it wraps
`sut_prompts` into a prompt-only screenplay and invokes `memstrata.production.run` in bench mode
(MemStrata's own read path, write path and memory). It is the only place that knows both packages.

The shared-generator **testbed conditions** (text-frame, seg-uniform, seg-DINO-key, seg-frame, no
memory, last-seen) are no longer run through MemStrata. They belong to the benchmark and run with
`python -m vmem_bench.testbed.run` (see `src/vmem_bench/testbed/README.md` and
`docs/TRACKB_PAPER_CONDITIONS.md`).

`--retrieval-variant` was removed here and in MemStrata `main`; passing it exits with a pointer to the
testbed. The frozen `prod_retr_*` artifacts stay re-runnable from the MemStrata and VMem-Bench provenance
tags listed in `docs/TRACKB_PAPER_CONDITIONS.md`.
