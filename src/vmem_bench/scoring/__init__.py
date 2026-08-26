"""VMem-Bench scoring package (new causal protocol).

The old gold-replay / ID-fidelity harness (``runner``/``metrics``/``visual`` + the ``__main__``
CLI, plus the ``vmem_bench.benchmark_run`` v1 embedder orchestration) was removed with the
gold-replay protocol. Track A now scores retrieved-memory coverage with a VLM judge:

* :mod:`vmem_bench.scoring.visual_coverage` — per-system VLM visual-coverage scorer (Track A).
* :mod:`vmem_bench.scoring.end2end_coverage` — Track B end-to-end generated-video judge. Reads
  the per-segment GT (``gt_version trackB-gt-2.0``) built by ``assets/trackB/complete_gt.py``
  and scores per-capability (memory abilities) + a gap-stratified recall decay curve.
* :mod:`vmem_bench.scoring.embedder` — pinned image embedders (DINOv3 / ArcFace / MegaLoc),
  reused by the scorer and by the self-contained retrieval baselines.

Retired: ``scoring._archive.trackb_gt`` — the old screenplay-derived, per-shot GT exporter.
Track B GT is now hand-authored under ``assets/trackB/gt_source/`` and compiled by
``assets/trackB/complete_gt.py``; SUT prompts by ``assets/trackB/get_sut_prompts.py``.

Import the submodule you need directly; this package intentionally re-exports nothing.
"""
