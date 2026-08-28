# Track A gold / annotation helpers

Public entry points in this folder:

| Area | Use |
|---|---|
| `core/` | Annotation and S5 crop drivers (needs your own VLM endpoint) |
| `servers/` | Optional one-rank VLM launcher |
| `maintenance/` | Gold layout / readiness helpers |
| `compare/` | Leaderboard builders |

Cluster-specific launchers (internal job queues, node maps) are **not** shipped.
Drive Track A evaluation with:

```bash
python3 scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py --help
python3 -m vmem_bench.scoring.visual_coverage --help
```

Source videos: [`docs/DATA.md`](../../docs/DATA.md).
