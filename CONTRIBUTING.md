# Contributing to VMem-Bench

Thanks for improving VMem-Bench. Keep `vmem_bench` self-contained: method
implementations belong in external SUT adapters and must not be imported by
the benchmark package.

## Before opening a pull request

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest -q
```

For a method or scorer change, run the CPU BYOM example in
[`scripts/evaluate_baselines/your_method/README.md`](scripts/evaluate_baselines/your_method/README.md).
Do not commit source videos, model weights, generated outputs, credentials, or
machine-specific absolute paths.

## Protocol and documentation

Preserve the fairness contract: identical SUT inputs, no test-set fitting, no
gold leakage, and causal `compose` before `observe`. Update `README.md`,
`README.zh.md`, and the relevant protocol document when a user-facing workflow
changes. Keep `paper-reproduction` frozen; new benchmark behavior belongs on
`main` unless it is explicitly backported without changing the frozen protocol.
