"""Track-B shared-generator testbed.

The testbed is part of the benchmark, not of any method: it holds the similarity-retrieval memories,
the no-memory and last-seen controls, the causal closed-loop runner, and the realization backends
(generation pipelines) every condition shares. Methods under test plug in through adapters under
``scripts/evaluate_baselines/`` instead.
"""
