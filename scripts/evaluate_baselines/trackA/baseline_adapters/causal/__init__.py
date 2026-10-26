"""New-protocol causal baseline adapters for MemStrata-Bench.

Modules here are run script-style (``runner.py`` puts this directory on
``sys.path`` and imports adapters by bare name), so vendored baseline checkouts
under ``baselines/Causal/<name>/`` stay pristine -- all bench glue lives here,
never inside the third-party code.

See ``README.md`` for the per-baseline hook contract and ``contract.py`` for the
:class:`CausalMemoryAdapter` interface.
"""
