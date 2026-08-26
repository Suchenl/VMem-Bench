# AGENTS.md · VMem-Bench

This directory **is the benchmark**. Its only job is to measure systems fairly.
Any shortcut that helps one system, or fits the eval data, invalidates every
number produced here. The rules below are hard prohibitions, not preferences.

## Fairness Contract (MUST)

1. **Identical inputs to every SUT.** For a given case + register/mode, every
   system-under-test — all baselines *and* MemStrata — MUST receive the
   byte-identical prompt stream and the identical video segment / timeline.
   No SUT may receive an input another SUT does not.
2. **No test-set fitting.** Adapter/method code MUST NOT contain anything
   derived from the evaluation content: no hardcoded entity lexicons, no
   entity→type maps, no lists of nouns/names that appear in the prompts or gold,
   no `story_id`/case-name branches, no thresholds tuned per case.
3. **Method fidelity.** An adapter may only expose capabilities the SUT actually
   has in its own paper/spec. It MUST NOT add, inside the adapter, a capability
   the method does not have. (Example of a violation: giving MemStrata a
   "parse entity names + guess type from the prompt" module, when the method
   paper states *"Naming is never inferred by the proposer."*)
4. **Symmetric preprocessing.** Any preprocessing (name parsing, entity
   detection, grounding, frame-sampling policy, …) offered to one SUT MUST be
   offered in the same form to all SUTs, or to none. Shared, neutral
   preprocessing belongs on the bench side, given identically to everyone —
   never hidden inside one SUT's adapter.
5. **No gold leakage.** No SUT ever sees gold (name rosters, `entity_registry`,
   gold frames/crops, answers). Gold is for bench-side scoring and timeline
   only; verify the adapter path never forwards it to a SUT.

## Prohibited — any of these invalidates the run

- Hardcoded entity word lists / CJK entity→type dicts / gt-derived noun lists in
  any adapter or method used for scoring.
- Special-casing a specific case/`story_id`, or per-case tuned parameters.
- Giving one SUT extra input, extra parsing, or extra hints not given to all.
- Importing, replicating, or reconstructing gold/answers inside an adapter.
- Presenting a config as the reported default while it depends on eval-fitted
  logic (e.g. the retired `name_source="deterministic"` extractor).

## Required audit before reporting ANY number

- `rg` the adapters for hardcoded entity strings, `frozenset` lexicons, CJK
  maps, and `story_id`/case branches — must be empty.
- Confirm all SUTs receive byte-identical inputs for each case+mode.
- Confirm no gold field reaches any SUT.

## Import boundary (self-containment)

VMem-Bench is self-contained: `src/vmem_bench` imports only `vmem_bench`. The
only sanctioned cross-boundary imports are baseline/SUT adapters under
`scripts/evaluate_baselines/`, which may import their own method as a black box.
`assets/` holds data only (videos, prompts, gt) — never adapter or method code.
