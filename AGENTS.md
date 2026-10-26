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

## Paper editing and visual preservation (MUST)

These rules apply to every agent editing manuscripts, teasers, figures, tables,
captions, or LaTeX layout in this repository, including
`assets/_paper/VMem-Bench/`.

1. **Preserve existing visual work by default.** Do not redesign, replace,
   shrink, expand, move to supplementary material, or delete an established
   figure or table merely because the paper's title, positioning, narrative,
   or venue changes. Preserve the information content, visual hierarchy,
   typography, figure dimensions, placement, and core results unless there
   is a documented substantive reason to change them.
2. **No large-scale visual changes without rendered verification.** Before
   materially restructuring existing figures, tables, or the page layout,
   compile the affected manuscript and inspect the rendered PDF **page by
   page**, including the first page, float placement, readability at actual
   paper size, whitespace, clipping, table density, cross-references, and
   page limits. Compare the proposed rendering directly against the
   previous rendering. Source-only inspection, raw TeX diffs, and successful
   compilation are **not** sufficient to claim visual improvement.
3. **If visual verification is unavailable, do not proceed with sweeping
   layout edits.** Limit changes to targeted textual/scientific corrections
   or clearly necessary, reversible, localized fixes. State the verification
   limitation honestly; do not assert that the manuscript looks better.
4. **Separate scientific revision from visual redesign.** First improve
   research questions, evidence interpretation, and prose while keeping
   established visual assets intact. Only change their presentation when
   the revised argument genuinely requires it, and retain all important
   diagnostic evidence rather than silently hiding full tables or
   qualitative figures in the appendix.
5. **After every approved layout change, audit the complete artifact.**
   Check that table-generation scripts cannot overwrite restored layouts,
   that all graphics and references still resolve, that figures and tables
   remain close to the relevant discussion, and that no data, baselines,
   or caveats were removed merely to create a cleaner-looking page.
   Commit source changes separately from regenerated PDF artifacts where
   practical; do not describe a PDF as current until it has been rebuilt.

The default rule is **change the argument without damaging the visual
presentation**. Any substantial redesign requires evidence from the
rendered pages, not an agent's unverified aesthetic judgment.

## Anonymous CVPR and public arXiv manuscripts

The CVPR submission stays anonymous. Public author details, Code/Dataset/Project
Page URLs belong only to the separately generated arXiv copy. Add public resource
links through `export_arxiv.py`; never insert them into the canonical CVPR abstract.

## Development and public release

The canonical editable checkout is the private VMem-Bench-dev repository.
The public VMem-Bench repository has a fresh history and a curated file set.
Generate a public candidate into an empty directory outside this checkout using
`scripts/packaging/export_public.py`; selection is defined by public_release.json.
Review public collaborator changes before updating a release. Do not copy code
changes back from generated exports or edit frozen source archives in place.
