# Public release contents

This is a curated export with a fresh Git history. The generation source archive
is an immutable generated artifact; the current working implementation is
`src/vmem_bench`, including the main shared-generator pipeline and subsequent
H3/second-judge support. Historical provenance is not a missing-feature claim.
`RELEASE_MANIFEST.json` records the development source commit and SHA256 of each
exported file, plus the files in the frozen source archive. A source commit by
itself is not a public download endpoint.

## Included

- Benchmark protocol, annotation tools, scorers, shared-generator testbed, method
  adapters, tests and lightweight interface example.
- Tracked text gold and Track-B authoring pool; `docs/cvpr27_story_ids.txt` identifies
  the 30-story evaluated cohort. The one-sample `freeze_manifest.json` does not
  certify the whole cohort.
- Paper table analysis snapshots and renderers under
  `assets/_paper/VMem-Bench/cvpr_27_submission/`. This directory contains results
  and analysis tools, not the manuscript, old submissions or rebuttal material.
- `reproduction/generation-source.tar.gz`: code and Track-B inputs from generation
  commit `0b41d3df8d5c2b53ecc491ad00c769db2fa24a8b`.

## Not included

- Original film frames/video, official LSMDC CSV/AD, model weights.
- Raw generated rollouts, per-segment score bundles and local result symlinks.
  Frozen aggregate JSON supports table rendering, not an independent rerun of
  the original judgments or statistical computation without the raw inputs.
- Development history, internal planning, old submissions, reviews and rebuttals.

The old `paper-reproduction` branch name refers to an earlier Track-A Stage-1
freeze. That branch is not present in this fresh repository and must not be
advertised as reproducing the current Track-B tables. No new multi-seed or
backend experiment was performed for release preparation.

## Reuse and maintenance

For the public checkout, follow [REPRODUCE.md](../REPRODUCE.md) and the
[paper checklist](CVPR27_REPRODUCTION_CHECKLIST.md). The source archive is included
so reproducing the recorded generation revision does not require private-repo access.
It preserves the historical implementation, including its known limitations.
New runs on the moving working implementation must record their own commit,
configuration, cohort, model versions and seed.

Maintainers generate exports using the development repository's
`scripts/packaging/export_public.py` and its explicit allowlist. Generate into a
fresh directory, review the diff against the public checkout, then commit the
approved changes there. Do not manually maintain a second implementation in an
export tree or overwrite collaborators' public changes with an unchecked export.
