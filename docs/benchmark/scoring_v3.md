# Track A visual-coverage 3.0

Status: production default after the frozen speed/effect pilot passed.
Implementation: `src/vmem_bench/scoring/visual_coverage.py`.

## Capability boundary

Version 3.0 measures whether each reference selected by the SUT visually covers
entities that the frozen gold says are present in the current segment. The
benchmark judge receives:

1. the gold-present entity roster;
2. exactly one unlabeled historical reference image.

It does not receive the current segment video or segment prompt. Those inputs
made the v2 judge jointly re-decide target presence and reference identity, and
mixed target-video frame indices with reference indices. Gold remains entirely
bench-side and is never forwarded to the SUT.

## Judge contract

Each reference is an independent request with `temperature=0` and an
OpenAI-compatible strict JSON-schema response format. The only accepted payload
is:

```json
{"entity_ids":["char_001","loc_001"]}
```

`entity_ids` must be a unique subset of the supplied roster. Empty means that
the reference does not visually support any gold-present entity. A single image
may cover multiple entities, including a visible character and a recognizable
location. The scorer performs no syntax repair and does not accept prose or
code fences.

The covered set is the union of validated per-reference labels. `missing` is
derived deterministically as `gold_present - covered`; the judge never authors
it.

## Metrics

For a segment with `n` selected references:

- precision is the fraction whose `entity_ids` is non-empty;
- continuity recall is the fraction of continuity entities in the covered set;
- `recall_all` applies the same formula to all gold-present entities;
- F1 is the harmonic mean of precision and continuity recall.

Multi-label references do not have a unique per-entity partition. Consequently
`redundancy_vlm`, `redundancy_sim`, and `selection_efficiency` are `null` in
3.0 rather than silently assigning a multi-label image to one arbitrary entity.
A future redundancy contract requires its own preregistered validation.

## Execution and caching

`--ref-workers` bounds independent reference requests inside a segment.
`--workers` controls segment concurrency. The endpoint pool remains the global
in-flight cap, so configure `--endpoint-slots` no higher than the serving
replica's `max_num_seqs`.

v3 enables a run-local content-addressed cache by default; `--judge-cache PATH`
selects an explicit shared location. Its key covers
the contract version, model, exact prompt, downscaled image payload, structured
response schema, and sampling temperature. Cache records are accepted only
after the response passes the same strict validator as a live call.

The result summary records `metric_version`, logical live judge requests, cache
hits, and per-segment scoring latency. Reference results are folded back in
original selection order regardless of completion order.

## Compatibility

`visual-coverage-3.0` is the CLI and Stage-2 service default. Replay a legacy
v2.2 artifact explicitly with:

```bash
--metric-version visual-coverage-2.2
```

Existing v2 artifacts are not rewritten. A v3 result must retain
`metric_version: visual-coverage-3.0` in `score.json`.
