# Track B `gt_source` → English translation contract

You are translating hand-authored Chinese story sources into **full English**, in
place, one file at a time. The English files rebuild the English Track B benchmark
through two deterministic scripts, so **structure is sacred**: if you change the
skeleton or break a cross-reference, the build crashes.

## Files

- Reference (Chinese, read-only): `../zh/gt_source/<story>.json`
- Target (write here, English): `en/gt_source/<story>.json`
  (currently an identical Chinese copy — overwrite it with the English version)

## What you TRANSLATE (human / SUT-visible free text → natural English)

- `title`, `premise`, `_comment`
- `scenes[].setting`
- `scenes[].actions[]` — the narrative string, or the `"action"` field of a dict item
- `entities[].name`
- `entities[].states{}` **values** (the appearance descriptions)
- `entities[].appearance`
- `lookalike_pairs[].features{}` **values**, and `lookalike_pairs[].note`
- `present[].anchor.phrase`
- `entities[].states{}` **keys** and every state name (see "State names" below)

## What you MUST keep byte-identical to zh (identifiers / structure)

- every `eid` string (`E1`, `E2`, …) wherever it appears
  (`present` lists, `lookalike_pairs[].pair`, `features` keys, `events[].eid`,
  `anchor.resolves_to`, `confusable_with`)
- `story_id`, scene `id` (`S1`, `S2`, …) and scene ORDER
- all numbers: `count`, `segment_sec`, `gap_long_threshold`,
  `avoidance_probe_window`, `probe_target_default`
- all enums: `kind` (character/prop/location), `events[].type`,
  `events[].reason`, `events[].shown`, `anchor.type`
- Do NOT add, delete, reorder, split, or merge any list item, entity, scene, or
  action. Same counts everywhere.

## State names (the one tricky part — full English requires CONSISTENT remap)

State names are dict KEYS **and** are referenced elsewhere. You must remap each
state name to the SAME English label in **all four** places, and keep order:

1. `entities[<eid>].states` keys  (e.g. `"青壮探险"` → `"prime_explorer"`)
2. `state_machines[<eid>]`         (same list, same order, same English labels)
3. `scenes[…].actions[…].events[].to`     (must be one of that entity's state labels)
4. `scenes[…].actions[…].present[].state` (must be one of that entity's state labels)

Use short snake_case-ish English state labels (e.g. `intact`, `shattered`,
`reassembled`, `dead`, `weathered_burn`). They are internal ids; the SUT never
sees them. Just be 100% consistent.

## Entity name ↔ action rule (critical — or the SUT loses the appearance)

The build injects an entity's appearance right after the entity's `name` the first
time it is introduced (and on state change). Therefore:

- Pick ONE English `name` per entity and use that EXACT string verbatim inside the
  `introduce` and `transform` action(s) for that entity.
  - e.g. `name: "Hasan the guide"` → the introducing action must literally contain
    `"Hasan the guide"`. Later recall actions may use just `"Hasan"`.
- Transliterate Chinese personal names consistently (艾拉→Aida, 哈桑→Hasan,
  娜迪娅→Nadia, 莎娜→Sana) and use the same spelling every time.

## Style

Natural, concise cinematic English. Faithful to the Chinese meaning. Consistent
terminology within a story (same prop/location always worded the same way).

## Self-check — REQUIRED before you finish each file

Run, from `benchmarks/VMem-Bench/assets/trackB/`:

```
python3 scripts/validate_en_source.py --story <story_id>
```

It compares your file to the Chinese skeleton, checks state-reference integrity,
verifies zero CJK remain, and actually runs the build. Fix every reported error and
re-run until it prints `PASS <story_id>`. Do not move to the next file until the
current one PASSES.
