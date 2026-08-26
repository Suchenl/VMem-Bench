# Optional dump hooks (NOT Track A)

Track A is online gold-replay (`docs/baselines/track_a.md`). Do **not** start here for
main-table numbers.

If you run a vendor's **official** inference for Track B / debugging and want to
import a selection dump into `*_dump` converters, use the schema below. Keep
hooks local to your run scripts; do not commit patches into vendor checkouts.

## Dump schema

```json
[
  {
    "chunk_id": 3,
    "selected": [
      {"source_chunk_id": 1, "frame_index": 12, "raw_ref": "kv_block_1"},
      {"block_index": 7, "raw_ref": "ltm_tile_7"}
    ]
  }
]
```

| dump converter | filenames |
|---|---|
| memflow_dump | `memflow_memory_dump.json`, `memory_dump.json` |
| decmem_dump | `decmem_memory_dump.json`, `ltm_topk_dump.json`, `memory_dump.json` |
| helios_dump | `helios_memory_dump.json`, `history_dump.json`, `memory_dump.json` |
| longlive_rag_dump | `*_memory_log.json` |
| iamflow_dump | `mapping_*.json` |

```bash
python -m vmem_bench.baseline_adapters.convert \
  --baseline memflow_dump --export-dir <dump_dir> --out /tmp/ev
```
