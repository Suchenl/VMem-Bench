# Track A gold (samples in git)

Track A measures **visual-memory retrieval** on a movie timeline. The causal protocol (not gold-replay) is in [`docs/trackA.md`](../../docs/trackA.md).

Each movie directory holds text gold only:

| File | Role |
|---|---|
| `gold/entity_registry.json` | Entity ids, names, kinds |
| `gold/chunk_annotations.json` | Per-chunk prompt, `present`, first appearances |
| `gold/chunk_index.json` | Chunk time spans (`seconds_span`) |

**No videos, no gold crops.** Point the runner at your own Blender / LSMDC files.

Full gold (all movies) lives on Hugging Face: [Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench) (`trackA/`). Git keeps copies for tests and local scoring smoke.
