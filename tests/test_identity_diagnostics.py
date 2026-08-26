"""CPU-only tests for the review-only identity candidate queue."""

from vmem_bench.annotation.pipeline_track_first.consolidation import Entity, Registry, Representation
from vmem_bench.annotation.pipeline_track_first.identity_diagnostics import identity_candidates


def test_identity_candidates_rank_alias_pair_without_mutating_registry() -> None:
    registry = Registry()
    for entity_id, chunk, vector in (("char_a", 0, [1.0, 0.0]), ("char_a_02", 3, [0.9, 0.1])):
        rep = Representation(f"{entity_id}@c{chunk:03d}", chunk, f"assets/{entity_id}.jpg",
                             embedding_key=f"{entity_id}@c{chunk:03d}", qa={"grounding_score": 0.9})
        registry.entities[entity_id] = Entity(entity_id, "character", "Grey Puff", "rabbit", chunk, [rep])
        registry.embeddings[rep.embedding_key] = vector
    before = list(registry.entities)
    candidates = identity_candidates(registry)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["recommendation"] == "review_merge"
    assert candidate["body_cos"] is not None
    assert candidate["left_chunk_span"] == [0, 0]
    assert candidate["right_representative_crop"].endswith("char_a_02.jpg")
    assert list(registry.entities) == before


def _run_all() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    _run_all()
