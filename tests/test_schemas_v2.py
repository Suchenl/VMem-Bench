"""Round-trip self-check for the v2 contracts (docs/design/bench/schemas_and_contracts.md)."""

import json

from vmem_bench.common.schemas import (
    ChunkAnnotation, ChunkAnnotations, ComposedContextRecord, Entity, EntityRegistry,
    ForbiddenRep, GoldInstruction, Instruction, InstructionItem, Observation,
    ObservationPacket, PromptPacket, Representation, SelectedAsset, StateEvent,
)


def _roundtrip(obj):
    cls = type(obj)
    return cls.from_dict(json.loads(json.dumps(obj.to_dict())))


def test_roundtrip_all_contracts():
    rep = Representation(representation_id="prop_apple_01@c003", chunk_id=3,
                         crop_path="chunks/crops/a.jpg", bbox=[10, 20, 500, 600],
                         bbox_source="grounding_dino", frame_index=1234,
                         embedding_key="prop_apple_01@c003",
                         qa={"verified": True, "rounds": 1, "flagged": False})
    evt = StateEvent(event_id="evt_apple_eaten", chunk_id=6,
                     description="the apple is eaten", deprecates=[rep.representation_id])
    ent = Entity(entity_id="prop_apple_01", kind="prop", name="Red Apple",
                 description="a shiny red apple", first_chunk=3,
                 representations=[rep], state_events=[evt])
    registry = EntityRegistry(movie_id="big_buck_bunny", entities=[ent])
    ann = ChunkAnnotation(chunk_id=7, shot_span=[12, 14], frame_span=[3021, 3140],
                          prompt="…", present=["prop_apple_01"], first_appearances=[],
                          gold_instructions=[GoldInstruction("prop_apple_01", "continuity")],
                          forbidden=[ForbiddenRep(rep.representation_id, "evt_apple_eaten")],
                          scenario_tags=["state-change"])
    record = ComposedContextRecord(
        chunk_id=7,
        selected=[SelectedAsset("prop_apple_01", [rep.representation_id], "subject", "required")],
        instruction=Instruction(per_asset=[InstructionItem("prop_apple_01", "continuity")],
                                exclusions=[rep.representation_id]),
        memory_keys=["prop_apple_01"])
    packet = ObservationPacket(chunk_id=7, chunk_video="chunks/chunk_007.mp4",
                               observations=[Observation("prop_apple_01", "prop", "Red Apple",
                                                          rep.representation_id, rep.crop_path)],
                               state_events=[evt])
    for obj in (rep, evt, ent, registry, ChunkAnnotations("big_buck_bunny", [ann]),
                record, packet, PromptPacket(7, "…")):
        assert _roundtrip(obj) == obj, type(obj).__name__

    # No inline embeddings anywhere (principle #10)
    assert "embedding" not in json.dumps(registry.to_dict()).replace("embedding_key", "")


if __name__ == "__main__":
    test_roundtrip_all_contracts()
    print("schemas v2 roundtrip OK")
