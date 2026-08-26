"""Pure-logic checks for the resident-service registry + GPU placement (no GPU needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vmem_bench.services.placement import CapacityError, GpuInfo, plan_placement
from vmem_bench.services.registry import enabled_services


def _cfg(**kw):
    base = dict(use_text_embed=True, use_face=True, use_crop_classify=True,
                crop_classify_method="prototype", perception_backend="gdino_track")
    base.update(kw)
    return SimpleNamespace(**base)


def test_enabled_services_drops_unused_and_external():
    # prototype crop-QA -> no siglip service; external VLM excluded by default.
    keys = {s.key for s in enabled_services(_cfg())}
    assert keys == {"text_embed", "gdino", "dino", "face"}
    assert "vlm" not in keys and "siglip" not in keys
    # turning features off removes their services; siglip method turns siglip on.
    keys2 = {s.key for s in enabled_services(_cfg(use_text_embed=False, use_face=False,
                                                  crop_classify_method="siglip"))}
    assert keys2 == {"gdino", "dino", "siglip"}
    # include_external surfaces the VLM.
    assert "vlm" in {s.key for s in enabled_services(_cfg(), include_external=True)}


def test_fastest_one_service_per_card_and_capacity():
    specs = enabled_services(_cfg())  # 4 services
    gpus = [GpuInfo(index=i, free_mib=80000, used_mib=0, n_procs=0) for i in (5, 6, 7, 4)]
    assign = plan_placement(specs, gpus, "fastest")
    assert len(assign) == 4
    assert len(set(assign.values())) == 4  # distinct cards
    # not enough cards -> CapacityError (3 services, 2 cards).
    three = [s for s in specs if s.key in ("gdino", "dino", "face")]
    with pytest.raises(CapacityError):
        plan_placement(three, [GpuInfo(index=0, free_mib=80000, used_mib=0, n_procs=0),
                               GpuInfo(index=1, free_mib=80000, used_mib=0, n_procs=0)], "fastest")


def test_packed_binpacks_onto_fewer_cards():
    specs = enabled_services(_cfg())
    gpus = [GpuInfo(index=0, free_mib=80000, used_mib=0, n_procs=0)]
    assign = plan_placement(specs, gpus, "packed")
    assert set(assign.values()) == {0}  # everything fits on one big card
    # a tiny card cannot hold the text_embed footprint -> CapacityError.
    with pytest.raises(CapacityError):
        plan_placement(specs, [GpuInfo(index=0, free_mib=8000, used_mib=0, n_procs=0)], "packed")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
