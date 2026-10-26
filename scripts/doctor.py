#!/usr/bin/env python3
"""Tell the user exactly what is missing to run VMem-Bench, instead of crashing later."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

CAUSAL = (
    Path(__file__).resolve().parent
    / "evaluate_baselines"
    / "trackA"
    / "baseline_adapters"
    / "causal"
)
sys.path.insert(0, str(CAUSAL))
from _local_roots import BENCH_ROOT, default_datasets_root, expand_dataset_root, find_memstrata_src  # noqa: E402

BBB_GOLD = BENCH_ROOT / "assets" / "trackA" / "BlenderOpenMovies" / "big_buck_bunny"


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _bad(msg: str) -> None:
    print(f"  MISSING  {msg}")


def main() -> int:
    print("VMem-Bench doctor")
    failed = 0

    if shutil.which("ffmpeg") is None:
        _bad("ffmpeg not on PATH")
        failed += 1
    else:
        _ok(f"ffmpeg={shutil.which('ffmpeg')}")

    if not (BBB_GOLD / "gold" / "entity_registry.json").is_file():
        _bad(f"bundled BBB gold at {BBB_GOLD}")
        failed += 1
    else:
        _ok(f"BBB gold={BBB_GOLD}")

    try:
        src = find_memstrata_src()
        _ok(f"MemStrata src={src}")
    except FileNotFoundError as exc:
        _bad(str(exc))
        failed += 1

    datasets = default_datasets_root()
    video_dir = expand_dataset_root("${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos") / "big_buck_bunny"
    vids = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")) + list(video_dir.glob("*.mkv"))
    if vids:
        _ok(f"BBB video={vids[0]}")
    else:
        _bad(f"BBB video under {video_dir}")
        print("         fix: bash scripts/prepare_blender.sh")
        print("         other films / LSMDC: see docs/DATA.md")
        failed += 1

    root = os.environ.get("PUBLIC_MODELS_ROOT", "")
    if not root:
        _bad("PUBLIC_MODELS_ROOT unset (needed for Track A perception / GPU generator, not for CPU MemStrata demo)")
        print("         export PUBLIC_MODELS_ROOT=/path/to/hf-style-models")
        print("         then: huggingface-cli download facebook/sam3 --local-dir $PUBLIC_MODELS_ROOT/facebook/sam3")
        print("               huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir $PUBLIC_MODELS_ROOT/facebook/dinov3-vitb16-pretrain-lvd1689m")
        failed += 1
    else:
        sam3 = Path(root).expanduser() / "facebook" / "sam3"
        if sam3.is_dir():
            _ok(f"SAM3={sam3}")
        else:
            _bad(f"SAM3 weights at {sam3}")
            print("         huggingface-cli download facebook/sam3 --local-dir \"$PUBLIC_MODELS_ROOT/facebook/sam3\"")
            failed += 1

    print()
    if failed:
        print(f"doctor: {failed} item(s) missing.")
        print("CPU path that never needs videos or weights:")
        print("  cd ../MemStrata && bash scripts/memstrata/cpu_demo.sh")
        return 1
    print("doctor: OK — Track A smoke:")
    print("  bash scripts/run_tracka_smoke.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
